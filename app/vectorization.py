import io
import json
import uuid
import pandas as pd

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .database import AsyncSessionLocal, async_engine, Base, embeddings, COLLECTION_NAME
from .models import Product

REQUIRED_COLUMNS = ["id", "name", "description", "price", "category", "image_url"]


async def _ensure_collection(conn) -> str:
    """Return the UUID of the named collection, creating it if needed."""
    row = await conn.execute(
        text("SELECT uuid FROM langchain_pg_collection WHERE name = :name"),
        {"name": COLLECTION_NAME},
    )
    existing = row.fetchone()
    if existing:
        return str(existing[0])
    new_uuid = str(uuid.uuid4())
    await conn.execute(
        text("INSERT INTO langchain_pg_collection (uuid, name, cmetadata) VALUES (:uuid, :name, :meta)"),
        {"uuid": new_uuid, "name": COLLECTION_NAME, "meta": "{}"},
    )
    return new_uuid


async def _ensure_pgvector_tables(conn) -> None:
    """Create langchain PGVector tables if they don't exist yet."""
    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS langchain_pg_collection (
            uuid UUID PRIMARY KEY,
            name VARCHAR NOT NULL,
            cmetadata JSON
        )
    """))
    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
            id UUID PRIMARY KEY,
            collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
            embedding vector,
            document TEXT,
            cmetadata JSON
        )
    """))


async def vectorize_products(file_content: bytes, filename: str) -> dict:
    """Ingest a CSV: upsert products into PostgreSQL and embed into pgvector."""
    df = pd.read_csv(io.BytesIO(file_content), dtype={"id": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Ensure SQLAlchemy ORM tables exist
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Upsert products
    async with AsyncSessionLocal() as db:
        for _, row in df.iterrows():
            stmt = pg_insert(Product).values(
                id=str(row["id"]),
                name=str(row["name"]),
                description=str(row["description"]),
                price=float(row["price"]),
                category=str(row["category"]),
                image_url=str(row["image_url"]),
            ).on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "name": str(row["name"]),
                    "description": str(row["description"]),
                    "price": float(row["price"]),
                    "category": str(row["category"]),
                    "image_url": str(row["image_url"]),
                },
            )
            await db.execute(stmt)
        await db.commit()

    # Build chunked documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs: list[Document] = []
    for _, row in df.iterrows():
        text_ = f"{row['name']} {row['description']}"
        chunks = splitter.create_documents(
            [text_],
            metadatas=[{"product_id": str(row["id"]), "category": str(row["category"])}],
        )
        docs.extend(chunks)

    # Embed via Cohere and write directly to pgvector tables (bypasses PGVector ORM)
    texts = [d.page_content for d in docs]
    vectors = await embeddings.aembed_documents(texts)

    async with async_engine.begin() as conn:
        await _ensure_pgvector_tables(conn)
        collection_id = await _ensure_collection(conn)
        for doc, vector in zip(docs, vectors):
            emb_literal = "[" + ",".join(str(float(v)) for v in vector) + "]"
            await conn.execute(
                text("""
                    INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
                    VALUES (:uid, :cid, CAST(:emb AS vector), :doc, :meta)
                    ON CONFLICT DO NOTHING
                """),
                {
                    "uid": str(uuid.uuid4()),
                    "cid": collection_id,
                    "emb": emb_literal,
                    "doc": doc.page_content,
                    "meta": json.dumps(doc.metadata),
                },
            )

    print(f"[vectorize] {len(df)} products → {len(docs)} chunks stored")
    return {"products": len(df), "chunks": len(docs)}