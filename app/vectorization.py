import io
import json
import uuid
import pandas as pd

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sqlalchemy import text, delete
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .database import AsyncSessionLocal, async_engine, Base, embeddings, COLLECTION_NAME
from .models import Product, ProductReview

REQUIRED_COLUMNS = ["id", "name", "description", "price", "category", "image_url"]
OPTIONAL_COLUMNS = ["rating", "review_count", "brand", "prime_eligible", "features", "reviews"]


def _parse_json_field(val, default):
    """Safely parse a JSON string column; return default on failure."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(str(val))
    except (json.JSONDecodeError, TypeError):
        return default


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
    """Ingest a CSV: upsert products + reviews into PostgreSQL and embed into pgvector."""
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
            features = _parse_json_field(row.get("features"), [])
            prime_raw = row.get("prime_eligible", False)
            if isinstance(prime_raw, str):
                prime_eligible = prime_raw.strip().lower() in ("true", "1", "yes")
            else:
                prime_eligible = bool(prime_raw)

            stmt = pg_insert(Product).values(
                id=str(row["id"]),
                name=str(row["name"]),
                description=str(row["description"]),
                price=float(row["price"]),
                category=str(row["category"]),
                image_url=str(row["image_url"]),
                rating=float(row.get("rating") or 0),
                review_count=int(row.get("review_count") or 0),
                brand=str(row.get("brand") or ""),
                prime_eligible=prime_eligible,
                features=features,
            ).on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "name": str(row["name"]),
                    "description": str(row["description"]),
                    "price": float(row["price"]),
                    "category": str(row["category"]),
                    "image_url": str(row["image_url"]),
                    "rating": float(row.get("rating") or 0),
                    "review_count": int(row.get("review_count") or 0),
                    "brand": str(row.get("brand") or ""),
                    "prime_eligible": prime_eligible,
                    "features": features,
                },
            )
            await db.execute(stmt)

            # Replace reviews for this product
            if "reviews" in df.columns:
                reviews_data = _parse_json_field(row.get("reviews"), [])
                if reviews_data:
                    await db.execute(
                        delete(ProductReview).where(ProductReview.product_id == str(row["id"]))
                    )
                    for rv in reviews_data:
                        if not isinstance(rv, dict):
                            continue
                        await db.execute(
                            pg_insert(ProductReview).values(
                                product_id=str(row["id"]),
                                reviewer_name=str(rv.get("reviewer_name") or "Anonymous"),
                                rating=float(rv.get("rating") or 0),
                                title=str(rv.get("title") or ""),
                                body=str(rv.get("body") or ""),
                                verified_purchase=bool(rv.get("verified_purchase", False)),
                                helpful_votes=int(rv.get("helpful_votes") or 0),
                            ).on_conflict_do_nothing()
                        )

        await db.commit()

    # Build chunked documents for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs: list[Document] = []
    for _, row in df.iterrows():
        brand_prefix = f"{row.get('brand', '')} " if row.get("brand") else ""
        text_ = f"{brand_prefix}{row['name']} {row['description']}"
        chunks = splitter.create_documents(
            [text_],
            metadatas=[{"product_id": str(row["id"]), "category": str(row["category"])}],
        )
        docs.extend(chunks)

    # Embed via Cohere and write directly to pgvector tables
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
