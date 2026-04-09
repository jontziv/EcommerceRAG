import io
import pandas as pd

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .database import get_vector_store, AsyncSessionLocal, async_engine, Base
from .models import Product

REQUIRED_COLUMNS = ["id", "name", "description", "price", "category", "image_url"]


async def vectorize_products(file_content: bytes, filename: str) -> dict:
    """Ingest a CSV: upsert products into PostgreSQL and embed into pgvector."""
    # Parse CSV from in-memory bytes
    df = pd.read_csv(io.BytesIO(file_content), dtype={"id": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Ensure SQLAlchemy-managed tables exist
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Upsert products (safe re-upload)
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

    # Build chunked LangChain documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs: list[Document] = []
    for _, row in df.iterrows():
        text = f"{row['name']} {row['description']}"
        chunks = splitter.create_documents(
            [text],
            metadatas=[{"product_id": str(row["id"]), "category": str(row["category"])}],
        )
        docs.extend(chunks)

    # Embed and store in pgvector
    store = get_vector_store()
    await store.aadd_documents(docs)

    print(f"[vectorize] {len(df)} products → {len(docs)} chunks stored")
    return {"products": len(df), "chunks": len(docs)}