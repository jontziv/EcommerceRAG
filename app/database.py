import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

Base = declarative_base()

_raw_url = os.getenv("DATABASE_URL", "")

# SQLAlchemy async engine — needs asyncpg driver
if _raw_url.startswith("postgresql+psycopg://"):
    _async_url = _raw_url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
elif _raw_url.startswith("postgresql://"):
    _async_url = "postgresql+asyncpg://" + _raw_url[len("postgresql://"):]
else:
    _async_url = _raw_url

async_engine = create_async_engine(_async_url, pool_pre_ping=True, future=True)
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Embeddings model (shared)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

COLLECTION_NAME = "shopwise_products"


def get_vector_store() -> PGVector:
    """Return a PGVector store backed by Supabase."""
    return PGVector(
        connection=_raw_url,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
    )