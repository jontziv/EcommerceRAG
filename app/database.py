import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

Base = declarative_base()

_raw_url = os.getenv("DATABASE_URL", "")

# ── URL variants ──────────────────────────────────────────────────────────────
# asyncpg  → for SQLAlchemy async engine
# psycopg3 → for LangChain PGVector (needs plain postgresql:// URI, no driver prefix)

def _to_asyncpg(url: str) -> str:
    """Derive an asyncpg-compatible URL for SQLAlchemy."""
    if url.startswith("postgresql+asyncpg://"):
        base = url
    elif url.startswith("postgresql+psycopg://"):
        base = "postgresql+asyncpg://" + url[len("postgresql+psycopg://"):]
    elif url.startswith("postgresql://"):
        base = "postgresql+asyncpg://" + url[len("postgresql://"):]
    else:
        base = url
    # asyncpg uses ssl=true query param
    if "ssl" not in base and "supabase" in base:
        sep = "&" if "?" in base else "?"
        base += sep + "ssl=true"
    return base


def _to_psycopg3(url: str) -> str:
    """Strip driver prefix — psycopg3 expects plain postgresql:// URIs."""
    for prefix in ("postgresql+psycopg://", "postgresql+asyncpg://"):
        if url.startswith(prefix):
            url = "postgresql://" + url[len(prefix):]
            break
    # psycopg3 uses sslmode=require
    if "sslmode" not in url and "supabase" in url:
        sep = "&" if "?" in url else "?"
        url += sep + "sslmode=require"
    return url

_async_url   = _to_asyncpg(_raw_url)
_pgvector_url = _to_psycopg3(_raw_url)

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
    # use_jsonb=False avoids SQLAlchemy cache-key hashing errors with dict metadata
    return PGVector(
        connection=_pgvector_url,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        use_jsonb=False,
    )