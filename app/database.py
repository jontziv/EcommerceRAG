import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain_cohere import CohereEmbeddings

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
    # asyncpg ssl param must be a mode name (not a boolean)
    if "ssl" not in base and "supabase" in base:
        sep = "&" if "?" in base else "?"
        base += sep + "ssl=require"
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
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
)

COLLECTION_NAME = "shopwise_products"