from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, Numeric, String, Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid

from .database import Base


# ── SQLAlchemy ORM models ────────────────────────────────────────────────────

class Product(Base):
    __tablename__ = "products"

    id             = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    name           = Column(Text, nullable=False)
    description    = Column(Text)
    price          = Column(Numeric(10, 2))
    category       = Column(Text)
    image_url      = Column(Text)
    rating         = Column(Numeric(3, 2), default=0)
    review_count   = Column(Integer, default=0)
    brand          = Column(Text, default="")
    prime_eligible = Column(Boolean, default=False)
    features       = Column(JSONB, default=list)


class ProductReview(Base):
    __tablename__ = "product_reviews"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    product_id        = Column(String(255), ForeignKey("products.id", ondelete="CASCADE"))
    reviewer_name     = Column(Text)
    rating            = Column(Numeric(3, 2))
    title             = Column(Text)
    body              = Column(Text)
    verified_purchase = Column(Boolean, default=False)
    helpful_votes     = Column(Integer, default=0)
    created_at        = Column(DateTime(timezone=True))


# ── Pydantic API schemas ─────────────────────────────────────────────────────

class SearchQuery(BaseModel):
    query: str
    category: Optional[str] = None


class SearchResult(BaseModel):
    answer: str
    products: List[Dict[str, Any]]
    contexts: List[str]
