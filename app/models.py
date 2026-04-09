from sqlalchemy import Column, String, Text, Numeric
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid

from .database import Base


# ── SQLAlchemy ORM model ─────────────────────────────────────────────────────

class Product(Base):
    __tablename__ = "products"

    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(Text, nullable=False)
    description = Column(Text)
    price = Column(Numeric(10, 2))
    category = Column(Text)
    image_url = Column(Text)


# ── Pydantic API schemas ─────────────────────────────────────────────────────

class SearchQuery(BaseModel):
    query: str
    category: Optional[str] = None


class SearchResult(BaseModel):
    answer: str
    products: List[Dict[str, Any]]
    contexts: List[str]