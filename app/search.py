import os
from typing import Optional, Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereRerank
from sqlalchemy import select

from .database import get_vector_store, AsyncSessionLocal
from .models import Product, SearchResult

SYSTEM = """You are a friendly product search assistant for ShopWise, a modern e-commerce store.
Help customers find the best products based on their lifestyle needs and context.
Use the provided product catalog to give concise, helpful recommendations (2-4 sentences).
Highlight why each product fits the customer's described situation.
If no products match, suggest refining the search. Do NOT invent products not in the catalog."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user",
     "Customer is looking for: {input}\n\n"
     "Available products:\n{context}\n\n"
     "Provide a short, friendly recommendation."),
])


def _format_docs(docs: Sequence[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


async def search_products(query: str, category: Optional[str] = None) -> SearchResult:
    store = get_vector_store()

    # 1. Vector similarity search — initial candidates
    initial_docs = await store.asimilarity_search(query, k=5)

    # 2. Cohere rerank — keep top 3 most relevant
    compressor = CohereRerank(top_n=3, model="rerank-multilingual-v3.0")
    docs = list(await compressor.acompress_documents(initial_docs, query))

    # 3. GPT-4o answer via LCEL
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))
    chain = PROMPT | llm | StrOutputParser()
    answer: str = await chain.ainvoke({"input": query, "context": _format_docs(docs)})

    # 4. Fetch full product details from Supabase
    product_ids = [
        doc.metadata.get("product_id")
        for doc in docs
        if doc.metadata.get("product_id")
    ]

    products = []
    if product_ids:
        async with AsyncSessionLocal() as db:
            q = select(Product).where(Product.id.in_(product_ids))
            if category:
                q = q.where(Product.category == category)
            rows = await db.execute(q)
            db_products = rows.scalars().all()
            products = [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "price": float(p.price) if p.price is not None else 0.0,
                    "category": p.category,
                    "image_url": p.image_url,
                }
                for p in db_products
            ]

    contexts = [d.page_content for d in docs]
    return SearchResult(answer=answer, products=products, contexts=contexts)