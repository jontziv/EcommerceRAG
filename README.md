# 🛍️ ShopWise — AI-Powered Electronics Store

ShopWise is a full-stack RAG (Retrieval-Augmented Generation) e-commerce application. Shoppers describe what they need in plain English — *"wireless headphones for the gym"*, *"laptop for a college student"* — and the AI finds and recommends the best-matching electronics using semantic vector search, Cohere reranking, and Groq LLM-generated recommendations.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

---

## ✨ Features

- **Natural language product search** — powered by Cohere embeddings + pgvector similarity
- **AI recommendations** — Llama 3.3 70B (via Groq) generates contextual recommendation blurbs
- **Cohere reranking** — top-k vector results re-ranked for precision before LLM generation
- **Product detail pages** — image, star ratings, Prime badge, feature bullets, customer reviews
- **Sliding cart drawer** — persistent localStorage cart with quantity controls
- **Full checkout flow** — address form → mock payment → order confirmation
- **HuggingFace dataset ingestion** — script to pull 100 real Electronics products from Amazon Reviews 2023
- **Dark mode** — system-aware with manual toggle, persisted to localStorage
- **Admin upload** — drag-and-drop CSV ingestion with live vectorisation progress
- **Render-ready** — `render.yaml` for one-click deploy

---

## 🏗️ Architecture

```
User query
    │
    ▼
Cohere embed-english-v3.0          ← query embedding
    │
    ▼
pgvector similarity search (k=5)   ← Supabase PostgreSQL
    │
    ▼
Cohere rerank-multilingual-v3.0    ← top 3 reranked docs
    │
    ▼
Groq Llama 3.3 70B                 ← recommendation blurb
    │
    ▼
FastAPI + Jinja2 + Tailwind CSS    ← rendered to browser
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI, SQLAlchemy (async), asyncpg |
| **Database** | Supabase (PostgreSQL + pgvector) |
| **Embeddings** | Cohere `embed-english-v3.0` |
| **Reranking** | Cohere `rerank-multilingual-v3.0` |
| **LLM** | Groq `llama-3.3-70b-versatile` |
| **Orchestration** | LangChain |
| **Frontend** | Jinja2 templates, Tailwind CSS, Vanilla JS |
| **Dataset** | HuggingFace McAuley-Lab/Amazon-Reviews-2023 |
| **Deployment** | Render |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- A [Supabase](https://supabase.com) project (free tier works)
- API keys for [Cohere](https://dashboard.cohere.com) and [Groq](https://console.groq.com)

### 1. Clone & install

```bash
git clone https://github.com/jontziv/EcommerceRAG.git
cd EcommerceRAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up Supabase

In your Supabase project → **SQL Editor**, run `init-db/init.sql`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
-- (full schema in init-db/init.sql)
```

> **Existing Supabase project?** Run `migrations/001_add_product_fields.sql` instead.

### 3. Configure environment

```bash
cp .env.example .env
# Fill in COHERE_API_KEY, GROQ_API_KEY, DATABASE_URL
```

Your `DATABASE_URL` should use the `postgresql+psycopg://` scheme (Session mode, port 5432):
```
DATABASE_URL=postgresql+psycopg://postgres:[PASSWORD]@db.[ref].supabase.co:5432/postgres
```

### 4. (Optional) Fetch real product data

Pull 100 Electronics products + reviews from HuggingFace:

```bash
# Requires datasets<3.0 — already pinned in requirements.txt
python -m scripts.fetch_electronics
```

This streams `McAuley-Lab/Amazon-Reviews-2023` and writes an enriched CSV to `uploads/product_real_data.csv`. Takes ~5–10 minutes depending on connection speed.

### 5. Run

```bash
uvicorn app.api:app --reload --port 8000
```

The app **auto-seeds** the database from `uploads/product_real_data.csv` on first startup if the products table is empty.

| URL | Description |
|---|---|
| `http://localhost:8000` | Storefront + AI search |
| `http://localhost:8000/product/{id}` | Product detail page |
| `http://localhost:8000/checkout` | Checkout flow |
| `http://localhost:8000/admin` | CSV upload & vectorisation |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | ✅ | Supabase connection string (`postgresql+psycopg://...`) |
| `COHERE_API_KEY` | ✅ | Cohere API key — embeddings & reranking |
| `GROQ_API_KEY` | ✅ | Groq API key — Llama 3.3 70B inference |

---

## 📤 CSV Format

### Required columns

| Column | Type | Example |
|---|---|---|
| `id` | string | `B09XYZ123` |
| `name` | string | `Sony WH-1000XM5 Headphones` |
| `description` | string | `Industry-leading noise cancelling...` |
| `price` | float | `279.99` |
| `category` | string | `Electronics` |
| `image_url` | string | `https://...` |

### Optional columns (enable full product experience)

| Column | Type | Description |
|---|---|---|
| `rating` | float | Average star rating (0–5) |
| `review_count` | int | Total number of ratings |
| `brand` | string | Manufacturer/brand name |
| `prime_eligible` | bool | Shows Prime badge |
| `features` | JSON array | Bullet point feature list |
| `reviews` | JSON array | Customer review objects |

---

## 🚢 Deploy to Render

1. Fork/push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service** → connect your repo
3. Render auto-detects `render.yaml`
4. Add the three environment variables in **Environment** tab
5. Click **Deploy**

> Run `init-db/init.sql` (or `migrations/001_add_product_fields.sql`) in Supabase before the first deploy. The app seeds itself automatically on first startup.

---

## 📁 Project Structure

```
shopwise/
├── app/
│   ├── api.py              # FastAPI routes (search, product detail, checkout, admin)
│   ├── database.py         # Async SQLAlchemy engine + Cohere embeddings
│   ├── models.py           # ORM models (Product, ProductReview) + Pydantic schemas
│   ├── search.py           # RAG pipeline: embed → pgvector → rerank → LLM
│   └── vectorization.py    # CSV ingestion: upsert products + reviews + embeddings
├── templates/
│   ├── base.html           # Layout, dark mode, cart drawer
│   ├── index.html          # AI search homepage
│   ├── product.html        # Product detail page
│   ├── admin.html          # CSV upload interface
│   └── checkout/
│       ├── address.html    # Shipping address form
│       ├── payment.html    # Mock payment form
│       └── confirmation.html # Order confirmation
├── scripts/
│   └── fetch_electronics.py  # HuggingFace dataset ingestion script
├── migrations/
│   └── 001_add_product_fields.sql  # Supabase migration for new fields
├── init-db/
│   └── init.sql            # Full schema for fresh Supabase projects
├── uploads/
│   └── product_real_data.csv  # Seed product data (auto-loaded on startup)
├── render.yaml             # Render deployment config
├── docker-compose.yml      # Local Docker setup (Postgres + Redis)
└── requirements.txt
```

---

## 🐳 Docker (local development)

```bash
docker-compose up --build
```

Starts FastAPI on port 8000, PostgreSQL with pgvector on 5432, and Redis on 6379. Requires `DATABASE_URL` pointing to the local Postgres instance in your `.env`.

---

## 🧪 RAG Evaluation

Quality evaluation using [RAGAS](https://docs.ragas.io) (faithfulness, answer relevancy, context precision, context recall):

```bash
python -m app.eval_rags
```

Runs against the sample Q&A pairs in `seed/qna_test.json`.
