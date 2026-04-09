# 🛍️ ShopWise — AI-Powered E-Commerce Search

ShopWise is a full-stack RAG (Retrieval-Augmented Generation) application that lets shoppers find products using natural language. Describe your situation — *"going to the beach"*, *"gift for my dad"* — and the AI finds the best matching products using semantic vector search, Cohere reranking, and a GPT-4o generated recommendation.

**Stack**: FastAPI · LangChain · OpenAI (embeddings + GPT-4o) · Cohere Rerank · Supabase (PostgreSQL + pgvector) · Tailwind CSS · Render

---

## 🚀 Supabase Setup

> Run once before your first deploy.

1. **Create a Supabase project** at [supabase.com](https://supabase.com).

2. **Run the init SQL** — go to your project → SQL Editor → paste and run `init-db/init.sql`:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE TABLE IF NOT EXISTS products ( ... );
   ```

3. **Get your connection string** — Project Settings → Database → Connection string → URI (Session mode, port 5432).  
   Change the scheme to `postgresql+psycopg://`:
   ```
   postgresql+psycopg://postgres:[PASSWORD]@db.[ref].supabase.co:5432/postgres
   ```

4. **Set env vars** — copy `.env.example` → `.env` and fill in your keys.

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (embeddings + GPT-4o) |
| `COHERE_API_KEY` | Cohere API key (reranking) |
| `DATABASE_URL` | Supabase connection string (`postgresql+psycopg://...`) |

---

## 🧪 Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in environment variables
cp .env.example .env

# 3. Start the server
uvicorn app.api:app --reload --port 8000
```

- Storefront: http://localhost:8000
- Admin upload: http://localhost:8000/admin

> Alternatively, use Docker: `docker-compose up --build` (still requires a Supabase DATABASE_URL in .env)

---

## 📤 Uploading Products

1. Go to `/admin`
2. Drag & drop or browse for your CSV file
3. Required columns: `id, name, description, price, category, image_url`
4. Click **Upload & Vectorize** — products are upserted into Supabase and embedded into pgvector

---

## 🔍 How Search Works

```
User query  →  OpenAI embeddings  →  pgvector similarity (k=5)
           →  Cohere rerank (top 3)
           →  GPT-4o generates recommendation blurb
           →  Fetch full product details from Supabase
           →  Return answer + product cards
```

---

## 🚢 Deploying on Render

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → New → Web Service → connect repo.
3. Render detects `render.yaml` automatically.
4. Set the three env vars in the Render dashboard (Environment tab).
5. Deploy — done.

---

## 🚀 Features

- Upload and parse product CSVs with metadata
- Generate embeddings using OpenAI models
- Store data and vectors in PostgreSQL (`pgvector`)
- Perform semantic vector search with LangChain
- Run using Docker and Docker Compose

---

## 📁 Project Structure

```
smartfind/
├── app/
│   ├── __init__.py              # App factory
│   ├── api/
│   │   ├── search.py            # Search API
│   │   └── vectorization.py     # File upload + embedding logic
│   ├── config.py                # Flask + environment configs
│   ├── database.py              # SQLAlchemy + PGVector init
│   ├── models.py                # SQLAlchemy models
│   ├── utils.py                 # Helpers (e.g., file type check)
│   └── templates/
│       └── index.html           # Frontend template
├── init-db/
│   └── init.sql                 # DB schema and index creation
├── product_real_data.csv        # Sample product data
├── .env                         # Secrets (OpenAI key, DB URL)
├── Dockerfile                   # Flask app Docker config
├── docker-compose.yml           # Multi-container setup
├── requirements.txt             # Python dependencies
└── run.py                       # Entrypoint for Flask app
```

---

## 🧪 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourname/smartfind.git
cd smartfind
```

### 2. Add Your API Key

Update `.env`:
```env
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/postgres
```

### 3. Start the App with Docker

```bash
docker-compose up --build
```

The app will be available at: [http://localhost:5000](http://localhost:5000)

---

## 📄 CSV Format

Ensure your uploaded CSV follows this format:

| id | name         | description             | price | category  | image_url           |
|----|--------------|--------------------------|-------|-----------|----------------------|
| 1  | Apple Watch  | Smart wearable device    | 299   | Electronics| http://example.com/1 |
| 2  | Leather Bag  | Stylish and durable bag  | 150   | Fashion   | http://example.com/2 |

---

## 🔍 How It Works

- **Upload Page**: Upload a CSV → parses data → generates embeddings
- **Database**: Vectors and metadata stored in `product_embeddings` table
- **Search**: Uses OpenAI + LangChain’s `PGVector` to return top results

---

## 🧱 Built With

- [Flask](https://flask.palletsprojects.com/)
- [LangChain](https://www.langchain.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [PostgreSQL](https://www.postgresql.org/) + [pgvector](https://github.com/pgvector/pgvector)
- [Docker](https://www.docker.com/)

---

## 🧠 Future Improvements

- Add authentication
- Advanced filtering (price range, category)
- Switch to `langchain_postgres` PGVector store
- UI polish and error handling

---