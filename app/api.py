import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select as sa_select

from .database import AsyncSessionLocal
from .models import SearchQuery, SearchResult, Product
from .search import search_products
from .vectorization import vectorize_products

_vectorize_lock = asyncio.Lock()


def _masked_url(url: str) -> str:
    """Return a log-safe version of the DB URL with password hidden."""
    try:
        p = urlparse(url)
        return f"{p.scheme}://***@{p.hostname}:{p.port}{p.path}"
    except Exception:
        return "(unparseable URL)"


async def _auto_seed():
    """On first deploy, vectorize the bundled product CSV automatically."""
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        print("[startup] ⚠️  DATABASE_URL is not set — skipping auto-seed")
        print("[startup] ⚠️  Add DATABASE_URL in Render > Environment before searches will work")
        return

    print(f"[startup] Using database: {_masked_url(db_url)}")

    csv_path = Path(__file__).parent.parent / "uploads" / "product_real_data.csv"
    if not csv_path.exists():
        print("[startup] Seed CSV not found — skipping auto-seed")
        return
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(sa_select(func.count()).select_from(Product))
            count = result.scalar()
        if count == 0:
            print("[startup] Empty database — seeding from product_real_data.csv…")
            async with _vectorize_lock:
                stats = await vectorize_products(csv_path.read_bytes(), csv_path.name)
            print(f"[startup] ✅ Seeded {stats['products']} products, {stats['chunks']} chunks")
        else:
            print(f"[startup] {count} products already in database — skipping seed")
    except Exception as exc:
        print(f"[startup] ❌ Auto-seed error: {exc}")
        print("[startup] Check DATABASE_URL value in Render > Environment tab")
        print(f"[startup] URL being used: {_masked_url(db_url)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _auto_seed()
    yield


app = FastAPI(title="ShopWise", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse({"error": str(exc)}, status_code=500)


# ── Pages ────────────────────────────────────────────────────────────────────

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.head("/")
async def head_home():
    """Explicit HEAD handler required for Starlette 1.x health checks."""
    return Response(status_code=200)


@app.get("/admin")
async def admin_page(request: Request):
    return templates.TemplateResponse(request, "admin.html")


# ── API ──────────────────────────────────────────────────────────────────────

@app.post("/admin/upload")
async def upload_products(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    start = time.perf_counter()
    async with _vectorize_lock:
        content = await file.read()
        try:
            stats = await vectorize_products(content, file.filename)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elapsed = time.perf_counter() - start
    print(f"[upload] {stats['products']} products, {stats['chunks']} chunks in {elapsed:.2f}s")
    return {"message": "Products vectorized successfully!", "stats": stats}


@app.post("/api/search", response_model=SearchResult)
async def api_search(q: SearchQuery):
    start = time.perf_counter()
    try:
        result = await search_products(q.query, category=q.category)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = time.perf_counter() - start
    print(f"[search] '{q.query}' → {len(result.products)} products in {elapsed:.2f}s")
    return result