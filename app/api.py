import asyncio
import time

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from .models import SearchQuery, SearchResult
from .search import search_products
from .vectorization import vectorize_products

app = FastAPI(title="ShopWise")
templates = Jinja2Templates(directory="templates")

_vectorize_lock = asyncio.Lock()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse({"error": str(exc)}, status_code=500)


# ── Pages ────────────────────────────────────────────────────────────────────

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin")
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


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