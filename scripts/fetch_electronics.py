"""
Fetch ~100 Electronics products from the HuggingFace McAuley-Lab/Amazon-Reviews-2023
dataset and write an enriched CSV to uploads/product_real_data.csv.

Usage (from project root):
    python -m scripts.fetch_electronics
    # or
    python scripts/fetch_electronics.py

Requirements: datasets>=2.0.0 (already in requirements.txt)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

def _check_datasets_version() -> None:
    """datasets ≥ 3.0 removed support for custom loading scripts, which the
    McAuley-Lab dataset relies on. Fail fast with a clear fix message."""
    try:
        import datasets as _ds
        major = int(_ds.__version__.split(".")[0])
        if major >= 3:
            print(
                f"\n[fetch] ❌ Incompatible datasets version: {_ds.__version__}\n"
                "The McAuley-Lab Amazon-Reviews-2023 dataset requires datasets<3.0.\n"
                "Fix:  pip install 'datasets>=2.14.0,<3.0.0'\n"
            )
            sys.exit(1)
    except ImportError:
        print("[fetch] ❌ datasets library not installed. Run: pip install 'datasets>=2.14.0,<3.0.0'")
        sys.exit(1)


_check_datasets_version()

N_PRODUCTS = 100
MIN_REVIEWS_PER_PRODUCT = 3
MAX_REVIEWS_PER_PRODUCT = 5
MAX_REVIEW_SCAN = 1_000_000   # stop scanning reviews after this many rows
OUTPUT_PATH = Path(__file__).parent.parent / "uploads" / "product_real_data.csv"


def _clean_price(raw) -> float | None:
    """Parse a price field like '$19.99' or 19.99 into a float, or None."""
    if raw is None:
        return None
    try:
        return float(str(raw).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _best_image(images: list) -> str:
    """Return the best available image URL from the Amazon images list."""
    for img in images or []:
        if not isinstance(img, dict):
            continue
        for key in ("large", "hi_res", "thumb"):
            url = img.get(key)
            if url and url.startswith("http"):
                return url
    return ""


def _join_list(val, max_chars: int = 600) -> str:
    """Join a list of strings, truncating to max_chars."""
    if isinstance(val, list):
        text = " ".join(str(v) for v in val if v)
    else:
        text = str(val or "")
    return text[:max_chars]


def fetch_metadata(n: int = N_PRODUCTS) -> list[dict]:
    """Stream the Electronics metadata config and collect n valid products."""
    from datasets import load_dataset  # lazy import

    print("[fetch] Loading raw_meta_Electronics (streaming)…")
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Electronics",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    products: list[dict] = []
    scanned = 0

    for item in ds:
        scanned += 1
        if len(products) >= n:
            break

        asin = item.get("parent_asin") or ""
        title = (item.get("title") or "").strip()
        if not asin or not title:
            continue

        price = _clean_price(item.get("price"))
        if price is None or price <= 0 or price > 5_000:
            continue

        image_url = _best_image(item.get("images") or [])
        if not image_url:
            image_url = (
                f"https://placehold.co/400x300/e0e7ff/6366f1?text={asin}"
            )

        description = _join_list(item.get("description") or [])
        if not description:
            description = title

        features_raw = item.get("features") or []
        features: list[str] = [str(f)[:200] for f in features_raw[:6] if f]

        brand = str(item.get("store") or "").strip()
        rating = float(item.get("average_rating") or 0)
        review_count = int(item.get("rating_number") or 0)

        products.append(
            {
                "id": asin,
                "name": title[:250],
                "description": description,
                "price": round(price, 2),
                "category": "Electronics",
                "image_url": image_url,
                "rating": round(rating, 2),
                "review_count": review_count,
                "brand": brand,
                "prime_eligible": True,
                "features": json.dumps(features),
                # populated in the reviews pass
                "reviews": "[]",
            }
        )
        if len(products) % 10 == 0:
            print(f"[fetch] Collected {len(products)}/{n} products  (scanned {scanned})")

    print(f"[fetch] Done collecting metadata: {len(products)} products (scanned {scanned} rows)")
    return products


def fetch_reviews(products: list[dict]) -> None:
    """Stream review dataset and attach up to MAX_REVIEWS_PER_PRODUCT reviews
    to each product dict in-place."""
    from datasets import load_dataset  # lazy import

    target_asins: set[str] = {p["id"] for p in products}
    reviews_by_asin: dict[str, list[dict]] = {p["id"]: [] for p in products}

    print("[fetch] Loading raw_review_Electronics (streaming)…")
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Electronics",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    scanned = 0
    collected = 0

    for row in ds:
        scanned += 1
        if scanned > MAX_REVIEW_SCAN:
            print(f"[fetch] Hit review scan limit ({MAX_REVIEW_SCAN:,}); stopping.")
            break

        asin = row.get("parent_asin") or ""
        if asin not in target_asins:
            continue

        bucket = reviews_by_asin[asin]
        if len(bucket) >= MAX_REVIEWS_PER_PRODUCT:
            continue

        reviewer_raw = str(row.get("user_id") or "")
        bucket.append(
            {
                "reviewer_name": "Customer " + reviewer_raw[-5:] if reviewer_raw else "Anonymous",
                "rating": float(row.get("rating") or 0),
                "title": str(row.get("title") or "")[:150],
                "body": str(row.get("text") or "")[:400],
                "verified_purchase": bool(row.get("verified_purchase", False)),
                "helpful_votes": int(row.get("helpful_vote") or 0),
            }
        )
        collected += 1

        if scanned % 50_000 == 0:
            filled = sum(1 for v in reviews_by_asin.values() if len(v) >= MIN_REVIEWS_PER_PRODUCT)
            print(
                f"[fetch] Reviews: scanned {scanned:,}, collected {collected}, "
                f"{filled}/{len(products)} products have ≥{MIN_REVIEWS_PER_PRODUCT} reviews"
            )

        # Early exit if every product has enough reviews
        if all(len(v) >= MIN_REVIEWS_PER_PRODUCT for v in reviews_by_asin.values()):
            print("[fetch] All products have sufficient reviews — stopping early.")
            break

    # Attach reviews back to product dicts
    for p in products:
        p["reviews"] = json.dumps(reviews_by_asin[p["id"]])

    products_with_reviews = sum(1 for v in reviews_by_asin.values() if v)
    print(f"[fetch] Attached reviews: {products_with_reviews}/{len(products)} products have at least one review")


def main() -> None:
    products = fetch_metadata(N_PRODUCTS)
    if not products:
        print("[fetch] ERROR: No products collected. Check your internet connection / HuggingFace access.")
        sys.exit(1)

    fetch_reviews(products)

    df = pd.DataFrame(products)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[fetch] ✅ Wrote {len(df)} products to {OUTPUT_PATH}")
    print(f"[fetch] Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
