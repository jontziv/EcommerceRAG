-- Enable pgvector extension
-- Run this in Supabase SQL Editor before first deploy
CREATE EXTENSION IF NOT EXISTS vector;

-- Product catalog (managed by ShopWise)
CREATE TABLE IF NOT EXISTS products (
    id              VARCHAR PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT,
    price           NUMERIC(10, 2),
    category        TEXT,
    image_url       TEXT,
    rating          NUMERIC(3, 2)  DEFAULT 0,
    review_count    INTEGER        DEFAULT 0,
    brand           TEXT           DEFAULT '',
    prime_eligible  BOOLEAN        DEFAULT false,
    features        JSONB          DEFAULT '[]'::jsonb
);

-- Product reviews
CREATE TABLE IF NOT EXISTS product_reviews (
    id                SERIAL PRIMARY KEY,
    product_id        VARCHAR(255) REFERENCES products(id) ON DELETE CASCADE,
    reviewer_name     TEXT,
    rating            NUMERIC(3, 2),
    title             TEXT,
    body              TEXT,
    verified_purchase BOOLEAN DEFAULT false,
    helpful_votes     INTEGER DEFAULT 0,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_product_reviews_product_id ON product_reviews(product_id);

-- LangChain PGVector will auto-create its own embedding tables
