-- Migration 001: Add electronics-specific fields to products table
-- Run this in Supabase SQL Editor BEFORE redeploying the app

ALTER TABLE products
ADD COLUMN IF NOT EXISTS rating        NUMERIC(3, 2) DEFAULT 0,
ADD COLUMN IF NOT EXISTS review_count  INTEGER       DEFAULT 0,
ADD COLUMN IF NOT EXISTS brand         TEXT          DEFAULT '',
ADD COLUMN IF NOT EXISTS prime_eligible BOOLEAN      DEFAULT false,
ADD COLUMN IF NOT EXISTS features      JSONB         DEFAULT '[]'::jsonb;

-- Product reviews table
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
