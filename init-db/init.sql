-- Enable pgvector extension
-- Run this in Supabase SQL Editor before first deploy
CREATE EXTENSION IF NOT EXISTS vector;

-- Product catalog (managed by ShopWise)
-- LangChain PGVector will auto-create its own embedding tables
CREATE TABLE IF NOT EXISTS products (
    id      VARCHAR PRIMARY KEY,
    name    TEXT NOT NULL,
    description TEXT,
    price   NUMERIC(10, 2),
    category TEXT,
    image_url TEXT
);
