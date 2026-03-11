import requests
import json
import os
import time
from src.config import (EMBEDDING_MODEL, HF_API_TOKEN, EMBEDDING_API_URL, EMBEDDING_BATCH_SIZE, EMBEDDING_DIMENSIONS, EMBEDDINGS_FILE, DATA_DIR,)

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json",
}

def embed_single_batch(texts: list[str], retry_count: int = 3) -> list[list[float]]:
    payload = {"inputs": texts}
    for attempt in range(retry_count):
        response = requests.post(EMBEDDING_API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 503:
            data = response.json()
            wait_time = data.get("estimated_time", 30)
            print(f"\n Model is loading... waiting {wait_time:.0f}s (attempt {attempt + 1}/{retry_count})")
            time.sleep(wait_time)
            continue
        if response.status_code == 429:
            print(f"\n Rate limited. Waiting 10s...")
            time.sleep(10)
            continue
        print(f"\n API Error {response.status_code}: {response.text}")
        if attempt < retry_count - 1:
            time.sleep(5)
            continue
        else:
            raise Exception(f"HuggingFace API failed after {retry_count} attempts: {response.text}")
    return []

def validate_token() -> bool:
    if not HF_API_TOKEN:
        print("HF_API_TOKEN not found!")
        print("1. Create a .env file in your project root")
        print("2. Add: HF_API_TOKEN=hf_your_token_here")
        print("3. Get your free token from: https://huggingface.co/settings/tokens")
        return False
    return True

def embed_chunks(chunks: list[dict]) -> list[dict]:
    if not validate_token():
        return []
    print(f"\n Generating embeddings for {len(chunks)} chunks...")
    print(f"Model: {EMBEDDING_API_URL.split('/')[-1]}")
    print(f"Dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"Batch size: {EMBEDDING_BATCH_SIZE}")
    total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    print(f"Batches: {total_batches}\n")
    embedded_chunks = []
    failed_chunks = 0
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
        batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
        texts = [chunk["text"] for chunk in batch]
        print(f"Batch {batch_num}/{total_batches} ({len(texts)} chunks)...", end=" ", flush=True)
        try:
            embeddings = embed_single_batch(texts)
            if len(embeddings) != len(batch):
                print(f"Expected {len(batch)} embeddings, got {len(embeddings)}")
                failed_chunks += len(batch)
                continue
            for chunk, embedding in zip(batch, embeddings):
                embedded_chunk = {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "embedding": embedding,
                }
                embedded_chunks.append(embedded_chunk)
            print(f"Done")
        except Exception as e:
            print(f"Error: {e}")
            failed_chunks += len(batch)
        if batch_num < total_batches:
            time.sleep(1)
    save_embeddings(embedded_chunks)
    print_embedding_summary(embedded_chunks, failed_chunks)
    return embedded_chunks

def save_embeddings(embedded_chunks: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)
    file_size = os.path.getsize(EMBEDDINGS_FILE) / (1024 * 1024)
    print(f"\n Saved {len(embedded_chunks)} embedded chunks to: {EMBEDDINGS_FILE}")
    print(f"File size: {file_size:.1f} MB")

def load_embeddings() -> list[dict]:
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"No embeddings file found at: {EMBEDDINGS_FILE}")
        print("Run the pipeline first: python ingest.py")
        return []
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        embedded_chunks = json.load(f)
    print(f"Loaded {len(embedded_chunks)} embedded chunks from: {EMBEDDINGS_FILE}")
    return embedded_chunks

def print_embedding_summary(embedded_chunks: list[dict], failed: int) -> None:
    if not embedded_chunks:
        return
    sample_dim = len(embedded_chunks[0]["embedding"])
    print("EMBEDDING SUMMARY")
    print(f"Chunks embedded: {len(embedded_chunks)}")
    print(f"Failed chunks: {failed}")
    print(f"Embedding dimensions: {sample_dim}")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Saved to: {EMBEDDINGS_FILE}")

    first_emb = embedded_chunks[0]["embedding"][:5]
    print(f"\n Sample embedding (first 5 dims of chunk_0):")
    print(f"{first_emb}")
    print(f"... ({sample_dim} dimensions total)")

def embed_query(text: str) -> list[float]:
    result = embed_single_batch([text])
    if result and len(result) > 0:
        return result[0]
    raise Exception("Failed to embed query text")