import chromadb
import json
import os
from src.config import (CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDINGS_FILE, DATA_DIR,)

def get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client

def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "description": "Wikipedia RAG articles"
        }
    )
    return collection

def store_embeddings(embedded_chunks: list[dict]) -> chromadb.Collection:
    print(f"\n Setting up ChromaDB vector store...")
    print(f"Persist directory: {CHROMA_PERSIST_DIR}")
    print(f"Collection name:  {COLLECTION_NAME}\n")
    client = get_chroma_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Deleted existing collection (fresh start)")
    except Exception:
        pass
    collection = get_or_create_collection(client)
    print(f"Preparing {len(embedded_chunks)} chunks for storage...")
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    for chunk in embedded_chunks:
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        safe_metadata = {}
        for key, value in chunk["metadata"].items():
            if isinstance(value, (str, int, float, bool)):
                safe_metadata[key] = value
            else:
                safe_metadata[key] = str(value)
        metadatas.append(safe_metadata)
    batch_size = 500
    total_batches = (len(ids) + batch_size - 1) // batch_size
    for i in range(0, len(ids), batch_size):
        batch_num = (i // batch_size) + 1
        end = min(i + batch_size, len(ids))
        print(f"Storing batch {batch_num}/{total_batches} (chunks {i}-{end - 1})...", end=" ", flush=True)
        collection.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        print("Done")
    count = collection.count()
    print_store_summary(count, len(embedded_chunks))
    return collection

def search(query_embedding: list[float], n_results: int = 5) -> dict:
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return results

def print_store_summary(stored_count: int, expected_count: int) -> None:
    status = "Done" if stored_count == expected_count else "Error"
    print("VECTOR STORE SUMMARY")
    print(f"Status: {status}")
    print(f"Chunks stored: {stored_count}")
    print(f"Expected: {expected_count}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Similarity: Cosine")
    print(f"Persist dir: {CHROMA_PERSIST_DIR}")

def get_collection_info() -> dict:
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        return {
            "name": COLLECTION_NAME,
            "count": collection.count(),
            "persist_dir": CHROMA_PERSIST_DIR,
        }
    except Exception:
        return {"error": "Collection not found. Run the pipeline first."}