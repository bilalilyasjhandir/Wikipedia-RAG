import os
from src.wikipedia_fetcher import fetch_all_articles, load_articles
from src.chunker import chunk_articles, load_chunks
from src.embedder import embed_chunks, load_embeddings
from src.vector_store import store_embeddings
from src.config import TOPIC, NUM_ARTICLES

def main():
    print("WIKIPEDIA RAG — INGESTION PIPELINE")
    
    print("\n STEP 1: Fetching Wikipedia Articles")
    if os.path.exists("data/articles.json"):
        print("Articles already fetched! Loading from file...")
        articles = load_articles()
    else:
        articles = fetch_all_articles(topic=TOPIC, num_articles=NUM_ARTICLES)
    if not articles:
        print("\n Pipeline failed at Step 1. No articles fetched.")
        return
    print(f"\n Step 1 Complete! {len(articles)} articles ready for processing.")

    print("\n STEP 2: Chunking Articles into Passages")
    if os.path.exists("data/chunks.json"):
        print("Chunks already created! Loading from file...")
        chunks = load_chunks()
    else:
        chunks = chunk_articles(articles)
    if not chunks:
        print("\n Pipeline failed at Step 2. No chunks created.")
        return
    print(f"\n Step 2 Complete! {len(chunks)} chunks ready for embedding.\n")

    print("\n STEP 3: Generating Embeddings using intfloat/multilingual-e5-large")
    if os.path.exists("data/embeddings.json"):
        print("Embeddings already generated! Loading from file...")
        from src.embedder import load_embeddings
        embedded_chunks = load_embeddings()
    else:
        embedded_chunks = embed_chunks(chunks)
    if not embedded_chunks:
        print("\n Pipeline failed at Step 3. No embeddings generated.")
        return
    print(f"\n Step 3 Complete! {len(embedded_chunks)} chunks embedded.\n")
    
    print("\n STEP 4: Storing in ChromaDB Vector Database")
    collection = store_embeddings(embedded_chunks)
    if not collection:
        print("\n Pipeline failed at Step 4. Vector store error.")
        return
    print(f"\n Step 4 Complete! Vector store ready for queries.\n")

    print("INGESTION PIPELINE COMPLETE!")

if __name__ == "__main__":
    main()