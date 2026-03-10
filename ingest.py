from src.wikipedia_fetcher import fetch_all_articles, load_articles
from src.chunker import chunk_articles
from src.config import TOPIC, NUM_ARTICLES
import os

def main():
    print("WIKIPEDIA RAG — INGESTION PIPELINE")
    
    print("\n STEP 1: Fetching Wikipedia Articles")
    articles = fetch_all_articles(topic=TOPIC, num_articles=NUM_ARTICLES)
    if not articles:
        print("\n Pipeline failed at Step 1. No articles fetched.")
        return
    print(f"\n Step 1 Complete! {len(articles)} articles ready for processing.")

    print("\n STEP 2: Chunking Articles into Passages")
    chunks = chunk_articles(articles)
    if not chunks:
        print("\n Pipeline failed at Step 2. No chunks created.")
        return
    print(f"\n Step 2 Complete! {len(chunks)} chunks ready for embedding.\n")

if __name__ == "__main__":
    main()