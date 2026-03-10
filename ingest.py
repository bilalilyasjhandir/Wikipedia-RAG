from src.wikipedia_fetcher import fetch_all_articles
from src.config import TOPIC, NUM_ARTICLES

def main():
    print("WIKIPEDIA RAG — INGESTION PIPELINE")
    print("\n STEP 1: Fetching Wikipedia Articles")
    articles = fetch_all_articles(topic=TOPIC, num_articles=NUM_ARTICLES)
    if not articles:
        print("\n Pipeline failed at Step 1. No articles fetched.")
        return
    print(f"\n Step 1 Complete! {len(articles)} articles ready for processing.")

if __name__ == "__main__":
    main()