import requests
import json
import os
import time
from src.config import (WIKI_API_URL, WIKI_BASE_URL, TOPIC, NUM_ARTICLES, SEARCH_RESULTS_PER_CALL, ARTICLES_DIR, ARTICLES_FILE,)

HEADERS = {"User-Agent": "WikipediaRAGBot/1.0 (https://github.com/bilalilyasjhandir/Wikipedia-RAG; bilalilyasjhandir@gmail.com)"}

def search_wikipedia(topic: str, num_results: int) -> list[dict]:
    print(f"\n Searching Wikipedia for '{topic}'...")
    print(f" Requesting {num_results} articles...\n")
    search_results = []
    offset = 0  #for pagination
    while len(search_results) < num_results:
        remaining = num_results - len(search_results)
        limit = min(remaining, SEARCH_RESULTS_PER_CALL)
        params = {
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "srnamespace": 0,
            "srlimit": limit,
            "sroffset": offset,
            "format": "json",
        }
        response = requests.get(WIKI_API_URL, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            print(f"No more results found. Got {len(search_results)} total.")
            break
        for result in results:
            search_results.append({
                "title": result["title"],
                "pageid": result["pageid"],
            })
        print(f"Found {len(search_results)} articles so far...")
        offset += limit
        time.sleep(0.5)
    print(f"\n Total articles found: {len(search_results)}")
    return search_results[:num_results]

def fetch_article_content(title: str, pageid: int) -> dict | None:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "exintro": False,
        "explaintext": True,
        "inprop": "url",
        "titles": title,
        "format": "json",
    }
    try:
        response = requests.get(WIKI_API_URL, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if int(page_id) < 0:
                print(f"Article not found: {title}")
                return None
            content = page_data.get("extract", "")
            if len(content) < 500:
                print(f"Skipping (too short): {title} ({len(content)} chars)")
                return None
            article = {
                "title": page_data.get("title", title),
                "pageid": int(page_id),
                "url": page_data.get("fullurl", f"{WIKI_BASE_URL}/{title.replace(' ', '_')}"),
                "content": content,
                "content_length": len(content),
                "word_count": len(content.split()),
            }
            return article
    except requests.RequestException as e:
        print(f"Error fetching '{title}': {e}")
        return None
    return None

def fetch_all_articles(topic: str = TOPIC, num_articles: int = NUM_ARTICLES) -> list[dict]:
    search_results = search_wikipedia(topic, num_articles)
    if not search_results:
        print("No articles found. Try a different topic.")
        return []
    print(f"\n Fetching full content for {len(search_results)} articles...\n")
    articles = []
    for i, result in enumerate(search_results, 1):
        title = result["title"]
        pageid = result["pageid"]
        print(f"   [{i}/{len(search_results)}] Fetching: {title}...", end=" ")
        article = fetch_article_content(title, pageid)
        if article:
            articles.append(article)
            print(f"({article['word_count']} words)")
        else:
            print("")
        time.sleep(0.5)
    save_articles(articles)
    print_summary(articles)
    return articles

def save_articles(articles: list[dict]) -> None:
    os.makedirs(ARTICLES_DIR, exist_ok=True)
    with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"\n Saved {len(articles)} articles to: {ARTICLES_FILE}")
    for article in articles:
        safe_title = article["title"].replace("/", "-").replace("\\", "-")
        filepath = os.path.join(ARTICLES_DIR, f"{safe_title}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(article, f, ensure_ascii=False, indent=2)
    print(f"Individual articles saved to: {ARTICLES_DIR}/")

def load_articles() -> list[dict]:
    if not os.path.exists(ARTICLES_FILE):
        print(f" No articles file found at: {ARTICLES_FILE}")
        print(" Run the fetcher first: python ingest.py")
        return []
    with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"Loaded {len(articles)} articles from: {ARTICLES_FILE}")
    return articles

def print_summary(articles: list[dict]) -> None:
    if not articles:
        return
    total_words = sum(a["word_count"] for a in articles)
    avg_words = total_words // len(articles)
    print(f"FETCH SUMMARY")
    print(f"Topic: {TOPIC}")
    print(f"Articles fetched: {len(articles)}")
    print(f"Total words: {total_words:,}")
    print(f"Average words: {avg_words:,} per article")
    print(f"Saved to: {ARTICLES_FILE}")
    print(f"\n Articles fetched:")
    for i, article in enumerate(articles, 1):
        print(f"{i:2d}. {article['title']}")
        print(f"{article['word_count']:,} words | {article['url']}")