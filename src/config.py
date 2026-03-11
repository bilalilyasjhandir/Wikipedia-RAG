import os
from dotenv import load_dotenv
load_dotenv()

TOPIC = "World War 2"
NUM_ARTICLES = 30
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_BASE_URL = "https://en.wikipedia.org/wiki"
SEARCH_RESULTS_PER_CALL = 50 #wikipedia allows 50 only

DATA_DIR = "data"
ARTICLES_DIR = f"{DATA_DIR}/articles"
ARTICLES_FILE = f"{DATA_DIR}/articles.json"
CHUNKS_FILE = f"{DATA_DIR}/chunks.json"
EMBEDDINGS_FILE = f"{DATA_DIR}/embeddings.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_API_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction"
EMBEDDING_DIMENSIONS = 384
EMBEDDING_BATCH_SIZE = 16 #HF limit

CHROMA_PERSIST_DIR = f"{DATA_DIR}/chroma_db"
COLLECTION_NAME = "wikipedia_articles"