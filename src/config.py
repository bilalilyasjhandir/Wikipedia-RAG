TOPIC = "World War 2"
NUM_ARTICLES = 30
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_BASE_URL = "https://en.wikipedia.org/wiki"
SEARCH_RESULTS_PER_CALL = 50 #wikipedia allows 50 only

DATA_DIR = "data"
ARTICLES_DIR = f"{DATA_DIR}/articles"
ARTICLES_FILE = f"{DATA_DIR}/articles.json"
CHUNKS_FILE = f"{DATA_DIR}/chunks.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200