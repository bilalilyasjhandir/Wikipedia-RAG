import json
import os
import re
from src.config import (ARTICLES_FILE, CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR,)

def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    cutoff_sections = [
        r"\n== See also ==.*",
        r"\n== References ==.*",
        r"\n== External links ==.*",
        r"\n== Further reading ==.*",
        r"\n== Notes ==.*",
        r"\n== Bibliography ==.*",
        r"\n== Sources ==.*",
    ]
    for pattern in cutoff_sections:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text

def extract_sections(text: str) -> list[dict]:
    parts = re.split(r"\n(={2,}\s*.+?\s*={2,})\n", text)
    sections = []
    current_section = "Introduction"
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        header_match = re.match(r"^={2,}\s*(.+?)\s*={2,}$", part)
        if header_match:
            current_section = header_match.group(1)
        else:
            if len(part) > 50:
                sections.append({
                    "section": current_section,
                    "content": part,
                })
    return sections

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(current_chunk) + len(paragraph) + 2 > chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            if len(paragraph) > chunk_size:
                sentence_chunks = _split_by_sentences(paragraph, chunk_size, overlap)
                chunks.extend(sentence_chunks)
                current_chunk = ""
            else:
                if chunks:
                    prev_chunk = chunks[-1]
                    overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
                    clean_break = overlap_text.find(". ")
                    if clean_break != -1:
                        overlap_text = overlap_text[clean_break + 2:]
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def _split_by_sentences(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            if chunks and overlap > 0:
                prev_chunk = chunks[-1]
                overlap_text = prev_chunk[-overlap:]
                clean_break = overlap_text.find(". ")
                if clean_break != -1:
                    overlap_text = overlap_text[clean_break + 2:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def chunk_articles(articles: list[dict]) -> list[dict]:
    print(f"\n Chunking {len(articles)} articles...")
    print(f" Chunk size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars\n")
    all_chunks = []
    chunk_id = 0
    for article in articles:
        title = article["title"]
        url = article["url"]
        content = article["content"]
        cleaned_content = clean_text(content)
        sections = extract_sections(cleaned_content)
        if not sections:
            sections = [{"section": "Full Article", "content": cleaned_content}]
        article_chunk_count = 0
        for section in sections:
            section_chunks = chunk_text(section["content"])
            for i, chunk_text_content in enumerate(section_chunks):
                if len(chunk_text_content) < 100:
                    continue
                chunk = {
                    "chunk_id": f"chunk_{chunk_id}",
                    "text": chunk_text_content,
                    "metadata": {
                        "source_title": title,
                        "source_url": url,
                        "section": section["section"],
                        "chunk_index": i,
                        "char_count": len(chunk_text_content),
                        "word_count": len(chunk_text_content.split()),
                    }
                }
                all_chunks.append(chunk)
                chunk_id += 1
                article_chunk_count += 1
        print(f"{title}: {article_chunk_count} chunks")
    save_chunks(all_chunks)
    print_chunk_summary(all_chunks)
    return all_chunks

def save_chunks(chunks: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\n Saved {len(chunks)} chunks to: {CHUNKS_FILE}")

def load_chunks() -> list[dict]:
    if not os.path.exists(CHUNKS_FILE):
        print(f"No chunks file found at: {CHUNKS_FILE}")
        print("Run the pipeline first: python ingest.py")
        return []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from: {CHUNKS_FILE}")
    return chunks

def print_chunk_summary(chunks: list[dict]) -> None:
    if not chunks:
        return
    total_chars = sum(c["metadata"]["char_count"] for c in chunks)
    total_words = sum(c["metadata"]["word_count"] for c in chunks)
    avg_chars = total_chars // len(chunks)
    avg_words = total_words // len(chunks)
    articles_count = {}
    for chunk in chunks:
        title = chunk["metadata"]["source_title"]
        articles_count[title] = articles_count.get(title, 0) + 1
    print("CHUNKING SUMMARY")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total words: {total_words:,}")
    print(f"Avg chars per chunk: {avg_chars:,}")
    print(f"Avg words per chunk: {avg_words:,}")
    print(f"Articles processed: {len(articles_count)}")
    print(f"Saved to: {CHUNKS_FILE}")
    print(f"\n Chunks per article:")
    for title, count in articles_count.items():
        print(f"{count:3d} chunks ← {title}")