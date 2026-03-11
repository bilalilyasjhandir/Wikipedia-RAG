from src.embedder import embed_query
from src.vector_store import search
from src.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD

def retrieve(question: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    print(f"\n Retrieving top-{top_k} relevant passages...")
    print(f"Question: \"{question}\"")
    print(f"Embedding question...", end=" ", flush=True)
    query_embedding = embed_query(question)
    print("Done")
    print(f"Searching vector store...", end=" ", flush=True)
    raw_results = search(query_embedding, n_results=top_k)
    print("Done")
    results = []
    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]
    ids = raw_results.get("ids", [[]])[0]

    for i, (doc, meta, dist, chunk_id) in enumerate(zip(documents, metadatas, distances, ids)):
        similarity = 1 - dist
        if similarity < SIMILARITY_THRESHOLD:
            continue
        results.append({
            "rank": i + 1,
            "chunk_id": chunk_id,
            "text": doc,
            "metadata": meta,
            "similarity": round(similarity, 4),
        })
    if results:
        print(f"\n Retrieved {len(results)} relevant passages:")
        for r in results:
            title = r["metadata"].get("source_title", "Unknown")
            section = r["metadata"].get("section", "Unknown")
            sim = r["similarity"]
            preview = r["text"][:80].replace("\n", " ") + "..."
            print(f"   [{r['rank']}] (similarity: {sim:.4f}) {title} → {section}")
            print(f"       \"{preview}\"")
        print("-" * 60)
    else:
        print(f"\n No passages passed the similarity threshold ({SIMILARITY_THRESHOLD})")
        print(f"This question is likely outside the knowledge base.")
    return results