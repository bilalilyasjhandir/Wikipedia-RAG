SYSTEM_PROMPT = """You are a World War 2 Q&A assistant. You answer questions ONLY using the provided Wikipedia article excerpts below.
STRICT RULES:
1. ONLY use information from the provided context below. Do NOT use your own knowledge.
2. For every claim you make, cite the source using [Source: Article Title].
3. If the context does not contain enough information to answer the question, say: "I don't have enough information in my sources to answer this question."
4. Be concise but thorough. Use 2-4 paragraphs.
5. If multiple sources discuss the same topic, synthesize them and cite all relevant sources.
6. NEVER make up facts or dates that are not in the provided context."""

def build_prompt(question: str, retrieved_chunks: list[dict]) -> list[dict]:
    context_parts = []
    sources_used = {}
    for chunk in retrieved_chunks:
        title = chunk["metadata"].get("source_title", "Unknown")
        section = chunk["metadata"].get("section", "Unknown")
        url = chunk["metadata"].get("source_url", "")
        text = chunk["text"]
        context_parts.append(
            f"[Source: {title} | Section: {section}]\n{text}"
        )
        if title not in sources_used:
            sources_used[title] = url
    context_block = "\n\n---\n\n".join(context_parts)
    source_list = "\n".join(
        f"  {i}. {title} — {url}"
        for i, (title, url) in enumerate(sources_used.items(), 1)
    )
    user_message = f"CONTEXT (Wikipedia Article Excerpts): {context_block}, AVAILABLE SOURCES: {source_list}, QUESTION: {question} Please answer the question using ONLY the context above. Cite your sources using [Source: Article Title]."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return messages

def print_prompt_preview(messages: list[dict]) -> None:
    user_msg = messages[-1]["content"]
    source_count = user_msg.count("[Source:")
    print(f"\n Prompt built:")
    print(f"System prompt: {len(messages[0]['content'])} chars")
    print(f"User prompt:   {len(user_msg)} chars")
    print(f"Sources in context: {source_count}")