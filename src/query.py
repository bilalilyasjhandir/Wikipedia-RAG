from src.retriever import retrieve
from src.prompt_builder import build_prompt, print_prompt_preview
from src.llm_client import generate_answer
from src.config import TOP_K_RESULTS, LLM_MODEL

def ask(question: str) -> str:
    print("WIKIPEDIA RAG")
    retrieved_chunks = retrieve(question, top_k=TOP_K_RESULTS)
    if not retrieved_chunks:
        answer = "I don't have any relevant information in my Wikipedia sources to answer this question. I can only answer questions about World War 2."
        print("ANSWER")
        print(f"\n{answer}")
        return answer
    messages = build_prompt(question, retrieved_chunks)
    print_prompt_preview(messages)
    answer = generate_answer(messages)
    print("ANSWER")
    print(f"\n{answer}")
    sources = {}
    for chunk in retrieved_chunks:
        title = chunk["metadata"].get("source_title", "Unknown")
        url = chunk["metadata"].get("source_url", "")
        if title not in sources:
            sources[title] = url
    print("\n Sources Available:")
    for i, (title, url) in enumerate(sources.items(), 1):
        print(f"{i}. {title}")
        print(f"{url}")
    return answer

def main():
    print("WIKIPEDIA RAG — WORLD WAR 2 Q&A BOT")
    print(f"Model: {LLM_MODEL}")
    print(f"Retrieval: Top-{TOP_K_RESULTS} chunks")
    print(f"\n Type your question and press Enter.")
    print(f"Type 'quit' or 'exit' to stop.\n")
    while True:
        question = input("\n Your question (Type 'quit' or 'exit' to stop.): ").strip()
        if not question:
            print("Please enter a question. (Type 'quit' or 'exit' to stop.) ")
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\n Goodbye!")
            break
        ask(question)

if __name__ == "__main__":
    main()