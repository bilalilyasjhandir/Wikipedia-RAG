import requests
import time
from src.config import (HF_API_TOKEN, LLM_MODEL, LLM_API_URL, LLM_MAX_TOKENS, LLM_TEMPERATURE,)

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json",
}

def generate_answer(messages: list[dict], retry_count: int = 3) -> str:
    print(f"\n Generating answer with {LLM_MODEL}...")
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "top_p": 0.9,
        "stream": False,
    }
    for attempt in range(retry_count):
        try:
            response = requests.post(
                LLM_API_URL,
                headers=HEADERS,
                json=payload,
                timeout=120,
            )
            if response.status_code == 200:
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                return answer.strip()
            if response.status_code == 503:
                data = response.json()
                wait_time = data.get("estimated_time", 30)
                print(f"Model loading... waiting {wait_time:.0f}s (attempt {attempt + 1}/{retry_count})")
                time.sleep(wait_time)
                continue
            if response.status_code == 429:
                print(f"Rate limited. Waiting 15s...")
                time.sleep(15)
                continue
            print(f"API Error {response.status_code}: {response.text}")
            if attempt < retry_count - 1:
                time.sleep(5)
                continue
        except requests.exceptions.Timeout:
            print(f"Request timed out. Retrying... (attempt {attempt + 1}/{retry_count})")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"Error: {e}")
            if attempt < retry_count - 1:
                time.sleep(5)
                continue
    return "Sorry, I couldn't generate an answer. The LLM service is currently unavailable. Please try again later."