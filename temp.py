import requests
import os
from dotenv import load_dotenv
load_dotenv()

token = os.getenv("HF_API_TOKEN")
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

# Test each model — uncomment one at a time and run

# Option 1: Qwen (HF native)
url = "https://router.huggingface.co/v1/chat/completions"
model = "Qwen/Qwen2.5-7B-Instruct"

# Option 2: Llama via Cerebras
# url = "https://router.huggingface.co/cerebras/v1/chat/completions"
# model = "meta-llama/Llama-3.1-8B-Instruct"

# Option 3: DeepSeek
# url = "https://router.huggingface.co/v1/chat/completions"
# model = "deepseek-ai/DeepSeek-R1"

payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50,
}

response = requests.post(url, headers=headers, json=payload, timeout=60)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")