import os
import time
import httpx
from dotenv import load_dotenv

# Load .env once at import time
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

# Debug logs – remove later if you want
print("[LLM] BASE_URL:", BASE_URL)
print("[LLM] MODEL_NAME:", MODEL_NAME)
print("[LLM] OPENAI_API_KEY present:", bool(OPENAI_API_KEY))

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Make sure it's in your .env and that load_dotenv() can see it."
    )

class LLMClient:
    def __init__(self):
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    def chat(self, messages, max_retries: int = 3):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
        }

        for attempt in range(1, max_retries + 1):
            resp = self._client.post("/chat/completions", json=payload)

            # Debug: print first bit of response text for errors
            if resp.status_code != 200:
                print(
                    f"[LLM] HTTP {resp.status_code}. Response: "
                    f"{resp.text[:300]}..."
                )

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("retry-after", "1"))
                print(
                    f"[LLM] 429 Too Many Requests. Attempt {attempt}/{max_retries}. "
                    f"Sleeping for {retry_after} seconds..."
                )
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        raise RuntimeError(
            "LLM rate-limited (429) after multiple attempts. "
            "Check OpenAI usage/limits."
        )
