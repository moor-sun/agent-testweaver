import os
import time
import httpx
from dotenv import load_dotenv

# Load env file once
load_dotenv()

import os
import time
import httpx
from dotenv import load_dotenv

# Load env file once
load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL")
MODEL_NAME = os.getenv("LLM_MODEL_NAME")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")

if not BASE_URL:
    raise RuntimeError("LLM_BASE_URL is missing in .env")

if not MODEL_NAME:
    raise RuntimeError("LLM_MODEL_NAME is missing in .env")

# Detect local LLM (so API key should be ignored)
IS_LOCAL = BASE_URL.startswith("http://localhost") or BASE_URL.startswith("http://127.0.0.1")

print("[LLM] BASE_URL =", BASE_URL)
print("[LLM] MODEL_NAME =", MODEL_NAME)
print("[LLM] API_KEY present =", bool(API_KEY))
print("[LLM] IS_LOCAL =", IS_LOCAL)


class LLMClient:
    def __init__(self):
        headers = {"Content-Type": "application/json"}

        # ONLY send key if not local (Ollama ignores Bearer anyway)
        if not IS_LOCAL:
            headers["Authorization"] = f"Bearer {API_KEY}"

        # Generous timeout for local CPU models
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(300.0, connect=30.0, read=300.0),
        )

    def chat(self, messages, max_retries=3):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
        }

        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.post("/chat/completions", json=payload)

            except httpx.ReadTimeout:
                print(f"[LLM] ReadTimeout on attempt {attempt}/{max_retries}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                raise RuntimeError("LLM timed out. Reduce prompt size or use faster model.")

            if resp.status_code != 200:
                print(f"[LLM] Error {resp.status_code}: {resp.text[:200]}")

            # Retry 429
            if resp.status_code == 429:
                wait = int(resp.headers.get("retry-after", "2"))
                print(f"[LLM] 429: retry after {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        raise RuntimeError("Failed after retries")


# Detect local LLM (so API key should be ignored)
IS_LOCAL = BASE_URL.startswith("http://localhost") or BASE_URL.startswith("http://127.0.0.1")

print("[LLM] BASE_URL =", BASE_URL)
print("[LLM] MODEL_NAME =", MODEL_NAME)
print("[LLM] API_KEY present =", bool(API_KEY))
print("[LLM] IS_LOCAL =", IS_LOCAL)


class LLMClient:
    def __init__(self):
        headers = {"Content-Type": "application/json"}

        # ONLY send key if not local (Ollama ignores Bearer anyway)
        if not IS_LOCAL:
            headers["Authorization"] = f"Bearer {API_KEY}"

        # Generous timeout for local CPU models
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(300.0, connect=30.0, read=300.0),
        )

    def chat(self, messages, max_retries=3):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
        }

        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.post("/chat/completions", json=payload)

            except httpx.ReadTimeout:
                print(f"[LLM] ReadTimeout on attempt {attempt}/{max_retries}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                raise RuntimeError("LLM timed out. Reduce prompt size or use faster model.")

            if resp.status_code != 200:
                print(f"[LLM] Error {resp.status_code}: {resp.text[:200]}")

            # Retry 429
            if resp.status_code == 429:
                wait = int(resp.headers.get("retry-after", "2"))
                print(f"[LLM] 429: retry after {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        raise RuntimeError("Failed after retries")
