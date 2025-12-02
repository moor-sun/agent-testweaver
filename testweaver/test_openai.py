import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load .env file

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
model = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

print("BASE_URL:", base_url)
print("MODEL:", model)
print("API KEY:", api_key[:8] + "********")  # Just to verify it's loaded

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say hello from TestWeaver"}],
)

print("REPLY:", resp.choices[0].message.content)
