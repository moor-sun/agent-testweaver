# main.py
import os
import uvicorn
from .api.http_api import app

if __name__ == "__main__":
    host = os.getenv("TESTWEAVER_HOST", "0.0.0.0")
    port = int(os.getenv("TESTWEAVER_PORT", "9090"))
    uvicorn.run(app, host=host, port=port)

# testweaver/main.py

def app():
    print("TestWeaver entrypoint – app() called")
#poetry run uvicorn testweaver.api.http_api:app --host 0.0.0.0 --port 9090