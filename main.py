# main.py
import os
import uvicorn
from api.http_api import app

if __name__ == "__main__":
    host = os.getenv("TESTWEAVER_HOST", "0.0.0.0")
    port = int(os.getenv("TESTWEAVER_PORT", "9090"))
    uvicorn.run(app, host=host, port=port)
