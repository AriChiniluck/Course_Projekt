"""Launch the Telekom Support API server.

Run:
    python run_api.py

The server starts on http://127.0.0.1:8000
Interactive docs available at http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import uvicorn
from api import app  # imports config, agents, graph — same as support_main.py

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
