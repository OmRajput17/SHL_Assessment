# Entry-point shim for HuggingFace Spaces and any platform that expects
# a top-level app.py.  The real application lives in app/main.py.
from app.main import app  # noqa: F401  (re-exported for uvicorn)
