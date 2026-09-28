"""
app.py — FastAPI backend for the Tanglish autocomplete UI.

Endpoints:
  GET  /           → static/index.html  (the chat/autocomplete UI)
  POST /complete   → {"text": "..."} → {"completion": "..."}
  GET  /health     → {"status": "ok", "device": "cpu|cuda"}

Run:
    uvicorn app:app --reload
    # or with a custom model path:
    MODEL_DIR=./models/tanglish-distilgpt2 uvicorn app:app --reload
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allow overriding model path via environment variable
# ---------------------------------------------------------------------------
_model_dir_env = os.getenv("MODEL_DIR")
if _model_dir_env:
    # Patch inference module's default path before importing
    import inference as _inf_mod
    from pathlib import Path as _Path
    _inf_mod._DEFAULT_MODEL_DIR = _Path(_model_dir_env)

from inference import generate_completion, generate_suggestions  # noqa: E402  (after optional patch)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Tanglish Autocomplete",
    description="DistilGPT2 fine-tuned on Tamil-English code-mixed text",
    version="1.0.0",
)

# ── CORS ────────────────────────────────────────────────────────────────────
# Allow all origins so the UI can call the API from any host
# (tighten in production by listing specific origins).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Static files ─────────────────────────────────────────────────────────────
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CompleteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, example="Naan gym ku pogaren")
    max_new_tokens: int = Field(default=15, ge=1, le=100)


class CompleteResponse(BaseModel):
    completion: str


class SuggestRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, example="Naan gym ku")


class SuggestResponse(BaseModel):
    words: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_ui():
    index = _static_dir / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="static/index.html not found")
    return FileResponse(str(index))


@app.get("/health")
async def health():
    import inference as _inf
    return {
        "status": "ok",
        "device": _inf._device,
    }


@app.post("/complete", response_model=CompleteResponse)
async def complete(req: CompleteRequest):
    try:
        completion = generate_completion(
            prompt=req.text,
            max_new_tokens=req.max_new_tokens,
        )
    except RuntimeError as e:
        # Model not yet extracted
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Generation error")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return CompleteResponse(completion=completion)


@app.post("/suggest", response_model=SuggestResponse)
async def suggest(req: SuggestRequest):
    """Return top 3 most likely next words."""
    try:
        from inference import generate_suggestions
        words = generate_suggestions(prompt=req.text, k=3)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Suggest error")
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {e}")

    return SuggestResponse(words=words)
