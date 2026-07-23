import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from rag import generate_answer

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Neural Ninjas Chatbot API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# in-memory session store (Phase 1). Swap for Supabase nn_chat_sessions later
# if you need memory to survive server restarts.
SESSIONS: dict[str, list[dict]] = {}
MAX_SESSIONS = 5000


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("8/minute")
def chat(request: Request, body: ChatRequest):
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    history = SESSIONS.get(body.session_id, [])
    result = generate_answer(body.message, history=history)

    history.append({"role": "user", "content": body.message})
    history.append({"role": "assistant", "content": result["answer"]})
    if len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.clear()  # crude cap, fine for Phase 1
    SESSIONS[body.session_id] = history[-10:]

    return {"answer": result["answer"], "sources": result["sources"]}


import threading

crawl_status = {"state": "idle", "pages_done": 0, "total_found": None, "current_url": None}


def _run_crawl(site_url: str):
    from crawler import crawl_site
    crawl_status["state"] = "running"
    crawl_status["pages_done"] = 0
    crawl_status["current_url"] = None
    try:
        crawl_site(site_url, status=crawl_status)
        crawl_status["state"] = "done"
    except Exception as e:
        crawl_status["state"] = f"error: {e}"


@app.post("/crawl")
def trigger_crawl(request: Request, secret: str):
    """Kicks off a crawl in the background and returns immediately, so the
    HTTP request doesn't sit open long enough to hit a gateway timeout.
    Check progress with GET /crawl-status."""
    expected = os.environ.get("CRAWL_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

    if crawl_status["state"] == "running":
        return {"status": "already running", "progress": crawl_status}

    site_url = os.environ.get("SITE_URL", "https://neuralninjas.in")
    thread = threading.Thread(target=_run_crawl, args=(site_url,), daemon=True)
    thread.start()
    return {"status": "crawl started in background", "check_progress_at": "/crawl-status"}


@app.get("/crawl-status")
def get_crawl_status():
    return crawl_status
