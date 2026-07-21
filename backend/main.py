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


@app.post("/crawl")
def trigger_crawl(request: Request, secret: str):
    """Call this after deploying, and again whenever you publish new content.
    Protect it with a shared secret so randoms can't trigger recrawls."""
    expected = os.environ.get("CRAWL_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

    from crawler import crawl_site
    site_url = os.environ.get("SITE_URL", "https://neuralninjas.in")
    crawl_site(site_url)
    return {"status": "crawl complete"}
