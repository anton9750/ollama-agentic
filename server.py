
import os
import secrets
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from team import Runtime


# --------------------------------------------------
# Application and data paths
# --------------------------------------------------

if getattr(sys, "frozen", False):
    # Bundled read-only resources
    RESOURCE_ROOT = Path(sys._MEIPASS).resolve()
else:
    # Project directory during development
    RESOURCE_ROOT = Path(__file__).resolve().parent

# Writable location for memory, history, and generated output
DATA_ROOT = Path(
    os.environ.get("AGENTIC_DATA_DIR", str(RESOURCE_ROOT))
).resolve()

DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Make bundled configuration and resource files available to Runtime.
# Runtime currently uses one root for both resources and writable data.
# Keep RESOURCE_ROOT here until we update Runtime to separate those paths.
ROOT = RESOURCE_ROOT


# --------------------------------------------------
# Security
# --------------------------------------------------

LOCAL_TOKEN = os.environ.get("LOCAL_COMPANION_TOKEN")

if not LOCAL_TOKEN:
    raise RuntimeError(
        "LOCAL_COMPANION_TOKEN is missing. "
        "Start the app using launcher.py instead of server.py."
    )


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(title="Local AI Companion API")

ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://anton9750.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# --------------------------------------------------
# AI runtime
# --------------------------------------------------

runtime = Runtime(root=ROOT)


# --------------------------------------------------
# Request models
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)


# --------------------------------------------------
# Authentication
# --------------------------------------------------

def verify_token(
    authorization: str | None = Header(default=None),
):
    expected = f"Bearer {LOCAL_TOKEN}"

    if (
        not authorization
        or not secrets.compare_digest(authorization, expected)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing companion token.",
        )


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", dependencies=[Depends(verify_token)])
def chat(body: ChatRequest):
    try:
        reply = runtime.quick(body.message)

        answer = reply if isinstance(reply, str) else str(reply)

        return {"reply": answer}

    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "The local AI failed to generate a response. "
                "Check the companion terminal."
            ),
        )
