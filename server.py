
import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from team import Runtime

# Project directory
ROOT = Path(__file__).resolve().parent

# The launcher must set this before importing server.py.
LOCAL_TOKEN = os.environ.get("LOCAL_COMPANION_TOKEN")

if not LOCAL_TOKEN:
    raise RuntimeError(
        "LOCAL_COMPANION_TOKEN is missing. "
        "Start the app using launcher.py instead of server.py."
    )

app = FastAPI(title="Local AI Companion API")

# CORS is not authentication. Keep this list restricted to sites you control.
# Replace the GitHub Pages example with your real username and repository.
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

runtime = Runtime(root=ROOT)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)


def verify_token(
    authorization: str | None = Header(default=None),
):
    expected = f"Bearer {LOCAL_TOKEN}"

    if not authorization or not secrets.compare_digest(
        authorization, expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing companion token.",
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", dependencies=[Depends(verify_token)])
def chat(body: ChatRequest):
    try:
        reply = runtime.quick(body.message)

        # The frontend expects a JSON object with a string named "reply".
        if isinstance(reply, str):
            answer = reply
        else:
            answer = str(reply)

        return {"reply": answer}

    except Exception:
        # Keep internal exception details out of the browser response.
        raise HTTPException(
            status_code=500,
            detail="The local AI failed to generate a response. Check the companion terminal.",
        )
