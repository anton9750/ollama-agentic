
import os
import secrets
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn


HOST = "127.0.0.1"
PORT = 8765
WEBSITE_URL = "https://anton9750.github.io/ollama-agentic/"


def main():
    token = secrets.token_urlsafe(32)
    os.environ["LOCAL_COMPANION_TOKEN"] = token

    if getattr(sys, "frozen", False):
        # Folder containing the packaged Windows executable
        app_dir = Path(sys.executable).resolve().parent

        # Permanent, writable storage for the current Windows user
        data_dir = (
            Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
            / "AgenticAICompanion"
        )
        data_dir.mkdir(parents=True, exist_ok=True)

        os.environ["AGENTIC_APP_ROOT"] = str(app_dir)
        os.environ["AGENTIC_DATA_DIR"] = str(data_dir)

    else:
        # Normal development mode: use the project directory
        app_dir = Path(__file__).resolve().parent
        os.environ["AGENTIC_APP_ROOT"] = str(app_dir)
        os.environ["AGENTIC_DATA_DIR"] = str(app_dir)

    from server import app

    config = uvicorn.Config(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )

    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(100):
        if server.started:
            break

        if not thread.is_alive():
            raise RuntimeError("The local server failed to start.")

        time.sleep(0.2)
    else:
        raise RuntimeError("The local server did not start in time.")

    webbrowser.open(
        WEBSITE_URL + "#local-companion-token=" + token
    )

    print("Local AI companion is running.")
    print("Keep this window open while using the assistant.")
    print("Press Ctrl+C to stop the companion.")

    try:
        while thread.is_alive():
            thread.join(timeout=1)
    except KeyboardInterrupt:
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == "__main__":
    main()
