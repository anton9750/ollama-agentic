
import os
import secrets
import threading
import time
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8765

# Replace these with your actual GitHub Pages URL.
WEBSITE_URL = "https://anton9750.github.io/ollama-agentic/"


def main():
    token = secrets.token_urlsafe(32)
    os.environ["LOCAL_COMPANION_TOKEN"] = token

    # Import only after setting the token.
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

    # Wait for the local server to start.
    for _ in range(100):
        if server.started:
            break
        if not thread.is_alive():
            raise RuntimeError("The local server failed to start.")
        time.sleep(0.2)
    else:
        raise RuntimeError("The local server did not start in time.")

    # The token is passed in the URL fragment, not sent in the HTTP request.
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
