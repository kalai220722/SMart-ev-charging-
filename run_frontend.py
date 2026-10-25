#!/usr/bin/env python3
"""
Launcher script for Safe Station Web Application & OpenEnv Backend Server.
Starts the server at http://localhost:8000 and opens the browser.
"""

import sys
import os
import time
import webbrowser
import uvicorn

# Inject paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
server_dir = os.path.join(current_dir, "server")
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)

def main():
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    url = f"http://localhost:{port}"

    print(f"\n=======================================================")
    print(f"  ⚡ SafeStation EV RL Environment Dashboard")
    print(f"  Developed for Meta PyTorch OpenEnv Hackathon 2026")
    print(f"=======================================================")
    print(f"  • Interactive Dashboard: {url}")
    print(f"  • OpenEnv API Docs:      {url}/docs")
    print(f"  • Backend Health:        {url}/health")
    print(f"=======================================================\n")

    # Open browser in a separate thread after a short pause
    import threading
    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()

    from server.app import app
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
