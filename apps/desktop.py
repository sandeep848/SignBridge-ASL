"""Desktop wrapper using same inference API as browser."""
import socket
import threading
import uvicorn
import webview
from .server import app

def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,host="127.0.0.1",port=port,log_level="warning"))
    worker=threading.Thread(target=server.run,daemon=True)
    worker.start()
    try:
        webview.create_window("SignBridge ASL",f"http://127.0.0.1:{port}",width=1150,height=820)
        webview.start()
    finally:
        server.should_exit=True
        worker.join(timeout=8)

if __name__=="__main__":
    main()
