"""Server HTTP lokal untuk laporan HTML SiODI.

Server hanya mendengarkan di 127.0.0.1. Server ini membuat laporan dapat
mengakses file hasil melalui HTTP, sehingga browser dapat menjalankan unduhan
secara konsisten tanpa batasan URL ``file://``.
"""
import functools
import http.server
import mimetypes
import os
import shutil
import threading
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


class _LoopbackServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class _ReportRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Melayani file biasa dan endpoint unduhan paksa ``/_download``."""

    def log_message(self, format, *args):
        # Akses laporan tidak perlu memenuhi terminal/log aplikasi.
        return

    def do_GET(self):
        request = urlparse(self.path)
        if request.path == "/_download":
            self._send_download(parse_qs(request.query).get("path", [""])[0])
            return
        super().do_GET()

    def _send_download(self, relative_path: str):
        root = Path(self.directory).resolve()
        target = (root / unquote(relative_path)).resolve()
        try:
            is_within_root = os.path.commonpath((str(root), str(target))) == str(root)
        except ValueError:
            is_within_root = False

        if not is_within_root or not target.is_file():
            self.send_error(404, "File tidak ditemukan")
            return

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        safe_filename = quote(target.name, safe="")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{safe_filename}")
        self.end_headers()
        with target.open("rb") as source:
            shutil.copyfileobj(source, self.wfile)


class ReportServer:
    """Menyajikan satu folder hasil SiODI secara lokal melalui HTTP."""

    def __init__(self, root_dir: str):
        self.root_dir = str(Path(root_dir).resolve())
        handler = functools.partial(_ReportRequestHandler, directory=self.root_dir)
        self._httpd = _LoopbackServer(("127.0.0.1", 0), handler)
        self._thread = None

    def start(self) -> str:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._httpd.serve_forever,
                name="SiODI-ReportServer",
                daemon=True,
            )
            self._thread.start()
        return self.base_url

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def url_for(self, relative_path: str) -> str:
        relative_url = quote(relative_path.replace("\\", "/"), safe="/")
        return f"{self.start()}/{relative_url.lstrip('/')}"

    def stop(self) -> None:
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join(timeout=2)
            self._thread = None
        self._httpd.server_close()
