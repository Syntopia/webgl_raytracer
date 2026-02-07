import asyncio
import logging
import mimetypes
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("image/vnd.radiance", ".hdr")
mimetypes.add_type("application/wasm", ".wasm")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


async def _read_bytes(path: Path) -> bytes:
    return await asyncio.to_thread(path.read_bytes)


async def app(scope, receive, send):
    if scope["type"] != "http":
        return

    method = scope.get("method", "GET").upper()
    if method not in {"GET", "HEAD"}:
        await send(
            {
                "type": "http.response.start",
                "status": 405,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")],
            }
        )
        await send({"type": "http.response.body", "body": b"Method not allowed."})
        logging.info("%s %s -> 405", method, scope.get("path"))
        return

    raw_path = unquote(scope.get("path", "/"))
    if raw_path == "/":
        raw_path = "/index.html"

    full_path = (ROOT / raw_path.lstrip("/")).resolve()
    if not full_path.is_file() or ROOT not in full_path.parents and full_path != ROOT:
        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")],
            }
        )
        await send({"type": "http.response.body", "body": b"Not found."})
        logging.info("%s %s -> 404", method, raw_path)
        return

    content_type = mimetypes.guess_type(full_path.name)[0] or "application/octet-stream"
    headers = [
        (b"content-type", content_type.encode("utf-8")),
        (b"cache-control", b"no-cache"),
    ]

    body = b""
    if method == "GET":
        body = await _read_bytes(full_path)

    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": body})
    logging.info("%s %s -> 200", method, raw_path)
