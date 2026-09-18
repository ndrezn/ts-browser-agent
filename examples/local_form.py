"""Fill and submit a form on a page you control, using `allow_private=True`.

Runs a tiny local HTTP server so this example has no external dependency and no
network flakiness, then drives it with `Agent` directly rather than the `browse_fast`
tool — `allow_private` is deliberately not exposed to a tool-calling model (see
`ts_browser_agent.safety`), so testing against a local dev server only makes sense from
trusted, non-agent code like this.

Usage:
    uv run --env-file .env python examples/local_form.py
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from ts_browser_agent import Agent

_PAGE = b"""<!doctype html>
<html><body>
  <form method="get" action="/">
    <label>Search <input name="q" /></label>
    <button type="submit">Go</button>
  </form>
  <div id="result"></div>
</body></html>"""


class _FormHandler(BaseHTTPRequestHandler):
    def do_get(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        body = _PAGE
        if "q" in query:
            result = f'<div id="result">You searched for: {query["q"][0]}</div>'.encode()
            body = _PAGE.replace(b'<div id="result"></div>', result)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_get  # http.server requires this exact name; keeps ruff's naming check happy

    def log_message(self, *_args: object) -> None:
        pass  # keep example output focused on the agent, not HTTP access logs


def main() -> None:
    server = HTTPServer(("127.0.0.1", 0), _FormHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"

    try:
        with Agent(url, "Search for 'langchain' and submit.", allow_private=True) as agent:
            for state in agent.run():
                print(state["step"], state["operation"], state["target"])
            print(f"Final status: {agent.status}")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
