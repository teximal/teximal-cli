"""
`teximal serve`: a Teximal model behind the request shape of OpenRouter's Decisions API (the one TypeSafe's Jev
is served through), so switching from Jev is a change of URL. Local, one model, requests answered one at a time.

  POST /api/alpha/decisions   {"model": "...", "state": "<text or fields>", "questions": {...}}
                              -> {"model": "...", "answers": {...}, "usage": {...}, "ms": ...}
  Bulk, Teximal's addition: "states": [...] in place of "state" -> {"results": [<one response per state>]}.
"""
import json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()


def make_handler(model):
    def respond(body):
        t0 = time.perf_counter()
        answers = model.decide(body["state"], body["questions"])
        return {"model": model.name, "answers": answers, "usage": {"output_tokens": 0, "cost": 0},
                "ms": round(1000 * (time.perf_counter() - t0), 1)}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send(200, {"model": model.name, "post": "/api/alpha/decisions"})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/api/alpha/decisions", "/v1/decisions"):
                return self.send(404, {"error": "POST /api/alpha/decisions"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                with LOCK:
                    res = ({"results": [respond(dict(body, state=s)) for s in body["states"]]}
                           if "states" in body else respond(body))
            except KeyError as e:
                return self.send(400, {"error": f"missing field {e.args[0]!r}"})
            except (ValueError, TypeError) as e:
                return self.send(400, {"error": str(e)})
            self.send(200, res)

    return Handler


def serve(model, host="127.0.0.1", port=8766):
    print(f"{model.name} at http://{host}:{port}/api/alpha/decisions", flush=True)
    ThreadingHTTPServer((host, port), make_handler(model)).serve_forever()
