"""A tiny API world speaking the AppWorld relay's JSON-lines protocol over a Unix socket, for
offline tests. It counts every call it receives, so tests can prove a replay never reached it."""
import json
import os
import secrets
import socketserver
import tempfile
import threading

DESCRIBE = {
    "supervisor": {"show_account_passwords": {"method": "GET", "parameters": [], "required": []},
                   "complete_task": {"method": "POST", "parameters": ["answer", "status"], "required": []}},
    "spotify": {"login": {"method": "POST", "parameters": ["username", "password"], "required": ["username", "password"]},
                "show_playlist": {"method": "GET", "parameters": ["access_token", "playlist"], "required": []},
                "like_song": {"method": "POST", "parameters": ["access_token", "song_id"], "required": []},
                "show_liked_songs": {"method": "GET", "parameters": ["access_token"], "required": []}},
}


class FakeWorld:
    def __init__(self, playlists=None, password=None):
        self.password = password or f"pw-{secrets.token_hex(4)}"
        self.playlists = playlists or {"Road Trip": [{"id": 11, "title": "Highway", "genre": "rock"},
                                                     {"id": 12, "title": "Desert", "genre": "jazz"},
                                                     {"id": 13, "title": "Coast", "genre": "rock"}],
                                       "Focus": [{"id": 21, "title": "Calm", "genre": "ambient"}]}
        self.tokens, self.liked = set(), []
        self.completed = None
        self.calls, self.changing_calls = 0, 0

    def call(self, app, api, kw):
        if app == "supervisor" and api == "show_account_passwords":
            return [{"account_name": "venmo", "password": "pw-venmo-0000"},
                    {"account_name": "spotify", "password": self.password}]
        if app == "supervisor" and api == "complete_task":
            self.completed = kw.get("answer")
            return {"message": "Marked the active task complete."}
        if app == "spotify" and api == "login":
            if kw.get("username") != "ada@example.com" or kw.get("password") != self.password:
                raise PermissionError("401: invalid credentials")
            tok = f"tok-{secrets.token_hex(6)}"
            self.tokens.add(tok)
            return {"access_token": tok, "token_type": "Bearer"}
        if kw.get("access_token") not in self.tokens:
            raise PermissionError("401: not authenticated")
        if api == "show_playlist":
            return self.playlists[kw["playlist"]]
        if api == "like_song":
            self.liked.append(kw["song_id"])
            return {"message": "liked"}
        if api == "show_liked_songs":
            return list(self.liked)
        raise KeyError(f"{app}.{api}")


class FakeRelay:
    def __init__(self, world: FakeWorld):
        self.world = world
        self.dir = tempfile.mkdtemp(prefix="cs-fakerelay-")
        self.path = os.path.join(self.dir, "relay.sock")
        relay = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                for line in self.rfile:
                    req = json.loads(line)
                    if req["op"] == "describe":
                        out = {"ok": True, "apps": DESCRIBE}
                    else:
                        w = relay.world
                        w.calls += 1
                        method = DESCRIBE.get(req["app"], {}).get(req["api"], {}).get("method", "POST")
                        w.changing_calls += method != "GET"
                        try:
                            out = {"ok": True, "status": "ok", "response": w.call(req["app"], req["api"], req["kwargs"])}
                        except Exception as exc:
                            out = {"ok": True, "status": "exception",
                                   "exception": {"type": type(exc).__name__, "message": str(exc)}}
                    self.wfile.write((json.dumps(out) + "\n").encode())

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True

        self.server = Server(self.path, Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        os.unlink(self.path)
        os.rmdir(self.dir)
