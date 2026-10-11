"""Local IDP/WebVPN/iCourse chain simulator for chain-level login tests.

Trimmed from the NIGHT4 login-testbench harness (archive result
product-night4-login-testbench-result-20260921). Zero real school
endpoints: binds 127.0.0.1:0 only. Mirrors the endpoint chain consumed by
courselens_worker.platform_session (_login_webvpn / _login_course):

  GET  /idp/authCenter/authenticate?service=...   redirect chain carrying lck
  POST /idp/authn/queryAuthMethods                authChainCode + requestType
  GET  /idp/authn/getJsPublicKey                  RSA public key (PKCS1 v1.5)
  POST /idp/authn/authExecute                     loginToken (single-use lck)
  POST /idp/authCenter/authnEngine                HTML locationValue ticket
  GET  /webvpn-ticket?ticket=...                  Set-Cookie + 302 portal
  GET  / (verify), /icourse/...                   portal + userinfo

WebVPN-wrapped URLs (/http/<iv><ct>/<path>, AES-128-CFB key/iv identical to
the product) are unwrapped for GETs; wrapped POSTs match by path suffix.

Failure modes (one roll per webvpn context entry):
  delivered      clean chain, full login succeeds
  not_delivered  authenticate returns a plain 200 page with no lck
  challenge      authenticate returns a JS challenge interstitial, no lck
  sso_direct     P65: valid IDP SSO skips the form, issues a ticket with no
                 lck anywhere in the chain (already-authed variant)
  sso_then_delivered  P65: first entry serves a stale (rejected) ticket, the
                 fresh-cookie retry lands on the delivered chain
  challenge_at_course  夜10-C: webvpn leg delivered, course entry serves the
                 risk-check interstitial (no lck) — independent zero-retry code
  expire_then_delivered  夜10-C: webvpn leg delivered, first course entry
                 lands on a plain login page (mid-flow expiry, no lck); the
                 outer ladder's full relogin recovers on the second entry
"""

import json
import socket
import struct
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlsplit

from Crypto.Cipher import AES, PKCS1_v1_5
from Crypto.PublicKey import RSA

VPN_KEY = b"wrdvpnisthebest!"
VPN_IV = b"wrdvpnisthebest!"
MODES = ("delivered", "not_delivered", "challenge",
         "sso_direct", "sso_then_delivered",
         "challenge_at_course", "expire_then_delivered")

_lock = threading.Lock()
_mode = {"mode": "delivered"}
STATS = {"webvpn_entries": 0, "requests": 0, "course_entries": 0}
SERVER = None

_key = RSA.generate(2048)
_PUB_B64 = "\n".join(_key.publickey().export_key(format="PEM")
                     .decode("ascii").strip().splitlines()[1:-1])

_LCKS = {}
_LOGIN_TOKENS = {}
_TICKETS = {}

_CHALLENGE_HTML = (
    "<html><head><meta http-equiv='refresh' content='2'></head>"
    "<body><div id='challenge'>risk-check</div>"
    "<script>document.cookie='risk=1';</script></body></html>")
_PLAIN_LOGIN_HTML = ("<html><body><form action='/login'>"
                     "<input name='password'></form>login page</body></html>")
# P65：陈旧票据的落地页——不是登录表单（无 action 落点）也不是挑战页，
# 正是「三者皆非」的已登录变体形态。
_STALE_TICKET_HTML = "<html><body>webvpn session expired</body></html>"
_USER = {"id": "u1", "account": "tester", "tenant_id": "222", "phone": "13800000000"}


def set_mode(mode):
    assert mode in MODES, mode
    with _lock:
        _mode["mode"] = mode
        STATS.update(webvpn_entries=0, requests=0, course_entries=0)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8", headers=None):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or []):
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200, headers=None):
        self._send(code, json.dumps(obj), "application/json", headers)

    def _redirect(self, location, cookie=None):
        headers = [("Location", location)]
        if cookie:
            headers.append(("Set-Cookie", cookie))
        self._send(302, b"", headers=headers)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def do_POST(self):
        with _lock:
            STATS["requests"] += 1
        path = urlsplit(self.path).path
        raw = self._body().decode("utf-8")
        # Wrapped POSTs keep the inner path as their suffix.
        if path.endswith("/idp/authn/queryAuthMethods"):
            self._json({"requestType": "chain_type",
                        "data": [{"moduleCode": "userAndPwd",
                                  "authChainCode": "chain-userpwd"}]})
            return
        if path.endswith("/idp/authn/authExecute"):
            data = json.loads(raw or "{}")
            with _lock:
                att_id = _LCKS.get(str(data.get("lck") or ""), "")
                if not att_id or att_id in _LOGIN_TOKENS.values():
                    self._json({"code": "409", "message": "lck consumed"})
                    return
                lt = "LT-" + uuid.uuid4().hex
                _LOGIN_TOKENS[lt] = att_id
            self._json({"code": "200", "loginToken": lt})
            return
        if path.endswith("/idp/authCenter/authnEngine"):
            form = parse_qs(raw or "")
            with _lock:
                lt = str((form.get("loginToken") or [""])[0])
                att_id = _LOGIN_TOKENS.get(lt, "")
                if not lt or not att_id or att_id in _TICKETS.values():
                    self._send(200, "<html>no ticket here</html>")
                    return
                ticket = "ST-" + uuid.uuid4().hex
                _TICKETS[ticket] = att_id
            host = self.headers.get("Host", "127.0.0.1")
            leg = "webvpn" if att_id.startswith("W-") else "course"
            url = f"http://{host}/{leg}-ticket?ticket={ticket}"
            self._send(200, f'<html><body><script>locationValue="{url}";'
                            "</script></body></html>")
            return
        self._json({"code": "404"}, 404)

    def do_GET(self):
        with _lock:
            STATS["requests"] += 1
        parts = urlsplit(self.path)
        path, query = parts.path, parse_qs(parts.query)
        if path == "/idp/authCenter/authenticate":
            self._entry_chain("webvpn")
            return
        if path == "/idp/ac/ctx":
            self._redirect(f"/idp/ac/login?lck={(query.get('next') or [''])[0]}")
            return
        if path == "/idp/authn/getJsPublicKey":
            self._json({"data": _PUB_B64})
            return
        if path == "/webvpn-ticket":
            self._ticket_follow("webvpn", query)
            return
        if path in ("/icourse-ticket", "/course-ticket"):
            self._ticket_follow("course", query)
            return
        if path in ("/", "/webvpn/portal"):
            cookie = self.headers.get("Cookie", "")
            if "wengine_vpn_ticket" in cookie:
                self._send(200, "<html>portal-ok</html>")
            else:
                self._redirect("/webvpn/login")
            return
        if path == "/icourse/portal":
            self._send(200, "<html>icourse-ok</html>")
            return
        # 夜10-C：直连拓扑别名（真实直连腿打 ICOURSE_BASE 根级路径，不经
        # /icourse 前缀也不经 vpn 包裹）——没有这两条别名时 _login_course_direct
        # 在模拟器上永远 404，直连腿整体不可测。
        if path in ("/icourse/casapi/index.php", "/casapi/index.php"):
            self._entry_chain("course")
            return
        if path in ("/icourse/userapi/v1/infosimple", "/userapi/v1/infosimple"):
            self._json({"code": 200, "params": _USER})
            return
        if path == "/webvpn/login":
            self._send(200, _PLAIN_LOGIN_HTML)
            return
        if path.startswith("/http/"):
            self._vpn_unwrap(parts)
            return
        self._send(404, "not found")

    def _entry_chain(self, leg):
        att_id = ("W-" if leg == "webvpn" else "C-") + uuid.uuid4().hex
        if leg == "webvpn":
            with _lock:
                STATS["webvpn_entries"] += 1
                mode = _mode["mode"]
            if mode == "sso_direct":
                # P65：IDP SSO 有效——跳过登录表单，不发票据 lck，直发票。
                ticket = "ST-" + uuid.uuid4().hex
                with _lock:
                    _TICKETS[ticket] = att_id
                self._redirect(f"/webvpn-ticket?ticket={ticket}")
                return
            if mode == "sso_then_delivered" and STATS["webvpn_entries"] == 1:
                # P65：首次进入撞陈旧票据（验证不过），清态重试后走 delivered。
                self._redirect("/webvpn-ticket?ticket=ST-bogus")
                return
            if mode in ("not_delivered", "challenge"):
                self._send(200, _CHALLENGE_HTML if mode == "challenge"
                           else _PLAIN_LOGIN_HTML)
                return
        if leg == "course":
            with _lock:
                STATS["course_entries"] += 1
                mode = _mode["mode"]
            if mode == "challenge_at_course":
                # 夜10-C：课程腿挑战页变体——webvpn 腿已成功，课程入口撞
                # 人机确认墙（无 lck），检验独立码零重试语义。
                self._send(200, _CHALLENGE_HTML)
                return
            if mode == "expire_then_delivered" and STATS["course_entries"] == 1:
                # 夜10-C：半途过期——webvpn 腿已成功，课程会话过期落在
                # 普通登录页（无 lck），外层梯全链重跑后应恢复。
                self._send(200, _PLAIN_LOGIN_HTML)
                return
        lck = "LCK-" + uuid.uuid4().hex
        with _lock:
            _LCKS[lck] = att_id
        self._redirect(f"/idp/ac/ctx?att={att_id}&leg={leg}&next={lck}")

    def _ticket_follow(self, leg, query):
        ticket = (query.get("ticket") or [""])[0]
        with _lock:
            att_id = _TICKETS.get(ticket, "")
            if not ticket or not att_id:
                self._send(200, _STALE_TICKET_HTML)
                return
        if leg == "webvpn":
            self._redirect("/webvpn/portal",
                           cookie="wengine_vpn_ticketwebvpn=1; Path=/")
            return
        token = uuid.uuid4().hex
        php = '{{i:0;s:6:"_token";i:1;s:32:"%s";}}' % token
        self._redirect("/icourse/portal",
                       cookie="S=" + quote(php, safe="") + "; Path=/")

    def _vpn_unwrap(self, parts):
        rest = parts.path[len("/http/"):]
        iv_hex, ct_hex = rest[:32], rest[32:]
        slash = ct_hex.find("/")
        inner_path = "/"
        if slash >= 0:
            ct_hex, inner_path = ct_hex[:slash], ct_hex[slash:]
        try:
            cipher = AES.new(VPN_KEY, AES.MODE_CFB,
                             bytes.fromhex(iv_hex), segment_size=128)
            host = cipher.decrypt(bytes.fromhex(ct_hex)).decode("utf-8", "ignore")
        except ValueError:
            self._send(404, "bad wrap")
            return
        if host != "127.0.0.1":
            self._send(404, "bad host")
            return
        if inner_path.endswith("/idp/authn/getJsPublicKey"):
            self._json({"data": _PUB_B64})
            return
        if inner_path.endswith("/userapi/v1/infosimple"):
            self._json({"code": 200, "params": _USER})
            return
        if "casapi/index.php" in inner_path:
            self._entry_chain("course")
            return
        if "/webvpn-ticket" in inner_path:
            self._ticket_follow("webvpn", parse_qs(parts.query))
            return
        self._send(404, "unwrapped no route: " + inner_path[:80])


def _quiet_handle_error(self, request, client_address):
    exc = sys.exc_info()[1]
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError,
                        BrokenPipeError, TimeoutError)):
        return  # abandoned keep-alive reads are by design
    print(json.dumps({"sim_error": repr(exc)[:200]}), flush=True)


def serve():
    global SERVER
    SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    SERVER.handle_error = _quiet_handle_error.__get__(SERVER)
    SERVER.serve_forever(poll_interval=0.2)


def start():
    """Start the simulator thread; return the base URL plus redirect helpers."""
    global SERVER
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while SERVER is None and time.time() < deadline:
        time.sleep(0.05)
    if SERVER is None:
        raise RuntimeError("sim did not start")
    return "http://127.0.0.1:%d" % SERVER.server_address[1]
