from __future__ import annotations

import unittest
from unittest.mock import patch

import requests

from courselens_worker.mailbox import API_ROOT, ISSUE_LABEL, TITLE_PREFIX, MailboxError, IssueMailbox
from courselens_worker.protocol import ProtocolError, chunk_envelope, join_envelope

TASK_ID = "0123456789abcdef0123456789abcdef"
ENVELOPE = {"schema": "mailbox.test.v1", "payload": {"n": 1}}


class FakeResponse:
    def __init__(self, status_code, payload=None, request_id="REQ-1"):
        self.status_code = status_code
        self._payload = payload
        self.headers = {"X-GitHub-Request-Id": request_id}

    def json(self):
        return self._payload


class FakeSession:
    """Records every call and replays per-(method, path) canned responses."""

    def __init__(self):
        self.calls = []
        self.responses = {}

    def expect(self, method, path, response):
        self.responses[(method, path)] = response

    def _reply(self, method, path, **kwargs):
        # The mailbox client passes absolute URLs to the session; record and
        # address calls by their repository-relative path.
        if path.startswith(API_ROOT):
            path = path[len(API_ROOT):]
        self.calls.append((method, path, kwargs))
        return self.responses[(method, path)]

    def get(self, path, params=None, timeout=None):
        return self._reply("GET", path, params=params, timeout=timeout)

    def post(self, path, json=None, timeout=None):
        return self._reply("POST", path, payload=json, timeout=timeout)

    def patch(self, path, json=None, timeout=None):
        return self._reply("PATCH", path, payload=json, timeout=timeout)


class RaisingSession:
    def get(self, path, params=None, timeout=None):
        raise requests.ConnectionError("connection reset")

    def post(self, path, json=None, timeout=None):
        raise requests.ConnectionError("connection reset")

    def patch(self, path, json=None, timeout=None):
        raise requests.ConnectionError("connection reset")


def _mailbox(session=None) -> IssueMailbox:
    # RR-FIX452-1：默认走不限速/零退避参数，既有用例不因发布节流变慢；
    # 节流与重试语义由 test_checkpoint_publish.py 专测。
    box = IssueMailbox(
        "student/jobs",
        "tok",
        publish_rate_per_minute=6e7,
        publish_burst=1e6,
        retry_waits=(0.0, 0.0, 0.0),
    )
    if session is not None:
        box.session = session
    return box


def _encoded() -> str:
    return chunk_envelope(ENVELOPE)[0]


def _issue_listing(number: int = 7) -> list[dict]:
    return [{"number": number, "title": f"{TITLE_PREFIX} {TASK_ID}"}]


class MailboxTests(unittest.TestCase):
    def test_constructor_requires_repo_and_token(self):
        for repo, token in (("", "tok"), ("student/jobs", "")):
            with self.subTest(repo=repo, token=token):
                with self.assertRaises(ValueError):
                    IssueMailbox(repo, token)

    def test_constructor_configures_session_headers_and_timeout(self):
        box = IssueMailbox("student/jobs", "tok", timeout=5)
        self.assertEqual(box.timeout, 5)
        self.assertEqual(box.session.headers["Authorization"], "Bearer tok")
        self.assertEqual(box.session.headers["X-GitHub-Api-Version"], "2026-03-10")
        self.assertEqual(box.session.headers["User-Agent"], "Fudan-CourseLens-Worker/2")
        self.assertIsNone(box.issue_number)
        self.assertIsNone(box.status_comment_id)

    def test_http_status_mappings_for_get_post_patch(self):
        session = FakeSession()
        session.expect("GET", "/repos/student/jobs/issues", FakeResponse(404, {}, request_id="REQ-9"))
        session.expect("POST", "/repos/student/jobs/issues/7/comments", FakeResponse(200, {}))
        session.expect("PATCH", "/repos/student/jobs/issues/comments/1", FakeResponse(201, {}))
        box = _mailbox(session)
        box.issue_number = 7
        box.status_comment_id = 1
        with self.subTest(method="get"):
            with self.assertRaisesRegex(MailboxError, r"HTTP 404 .*REQ-9"):
                box._get("/repos/student/jobs/issues")
        with self.subTest(method="post"):
            with self.assertRaisesRegex(MailboxError, "HTTP 200"):
                box._post("/repos/student/jobs/issues/7/comments", payload={"body": "x"})
        with self.subTest(method="patch"):
            with self.assertRaisesRegex(MailboxError, "HTTP 201"):
                box._patch("/repos/student/jobs/issues/comments/1", payload={"body": "x"})
        ok = FakeSession()
        ok.expect("GET", "/repos/student/jobs/issues", FakeResponse(200, {"hello": "yes"}))
        ok.expect("POST", "/repos/student/jobs/issues/7/comments", FakeResponse(201, {"id": 3}))
        ok.expect("PATCH", "/repos/student/jobs/issues/comments/3", FakeResponse(200, {"id": 3}))
        good = _mailbox(ok)
        self.assertEqual(good._get("/repos/student/jobs/issues"), {"hello": "yes"})
        self.assertEqual(
            good._post("/repos/student/jobs/issues/7/comments", payload={"body": "x"}), {"id": 3}
        )
        self.assertEqual(
            good._patch("/repos/student/jobs/issues/comments/3", payload={"body": "x"}), {"id": 3}
        )

    def test_transport_exception_wrapped_as_mailbox_error(self):
        box = _mailbox(RaisingSession())
        with self.assertRaisesRegex(MailboxError, "ConnectionError"):
            box._get("/repos/student/jobs/issues")

    def test_read_selects_issue_by_exact_title_and_records_issue_number(self):
        session = FakeSession()
        session.expect("GET", "/repos/student/jobs/issues", FakeResponse(200, [
            {"number": 5, "title": f"{TITLE_PREFIX} other-task"},
            {"number": 7, "title": f"{TITLE_PREFIX} {TASK_ID}"},
        ]))
        whole = _encoded()
        session.expect(
            "GET", "/repos/student/jobs/issues/7/comments",
            FakeResponse(200, [{"body": f"part 1/1\n{whole}"}]),
        )
        box = _mailbox(session)
        self.assertEqual(box.read(TASK_ID.upper()), ENVELOPE)
        self.assertEqual(box.issue_number, 7)
        method, path, kwargs = session.calls[0]
        self.assertEqual((method, path), ("GET", "/repos/student/jobs/issues"))
        self.assertEqual(
            kwargs["params"],
            {"state": "open", "labels": ISSUE_LABEL, "per_page": 100},
        )

    def test_read_without_matching_issue_is_unavailable(self):
        session = FakeSession()
        session.expect("GET", "/repos/student/jobs/issues", FakeResponse(200, []))
        with self.assertRaisesRegex(MailboxError, "not available"):
            _mailbox(session).read(TASK_ID)

    def test_read_joins_out_of_order_prefixed_parts_and_skips_noise(self):
        whole = _encoded()
        half = len(whole) // 2
        session = FakeSession()
        session.expect("GET", "/repos/student/jobs/issues", FakeResponse(200, _issue_listing()))
        session.expect("GET", "/repos/student/jobs/issues/7/comments", FakeResponse(200, [
            {"body": "operator note"},
            {"body": f"part 2/2\n{whole[half:]}"},
            {"body": f"job part 1/2\n{whole[:half]}"},
        ]))
        self.assertEqual(_mailbox(session).read(TASK_ID), ENVELOPE)

    def test_read_rejects_conflicting_part_totals(self):
        session = FakeSession()
        session.expect("GET", "/repos/student/jobs/issues", FakeResponse(200, _issue_listing()))
        session.expect("GET", "/repos/student/jobs/issues/7/comments", FakeResponse(200, [
            {"body": "part 1/3\naaaa"},
            {"body": "part 2/2\nbbbb"},
        ]))
        with self.assertRaisesRegex(MailboxError, "part count mismatch"):
            _mailbox(session).read(TASK_ID)

    def test_read_rejects_incomplete_part_set(self):
        for bodies in ([], [{"body": "part 1/2\naaaa"}], [{"body": "unparsable"}]):
            with self.subTest(bodies=bodies):
                session = FakeSession()
                session.expect("GET", "/repos/student/jobs/issues", FakeResponse(200, _issue_listing()))
                session.expect(
                    "GET", "/repos/student/jobs/issues/7/comments",
                    FakeResponse(200, list(bodies)),
                )
                with self.assertRaisesRegex(MailboxError, "incomplete"):
                    _mailbox(session).read(TASK_ID)

    def test_read_rejects_malformed_task_id(self):
        with self.assertRaises(ProtocolError):
            _mailbox().read("NOT-A-TASK-ID")

    def test_wait_returns_first_successful_read(self):
        session = FakeSession()
        session.expect("GET", "/repos/student/jobs/issues", FakeResponse(200, _issue_listing()))
        whole = _encoded()
        session.expect(
            "GET", "/repos/student/jobs/issues/7/comments",
            FakeResponse(200, [{"body": f"part 1/1\n{whole}"}]),
        )
        self.assertEqual(_mailbox(session).wait(TASK_ID, timeout_seconds=5), ENVELOPE)

    def test_wait_raises_last_error_after_bounded_timeout(self):
        session = FakeSession()
        session.expect("GET", "/repos/student/jobs/issues", FakeResponse(200, []))
        with self.assertRaisesRegex(MailboxError, "not available"):
            _mailbox(session).wait(TASK_ID, timeout_seconds=1, poll_seconds=0.2)

    def test_publish_control_requires_issue_and_posts_numbered_parts(self):
        box = _mailbox(FakeSession())
        with self.assertRaisesRegex(MailboxError, "job issue is not available"):
            box.publish_control(1, ENVELOPE)
        session = FakeSession()
        session.expect("POST", "/repos/student/jobs/issues/7/comments", FakeResponse(201, {"id": 11}))
        box = _mailbox(session)
        box.issue_number = 7
        box.publish_control(4, ENVELOPE)
        self.assertEqual(len(session.calls), 1)
        _, path, kwargs = session.calls[0]
        self.assertEqual(path, "/repos/student/jobs/issues/7/comments")
        self.assertEqual(kwargs["payload"], {"body": f"control 4 part 1/1\n{_encoded()}"})
        with patch("courselens_worker.mailbox.chunk_envelope", return_value=["AAA", "BBB"]):
            box.publish_control(9, ENVELOPE)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(
            session.calls[1][2]["payload"], {"body": "control 9 part 1/2\nAAA"}
        )
        self.assertEqual(
            session.calls[2][2]["payload"], {"body": "control 9 part 2/2\nBBB"}
        )

    def test_publish_status_creates_once_then_patches_same_comment(self):
        box = _mailbox(FakeSession())
        with self.assertRaisesRegex(MailboxError, "job issue is not available"):
            box.publish_status(1, ENVELOPE)
        session = FakeSession()
        session.expect("POST", "/repos/student/jobs/issues/7/comments", FakeResponse(201, {"id": 42}))
        session.expect("PATCH", "/repos/student/jobs/issues/comments/42", FakeResponse(200, {"id": 42}))
        box = _mailbox(session)
        box.issue_number = 7
        box.publish_status(1, ENVELOPE)
        self.assertEqual(box.status_comment_id, 42)
        box.publish_status(2, ENVELOPE)
        posts = [call for call in session.calls if call[0] == "POST"]
        patches = [call for call in session.calls if call[0] == "PATCH"]
        self.assertEqual(len(posts), 1)
        self.assertEqual(len(patches), 1)
        self.assertEqual(
            patches[0][1],
            "/repos/student/jobs/issues/comments/42",
        )
        self.assertEqual(
            patches[0][2]["payload"], {"body": f"status 2\n{_encoded()}"}
        )
        with patch("courselens_worker.mailbox.chunk_envelope", return_value=["AAA", "BBB"]):
            with self.assertRaisesRegex(MailboxError, "unexpectedly large"):
                box.publish_status(3, ENVELOPE)
        self.assertEqual(len(session.calls), 2)


if __name__ == "__main__":
    unittest.main()
