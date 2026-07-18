"""Read-only encrypted job mailbox backed by private GitHub Issues."""

from __future__ import annotations

import re
import time
from typing import Any

import requests

from .protocol import join_envelope, validate_task_id

API_ROOT = "https://api.github.com"
ISSUE_LABEL = "courselens-job"
TITLE_PREFIX = "[courselens-job]"
_PART_RE = re.compile(r"^part (\d+)/(\d+)\n([A-Za-z0-9+/=]+)$")


class MailboxError(RuntimeError):
    pass


class IssueMailbox:
    def __init__(self, repo: str, token: str, *, timeout: int = 30):
        if not repo or not token:
            raise ValueError("private job repository and token are required")
        self.repo = repo
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Fudan-CourseLens-Worker/1",
        })

    def _get(self, path: str, *, params=None):
        try:
            response = self.session.get(f"{API_ROOT}{path}", params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise MailboxError(f"GitHub mailbox request failed: {type(exc).__name__}") from exc
        if response.status_code != 200:
            request_id = response.headers.get("X-GitHub-Request-Id", "unknown")
            raise MailboxError(f"GitHub mailbox returned HTTP {response.status_code} ({request_id})")
        return response.json()

    def wait(self, task_id: str, *, timeout_seconds: int = 600, poll_seconds: float = 2.0) -> dict[str, Any]:
        task_id = validate_task_id(task_id)
        deadline = time.monotonic() + max(1, timeout_seconds)
        last_error = "encrypted job is not available"
        while time.monotonic() < deadline:
            try:
                return self.read(task_id)
            except MailboxError as exc:
                last_error = str(exc)
                time.sleep(max(0.2, poll_seconds))
        raise MailboxError(last_error)

    def read(self, task_id: str) -> dict[str, Any]:
        task_id = validate_task_id(task_id)
        issues = self._get(
            f"/repos/{self.repo}/issues",
            params={"state": "open", "labels": ISSUE_LABEL, "per_page": 100},
        )
        expected_title = f"{TITLE_PREFIX} {task_id}"
        issue = next((item for item in issues if str(item.get("title") or "") == expected_title), None)
        if issue is None:
            raise MailboxError("encrypted job is not available")
        comments = self._get(
            f"/repos/{self.repo}/issues/{int(issue['number'])}/comments",
            params={"per_page": 100},
        )
        pieces: dict[int, str] = {}
        total = 0
        for comment in comments:
            match = _PART_RE.fullmatch(str(comment.get("body") or "").strip())
            if not match:
                continue
            index, candidate_total = int(match.group(1)), int(match.group(2))
            if total and total != candidate_total:
                raise MailboxError("encrypted job part count mismatch")
            total = candidate_total
            pieces[index] = match.group(3)
        if total <= 0 or sorted(pieces) != list(range(1, total + 1)):
            raise MailboxError("encrypted job is incomplete")
        return join_envelope(pieces[index] for index in range(1, total + 1))
