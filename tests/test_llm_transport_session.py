"""N13：LLM 传输层线程本地 Session 连接复用（R6 Q9a 修正版）。

讲级 20-40 次 _chat 调用（窗环 max_workers=2+主线程）此前每次裸
requests.post=新 TCP+TLS 握手（纯开销 5-20s+延迟抖动挤占 merge 超时预算）。
Session 官方未承诺跨线程共享安全 → 线程本地单例（各线程池化生效在自己
线程内），认证头仍逐调用传参，Session 零全局状态。
"""

from __future__ import annotations

import json
import os
import threading
import unittest
from unittest.mock import Mock, patch

from courselens_worker import llm as llm_mod
from courselens_worker.llm import REVIEW_VIEWS_ENV


def _fake_response() -> Mock:
    response = Mock()
    response.status_code = 200
    response.headers = {}
    response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return response


class TransportSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        llm_mod.reset_usage()

    def test_session_is_thread_local_singleton(self):
        session_main = llm_mod._http_session()
        self.assertIs(llm_mod._http_session(), session_main)
        seen: dict[str, object] = {}

        def _grab() -> None:
            seen["first"] = llm_mod._http_session()
            seen["second"] = llm_mod._http_session()

        thread = threading.Thread(target=_grab)
        thread.start()
        thread.join()
        # 同线程内单例；跨线程各自实例（官方未承诺共享安全）。
        self.assertIs(seen["first"], seen["second"])
        self.assertIsNot(seen["first"], session_main)
        self.assertIs(llm_mod._http_session(), session_main)

    def test_chat_posts_through_thread_session(self):
        session = Mock()
        session.post.return_value = _fake_response()
        with patch.object(llm_mod, "_http_session", return_value=session):
            value = llm_mod._chat("sk-synthetic", [{"role": "user", "content": "hi"}])
        self.assertEqual(value, "ok")
        self.assertEqual(session.post.call_count, 1)
        # 缺省超时与鉴权头逐调用传参（Session 零全局状态）。
        kwargs = session.post.call_args.kwargs
        self.assertEqual(kwargs["timeout"], 180.0)
        self.assertIn("Authorization", kwargs["headers"])

    def test_chat_sessions_differ_across_concurrent_threads(self):
        sessions: list[object] = []

        def _post_once() -> None:
            llm_mod._chat("sk-synthetic", [{"role": "user", "content": "hi"}])
            sessions.append(llm_mod._HTTP_LOCAL.session)

        with patch.object(llm_mod.requests.Session, "post", return_value=_fake_response()) as post:
            threads = [threading.Thread(target=_post_once) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(post.call_count, 2)
        self.assertEqual(len(sessions), 2)
        self.assertIsNot(sessions[0], sessions[1])


class TimeoutBudgetTests(unittest.TestCase):
    """N16：merge=420s；思考档 question/judge=240s；其余=缺省 180s（逐调用
    capture 钉）。背景：merge 32k 输出帽+全 parts 输入在 180s 缺省下慢速段
    超时→重试再超时→llm_pending 回炉（学生感知=总结突然重跑几分钟）。"""

    def setUp(self) -> None:
        llm_mod.reset_usage()

    def _timeouts_of(self, calls) -> list[float]:
        return [kwargs.get("timeout") for _, kwargs in calls]

    def test_merge_call_gets_extended_timeout_and_windows_stay_default(self):
        captured: list[tuple[list, dict]] = []

        def fake_chat(api_key, messages, **kwargs):
            captured.append((messages, kwargs))
            if messages[0]["content"] == llm_mod._SUMMARY_MERGE_PROMPT:
                return json.dumps({"markdown": "combined", "chapters": []})
            return json.dumps({"markdown": "part", "chapters": []})

        segments = [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"第{index}段"}
            for index in range(130)
        ]
        with (
            patch.dict(os.environ, {REVIEW_VIEWS_ENV: "off"}),
            patch("courselens_worker.llm._chat", side_effect=fake_chat),
        ):
            llm_mod.create_summary("sk-synthetic", title="t", transcript=segments, ppt_pages=[])
        merge_calls = [
            kwargs for messages, kwargs in captured
            if messages[0]["content"] == llm_mod._SUMMARY_MERGE_PROMPT
        ]
        window_calls = [
            kwargs for messages, kwargs in captured
            if messages[0]["content"] != llm_mod._SUMMARY_MERGE_PROMPT
        ]
        self.assertEqual(len(merge_calls), 1)
        self.assertEqual(merge_calls[0]["timeout"], llm_mod._SUMMARY_MERGE_TIMEOUT_SECONDS)
        self.assertTrue(window_calls)
        # 只有 merge 覆写：窗调用不吃 420 预算（缺省留在 _chat 签名里）。
        self.assertTrue(all("timeout" not in kwargs for kwargs in window_calls))

    def test_question_timeout_follows_thinking_tier(self):
        response = json.dumps({"answer": "答案", "grounded": True, "citations": []})
        evidence = [{"citation_id": "c1", "text": "证据"}]

        def fake_chat(api_key, messages, **kwargs):
            return response

        with patch.dict(os.environ, {"COURSELENS_QUESTION_THINKING": "low"}), patch(
            "courselens_worker.llm._chat", side_effect=fake_chat
        ) as chat:
            llm_mod.answer_question("sk-synthetic", query="问", evidence=evidence)
            self.assertEqual(chat.call_args.kwargs["timeout"], 240.0)
        with patch.dict(os.environ, {}, clear=True), patch(
            "courselens_worker.llm._chat", side_effect=fake_chat
        ) as chat:
            llm_mod.answer_question("sk-synthetic", query="问", evidence=evidence)
            self.assertEqual(chat.call_args.kwargs["timeout"], 180.0)

    def test_judge_timeout_follows_thinking_tier(self):
        flash = json.dumps({"findings": [
            {"target": "segment", "position": 0, "dimension": "readability",
             "code": "garbled", "severity": "warn"},
        ]})

        def fake_chat(api_key, messages, **kwargs):
            if "flash_findings" not in json.loads(messages[1]["content"]):
                return flash
            return json.dumps({"findings": []})

        sample = {
            "segments": [
                {"start_ms": 0, "end_ms": 1000, "text": "第一段", "correction": "none"}
            ],
        }
        # THINK-LADDER-2 两段式：筛段恒关思考，vars 思考档只作用于裁段——
        # 筛跳恒 180s 基础预算，裁跳随档位享 240s 长输出预算。
        with patch.dict(os.environ, {"COURSELENS_JUDGE_THINKING": "low"}), patch(
            "courselens_worker.llm._chat", side_effect=fake_chat
        ) as chat:
            report = llm_mod.judge_lecture_quality("sk-synthetic", subtitle_sample=sample)
            self.assertIsNotNone(report["subtitle"])
            self.assertEqual(chat.call_args_list[0].kwargs["timeout"], 180.0)
            self.assertEqual(chat.call_args.kwargs["timeout"], 240.0)
        with patch.dict(os.environ, {}, clear=True), patch(
            "courselens_worker.llm._chat", side_effect=fake_chat
        ) as chat:
            llm_mod.judge_lecture_quality("sk-synthetic", subtitle_sample=sample)
            self.assertEqual(chat.call_args.kwargs["timeout"], 180.0)


if __name__ == "__main__":
    unittest.main()
