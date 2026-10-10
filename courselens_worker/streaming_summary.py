"""滚动摘要状态机（D13-PROD 施工清单第 4 步）.

复用 ``llm.create_summary`` 的检查点语义（llm.py 批量摘要链已验证的形状）：

- **窗口计划指纹**（``summary_window_plan`` 同构）：每跳窗口以
  「段数+首末毫秒锚」做确定性 sha256 指纹；恢复时计划比对，首个不一致窗
  之后的旧计数作废（与 create_summary 的重跑语义同源——transcript 变了
  绝不拿旧 parts 冒充）。
- **completed 计数 + parts 累积**：``summary_completed_windows``/
  ``summary_parts`` 同构字段，前缀 ``streaming_``。
- **单窗降级**（SUMMARY-FIX-1 同款）：一跳重试梯穷尽仍败 → 记 skipped
  降级继续，绝不拖垮整讲；**全部跳失败 → fail-closed 抛错**，绝不产出
  completed 空笔记。
- **末跳 merge**：下课最后一跳把滚动 parts 合并成终稿（deepseek-flash
  128K 整讲单调用可容，课后 LLM 开销=恰好一跳）；merge 输出走与
  create_summary merge 同一形状门（对象+非空 markdown+chapters 列表）。

LLM 调用可注入（``refine``/``merge`` 两个可调用），测试无需网络；生产接线
把 ``llm._chat`` 与滚动提示词包进这两个闭包。**evidence 命名空间共存**
（清单第 6 步）：全部检查点键用 ``streaming_`` 前缀，与批量链
``summary_*``/``raw_*`` 键零碰撞——流式快稿与录播现链精修稿互不覆写。
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable

# 滚动窗口节奏（D13-RESEARCH §3.2：上课中每 5-10min 一跳；缺省取上限档）。
DEFAULT_WINDOW_SECONDS = 600.0
DEFAULT_ATTEMPTS = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0

CHECKPOINT_KEY_COMPLETED = "streaming_summary_completed_windows"
CHECKPOINT_KEY_PLAN = "streaming_summary_window_plan"
CHECKPOINT_KEY_PARTS = "streaming_summary_parts"
CHECKPOINT_KEY_SKIPPED = "streaming_summary_skipped_windows"
CHECKPOINT_KEY_ANCHORED_MS = "streaming_transcript_anchored_ms"

CODE_ALL_FAILED = "streaming_summary_all_failed"
CODE_MERGE_FAILED = "streaming_summary_merge_failed"
CODE_INVALID_PART = "streaming_summary_invalid_part"


class StreamingSummaryError(RuntimeError):
    """滚动摘要闭集失败：code 只取上表，绝不携带提示词/响应内容。"""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(code if not detail else f"{code}: {detail}")
        self.code = code


def valid_summary_part(part: Any) -> bool:
    """与 llm.create_summary 同一形状门：对象+非空 markdown+chapters 列表。"""
    return (
        isinstance(part, dict)
        and isinstance(part.get("markdown"), str)
        and bool(part["markdown"].strip())
        and isinstance(part.get("chapters"), list)
    )


def window_fingerprint(segments: list[dict[str, Any]]) -> str:
    """滚动窗口的确定性指纹（段数+首末锚），供恢复时计划比对。"""
    if not segments:
        return "empty"
    first = int(segments[0].get("start_ms") or 0)
    last = int(segments[-1].get("end_ms") or first)
    digest = hashlib.sha256(f"{len(segments)}:{first}:{last}".encode("ascii"))
    return digest.hexdigest()[:12]


class RollingSummarizer:
    """上课中每窗一跳的滚动摘要；下课末跳 merge 终稿。

    ``refine(window_payload) -> part``：滚动窗 LLM 调用（生产接线包
    ``llm._chat``）；``merge(merge_payload) -> note``：末跳合并调用。
    ``prior_checkpoint`` 恢复已完成的滚动状态（键见模块常量）。
    """

    def __init__(
        self,
        *,
        refine: Callable[[dict[str, Any]], dict[str, Any]],
        merge: Callable[[dict[str, Any]], dict[str, Any]],
        prior_checkpoint: dict[str, Any] | None = None,
        emit: Callable[[str], None] | None = None,
        attempts: int = DEFAULT_ATTEMPTS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
    ):
        self._refine = refine
        self._merge = merge
        self._emit = emit or (lambda line: None)
        self._attempts = max(1, int(attempts))
        self._backoff = max(0.0, float(retry_backoff_seconds))
        self.window_seconds = float(window_seconds)
        prior = prior_checkpoint if isinstance(prior_checkpoint, dict) else {}
        self.parts: list[dict[str, Any]] = [
            part for part in (prior.get(CHECKPOINT_KEY_PARTS) or []) if valid_summary_part(part)
        ]
        self.completed_windows = max(0, int(prior.get(CHECKPOINT_KEY_COMPLETED) or 0))
        self.skipped_windows = max(0, int(prior.get(CHECKPOINT_KEY_SKIPPED) or 0))
        self.anchored_ms = max(0, int(prior.get(CHECKPOINT_KEY_ANCHORED_MS) or 0))
        plan = prior.get(CHECKPOINT_KEY_PLAN)
        self.window_plan: list[str] = [
            str(marker) for marker in plan if isinstance(marker, str)
        ] if isinstance(plan, list) else []
        self._pending: list[dict[str, Any]] = []

    # -- 上课中滚动跳 -----------------------------------------------------

    def buffer_segments(self, segments: list[dict[str, Any]]) -> None:
        """累积自上一次跳以来的新转写段（调用方在喂流循环里持续投喂）。"""
        self._pending.extend(segments)

    def due(self) -> bool:
        """滚动窗是否已攒够（按段覆盖的音频跨度估）。"""
        if not self._pending:
            return False
        span_ms = int(self._pending[-1].get("end_ms") or 0) - int(
            self._pending[0].get("start_ms") or 0
        )
        return span_ms >= self.window_seconds * 1000

    def jump(self) -> dict[str, Any] | None:
        """把攒好的滚动窗送一次 LLM 精炼；成功返回 part，失败降级 ``None``。

        语义对齐 create_summary：part 过形状门才入 parts；重试梯穷尽记
        skipped；检查点状态在每一跳后可取（``checkpoint()``）。
        """
        segments = self._pending
        self._pending = []
        if not segments:
            return None
        marker = window_fingerprint(segments)
        window = {"transcript": segments, "rolling_index": len(self.window_plan)}
        part = self._attempt(self._refine, window, "jump")
        if part is None:
            self.skipped_windows += 1
            self._emit(
                f"stage=streaming-summary-jump skipped={self.skipped_windows} "
                f"completed={self.completed_windows}"
            )
            return None
        self.parts.append(part)
        self.completed_windows += 1
        self.window_plan.append(marker)
        self.anchored_ms = max(self.anchored_ms, int(segments[-1].get("end_ms") or 0))
        self._emit(
            f"stage=streaming-summary-jump completed={self.completed_windows} "
            f"windows={len(self.window_plan)}"
        )
        return part

    def checkpoint(self) -> dict[str, Any]:
        """create_summary 同构检查点（``streaming_`` 前缀命名空间）。"""
        return {
            CHECKPOINT_KEY_COMPLETED: self.completed_windows,
            CHECKPOINT_KEY_PLAN: list(self.window_plan),
            CHECKPOINT_KEY_PARTS: list(self.parts),
            CHECKPOINT_KEY_SKIPPED: self.skipped_windows,
            CHECKPOINT_KEY_ANCHORED_MS: self.anchored_ms,
        }

    # -- 下课末跳 ---------------------------------------------------------

    def finalize(
        self,
        *,
        title: str,
        course_context: dict[str, Any] | None = None,
        glossary: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """末跳 merge 终稿；无任何成功滚动 part 时 fail-closed 抛错。"""
        if not self.parts:
            raise StreamingSummaryError(
                CODE_ALL_FAILED,
                f"completed={self.completed_windows} skipped={self.skipped_windows}",
            )
        merge_input: dict[str, Any] = {
            "title": title,
            "parts": self.parts,
            "rolling": True,
        }
        if course_context:
            merge_input["course_context"] = course_context
        terms = [str(term).strip() for term in (glossary or ()) if str(term).strip()]
        if terms:
            merge_input["glossary"] = terms
        note = self._attempt(self._merge, merge_input, "merge")
        if not valid_summary_part(note):
            raise StreamingSummaryError(CODE_MERGE_FAILED, "invalid merged note shape")
        # 命名空间标记：这份笔记产自流式快稿链（录播精修链可异步替换）。
        note = dict(note)
        note["streaming_generated"] = True
        self._emit(
            f"stage=streaming-summary-merge completed={self.completed_windows} "
            f"chapters={len(note.get('chapters') or [])}"
        )
        return note

    # -- 重试梯 -----------------------------------------------------------

    def _attempt(
        self,
        call: Callable[[dict[str, Any]], dict[str, Any]],
        payload: dict[str, Any],
        label: str,
    ) -> dict[str, Any] | None:
        last_code = CODE_INVALID_PART
        for attempt in range(self._attempts):
            try:
                candidate = call(payload)
            except StreamingSummaryError as exc:
                last_code = exc.code
            except Exception as exc:  # noqa: BLE001 - 收拢为闭集码
                last_code = CODE_INVALID_PART
                self._emit(
                    f"stage=streaming-summary-{label}-error attempt={attempt + 1} "
                    f"type={type(exc).__name__}"
                )
                candidate = None
            else:
                if valid_summary_part(candidate):
                    return candidate
                last_code = CODE_INVALID_PART
            if attempt + 1 < self._attempts:
                time.sleep(self._backoff)
        self._emit(
            f"stage=streaming-summary-{label}-failed attempts={self._attempts} "
            f"code={last_code}"
        )
        return None
