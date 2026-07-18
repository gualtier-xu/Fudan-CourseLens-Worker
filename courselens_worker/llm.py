"""DeepSeek-backed proofreading, summary, and chapter derivation."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import requests

from .formats import normalize_segments

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"


class LLMError(RuntimeError):
    pass


def _chat(api_key: str, messages: list[dict[str, str]], *, max_tokens: int = 8192) -> str:
    if not api_key:
        raise LLMError("the encrypted job does not contain an AI API key")
    payload = {"model": MODEL, "messages": messages, "temperature": 0.1, "max_tokens": max_tokens}
    last_status = 0
    for attempt in range(3):
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
        except requests.RequestException as exc:
            if attempt == 2:
                raise LLMError(f"AI request failed: {type(exc).__name__}") from exc
            time.sleep(2 ** attempt)
            continue
        last_status = response.status_code
        if response.status_code == 200:
            try:
                return str(response.json()["choices"][0]["message"]["content"])
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise LLMError("AI response shape is invalid") from exc
        if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
            break
        time.sleep(2 ** attempt)
    raise LLMError(f"AI request returned HTTP {last_status or 'unknown'}")


def _json_content(text: str) -> Any:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise LLMError("AI response is not valid JSON") from exc


def proofread_segments(
    api_key: str,
    sensevoice: list[dict[str, Any]],
    firered: list[dict[str, Any]],
    *,
    prior_checkpoint: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    count = max(len(sensevoice), len(firered))
    prior = dict(prior_checkpoint or {})
    output: list[dict[str, Any]] = list(prior.get("proofread_segments") or [])
    total_windows = (count + 19) // 20
    completed_windows = max(
        0, min(total_windows, int(prior.get("proofread_completed_windows") or 0))
    )
    for window_index in range(completed_windows, total_windows):
        start = window_index * 20
        pairs = []
        for index in range(start, min(count, start + 20)):
            sense = sensevoice[index] if index < len(sensevoice) else {}
            fire = firered[index] if index < len(firered) else {}
            pairs.append({
                "index": index,
                "start_ms": int(sense.get("start_ms") or fire.get("start_ms") or 0),
                "end_ms": int(sense.get("end_ms") or fire.get("end_ms") or 0),
                "sensevoice": str(sense.get("text") or ""),
                "firered": str(fire.get("text") or ""),
            })
        raw = _chat(api_key, [
            {"role": "system", "content": "你是严谨的中文课程字幕校对器。综合两个识别结果，只修正识别错误，不扩写，不总结。输出 JSON 数组，每项仅含 index 和 text。"},
            {"role": "user", "content": json.dumps(pairs, ensure_ascii=False)},
        ])
        corrected = _json_content(raw)
        if not isinstance(corrected, list):
            raise LLMError("proofreading response must be a JSON array")
        by_index = {int(item["index"]): str(item.get("text") or "") for item in corrected if isinstance(item, dict) and "index" in item}
        for pair in pairs:
            text = by_index.get(pair["index"]) or pair["firered"] or pair["sensevoice"]
            output.append({"start_ms": pair["start_ms"], "end_ms": pair["end_ms"], "text": text})
        if checkpoint is not None:
            checkpoint({
                "proofread_completed_windows": window_index + 1,
                "proofread_total_windows": total_windows,
                "proofread_segments": normalize_segments(output),
            })
    return normalize_segments(output)


def create_summary(
    api_key: str,
    *,
    title: str,
    transcript: list[dict[str, Any]],
    ppt_pages: list[dict[str, Any]],
    prior_checkpoint: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    transcript_windows = [
        transcript[start:start + 120] for start in range(0, len(transcript), 120)
    ]
    if not transcript_windows:
        transcript_windows = [[] for _ in range(max(1, (len(ppt_pages) + 19) // 20))]
    sources = []
    for index, transcript_window in enumerate(transcript_windows):
        if transcript_window:
            lower = int(transcript_window[0].get("start_ms") or 0)
            upper = int(transcript_window[-1].get("end_ms") or lower)
            page_window = [
                page for page in ppt_pages
                if lower <= int(page.get("created_sec") or 0) * 1000 <= upper
            ]
        else:
            page_window = ppt_pages[index * 20:(index + 1) * 20]
        sources.append({"transcript": transcript_window, "ppt_pages": page_window})

    prior = dict(prior_checkpoint or {})
    parts: list[dict[str, Any]] = list(prior.get("summary_parts") or [])
    completed_windows = max(
        0, min(len(sources), int(prior.get("summary_completed_windows") or 0))
    )
    for index in range(completed_windows, len(sources)):
        raw_part = _chat(api_key, [
            {"role": "system", "content": "你是严谨的课程学习助理。仅依据输入整理当前窗口，输出 JSON 对象，字段为 markdown 和 chapters；chapters 每项包含 title、start_ms、summary，start_ms 必须来自输入。"},
            {"role": "user", "content": json.dumps(sources[index], ensure_ascii=False)},
        ])
        part = _json_content(raw_part)
        if not isinstance(part, dict) or not isinstance(part.get("markdown"), str) or not isinstance(part.get("chapters"), list):
            raise LLMError("summary window response has an invalid shape")
        parts.append(part)
        if checkpoint is not None:
            checkpoint({
                "stage": "summary",
                "completed_chunks": index + 1,
                "total_chunks": len(sources) + 1,
                "summary_completed_windows": index + 1,
                "summary_parts": parts,
            })

    raw = _chat(api_key, [
        {"role": "system", "content": "合并各窗口笔记为完整中文学习笔记。不得增加输入外事实。输出 JSON 对象，字段为 markdown 和 chapters；保留原有合法 start_ms。"},
        {"role": "user", "content": json.dumps({"title": title, "parts": parts}, ensure_ascii=False)},
    ], max_tokens=12_000)
    value = _json_content(raw)
    if not isinstance(value, dict) or not isinstance(value.get("markdown"), str) or not isinstance(value.get("chapters"), list):
        raise LLMError("summary response has an invalid shape")
    valid_anchors = {int(item.get("start_ms") or 0) for item in transcript}
    valid_anchors.update(int(item.get("created_sec") or 0) * 1000 for item in ppt_pages)
    chapters = []
    for item in value["chapters"]:
        if not isinstance(item, dict):
            continue
        start_ms = int(item.get("start_ms") or 0)
        if start_ms not in valid_anchors:
            continue
        chapters.append({
            "title": str(item.get("title") or "").strip(),
            "start_ms": start_ms,
            "summary": str(item.get("summary") or "").strip(),
        })
    return {"model": MODEL, "markdown": value["markdown"].strip(), "chapters": chapters}
