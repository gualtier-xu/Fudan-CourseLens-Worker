"""DeepSeek-backed proofreading, grounded answers, and summaries."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
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
    for attempt in range(4):
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
        except requests.RequestException as exc:
            if attempt == 3:
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
        retry_after = response.headers.get("Retry-After")
        try:
            delay = min(30.0, float(retry_after)) if retry_after else float(2 ** attempt)
        except ValueError:
            delay = float(2 ** attempt)
        time.sleep(delay)
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
    completed = max(0, min(total_windows, int(prior.get("proofread_completed_windows") or 0)))

    def pairs_for(window_index: int) -> list[dict[str, Any]]:
        pairs = []
        for index in range(window_index * 20, min(count, window_index * 20 + 20)):
            sense = sensevoice[index] if index < len(sensevoice) else {}
            fire = firered[index] if index < len(firered) else {}
            pairs.append({
                "index": index,
                "start_ms": int(sense.get("start_ms") or fire.get("start_ms") or 0),
                "end_ms": int(sense.get("end_ms") or fire.get("end_ms") or 0),
                "sensevoice": str(sense.get("text") or ""),
                "firered": str(fire.get("text") or ""),
            })
        return pairs

    def request_window(pairs: list[dict[str, Any]]) -> Any:
        return _json_content(_chat(api_key, [
            {"role": "system", "content": "你是严谨的中文课程字幕校对器。综合两份识别结果，只修正识别错误，不扩写、不总结。输出 JSON 数组，每项仅包含 index 和 text。"},
            {"role": "user", "content": json.dumps(pairs, ensure_ascii=False)},
        ]))

    for batch_start in range(completed, total_windows, 2):
        indices = list(range(batch_start, min(total_windows, batch_start + 2)))
        batch = {index: pairs_for(index) for index in indices}
        with ThreadPoolExecutor(max_workers=min(2, len(indices)), thread_name_prefix="llm-proofread") as executor:
            futures = {index: executor.submit(request_window, batch[index]) for index in indices}
            corrected_by_window = {index: futures[index].result() for index in indices}
        for window_index in indices:
            corrected = corrected_by_window[window_index]
            if not isinstance(corrected, list):
                raise LLMError("proofreading response must be a JSON array")
            by_index = {
                int(item["index"]): str(item.get("text") or "")
                for item in corrected if isinstance(item, dict) and "index" in item
            }
            for pair in batch[window_index]:
                text = by_index.get(pair["index"]) or pair["firered"] or pair["sensevoice"]
                output.append({"start_ms": pair["start_ms"], "end_ms": pair["end_ms"], "text": text})
            if checkpoint is not None:
                checkpoint({
                    "stage": "proofread",
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
    transcript_windows = [transcript[start:start + 120] for start in range(0, len(transcript), 120)]
    if not transcript_windows:
        transcript_windows = [[] for _ in range(max(1, (len(ppt_pages) + 19) // 20))]
    sources = []
    for index, transcript_window in enumerate(transcript_windows):
        if transcript_window:
            lower = int(transcript_window[0].get("start_ms") or 0)
            upper = int(transcript_window[-1].get("end_ms") or lower)
            pages = [page for page in ppt_pages if lower <= int(page.get("created_sec") or 0) * 1000 <= upper]
        else:
            pages = ppt_pages[index * 20:(index + 1) * 20]
        sources.append({"transcript": transcript_window, "ppt_pages": pages})

    prior = dict(prior_checkpoint or {})
    parts: list[dict[str, Any]] = list(prior.get("summary_parts") or [])
    completed = max(0, min(len(sources), int(prior.get("summary_completed_windows") or 0)))

    def summarize_window(index: int) -> dict[str, Any]:
        part = _json_content(_chat(api_key, [
            {"role": "system", "content": "你是严谨的课程学习助理。仅依据输入整理当前窗口，输出 JSON 对象，字段为 markdown 和 chapters；chapters 每项包含 title、start_ms、summary，start_ms 必须来自输入。"},
            {"role": "user", "content": json.dumps(sources[index], ensure_ascii=False)},
        ]))
        if not isinstance(part, dict) or not isinstance(part.get("markdown"), str) or not isinstance(part.get("chapters"), list):
            raise LLMError("summary window response has an invalid shape")
        return part

    for batch_start in range(completed, len(sources), 2):
        indices = list(range(batch_start, min(len(sources), batch_start + 2)))
        with ThreadPoolExecutor(max_workers=min(2, len(indices)), thread_name_prefix="llm-summary") as executor:
            futures = {index: executor.submit(summarize_window, index) for index in indices}
            values = {index: futures[index].result() for index in indices}
        for index in indices:
            parts.append(values[index])
            if checkpoint is not None:
                checkpoint({
                    "stage": "summary",
                    "completed_chunks": index + 1,
                    "total_chunks": len(sources) + 1,
                    "summary_completed_windows": index + 1,
                    "summary_parts": parts,
                })

    value = _json_content(_chat(api_key, [
        {"role": "system", "content": "合并各窗口笔记为完整中文学习笔记，不得增加输入外事实。输出 JSON 对象，字段为 markdown 和 chapters；保留原有合法 start_ms。"},
        {"role": "user", "content": json.dumps({"title": title, "parts": parts}, ensure_ascii=False)},
    ], max_tokens=12_000))
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


def answer_question(
    api_key: str,
    *,
    query: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Answer only from caller-provided evidence and preserve citation IDs."""
    allowed = [item for item in evidence if isinstance(item, dict) and item.get("citation_id") and item.get("text")]
    if not allowed:
        return {"answer": "资料不足，无法根据当前课程资料回答。", "citations": [], "grounded": False}
    raw = _json_content(_chat(api_key, [
        {
            "role": "system",
            "content": (
                "你是严谨的课程问答助手。只能依据用户提供的 evidence 回答，禁止补充外部事实。"
                "输出 JSON 对象，字段为 answer、grounded、citations。citations 只能填写输入中的 citation_id；"
                "证据不足时 answer 必须为‘资料不足，无法根据当前课程资料回答。’，grounded 为 false。"
            ),
        },
        {"role": "user", "content": json.dumps({"query": str(query), "evidence": allowed}, ensure_ascii=False)},
    ], max_tokens=4096))
    if not isinstance(raw, dict) or not isinstance(raw.get("answer"), str) or not isinstance(raw.get("citations"), list):
        raise LLMError("answer response has an invalid shape")
    allowed_ids = {str(item["citation_id"]) for item in allowed}
    citations = [str(value) for value in raw["citations"] if str(value) in allowed_ids]
    grounded = bool(raw.get("grounded")) and bool(citations)
    if not grounded:
        return {"answer": "资料不足，无法根据当前课程资料回答。", "citations": [], "grounded": False}
    return {"answer": raw["answer"].strip(), "citations": citations[:8], "grounded": True}
