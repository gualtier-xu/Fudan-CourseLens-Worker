"""ct-punc 本地标点恢复适配器（V4NONTHINK-1 件7 本地质量网）。

FunASR CT-Transformer 标点恢复模型（modelscope
``punc_ct-transformer_zh-cn-common-vocab272727``，onnx 导出版）经可选依赖
``funasr_onnx`` 加载：零 torch、零 API 成本，模型走 install_models 钉。

运行时门（CTPUNC-DEF-1 起 fill 缺省启用，off 可退）：
- ``COURSELENS_SUBTITLE_CTPUNC``：``off`` | ``fill``（缺省，A 序：仅补 LLM
  未标点的 cue 缺口）| ``full``（B 序：全部 cue 重标点）。
- ``CTPUNC_MODEL_DIR``：模型目录（缺席/引擎不可用=一次探测后安静跳过整讲，
  闭集遥测记账，不逐段报错、不逐段重试）。

安全纪律：适配是「标点重写」，内容必须逐字不变——输出经去标点内容等值门，
不等值（模型幻觉/换字）一律保留原文 fail-closed；推理异常按段跳过并闭集
遥测记账，绝不失败任务。推理器懒加载、进程内缓存一份；加载失败进程内
缓存失败标记，同目录零重试。
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

CTPUNC_ENV = "COURSELENS_SUBTITLE_CTPUNC"
CTPUNC_MODE_OFF = "off"
CTPUNC_MODE_FILL = "fill"
CTPUNC_MODE_FULL = "full"
CTPUNC_MODEL_DIR_ENV = "CTPUNC_MODEL_DIR"

# 内容等值门基准闭集（与 llm._DEEP_PUNCT_SET 同族语义，独立定义避免环依赖：
# 适配层只要求「去掉这些标点后内容逐字相同」）。
_PUNCT_SET = frozenset("，。？！、；：,.!?;:· ")
_HAS_PUNCT_RE = re.compile(r"[，。？！、；：,.!?;:]")

_lock = threading.Lock()
_engine: Any = None
_engine_dir: str = ""
# CTPUNC-DEF-1 件3：进程内一次探测缓存——加载失败按目录记名，同目录零重试，
# 杜绝默认开+缺依赖/坏模型环境下逐段重试 import/加载的浪费与噪音。
_failed_dir: str = ""


def ct_punc_mode() -> str:
    raw = os.environ.get(CTPUNC_ENV, "").strip().lower()
    if raw in {CTPUNC_MODE_FILL, CTPUNC_MODE_FULL}:
        return raw
    # 缺省=fill（A 序缺口填充）；未知值随缺省（与 vad_engine「未知回落缺省」
    # 同构，fill 有内容等值门 fail-closed 兜底），显式 off 才关。
    return CTPUNC_MODE_OFF if raw == CTPUNC_MODE_OFF else CTPUNC_MODE_FILL


def ct_punc_model_dir() -> Path | None:
    value = os.environ.get(CTPUNC_MODEL_DIR_ENV, "").strip()
    if not value:
        return None
    directory = Path(value)
    if directory.is_file():
        return directory
    model = directory / "model.onnx"
    return model if model.is_file() else directory if directory.is_dir() else None


def _load_engine(model_dir: Path) -> Any:
    """懒加载 funasr_onnx CT_Transformer；进程内单例，失败抛错由调用方记账。

    同目录此前探测失败过则直接快速抛错（不重试 import/构造）——缺依赖或
    坏模型面只探测一次，后续调用零成本回落。
    """
    global _engine, _engine_dir
    with _lock:
        if _engine is not None and _engine_dir == str(model_dir):
            return _engine
        if _failed_dir == str(model_dir):
            raise RuntimeError("ct-punc engine unavailable (probe failed earlier)")
        from funasr_onnx import CT_Transformer  # 可选依赖：缺席=特性不可用

        engine = CT_Transformer(str(model_dir), quantize=True)
        _engine, _engine_dir = engine, str(model_dir)
        return engine


def _engine_ready(model_dir: Path) -> bool:
    """每讲一次探测：成功则引擎已入进程缓存；失败记入 _failed_dir 零重试。"""
    global _failed_dir
    if _engine is not None and _engine_dir == str(model_dir):
        return True
    try:
        _load_engine(model_dir)
    except Exception:  # noqa: BLE001 - 探测面任何异常都按模型不可用记账
        _failed_dir = str(model_dir)
        return False
    return True


def _content_core(text: str) -> str:
    return "".join(ch for ch in str(text) if ch not in _PUNCT_SET)


def restore_text(text: str, model_dir: Path) -> str:
    """单条文本 ct-punc 重标点；内容不等值或推理失败时原样返回。"""
    source = " ".join(str(text or "").split())
    if not source or not _content_core(source):
        return source
    try:
        engine = _load_engine(model_dir)
        result = engine(source)
    except Exception:  # noqa: BLE001 - 推理面任何异常都保留原文（fail-closed）
        return source
    candidate = ""
    if isinstance(result, (list, tuple)) and result:
        item = result[0]
        if isinstance(item, (list, tuple)) and item:
            # funasr_onnx 形态：(标点后文本, 逐字标签列表)
            if len(item) >= 2 and isinstance(item[0], str) and not isinstance(item[1], str):
                candidate = item[0]
            elif isinstance(item[-1], str):
                candidate = item[-1]
        elif isinstance(item, str):
            candidate = item
    candidate = " ".join(str(candidate).split())
    if not candidate or _content_core(candidate) != _content_core(source):
        return source
    return candidate


def apply_ct_punc(
    segments: list[dict[str, Any]],
    *,
    telemetry: "list[str] | None" = None,
) -> dict[str, int]:
    """就地按 env 模式对字幕段做 ct-punc 重标点；返回遥测计数。

    fill：只处理没有任何句读标点的段（A 序缺口填充）；full：全部段重标点
    （B 序，LLM 侧标点职责由调用方在提示词层摘除）。模型目录缺席或引擎
    探测失败各记一次闭集回落（model_missing / model_load_failed）后整讲
    安静跳过——不逐段报错、不逐段重试。模式 off 零操作。
    """
    mode = ct_punc_mode()
    stats = {
        "mode_off": 0,
        "model_missing": 0,
        "model_load_failed": 0,
        "applied": 0,
        "kept": 0,
        "skipped": 0,
    }
    if mode == CTPUNC_MODE_OFF or not segments:
        stats["mode_off" if mode == CTPUNC_MODE_OFF else "skipped"] = 1
        return stats
    model_dir = ct_punc_model_dir()
    if model_dir is None:
        stats["model_missing"] = 1
        if telemetry is not None:
            telemetry.append("stage=ct-punc-fallback reason=model_missing")
        return stats
    if not _engine_ready(model_dir):
        stats["model_load_failed"] = 1
        if telemetry is not None:
            telemetry.append("stage=ct-punc-fallback reason=model_load_failed")
        return stats
    for segment in segments:
        text = str(segment.get("text") or "")
        if not text.strip():
            continue
        if mode == CTPUNC_MODE_FILL and _HAS_PUNCT_RE.search(text):
            continue
        restored = restore_text(text, model_dir)
        if restored != text:
            segment["text"] = restored
            stats["applied"] += 1
        else:
            stats["kept"] += 1
    return stats
