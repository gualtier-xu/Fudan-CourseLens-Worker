"""Install pinned upstream ASR models with SHA-256 verification."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import requests


MODELS = {
    "sensevoice": {
        "archive": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
        "sha256": "7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e",
        "default": True,
    },
    # M4 Paraformer（ASRBENCH-1 A5 立项，M4-ENABLE-1 U1 实测钉，234 MB tar.bz2）。
    "paraformer": {
        "archive": "sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2",
        "sha256": "9c49fd9c6fb63de8e18c1054cf3d100f804741b7e608e187923cd8ff09fa9f03",
        "default": True,
    },
    # SUBTITLE-DEEP-1 Phase B：zipformer-transducer 热词精修腿（309 MB tar.bz2，
    # BENCH-ASR-1 热词 -35% 术语错实证；仅 SUBTITLE_BACKENDS 显式启用时下载）。
    "zipformer": {
        "archive": "sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.tar.bz2",
        "sha256": "c4925a6b0f998800d16f80caf90d2decff7b7a8c156d044c6cffdf141c847d94",
        "default": True,
    },
    # D13-PROD：流式转写腿（sherpa-onnx OnlineRecognizer + 流式 Paraformer 双语
    # zh-en，POC 实测 RTF 0.04-0.17）。GitHub releases 资产为全量包（≈1GB，
    # 含 FP32+int8 双变体+test_wavs）；工作载荷=int8 三件 ≈237MB。非默认集
    # （沿 silero-vad 先例）：237MB 加重 Actions cache，仅流式任务显式启用时
    # 经 COURSELENS_INSTALL_MODELS 点名安装。实测钉：下载件本地复算 sha256，
    # 解包 int8 三件与 D13-IMPL POC 实测缓存逐字节一致（cmp 三件 IDENTICAL）。
    "streaming-paraformer": {
        "archive": "sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2",
        "sha256": "5462a1fce42693deae572af1e8c4687124b12aa85fe61ff4d3168bb5280e205f",
        "env": "STREAMING_MODEL_DIR",
        "default": False,
    },
    # V4NONTHINK-1 件7：本地质量网双钉（模型下载仅限本清单）。silero-vad=v5 段界
    # （COURSELENS_ASR_VAD_ENGINE=silero 显式启用，缺省关）；ct-punc=CT-Transformer
    # 标点恢复 onnx 导出版（CTPUNC-DEF-1 起 COURSELENS_SUBTITLE_CTPUNC 缺省 fill，
    # off 可退）。两腿都有 fail-closed 回落（能量 VAD / 保留 LLM 标点），绝不失败任务。
    # 默认集：default=True 的条目随 main() 默认安装；silero-vad 钉保留在册但
    # 不进默认集（翻默认前置=全讲级验证），经 COURSELENS_INSTALL_MODELS 显式
    # 点名（逗号分隔名或 all）才装。
    "silero-vad": {
        "archive": "silero_vad.onnx",
        "sha256": "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        "raw_file": "silero_vad.onnx",
        "env": "SILERO_MODEL_DIR",
        "default": False,
    },
    "ct-punc": {
        # 元组对（非 dict）：`"tokens.json": <sha256>` 的赋值形态会被公共仓
        # gitleaks generic-api-key 规则误报为密钥（值为模型文件完整性哈希）。
        "files": (
            ("model_quant.onnx", "e6cd8399bf7d0e75f8d9af4a107310e1968ecab1d50135e765b8f0265b27a83d"),
            ("tokens.json", "c960ab87bccea4aa15cf49a59f71973c2c330b46668048cd8da253749ec71ee3"),
            ("config.yaml", "a56ec10925b06fa976ad51af373396be2b13e1eb8dc62a5426b5adebaba7071d"),
            ("configuration.json", "16097d3034818080e39331fa08909dffe75189f5c986f74155afd90b5f531ee4"),
        ),
        "base": "https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx/resolve/master",
        "dir_name": "punc_ct-transformer_zh-cn-common-vocab272727-onnx",
        "sha256": "5dfb9eddce4b90be07ad2442ffd3454b993eb0b703866eb5f7f1e86ad78b17d7",
        "env": "CTPUNC_MODEL_DIR",
        "default": True,
    },
}
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

INSTALL_ALL_ENV = "COURSELENS_INSTALL_MODELS"


def _model_directories(root: Path, name: str) -> list[Path]:
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and (path / "tokens.txt").is_file()
    ] if root.is_dir() else []
    markers = {
        "sensevoice": "sense-voice",
        "paraformer": "paraformer",
        "zipformer": "zipformer",
        "streaming-paraformer": "streaming-paraformer",
    }
    marker = markers[name]
    return sorted(path for path in candidates if marker in path.name)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as handle:
        members: list[tuple[tarfile.TarInfo, Path]] = []
        for member in handle.getmembers():
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or member.issym()
                or member.islnk()
                or member.isdev()
                or member.isfifo()
                or not (member.isdir() or member.isfile())
            ):
                raise RuntimeError("model archive contains an unsafe member")
            target = destination.joinpath(*relative.parts).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError("model archive contains an unsafe path")
            members.append((member, target))
        for member, target in members:
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise RuntimeError("model archive member could not be read")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _files_marker_digest(spec: dict) -> str:
    """多文件钉的聚合标记=逐文件 sha256 清单的确定性总摘要（非占位）。"""
    combined = hashlib.sha256()
    for name, digest in sorted(spec["files"]):
        combined.update(f"{name}:{digest}\n".encode("ascii"))
    return combined.hexdigest()


def _install_files(name: str, spec: dict, root: Path) -> Path:
    """modelscope 多文件模型：逐文件下载+校验到 root/<dir_name>/。"""
    marker_digest = _files_marker_digest(spec)
    marker = root / f".{name}-{marker_digest}.ready"
    destination = root / spec["dir_name"]
    model_hint = destination / "model_quant.onnx"
    if marker.is_file() and model_hint.is_file():
        return destination
    root.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, digest in sorted(spec["files"]):
        target = destination / filename
        if target.is_file():
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual == digest:
                continue
            target.unlink()
        temporary = target.with_suffix(target.suffix + ".part")
        file_digest = hashlib.sha256()
        with requests.get(f"{spec['base']}/{filename}", stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    file_digest.update(block)
                    output.write(block)
        if file_digest.hexdigest() != digest:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"{name} model file checksum mismatch: {filename}")
        temporary.replace(target)
    marker.write_text(marker_digest, encoding="ascii")
    return destination


def _install_raw(name: str, spec: dict, root: Path) -> Path:
    """单文件裸模型（非压缩包）：下载+校验到 root/<raw_file>。"""
    marker = root / f".{name}-{spec['sha256']}.ready"
    target = root / spec["raw_file"]
    if marker.is_file() and target.is_file():
        return root
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    temporary = target.with_suffix(target.suffix + ".part")
    with requests.get(f"{BASE}/{spec['archive']}", stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for block in response.iter_content(1024 * 1024):
                digest.update(block)
                output.write(block)
    if digest.hexdigest() != spec["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{name} model checksum mismatch")
    temporary.replace(target)
    marker.write_text(spec["sha256"], encoding="ascii")
    return root


def _install(name: str, spec: dict[str, str], root: Path) -> Path:
    marker = root / f".{name}-{spec['sha256']}.ready"
    if marker.is_file():
        directories = _model_directories(root, name)
        if directories:
            return directories[0]
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-install-") as temporary:
        archive = Path(temporary) / spec["archive"]
        digest = hashlib.sha256()
        with requests.get(f"{BASE}/{spec['archive']}", stream=True, timeout=60) as response:
            response.raise_for_status()
            with archive.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    digest.update(block)
                    output.write(block)
        if digest.hexdigest() != spec["sha256"]:
            raise RuntimeError(f"{name} model checksum mismatch")
        _safe_extract(archive, root)
    marker.write_text(spec["sha256"], encoding="ascii")
    directories = _model_directories(root, name)
    if not directories:
        raise RuntimeError(f"{name} model directory was not extracted")
    return directories[0]


def _select_models() -> dict[str, dict]:
    """默认集=default=True 条目；COURSELENS_INSTALL_MODELS 显式点名覆盖。

    点名值：逗号分隔的模型名，或 ``all``（含 silero-vad 等非默认钉）；
    未知名一律拒绝（闭集外不开口子）。
    """
    raw = os.environ.get(INSTALL_ALL_ENV, "").strip().lower()
    if not raw:
        return {name: spec for name, spec in MODELS.items() if spec.get("default")}
    if raw == "all":
        return dict(MODELS)
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(names) - set(MODELS))
    if unknown:
        raise SystemExit(f"unknown model name(s): {', '.join(unknown)}")
    return {name: MODELS[name] for name in names}


def main() -> None:
    root = Path(os.environ.get("COURSELENS_MODEL_ROOT", ".models")).resolve()
    # 空 sha256 = 条目尚未实测钉，整条跳过。
    installed = {}
    for name, spec in _select_models().items():
        if not spec["sha256"]:
            continue
        if "files" in spec:
            installed[name] = _install_files(name, spec, root)
        elif "raw_file" in spec:
            installed[name] = _install_raw(name, spec, root)
        else:
            installed[name] = _install(name, spec, root)
    environment = Path(os.environ.get("GITHUB_ENV", root / "models.env"))
    with environment.open("a", encoding="utf-8") as output:
        for name, directory in installed.items():
            env_name = MODELS[name].get("env") or f"{name.upper()}_MODEL_DIR"
            output.write(f"{env_name}={directory}\n")


if __name__ == "__main__":
    main()
