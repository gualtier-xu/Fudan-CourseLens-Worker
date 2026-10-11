"""A3（N6W-ALL）：WorkerTwinParityTests 在带真 numpy 的套件里真跑。

客户端正典套件（venv-py310 无 numpy，且全量套件存在 numpy 替身污染风险）
里这对 parity 双子按设计 skip；本文件把客户端测试模块里的
``WorkerTwinParityTests`` 原样挂进 worker 套件（anaconda 真numpy），使
worker 全量每次收口都至少真跑一次两端等价断言。零复制过滤逻辑——测试类
本体只活在 ``tests/test_courseware_pdf_junk_local.py`` 一处，两端阈值漂移
在这对断言下必红。
"""

from __future__ import annotations

import sys
from pathlib import Path

CLIENT_ROOT = Path(__file__).resolve().parents[2]
if str(CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIENT_ROOT))

try:
    from tests.test_courseware_pdf_junk_local import WorkerTwinParityTests  # noqa: F401
except ImportError:
    # The public worker mirror ships no client tree: the parity run is a
    # dev-side suite feature, so the twin stays absent there instead of
    # failing collection.
    WorkerTwinParityTests = None
