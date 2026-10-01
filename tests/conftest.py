"""夜10-C T23：worker 测试进程级可选依赖桩。

此前各测试文件用 ``with patch.dict(sys.modules, {"sherpa_onnx": Mock()})``
包裹模块导入——patch.dict 退出时整体还原 sys.modules，把 with 块内顺带导入
的 numpy 等重依赖一并剥掉；下一个测试模块再导入时就得到第二个 numpy 实例，
跨实例数组进入 ufunc 即以 ``_NoValueType`` 哨兵失配炸穿（实测
test_platform_rough_source → test_asr_timing 同进程 16-17 红）。

修法：在 conftest（pytest 收集前恰一次）按需安装缺失的可选依赖桩，且**只增
不整体还原**——全部测试模块共享同一 sys.modules、同一 asr/numpy 单实例。
真实 sherpa_onnx/numpy 在环境里存在时零触碰。
"""

from __future__ import annotations

import importlib.util
import sys
from unittest.mock import Mock


def _stub_if_missing(module_name: str) -> None:
    if module_name in sys.modules:
        return
    if importlib.util.find_spec(module_name) is not None:
        return
    sys.modules[module_name] = Mock()


# rapidocr_onnxruntime：learning_pack 分支入口局部导入 ocr；answer-only 钉
# （RR-PARK-1 P2）不触 OCR 但需模块可导入（客户端 venv 无重依赖，与
# sherpa_onnx 同款桩纪律）。
for _name in ("sherpa_onnx", "numpy", "rapidocr_onnxruntime"):
    _stub_if_missing(_name)
