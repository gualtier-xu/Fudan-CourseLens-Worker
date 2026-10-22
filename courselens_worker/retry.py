"""Bounded retry ladder, single-sourced.

ARCH-P1-2（product-v010archaudit-result-20261014.md P1 波 2）：llm.py /
platform_session.py / asr.py 的站级重试梯（有界循环 + 末次不重试门 + 退避
睡眠）此前逐站手写、attempts/backoff 参数散落；本模块收拢为单一驱动。

等价纪律：每站的尝试数、退避参数、重试谓词、遥测行与耗尽处置全部由调用方
显式给出，驱动自身零策略、零第三方依赖。``sleep`` 由站点传入自己模块的
``time.sleep``——测试对 ``{site模块}.time.sleep`` 的补丁在调用期解析，
补钉兼容语义与收敛前逐字节一致。
"""

from __future__ import annotations

from typing import Any, Callable, Union

SecondsBetween = Union[int, float, Callable[[int], float]]


class LadderFailure(Exception):
    """站点内部信号：本次尝试按站点谓词判定为「可重试失败」。

    ``error`` 携带站点选定的原始失败载体（通常即捕获到的异常）；驱动在梯尽
    时原样上抛 ``error``。其余异常驱动一律不捕获——站点的 no-retry 通道
    （闭集码直抛）与确定性失败原样穿透，语义与手写梯一致。
    """

    def __init__(self, error: BaseException) -> None:
        self.error = error
        super().__init__(str(error))


class LadderExhausted(Exception):
    """无天然异常对象的失败（校验未过/未达标）的上梯载体。

    站点以它作 ``LadderFailure.error``，并在梯尽后按站点语义收口（降级/
    换码），本类型不向调用方泄露。
    """


def run_ladder(
    attempt_fn: Callable[[int], Any],
    attempts: int,
    *,
    seconds_between: SecondsBetween,
    sleep: Callable[[float], None],
    before_retry: Callable[[int, BaseException], None] | None = None,
) -> Any:
    """Run one bounded retry ladder; return the first successful attempt.

    ``attempt_fn(尝试号)`` 返回即整梯返回。抛 ``LadderFailure`` 记为一次可
    重试失败：若仍有剩余尝试，先调 ``before_retry(下一次尝试号, 失败载体)``
    （遥测/计数/重建动作），再以 ``seconds_between`` 退避——常数直接透传给
    ``sleep``；可调用形态则以本次失败的尝试号（0 起）调用取回延迟。梯尽时
    上抛最后一次的 ``error``。其余异常原样穿透。前置条件 ``attempts >= 1``
    （全部调用面均为编译期常量）。
    """
    last: LadderFailure | None = None
    for attempt in range(attempts):
        try:
            return attempt_fn(attempt)
        except LadderFailure as failure:
            # except-as 名字在块尾即被解释器解绑，梯间引用一律走 last。
            last = failure
        if attempt + 1 < attempts:
            if before_retry is not None:
                before_retry(attempt + 1, last.error)
            delay: SecondsBetween = seconds_between
            if callable(delay):
                delay = delay(attempt)
            sleep(delay)
    assert last is not None  # attempts >= 1 时梯尽必有失败记录
    raise last.error
