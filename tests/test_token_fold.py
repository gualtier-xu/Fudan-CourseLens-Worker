"""夜10-C 第九波任务1：邻接重复折叠单元钉（worker 转写后处理面）。

用户实测：口齿不清导致的 ASR 连续重复字词未有效剔除。折叠三规则：
R3 单字残叠 X+XY→XY；R1 立即同 token 串 run≥2→1（合法叠词保留）；
R2 ABAB≥2→AB；跨段相邻同文（≤5s）合并。确定性、零增字。
"""

from __future__ import annotations

import unittest

from courselens_worker.asr import (
    collapse_repeated_tokens,
    fold_transcript_repetitions,
)


class CollapseRepeatedTokenTests(unittest.TestCase):
    def test_immediate_word_run_collapses_to_single(self):
        self.assertEqual(
            collapse_repeated_tokens("我就不用这个不用这个话筒了")[0],
            "我就不用这个话筒了",
        )

    def test_single_char_run_collapses(self):
        self.assertEqual(collapse_repeated_tokens("那那那么我们开始")[0], "那么我们开始")
        self.assertEqual(collapse_repeated_tokens("呃呃然后")[0], "呃然后")

    def test_first_char_stutter_before_word_collapses(self):
        self.assertEqual(collapse_repeated_tokens("当当然一会考核")[0], "当然一会考核")
        # 尤尤其是=尤其是+尤其是 的口吃变体：R3 把 是尤其是 收敛为 尤其是
        self.assertEqual(collapse_repeated_tokens("尤尤其是这本书")[0], "尤其是这本书")

    def test_abab_stutter_collapses_with_trailing_residue(self):
        self.assertEqual(collapse_repeated_tokens("我我也我也没上过")[0], "我也没上过")
        self.assertEqual(
            collapse_repeated_tokens("不用这个不用这个话筒")[0], "不用这个话筒"
        )

    def test_legit_reduplication_is_preserved(self):
        self.assertEqual(collapse_repeated_tokens("慢慢来不要太急")[0], "慢慢来不要太急")
        self.assertEqual(collapse_repeated_tokens("刚刚到")[0], "刚刚到")

    def test_plain_text_is_untouched_and_zero_folds(self):
        self.assertEqual(collapse_repeated_tokens("今天我们讲数字集成电路")[1], 0)
        self.assertEqual(
            collapse_repeated_tokens("今天我们讲数字集成电路")[0], "今天我们讲数字集成电路"
        )


class FoldTranscriptRepetitionsTests(unittest.TestCase):
    def test_adjacent_identical_segments_merge_within_window(self):
        segments = [
            {"start_ms": 0, "end_ms": 2000, "text": "我们看下一页"},
            {"start_ms": 2600, "end_ms": 4600, "text": "我们看下一页"},
        ]
        stats = fold_transcript_repetitions(segments)
        self.assertEqual(stats["merged_adjacent"], 1)
        self.assertEqual(stats["segments"], 1)
        self.assertEqual(segments[0]["end_ms"], 4600)

    def test_beyond_window_identical_segments_are_kept(self):
        segments = [
            {"start_ms": 0, "end_ms": 2000, "text": "重复句"},
            {"start_ms": 60_000, "end_ms": 62_000, "text": "重复句"},
        ]
        stats = fold_transcript_repetitions(segments)
        self.assertEqual(stats["merged_adjacent"], 0)
        self.assertEqual(stats["segments"], 2)

    def test_in_place_mutation_reports_telemetry_only_counts(self):
        segments = [{"start_ms": 0, "end_ms": 2000, "text": "这个这个这样"}]
        stats = fold_transcript_repetitions(segments)
        self.assertEqual(segments[0]["text"], "这个这样")
        self.assertEqual(stats["folded_tokens"], 1)
        self.assertEqual(stats["segments"], 1)


if __name__ == "__main__":
    unittest.main()
