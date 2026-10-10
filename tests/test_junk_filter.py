from __future__ import annotations

import io
import unittest
from unittest.mock import Mock, patch

# The private Windows CI installs the real Worker OCR stack; lighter runtimes
# (the public mirror unit job) only provide the signing dependencies, so these
# contracts skip there instead of failing to import.
try:
    from PIL import Image, ImageDraw, ImageFilter
except ModuleNotFoundError:
    Image = None
try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    numpy = None

from courselens_worker.junk_filter import (
    JUNK_HAMMING_THRESHOLD,
    JUNK_PAGE_BLACKLIST,
    JUNK_PAGE_SKIP_REASON,
    adjudicate_deck,
    chrome_verdict,
    dual_hash,
    entry_for,
    featureless_verdict,
    frame_features,
    hamming_distance,
    junk_match,
    near_duplicate_families,
    semantic_hit,
)

if Image is not None:
    BASE_W, BASE_H = 800, 450

    def _png_bytes(image: "Image.Image") -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _notice_master(seed: int) -> "Image.Image":
        """A synthetic browser-notice screenshot: chrome bands around a table.

        Mirrors the real junk pages: a colored banner and window chrome in the
        outer bands (what changes per capture), a deterministic bordered table
        in the centre content region (what survives the ROI crop).
        """
        rng = numpy.random.default_rng(seed)
        image = Image.new("RGB", (BASE_W, BASE_H), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, BASE_W, int(BASE_H * 0.11)], fill=(40, 44, 52))
        draw.rectangle([0, int(BASE_H * 0.90), BASE_W, BASE_H], fill=(190, 194, 199))
        draw.rectangle(
            [int(BASE_W * 0.05), int(BASE_H * 0.03), int(BASE_W * 0.40), int(BASE_H * 0.10)],
            fill=(200, 30, 40),
        )
        left, top = int(BASE_W * 0.22), int(BASE_H * 0.18)
        right, bottom = int(BASE_W * 0.78), int(BASE_H * 0.80)
        draw.rectangle([left, top, right, bottom], outline=(120, 120, 120), width=3)
        rows, columns = 6, 4
        for row in range(1, rows):
            y = top + (bottom - top) * row // rows
            draw.line([left, y, right, y], fill=(150, 150, 150), width=2)
        for column in range(1, columns):
            x = left + (right - left) * column // columns
            draw.line([x, top, x, bottom], fill=(150, 150, 150), width=2)
        for row in range(rows):
            for column in range(columns - 1):
                x0 = left + (right - left) * column // columns + 8
                y0 = top + (bottom - top) * row // rows + 8
                x1 = x0 + 30 + int(rng.integers(0, 40))
                y1 = y0 + 6
                if x1 < left + (right - left) * (column + 1) // columns - 4:
                    draw.rectangle([x0, y0, x1, y1], fill=(60, 60, 66))
        return image

    def _peripheral_variant(image: "Image.Image", seed: int) -> "Image.Image":
        """Overwrite the outer bands with per-capture noise: another PC's chrome."""
        rng = numpy.random.default_rng(seed)
        array = numpy.asarray(image).copy()
        top = int(BASE_H * 0.15)
        bottom = int(BASE_H * 0.85)
        left = int(BASE_W * 0.20)
        right = int(BASE_W * 0.80)
        array[:top, :, :] = rng.integers(0, 256, size=(top, BASE_W, 3), dtype=numpy.uint8)
        array[bottom:, :, :] = rng.integers(0, 256, size=(BASE_H - bottom, BASE_W, 3), dtype=numpy.uint8)
        array[:, :left, :] = rng.integers(0, 256, size=(BASE_H, left, 3), dtype=numpy.uint8)
        array[:, right:, :] = rng.integers(0, 256, size=(BASE_H, BASE_W - right, 3), dtype=numpy.uint8)
        return Image.fromarray(array)

    def _noisy_variant(image: "Image.Image", seed: int, sigma: int = 20) -> "Image.Image":
        rng = numpy.random.default_rng(seed)
        array = numpy.asarray(image).astype(numpy.int16)
        noise = rng.normal(0, sigma, size=array.shape)
        return Image.fromarray(numpy.clip(array + noise, 0, 255).astype(numpy.uint8))

    def _scaled_variant(image: "Image.Image", factor: float) -> "Image.Image":
        return image.resize((int(BASE_W * factor), int(BASE_H * factor)))

    def _solid(color: tuple[int, int, int]) -> "Image.Image":
        return Image.new("RGB", (BASE_W, BASE_H), color)

    def _normal_slides() -> list["Image.Image"]:
        """Ten distinct courseware-style frames that must never be filtered."""
        slides: list["Image.Image"] = [
            _solid((255, 255, 255)),
            _solid((12, 12, 16)),
            _solid((245, 240, 230)),
        ]
        title = Image.new("RGB", (BASE_W, BASE_H), "white")
        ImageDraw.Draw(title).rectangle([80, 60, 520, 110], fill=(20, 20, 24))
        slides.append(title)
        paragraph = Image.new("RGB", (BASE_W, BASE_H), (252, 252, 250))
        draw = ImageDraw.Draw(paragraph)
        for index in range(9):
            y = 50 + index * 38
            draw.rectangle([60, y, 740 - (index % 3) * 90, y + 14], fill=(50, 54, 60))
        slides.append(paragraph)
        gradient = Image.new("RGB", (BASE_W, BASE_H))
        draw = ImageDraw.Draw(gradient)
        for y in range(BASE_H):
            shade = int(240 - 200 * y / BASE_H)
            draw.line([(0, y), (BASE_W, y)], fill=(shade, shade, min(255, shade + 8)))
        slides.append(gradient)
        checker = Image.new("RGB", (BASE_W, BASE_H), "white")
        draw = ImageDraw.Draw(checker)
        for row in range(9):
            for column in range(16):
                if (row + column) % 2 == 0:
                    draw.rectangle([column * 50, row * 50, column * 50 + 49, row * 50 + 49], fill=(70, 70, 80))
        slides.append(checker)
        photo = Image.fromarray(
            numpy.random.default_rng(7).integers(0, 256, size=(BASE_H, BASE_W, 3), dtype=numpy.uint8)
        )
        slides.append(photo)
        dark_deck = Image.new("RGB", (BASE_W, BASE_H), (24, 26, 34))
        draw = ImageDraw.Draw(dark_deck)
        for index in range(6):
            y = 70 + index * 56
            draw.rectangle([90, y, 620, y + 22], fill=(210, 214, 222))
        slides.append(dark_deck)
        diagram = Image.new("RGB", (BASE_W, BASE_H), "white")
        draw = ImageDraw.Draw(diagram)
        draw.ellipse([260, 90, 540, 370], outline=(180, 30, 40), width=6)
        draw.rectangle([90, 180, 240, 280], outline=(30, 60, 180), width=5)
        draw.line([240, 230, 260, 230], fill=(30, 60, 180), width=5)
        slides.append(diagram)
        return slides


@unittest.skipIf(Image is None or numpy is None, "Pillow and numpy runtimes required")
class JunkFilterUnitTests(unittest.TestCase):
    def test_shipped_blacklist_is_wellformed_and_pinned(self):
        self.assertEqual(
            [(row["dhash"], row["ahash"], row["label"]) for row in JUNK_PAGE_BLACKLIST],
            [
                ("662f5ce3e3f19292", "1f81e4383178f8fc", "course-notice-page"),
                ("6e0754e363f182d1", "2381fc3839f8fafc", "ce-evaluation-notice"),
                ("3d23235163530504", "808191f9f0f9ffff", "service-hall-portal"),
                ("b4661de163719251", "1f1f807839fcfff8", "notice-variant-b"),
            ],
        )
        for row in JUNK_PAGE_BLACKLIST:
            self.assertRegex(row["dhash"], r"^[0-9a-f]{16}$")
            self.assertRegex(row["ahash"], r"^[0-9a-f]{16}$")

    def test_dual_hash_is_64bit_and_deterministic(self):
        master = _notice_master(0)
        first = dual_hash(master)
        second = dual_hash(master.copy())
        self.assertEqual(first, second)
        for value in first:
            self.assertRegex(value, r"^[0-9a-f]{16}$")

    def test_hamming_distance_counts_bit_differences(self):
        self.assertEqual(hamming_distance("00ff", "00ff"), 0)
        self.assertEqual(hamming_distance("0000", "ffff"), 16)
        self.assertEqual(hamming_distance("0f0f", "f0f0"), 16)
        self.assertEqual(hamming_distance("abcd", "abcd"), 0)
        self.assertEqual(hamming_distance("0001", "0000"), 1)

    def test_entry_for_matches_dual_hash(self):
        master = _notice_master(1)
        entry = entry_for(master, "synthetic")
        dhash_hex, ahash_hex = dual_hash(master)
        self.assertEqual(entry, {"dhash": dhash_hex, "ahash": ahash_hex, "label": "synthetic"})


@unittest.skipIf(Image is None or numpy is None, "Pillow and numpy runtimes required")
class JunkMatchMatrixTests(unittest.TestCase):
    """Synthetic interference matrix: 28 frames across the contract cases.

    The shipped blacklist pins the real screenshots' hashes; the images
    themselves stay out of the repository.  So the interference cases run
    against a blacklist dynamically registered from the synthetic masters
    through the same ``entry_for`` path a maintainer would use, while the
    precision cases (normal frames never filtered) run against the shipped
    blacklist as well.
    """

    @classmethod
    def _synthetic_blacklist(cls) -> tuple[dict[str, str], ...]:
        return tuple(
            entry_for(_notice_master(seed), f"synthetic-{seed}") for seed in (0, 1)
        )

    def test_exact_reproductions_are_all_filtered(self):
        blacklist = self._synthetic_blacklist()
        for seed in (0, 1):
            with self.subTest(seed=seed):
                matched, label = junk_match(_notice_master(seed), blacklist=blacklist)
                self.assertTrue(matched)
                self.assertTrue(label)

    def test_peripheral_changes_are_still_filtered(self):
        blacklist = self._synthetic_blacklist()
        for seed in (0, 1):
            for edge_seed in (11, 22, 33):
                with self.subTest(master=seed, edges=edge_seed):
                    matched, _label = junk_match(
                        _peripheral_variant(_notice_master(seed), edge_seed),
                        blacklist=blacklist,
                    )
                    self.assertTrue(matched)

    def test_noise_scale_and_blur_are_still_filtered(self):
        blacklist = self._synthetic_blacklist()
        variants = {
            "noise": lambda im: _noisy_variant(im, 5),
            "scale-0.5": lambda im: _scaled_variant(im, 0.5),
            "scale-1.5": lambda im: _scaled_variant(im, 1.5),
            "blur": lambda im: im.filter(ImageFilter.GaussianBlur(1.5)),
            "combined": lambda im: _noisy_variant(_scaled_variant(_peripheral_variant(im, 9), 0.75), 9),
        }
        for seed in (0, 1):
            for name, mutate in variants.items():
                with self.subTest(master=seed, variant=name):
                    matched, _label = junk_match(mutate(_notice_master(seed)), blacklist=blacklist)
                    self.assertTrue(matched)

    def test_normal_slides_are_never_filtered(self):
        for index, slide in enumerate(_normal_slides()):
            with self.subTest(slide=index):
                matched, label = junk_match(slide)
                self.assertFalse(matched, f"normal slide {index} hit shipped {label!r}")
                matched, label = junk_match(slide, blacklist=self._synthetic_blacklist())
                self.assertFalse(matched, f"normal slide {index} hit synthetic {label!r}")
                self.assertEqual(label, "")

    def test_solid_frames_stay_far_from_the_blacklist(self):
        for color in ((255, 255, 255), (0, 0, 0), (128, 128, 128)):
            with self.subTest(color=color):
                self.assertFalse(junk_match(_solid(color))[0])

    def test_boundary_distances_split_exactly_at_threshold(self):
        master = _notice_master(0)
        candidate_d, candidate_a = dual_hash(master)

        def _shifted(distance: int, base: str) -> str:
            return f"{int(base, 16) ^ ((1 << distance) - 1):016x}"

        d_edge = {
            "dhash": _shifted(JUNK_HAMMING_THRESHOLD, candidate_d),
            "ahash": _shifted(JUNK_HAMMING_THRESHOLD + 1, candidate_a),
            "label": "d-edge",
        }
        self.assertTrue(junk_match(master, blacklist=(d_edge,))[0])
        both_beyond = {
            "dhash": _shifted(JUNK_HAMMING_THRESHOLD + 1, candidate_d),
            "ahash": _shifted(JUNK_HAMMING_THRESHOLD + 1, candidate_a),
            "label": "beyond",
        }
        self.assertFalse(junk_match(master, blacklist=(both_beyond,))[0])
        a_edge = {
            "dhash": _shifted(JUNK_HAMMING_THRESHOLD + 1, candidate_d),
            "ahash": _shifted(JUNK_HAMMING_THRESHOLD - 1, candidate_a),
            "label": "a-edge",
        }
        self.assertTrue(junk_match(master, blacklist=(a_edge,))[0])


@unittest.skipIf(Image is None or numpy is None, "Pillow and numpy runtimes required")
class JunkPipelineTests(unittest.TestCase):
    """Pipeline placement: junk pages are skipped before OCR, never emitted.

    The real screenshots stay out of the repository, so the pipeline runs
    with ``JUNK_PAGE_BLACKLIST`` substituted by entries registered from the
    synthetic masters through the same ``entry_for`` path a maintainer uses;
    the shipped blacklist itself is exercised by the matrix tests above.
    """

    SYNTHETIC_BLACKLIST = tuple(
        entry_for(_notice_master(seed), f"synthetic-{seed}") for seed in (0, 1)
    )

    def _run(self, bodies: list, *, prior: dict | None = None):
        from courselens_worker.ocr import process_slides

        engine = Mock(return_value=([["box", "text"]], 0.1))
        responses = {f"https://example.invalid/{index}": body for index, body in enumerate(bodies)}
        slides = [
            {
                "page_num": index + 1,
                "created_sec": index * 10,
                "source": {"url": f"https://example.invalid/{index}"},
            }
            for index in range(len(bodies))
        ]
        with patch("courselens_worker.junk_filter.JUNK_PAGE_BLACKLIST", self.SYNTHETIC_BLACKLIST), \
             patch("courselens_worker.ocr.fetch_bytes", side_effect=lambda source: responses[str(source.get("url"))]), \
             patch("courselens_worker.ocr._engine", return_value=engine):
            pages, skipped = process_slides(slides, progress=lambda *_args: None, prior_checkpoint=prior)
        return pages, skipped, engine

    def test_junk_page_is_skipped_before_ocr_and_never_emitted(self):
        pages, skipped, engine = self._run([_png_bytes(_notice_master(0))])
        self.assertEqual(pages, [])
        self.assertEqual(skipped, {JUNK_PAGE_SKIP_REASON: 1})
        self.assertEqual(engine.call_count, 0)

    def test_mixed_deck_keeps_content_and_counts_junk(self):
        junk_png = _png_bytes(_notice_master(0))
        normal_png = _png_bytes(_normal_slides()[3])
        pages, skipped, engine = self._run([junk_png, normal_png, junk_png])
        self.assertEqual([page["page_num"] for page in pages], [2])
        self.assertEqual(skipped, {JUNK_PAGE_SKIP_REASON: 2})
        self.assertEqual(engine.call_count, 1)
        self.assertEqual(pages[0]["text"], "text")

    def test_junk_page_does_not_shift_surviving_page_numbers(self):
        junk_png = _png_bytes(_notice_master(1))
        pages, skipped, _engine = self._run([
            _png_bytes(_normal_slides()[3]),
            junk_png,
            _png_bytes(_normal_slides()[4]),
        ])
        self.assertEqual([page["page_num"] for page in pages], [1, 3])
        self.assertEqual(skipped, {JUNK_PAGE_SKIP_REASON: 1})

    def test_blank_solid_page_is_skipped_by_featureless_stage(self):
        # V2 四级流水（NIGHT5-U4）：纯色空页在管线层按无特征规则计入 junk_page；
        # 黑名单哈希路径的「纯色远离黑名单」钉不受影响（两层各司其职）。
        pages, skipped, engine = self._run([_png_bytes(_solid((255, 255, 255)))])
        self.assertEqual(pages, [])
        self.assertEqual(skipped, {JUNK_PAGE_SKIP_REASON: 1})
        self.assertEqual(engine.call_count, 0, "无特征页在 OCR 之前落刀")

    def test_junk_skip_counts_survive_checkpoint_resume(self):
        junk_png = _png_bytes(_notice_master(0))
        normal_png = _png_bytes(_normal_slides()[3])
        prior = {
            "ocr_completed_items": 1,
            "ppt_pages": [],
            "ppt_skipped": {JUNK_PAGE_SKIP_REASON: 1},
        }
        pages, skipped, engine = self._run([junk_png, normal_png], prior=prior)
        self.assertEqual([page["page_num"] for page in pages], [2])
        self.assertEqual(skipped, {JUNK_PAGE_SKIP_REASON: 1})
        self.assertEqual(engine.call_count, 1)

    def test_active_blacklist_leaves_plain_content_pages_alone(self):
        normal_png = _png_bytes(_normal_slides()[3])
        pages, skipped, _engine = self._run([normal_png, normal_png])
        self.assertEqual(len(pages), 2)
        self.assertEqual(skipped, {})


@unittest.skipIf(Image is None or numpy is None, "Pillow and numpy runtimes required")
class JunkFilterV2StageTests(unittest.TestCase):
    """V2 四级流水的合成钉（NIGHT5-U4）：阈值已在六份真实样本上校准冻结，
    这里钉住每一级的闭集行为——空白杀、板书豁免、chrome 形状门、家族聚簇
    与语义裁决的联合落刀（语义永不单独杀人）。"""

    def _featureless_kills(self, image):
        return featureless_verdict(frame_features(image))

    def test_blank_and_solid_pages_die_by_featureless_rule(self):
        for color in ((255, 255, 255), (250, 250, 249), (18, 18, 18), (128, 128, 128)):
            with self.subTest(color=color):
                self.assertTrue(self._featureless_kills(_solid(color)))

    def test_dark_board_with_sparse_chalk_survives(self):
        rng = numpy.random.default_rng(7)
        image = Image.new("RGB", (BASE_W, BASE_H), (32, 34, 38))
        draw = ImageDraw.Draw(image)
        for _ in range(24):
            x0 = int(rng.integers(60, BASE_W - 160))
            y0 = int(rng.integers(60, BASE_H - 120))
            draw.line([x0, y0, x0 + int(rng.integers(40, 140)), y0 + int(rng.integers(-10, 10))],
                      fill=(238, 236, 228), width=3)
        self.assertFalse(self._featureless_kills(image), "深底板书（稀疏笔画）不得被无特征规则误杀")

    def test_chrome_gate_kills_full_window_and_spares_content(self):
        window = Image.new("RGB", (BASE_W, BASE_H), "white")
        draw = ImageDraw.Draw(window)
        top_height = int(BASE_H * 0.12)
        for y in range(top_height):
            tone = 200 if (y % 4) < 2 else 180  # 细条纹：行均值交替约 20 灰阶
            draw.line([(0, y), (BASE_W, y)], fill=(tone, tone, tone))
        draw.rectangle([int(BASE_W * 0.2), int(BASE_H * 0.2), int(BASE_W * 0.8), int(BASE_H * 0.7)],
                       outline=(60, 60, 60), width=2)
        draw.rectangle([0, int(BASE_H * 0.92), BASE_W, int(BASE_H * 0.96)], fill=(225, 225, 225))
        draw.rectangle([0, int(BASE_H * 0.96), BASE_W, BASE_H], fill=(40, 40, 40))
        self.assertTrue(chrome_verdict(frame_features(window)), "带顶带条纹+任务栏的整窗截图应被 chrome 门识别")
        content = _notice_master(3)
        features = frame_features(content)
        # 通知页母版顶带不是细条纹形态：chrome 门不认，黑名单才是它的归宿
        self.assertIsNot(features["chrome"], True)

    def test_semantic_vocabulary_is_closed(self):
        for word in ("通知", "公告", "评教", "提醒", "服务大厅", "门户"):
            self.assertTrue(semantic_hit(f"关于{word}的说明"))
        self.assertFalse(semantic_hit("拉普拉斯方程：球坐标系\n勒让德多项式及其应用\n复旦大学"))
        self.assertFalse(semantic_hit(""))

    @staticmethod
    def _near_record(seed_d: int, seed_a: int, *, near=15, weak=True, text=""):
        def _hex(value):
            return f"{value:016x}"
        return {
            "dhash": _hex(seed_d), "ahash": _hex(seed_a), "near": near,
            "weak_chrome": weak, "text": text,
        }

    def test_family_dies_only_on_semantic_confession(self):
        base_d, base_a = 0x00FF00FF00FF00FF, 0x0F0F0F0F0F0F0F0F
        records = [
            self._near_record(base_d, base_a, text="服务大厅使用通知"),
            self._near_record(base_d ^ 0b0011, base_a ^ 0b0100),
            self._near_record(base_d ^ 0b1100, base_a ^ 0b1000),
        ]
        self.assertEqual([len(m) for m in near_duplicate_families(records)], [3])
        self.assertEqual(adjudicate_deck(records), [True, True, True], "家族+语义命中→全族落刀")
        for record in records:
            record["text"] = "拉普拉斯方程与勒让德多项式"
        self.assertEqual(adjudicate_deck(records), [False, False, False], "真实讲义重复页：家族不成罪，语义不落刀")

    def test_singleton_needs_weak_chrome_and_wording_together(self):
        base_d, base_a = 0x1234567890ABCDEF, 0xF0F0F0F0F0F0F0F0
        suspicious = self._near_record(base_d, base_a, near=30, weak=True, text="评教提醒")
        self.assertEqual(adjudicate_deck([suspicious]), [True], "弱 chrome+语义双证→落刀")
        plain = self._near_record(base_d, base_a, near=30, weak=False, text="评教提醒")
        self.assertEqual(adjudicate_deck([plain]), [False], "无形状证据的语义词绝不单独杀人")


if __name__ == "__main__":
    unittest.main()
