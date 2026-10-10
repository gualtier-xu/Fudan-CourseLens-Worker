"""Blacklist filter for recurring browser junk pages in slide decks.

Some platform slide URLs serve full browser-window screenshots of public
notice pages (tab bar, address bar, and taskbar in frame) instead of real
courseware.  Those pages are pure noise for PPT extraction, so each decoded
frame is reduced and compared against a small blacklist before the OCR
stage runs; a hit is skipped with a closed-set reason and never becomes a
slide page.

The match is deliberately robust to everything that changes between
captures of the same junk page — browser chrome, per-PC peripherals,
compression noise, mild blur, moderate scale drift.  Each frame is reduced
to its centre content region (``JUNK_ROI``), normalized to a fixed size,
lightly blurred, and condensed into two 64-bit perceptual hashes.  A frame
is junk when EITHER hash lands within ``JUNK_HAMMING_THRESHOLD`` bits of a
blacklist entry: the OR tolerates each single algorithm's blind spots.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# 如何新增黑名单条目：取一张垃圾页截图（图片不入仓）→
#   entry = entry_for(image, label="简短标识")
# → 把返回的 {"dhash", "ahash", "label"} 追加到下方 JUNK_PAGE_BLACKLIST
#   一行即可。哈希注册与线上判定走同一条 dual_hash 路径，无需另行校准。

JUNK_PAGE_SKIP_REASON = "junk_page"
JUNK_HAMMING_THRESHOLD = 12
JUNK_NORM_SIZE = 64
# Centre content region as (left, top, right, bottom) fractions: excludes the
# tab bar, address bar, taskbar, and window edges that differ per capture.
JUNK_ROI = (0.20, 0.15, 0.80, 0.85)

JUNK_PAGE_BLACKLIST: tuple[dict[str, str], ...] = (
    {"dhash": "662f5ce3e3f19292", "ahash": "1f81e4383178f8fc", "label": "course-notice-page"},
    {"dhash": "6e0754e363f182d1", "ahash": "2381fc3839f8fafc", "label": "ce-evaluation-notice"},
    {"dhash": "3d23235163530504", "ahash": "808191f9f0f9ffff", "label": "service-hall-portal"},
    {"dhash": "b4661de163719251", "ahash": "1f1f807839fcfff8", "label": "notice-variant-b"},
)

# ---- V2 四级流水（NIGHT5-U4）------------------------------------------------
# 背景：黑名单只能冻结已见过的页面；平台侧每个学校的门户皮肤不同、截图设备
# 不同，同一张通知页的变体可以无限多。V2 在黑名单之外叠加三层不依赖具体皮肤
# 的通用判定，全部阈值取自 2026-09 六样本真页分析（757/68、94 零死亡、单例
# 正裁、17+ 变体、27/27 六钉），在真实样本上校准后冻结：
#   ① 无特征页：灰度方差过低 / dhash 全零 / ahash 置位过少 —— 空白页与近纯色
#     截图；深底板书（稀疏粉笔笔画）靠动态范围豁免，不得误杀。
#   ② 浏览器 chrome 结构门：顶带（标签+地址栏）行方差低且横向强边缘足够多，
#     且底带（任务栏）呈条带结构 —— 这是"整窗截图"的形状证据，与皮肤无关。
#   ③ 组内近重复家族：黑名单近失（13~18 位）的幸存页按种子半径聚簇，≥3 张
#     成族 —— 同一张垃圾页的变体天然聚簇，真实讲义不会。
#   ④ OCR 语义裁决：家族代表页与带弱信号的孤页，正文命中闭集词表才落刀；
#     语义永远不单独杀人，必须伴随形状或聚簇证据。
JUNK_FEATURELESS_GV_MAX = 8.0
JUNK_AHASH_MIN_BITS = 6
JUNK_BOARD_RANGE_MIN = 60.0
JUNK_CHROME_TOP_FRAC = 0.12
JUNK_CHROME_TOP_EDGE_FRAC = 0.15
JUNK_CHROME_BOTTOM_FRAC = 0.08
JUNK_CHROME_ROWVAR_MAX = 300.0
JUNK_CHROME_EDGE_MIN = 10
JUNK_CHROME_EDGE_DELTA = 8.0
JUNK_CHROME_BOTTOM_VAR_MIN = 1500.0
JUNK_FAMILY_RADIUS = 8
JUNK_FAMILY_MIN_MEMBERS = 3
JUNK_BLACKLIST_NEAR_MAX = 18
JUNK_SEMANTIC_WORDS: tuple[str, ...] = ("通知", "公告", "评教", "提醒", "服务大厅", "门户")


def _normalized(image: Image.Image) -> Image.Image:
    """ROI-crop, normalize, and denoise one frame into the hashing domain."""
    left, top, right, bottom = JUNK_ROI
    width, height = image.size
    box = (
        int(width * left),
        int(height * top),
        int(width * right),
        int(height * bottom),
    )
    crop = image
    if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
        crop = image.crop(box)
    gray = crop.convert("L").resize((JUNK_NORM_SIZE, JUNK_NORM_SIZE))
    return gray.filter(ImageFilter.GaussianBlur(1))


def dual_hash(image: Image.Image) -> tuple[str, str]:
    """Return ``(dhash_hex, ahash_hex)`` — two 64-bit perceptual hashes."""
    normalized = _normalized(image)
    dgrid = np.asarray(normalized.resize((9, 8)))
    dvalue = 0
    for bit in (dgrid[:, 1:] > dgrid[:, :-1]).flatten():
        dvalue = (dvalue << 1) | int(bit)
    agrid = np.asarray(normalized.resize((8, 8)))
    avalue = 0
    for bit in (agrid > agrid.mean()).flatten():
        avalue = (avalue << 1) | int(bit)
    return f"{dvalue:016x}", f"{avalue:016x}"


def hamming_distance(left: str, right: str) -> int:
    """Count differing bits between two same-width hex hashes."""
    return (int(left, 16) ^ int(right, 16)).bit_count()


def entry_for(image: Image.Image, label: str) -> dict[str, str]:
    """Build one blacklist row from a junk-page screenshot."""
    dhash_hex, ahash_hex = dual_hash(image)
    return {"dhash": dhash_hex, "ahash": ahash_hex, "label": label}


def junk_match(
    image: Image.Image,
    blacklist: tuple[dict[str, str], ...] | None = None,
) -> tuple[bool, str]:
    """Return ``(is_junk, matched_label)`` under the dual-hash OR rule.

    ``blacklist`` defaults to the shipped ``JUNK_PAGE_BLACKLIST``, read at
    call time so tests (and future dynamic sources) can substitute their own
    rows without rebinding this module.
    """
    rows = JUNK_PAGE_BLACKLIST if blacklist is None else blacklist
    dhash_hex, ahash_hex = dual_hash(image)
    for row in rows:
        if (
            hamming_distance(dhash_hex, row["dhash"]) <= JUNK_HAMMING_THRESHOLD
            or hamming_distance(ahash_hex, row["ahash"]) <= JUNK_HAMMING_THRESHOLD
        ):
            return True, str(row.get("label") or "")
    return False, ""


def is_junk_page(image: Image.Image) -> bool:
    """Pipeline-facing verdict against the shipped blacklist."""
    matched, _label = junk_match(image)
    return matched


def frame_features(image: Image.Image) -> dict[str, object]:
    """Compute every V2 signal for one frame in a single pass.

    Returns a plain dict so it can travel with the page record through the
    OCR pipeline and feed the deck-level adjudication later.  Hashes reuse
    :func:`dual_hash` so the blacklist, the family clustering, and the
    registration helper all live in one perceptual domain.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    height, _width = gray.shape
    top_band = gray[: int(height * JUNK_CHROME_TOP_FRAC), :]
    edge_band = gray[: int(height * JUNK_CHROME_TOP_EDGE_FRAC), :]
    bottom_band = gray[int(height * (1.0 - JUNK_CHROME_BOTTOM_FRAC)):, :]
    row_means_top = top_band.mean(axis=1)
    row_edges = np.abs(np.diff(edge_band, axis=0)).mean(axis=1)
    row_means_bottom = bottom_band.mean(axis=1)
    dhash_hex, ahash_hex = dual_hash(image)
    top_var = float(row_means_top.var())
    strong_edges = int((row_edges > JUNK_CHROME_EDGE_DELTA).sum())
    bottom_var = float(row_means_bottom.var())
    near = min(
        min(hamming_distance(dhash_hex, row["dhash"]), hamming_distance(ahash_hex, row["ahash"]))
        for row in JUNK_PAGE_BLACKLIST
    )
    return {
        "gv": float(gray[::4, ::4].var()),
        "range": float(np.percentile(gray, 99) - np.percentile(gray, 1)),
        "dhash": dhash_hex,
        "ahash": ahash_hex,
        "ahash_bits": bin(int(ahash_hex, 16)).count("1"),
        "top_var": top_var,
        "strong_edges": strong_edges,
        "bottom_var": bottom_var,
        "chrome": (
            top_var < JUNK_CHROME_ROWVAR_MAX
            and strong_edges >= JUNK_CHROME_EDGE_MIN
            and bottom_var > JUNK_CHROME_BOTTOM_VAR_MIN
        ),
        "weak_chrome": (
            (top_var < JUNK_CHROME_ROWVAR_MAX and strong_edges >= JUNK_CHROME_EDGE_MIN)
            or bottom_var > JUNK_CHROME_BOTTOM_VAR_MIN
        ),
        "near": int(near),
    }


def featureless_verdict(features: dict[str, object]) -> bool:
    """Stage ① — blank/near-solid pages, with the dark-board exemption.

    A blackboard photo has low variance overall but a wide dynamic range
    (chalk strokes against a dark ground), so a wide range always survives;
    a blank scan or a near-solid screenshot does not.
    """
    if float(features["range"]) >= JUNK_BOARD_RANGE_MIN:
        return False
    if float(features["gv"]) < JUNK_FEATURELESS_GV_MAX:
        return True
    if int(features["dhash"], 16) == 0:
        return True
    return int(features["ahash_bits"]) < JUNK_AHASH_MIN_BITS


def chrome_verdict(features: dict[str, object]) -> bool:
    """Stage ② — full-window browser screenshot by band structure alone."""
    return bool(features["chrome"])


def semantic_hit(text: str) -> bool:
    """Stage ④ — closed-vocabulary notice-page wording."""
    hay = str(text or "")
    return any(word in hay for word in JUNK_SEMANTIC_WORDS)


def near_duplicate_families(
    records: list[dict[str, object]],
    radius: int = JUNK_FAMILY_RADIUS,
    min_members: int = JUNK_FAMILY_MIN_MEMBERS,
) -> list[list[int]]:
    """Stage ③ — seed-cluster the near-miss survivors; return index groups.

    Only pages whose hashes sit within ``JUNK_BLACKLIST_NEAR_MAX`` of some
    blacklist row are eligible: this is the "variant of a known junk page"
    zone, so real repeated slides never even enter the clustering.
    """
    eligible = [index for index, record in enumerate(records) if int(record.get("near") or 0) <= JUNK_BLACKLIST_NEAR_MAX]
    families: list[list[int]] = []
    pool = list(eligible)
    while pool:
        seed = pool.pop(0)
        family, rest = [seed], []
        for index in pool:
            record = records[index]
            seed_record = records[seed]
            if (
                hamming_distance(str(seed_record["dhash"]), str(record["dhash"])) <= radius
                or hamming_distance(str(seed_record["ahash"]), str(record["ahash"])) <= radius
            ):
                family.append(index)
            else:
                rest.append(index)
        pool = rest
        if len(family) >= min_members:
            families.append(family)
    return families


def adjudicate_deck(records: list[dict[str, object]]) -> list[bool]:
    """Deck-level pass: stages ③+④ over one run's survivors.

    ``records`` carries one feature dict plus the recognized ``text`` per
    page.  Returns a keep/kill verdict per record.  A family dies when its
    seed page's wording hits the closed notice vocabulary; a lone page dies
    only when weak chrome structure AND notice wording both agree.  Neither
    signal ever kills alone, so real lecture pages (no chrome bands, no
    near-miss clustering) are unreachable by this pass.
    """
    verdicts = [False] * len(records)
    for family in near_duplicate_families(records):
        seed = family[0]
        if not semantic_hit(str(records[seed].get("text") or "")):
            continue
        for index in family:
            verdicts[index] = True
    for index, record in enumerate(records):
        if verdicts[index]:
            continue
        if record.get("weak_chrome") and semantic_hit(str(record.get("text") or "")):
            verdicts[index] = True
    return verdicts
