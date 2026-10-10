"""courseware_plan.v1 derivation: allowlist, collapse, retention, ordering."""

from __future__ import annotations

import json
import unittest

from courselens_worker.cloud_automation import (
    PLAN_SCHEMA,
    _dedupe_key,
    _plan_page_label,
    _plan_record_id,
    build_courseware_plan,
)
from courselens_worker.protocol import canonical_json, sha256_hex


def _page(num, sha, *, created=0, dhash="", text=""):
    return {
        "page_num": num, "created_sec": created, "source_sha256": sha,
        "dhash": dhash, "text": text,
    }


def _inventory(*rows):
    return [
        {"page_num": index + 1, "created_sec": row[0], "record_id": row[1]}
        for index, row in enumerate(rows)
    ]


class CoursewarePlanTests(unittest.TestCase):
    def test_field_allowlist_has_no_url_image_or_ocr_leakage(self):
        plan = build_courseware_plan(
            course_id="36941", sub_id="l-1",
            pages=[
                _page(1, "a" * 64, created=30, text="<html>auth page</html> 第1页"),
                _page(2, "b" * 64, created=60, dhash="f" * 16,
                      text="机密讲义正文 11/60"),
            ],
            inventory=_inventory(
                (30, "https://leak.invalid/1?sig=secret"),
                (60, "101"),
            ),
        )
        rendered = json.dumps(plan, ensure_ascii=False)
        for forbidden in (
            "http", "://", "pptimgurl", "cookie", "Cookie", "@",
            "机密讲义正文", "<html>", "url", "image", "thumb",
        ):
            self.assertNotIn(forbidden, rendered, f"plan must not leak {forbidden}")
        # A URL-shaped inventory record id is dropped, not carried.
        self.assertEqual(plan["entries"][0]["record_id"], "")
        self.assertEqual(plan["entries"][1]["record_id"], "101")
        # Only the normalized page label survives from OCR text.
        self.assertEqual(plan["entries"][1]["page_label"], "11/60")
        self.assertEqual(plan["entries"][0]["page_label"], "1")
        # Closed field sets on every structure.
        self.assertEqual(
            set(plan),
            {"schema", "policy_version", "pipeline", "course_id", "sub_id",
             "inventory_digest", "entries", "excluded", "ordering", "counts"},
        )
        for entry in plan["entries"]:
            self.assertEqual(
                set(entry),
                {"output_position", "capture_position", "record_id",
                 "capture_time", "capture_ordinal",
                 "source_sha256", "page_label", "page_label_source", "annotation",
                 "keep_reason", "version_of_position", "duplicate_count"},
            )
            self.assertEqual(
                set(entry["annotation"]), {"class", "confidence"},
            )
        for item in plan["excluded"]:
            self.assertEqual(
                set(item),
                {"kept_position", "capture_time", "capture_ordinal",
                 "source_sha256", "reason"},
            )

    def test_only_exact_source_sha256_duplicates_collapse(self):
        plan = build_courseware_plan(
            course_id="c", sub_id="l",
            pages=[
                _page(1, "a" * 64, created=10),
                _page(2, "a" * 64, created=20),
                _page(3, "a" * 64, created=30),
                _page(4, "b" * 64, created=40),
            ],
            inventory=_inventory((10, "1"), (20, "2"), (30, "3"), (40, "4")),
        )
        self.assertEqual(plan["counts"]["kept"], 2)
        self.assertEqual(plan["counts"]["exact_duplicates"], 2)
        excluded = plan["excluded"]
        self.assertTrue(all(item["reason"] == "exact_duplicate" for item in excluded))
        self.assertEqual({item["kept_position"] for item in excluded}, {1})
        # Every capture time stays auditable, including collapsed ones.
        self.assertEqual(
            sorted(item["capture_time"] for item in excluded), [20, 30],
        )
        self.assertEqual(plan["entries"][0]["duplicate_count"], 2)

    def test_distinct_annotated_and_unknown_variants_are_retained(self):
        plan = build_courseware_plan(
            course_id="c", sub_id="l",
            pages=[
                _page(1, "a" * 64, created=10, dhash="0" * 16, text="第1页"),
                _page(2, "b" * 64, created=20, dhash="f" * 16, text="第1页 批注"),
                _page(3, "c" * 64, created=30, dhash="1" * 16, text="无标签"),
            ],
            inventory=_inventory((10, "1"), (20, "2"), (30, "3")),
        )
        self.assertEqual(plan["counts"]["kept"], 3)
        self.assertEqual(plan["counts"]["exact_duplicates"], 0)
        by_sha = {entry["source_sha256"]: entry for entry in plan["entries"]}
        # Same page label with a visual difference: protected annotated variant.
        self.assertEqual(
            by_sha["b" * 64]["annotation"],
            {"class": "annotated_candidate", "confidence": 0.55},
        )
        self.assertEqual(by_sha["b" * 64]["keep_reason"], "version_variant_retained")
        self.assertEqual(by_sha["b" * 64]["version_of_position"], 1)
        # Unlabeled distinct content stays unknown and kept.
        self.assertEqual(
            by_sha["c" * 64]["annotation"], {"class": "unknown", "confidence": 0.0},
        )

    def test_page_label_ordering_requires_full_confidence_gate(self):
        # Capture order is deliberately shuffled (2,3,1,4) so only a passing
        # gate can reorder the output positions.
        capture_times = [10, 20, 30, 40]
        numbers = [2, 3, 1, 4]
        pages = [
            _page(index + 1, f"{index}" * 64, created=capture_times[index],
                  text=f"内容 {numbers[index]}/4")
            for index in range(4)
        ]
        inventory = _inventory((10, "1"), (20, "2"), (30, "3"), (40, "4"))
        plan = build_courseware_plan(
            course_id="c", sub_id="l", pages=pages, inventory=inventory,
        )
        self.assertEqual(plan["ordering"], {"mode": "page_label", "confidence": 0.9})
        self.assertEqual(
            [(entry["output_position"], entry["capture_time"]) for entry in plan["entries"]],
            [(1, 30), (2, 10), (3, 20), (4, 40)],
        )
        # capture_position stays the stable first-capture identity even when
        # page-label ordering reorders the output (entries are listed in
        # output order, so capture positions appear shuffled).
        self.assertEqual(
            [(entry["capture_position"], entry["capture_time"]) for entry in plan["entries"]],
            [(3, 30), (1, 10), (2, 20), (4, 40)],
        )

    def test_missing_label_or_denominator_conflict_falls_back_to_capture_order(self):
        complete = [
            _page(index + 1, f"{index}" * 64, created=index * 10,
                  text=f"{index + 1}/4")
            for index in range(4)
        ]
        inventory = _inventory(*[(index * 10, str(index + 1)) for index in range(4)])
        missing_label = list(complete)
        missing_label[3] = _page(4, "9" * 64, created=30, text="no label")
        plan = build_courseware_plan(
            course_id="c", sub_id="l", pages=missing_label, inventory=inventory,
        )
        self.assertEqual(plan["ordering"]["mode"], "capture_order")

        conflict = list(complete)
        conflict[3] = _page(4, "9" * 64, created=30, text="4/5")
        plan = build_courseware_plan(
            course_id="c", sub_id="l", pages=conflict, inventory=inventory,
        )
        self.assertEqual(plan["ordering"]["mode"], "capture_order")

        repeated = [
            _page(1, "a" * 64, created=0, text="1/4"),
            _page(2, "b" * 64, created=10, text="1/4"),
            _page(3, "c" * 64, created=20, text="2/4"),
            _page(4, "d" * 64, created=30, text="3/4"),
        ]
        plan = build_courseware_plan(
            course_id="c", sub_id="l", pages=repeated, inventory=inventory,
        )
        self.assertEqual(plan["ordering"]["mode"], "capture_order")

    def test_capture_ordinal_counts_same_time_occurrences(self):
        plan = build_courseware_plan(
            course_id="c", sub_id="l",
            pages=[
                _page(1, "a" * 64, created=10),
                _page(2, "b" * 64, created=10),
                _page(3, "c" * 64, created=10),
            ],
            inventory=_inventory((10, "1"), (10, "2"), (10, "3")),
        )
        self.assertEqual(
            [entry["capture_ordinal"] for entry in plan["entries"]], [0, 1, 2],
        )

    def test_record_id_joins_only_through_capture_time_guard(self):
        plan = build_courseware_plan(
            course_id="c", sub_id="l",
            pages=[
                _page(1, "a" * 64, created=10),
                _page(2, "b" * 64, created=999),
            ],
            inventory=_inventory((10, "1"), (20, "2")),
        )
        # page 1 matches the inventory time -> record id joins.
        self.assertEqual(plan["entries"][0]["record_id"], "1")
        # page 2's time disagrees with its inventory row -> no record id.
        self.assertEqual(plan["entries"][1]["record_id"], "")

    def test_plan_digest_is_stable_and_binding_is_recorded(self):
        pages = [_page(1, "a" * 64, created=10, text="1/2")]
        inventory = _inventory((10, "1"))
        first = build_courseware_plan(
            course_id="c", sub_id="l", pages=pages, inventory=inventory,
        )
        second = build_courseware_plan(
            course_id="c", sub_id="l", pages=pages, inventory=inventory,
        )
        self.assertEqual(first, second)
        self.assertEqual(sha256_hex(canonical_json(first)), sha256_hex(canonical_json(second)))
        self.assertEqual(first["schema"], PLAN_SCHEMA)
        self.assertEqual(first["pipeline"], "cloud-automation.v3")
        # A different lecture binding yields a different plan document.
        other = build_courseware_plan(
            course_id="c", sub_id="l-2", pages=pages, inventory=inventory,
        )
        self.assertNotEqual(
            sha256_hex(canonical_json(first)), sha256_hex(canonical_json(other)),
        )

    def test_record_id_and_label_sanitizers_are_closed(self):
        self.assertEqual(_plan_record_id(" 101 "), "101")
        self.assertEqual(_plan_record_id("https://x.invalid/a"), "")
        self.assertEqual(_plan_record_id("a@b"), "")
        self.assertEqual(_plan_record_id("a b"), "")
        self.assertEqual(_plan_record_id("x" * 65), "")
        self.assertEqual(_plan_record_id(None), "")
        self.assertEqual(_plan_page_label("第 12 页"), "12")
        self.assertEqual(_plan_page_label("3/48"), "3/48")
        self.assertEqual(_plan_page_label("99/98"), "")
        self.assertEqual(_plan_page_label("plain text"), "")

    def test_plan_keys_are_orthogonal_to_dedupe_identity(self):
        """计划绑定课程/讲次；完成工作身份绑定账户/束/管线，两者互不混淆。"""
        rules = {"account_id": "a", "schema": "cloud-automation.v3"}
        key = _dedupe_key(rules, "c", "l")
        plan = build_courseware_plan(
            course_id="c", sub_id="l", pages=[], inventory=[],
        )
        self.assertNotIn(key, json.dumps(plan))
        self.assertEqual(plan["counts"]["kept"], 0)
        self.assertEqual(plan["ordering"]["mode"], "capture_order")


if __name__ == "__main__":
    unittest.main()
