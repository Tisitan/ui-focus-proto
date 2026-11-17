#!/usr/bin/env python3
"""Unit + sanitization tests for ui-focus-proto."""

from __future__ import annotations

import json
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ui_focus  # noqa: E402
from vision_nodes import SnapshotTracker, build_vision_nodes  # noqa: E402


class _FakeHTTPResponse:
    def __init__(self, payload: dict, status: int = 200):
        self.status = status
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestVisionNodes(unittest.TestCase):
    def test_icon_dedup_and_text_absorption(self):
        regions = [
            {"id": "i1", "kind": "icon", "label": "btn", "confidence": 0.70, "bounds": {"x": 100, "y": 500, "width": 300, "height": 100}},
            {"id": "i2", "kind": "icon", "label": "btn", "confidence": 0.95, "bounds": {"x": 102, "y": 502, "width": 298, "height": 98}},
            {"id": "t1", "kind": "text", "text": "Confirm", "confidence": 0.99, "bounds": {"x": 160, "y": 525, "width": 180, "height": 48}},
            {"id": "tiny", "kind": "icon", "label": "dot", "confidence": 0.50, "bounds": {"x": 10, "y": 10, "width": 4, "height": 4}},
        ]
        nodes = build_vision_nodes(regions, 1080, 2400)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["ref"], "@v1")
        self.assertEqual(nodes[0]["text"], "Confirm")
        self.assertAlmostEqual(nodes[0]["confidence"], 0.95)

    def test_cid_guard_rejects_stale_snapshot(self):
        tracker = SnapshotTracker()
        nodes = build_vision_nodes(
            [{"kind": "text", "text": "OK", "bounds": {"x": 100, "y": 200, "width": 80, "height": 40}}],
            1080,
            2400,
        )
        cid1, _ = tracker.issue(nodes)
        self.assertEqual(tracker.resolve_click("@v1", cid1), (140, 220))
        cid2, _ = tracker.issue(nodes)
        with self.assertRaises(ValueError):
            tracker.resolve_click("@v1", cid1)
        self.assertEqual(tracker.resolve_click("@v1", cid2), (140, 220))


class TestUIFocus(unittest.TestCase):
    def setUp(self):
        ui_focus.reset_key_cache()

    def test_top_p_filtering_and_aria_noise_stripping(self):
        snap = "\n".join([
            "@view 1440x900",
            "L1 page",
            "  listitem",
            '    @e1 link "Home"',
            "  listitem",
            '    @e2 link "Billing & Invoices"',
            '    @e3 button "Sign Out"',
            '    @e4 link "Docs"',
            '    @e5 link "Status"',
            '    @e6 link "Privacy"',
        ])
        payload = {
            "answers": {
                "target": {
                    "probabilities": {
                        "e2": 0.92,
                        "e1": 0.02,
                        "e3": 0.02,
                        "e4": 0.02,
                        "e5": 0.01,
                        "e6": 0.01,
                    }
                }
            }
        }
        with patch.object(ui_focus, "_urlopen", return_value=_FakeHTTPResponse(payload)):
            res = ui_focus.focus_text(snap, "open billing", key="secret-test-key-123")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["kept"], 5)
        self.assertIn("★ [0.92] @e2 link \"Billing & Invoices\"", res["text"])
        self.assertNotIn("listitem", res["text"])
        self.assertNotIn("secret-test-key-123", res["text"])

    def test_fail_open_on_missing_key_or_timeout(self):
        snap = "[cid=c0001]\n[@v1] Vision:text \"A\"\n[@v2] Vision:text \"B\""
        with patch.object(ui_focus, "load_key", return_value=None):
            res = ui_focus.focus_text(snap, "tap B")
        self.assertEqual(res["status"], "fallback")
        self.assertEqual(res["code"], "no_key")
        self.assertIn("[@v1] Vision:text \"A\"", res["text"])
        self.assertIn("[@v2] Vision:text \"B\"", res["text"])

        with patch.object(ui_focus, "_urlopen", side_effect=TimeoutError("timed out")):
            res2 = ui_focus.focus_text(snap, "tap B", key="secret-test-key-456")
        self.assertEqual(res2["status"], "fallback")
        self.assertEqual(res2["code"], "timeout")
        self.assertNotIn("secret-test-key-456", res2["text"])

    def test_sharding_above_200_candidates(self):
        lines = ["[cid=c0009]"] + [f'[@e{i}] Button "Item {i}"' for i in range(1, 251)]
        snap = "\n".join(lines)

        def fake_post(focus, cands, key, timeout_s=None):
            probs = {}
            for c in cands:
                probs[c["key"]] = 0.85 if c["key"] == "e242" else 0.15 / max(1, len(cands) - 1)
            return probs, ""

        with patch.object(ui_focus, "post_decision", side_effect=fake_post):
            res = ui_focus.focus_text(snap, "Item 242", key="dummy")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["total"], 250)
        self.assertIn('★ [0.85] [@e242] Button "Item 242"', res["text"])


if __name__ == "__main__":
    unittest.main()
