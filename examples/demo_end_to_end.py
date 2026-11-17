#!/usr/bin/env python3
"""End-to-end demo: OCR/Icon fusion -> Semantic Focus -> Deterministic cid click.

Runs with live Gemini Decisions API if GEMINI_API_KEY is set; otherwise runs an
offline mock decision pass so the full pipeline can be inspected without network.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ui_focus  # noqa: E402
from vision_nodes import SnapshotTracker, build_vision_nodes  # noqa: E402


RAW_REGIONS = [
    {"id": "r1", "kind": "text", "text": "9:41", "confidence": 0.99, "bounds": {"x": 60, "y": 32, "width": 80, "height": 36}},
    {"id": "r2", "kind": "icon", "label": "back_arrow", "confidence": 0.95, "bounds": {"x": 48, "y": 140, "width": 64, "height": 64}},
    {"id": "r3", "kind": "text", "text": "Edit Profile", "confidence": 0.98, "bounds": {"x": 520, "y": 146, "width": 320, "height": 52}},
    {"id": "r4", "kind": "text", "text": "Display Name", "confidence": 0.97, "bounds": {"x": 80, "y": 420, "width": 260, "height": 44}},
    {"id": "r5", "kind": "icon", "label": "input_box", "confidence": 0.91, "bounds": {"x": 80, "y": 480, "width": 1280, "height": 96}},
    {"id": "r6", "kind": "text", "text": "Alice", "confidence": 0.98, "bounds": {"x": 110, "y": 506, "width": 120, "height": 44}},
    {"id": "r7", "kind": "icon", "label": "row_item", "confidence": 0.93, "bounds": {"x": 80, "y": 660, "width": 1280, "height": 110}},
    {"id": "r8", "kind": "text", "text": "Banner Color", "confidence": 0.99, "bounds": {"x": 112, "y": 692, "width": 280, "height": 46}},
    {"id": "r9", "kind": "text", "text": "#2B2D31", "confidence": 0.96, "bounds": {"x": 1120, "y": 692, "width": 180, "height": 44}},
    {"id": "r10", "kind": "text", "text": "Avatar Decoration", "confidence": 0.97, "bounds": {"x": 80, "y": 840, "width": 340, "height": 46}},
    {"id": "r11", "kind": "text", "text": "Privacy & Safety", "confidence": 0.97, "bounds": {"x": 80, "y": 980, "width": 320, "height": 46}},
    {"id": "r12", "kind": "icon", "label": "button", "confidence": 0.96, "bounds": {"x": 960, "y": 2740, "width": 400, "height": 120}},
    {"id": "r13", "kind": "text", "text": "Save Changes", "confidence": 0.99, "bounds": {"x": 1030, "y": 2776, "width": 260, "height": 48}},
]


def main() -> None:
    # Step 1: Fuse raw OCR + icon detector boxes into @v1..@vN lines
    nodes = build_vision_nodes(RAW_REGIONS, screen_w=1440, screen_h=3120)
    tracker = SnapshotTracker()
    cid, snapshot_text = tracker.issue(nodes, header_meta="app=example.settings screen=1440x3120")

    print("=== 1. Fused Snapshot (@vN + cid) ===")
    print(snapshot_text)

    # Step 2: Semantic focus filter
    intent = "change the profile banner color"
    live_key = ui_focus.load_key()
    if live_key:
        focused = ui_focus.focus_text(snapshot_text, intent)
    else:
        # Offline demonstration distribution when GEMINI_API_KEY is not set
        def fake_post(focus, cands, key, timeout_s=None):
            probs = {c["key"]: (0.91 if "Banner Color" in c["label"] else 0.09 / max(1, len(cands) - 1)) for c in cands}
            return probs, ""

        with patch.object(ui_focus, "post_decision", side_effect=fake_post):
            focused = ui_focus.focus_text(snapshot_text, intent, key="demo-offline")

    print(f"\n=== 2. Focused Output (intent={intent!r}) ===")
    print(focused["text"])

    # Step 3: Main model picks the top-starred @vN and executes a cid-guarded click
    target_ref = next(n["ref"] for n in nodes if "Banner Color" in n["text"])
    x, y = tracker.resolve_click(target_ref, cid=cid)
    print(f"\n=== 3. Deterministic Click Resolution ===")
    print(f"click {target_ref} --cid {cid}  ->  tap({x}, {y})")


if __name__ == "__main__":
    main()
