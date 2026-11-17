#!/usr/bin/env python3
"""vision_nodes.py — Deterministic OCR + Icon box fusion (@v1..@vN) & cid snapshot guard.

Zero third-party dependencies (Python 3.9+ stdlib only).

Why this layer exists:
- Browser / native apps usually expose DOM or Accessibility trees (`@e1..@eN`).
- Custom-drawn canvases, games, mini-programs, or WebViews often return empty trees.
  Instead of feeding full-screen screenshots into a slow VLM on every step, run a fast
  local detector + OCR (e.g., OmniParser v2 + PP-OCRv5 ONNX), fuse overlapping OCR text
  into its enclosing icon/button box, and emit plain-text `@v1..@vN` lines identical in
  shape to `@e1..@eN` lines.
- Pair every snapshot with a monotonic `cid` (`c0001`, `c0002`, ...) so `click @v12 --cid c0001`
  fails closed if the screen changed while the model was thinking.
"""

from __future__ import annotations

VISION_MIN_PX = 8
ICON_IOU_DEDUP = 0.80
TEXT_ABSORB_RATIO = 0.75
MAX_CARD_RATIO = 0.25
ROW_BAND_PX = 40
MERGE_BAND_PX = 24


def _rect_area(n: dict) -> int:
    return max(0, n["x2"] - n["x1"]) * max(0, n["y2"] - n["y1"])


def _inter_area(a: dict, b: dict) -> int:
    w = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
    h = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
    return w * h if w > 0 and h > 0 else 0


def _iou(a: dict, b: dict) -> float:
    inter = _inter_area(a, b)
    if inter <= 0:
        return 0.0
    union = _rect_area(a) + _rect_area(b) - inter
    return inter / union if union > 0 else 0.0


def build_vision_nodes(
    regions: list[dict],
    screen_w: int,
    screen_h: int,
    *,
    merge: bool = True,
) -> list[dict]:
    """Normalize raw detector/OCR boxes into reading-order `@v1..@vN` nodes.

    Each input region dict expects:
        {
            "id": "r1",                      # optional raw detector id
            "kind": "icon" | "text",
            "text": "Submit",                # OCR text (for kind="text")
            "label": "button",               # detector class/caption (for kind="icon")
            "confidence": 0.94,
            "bounds": {"x": 100, "y": 200, "width": 160, "height": 56}
        }
    """
    w_lim = max(1, int(screen_w))
    h_lim = max(1, int(screen_h))
    screen_area = w_lim * h_lim
    raw_nodes: list[dict] = []

    for reg in regions or []:
        if not isinstance(reg, dict):
            continue
        b = reg.get("bounds")
        if not isinstance(b, dict):
            continue
        try:
            x1 = max(0, min(w_lim, int(b["x"])))
            y1 = max(0, min(h_lim, int(b["y"])))
            x2 = max(0, min(w_lim, int(b["x"]) + int(b["width"])))
            y2 = max(0, min(h_lim, int(b["y"]) + int(b["height"])))
        except (KeyError, TypeError, ValueError):
            continue
        if (x2 - x1) < VISION_MIN_PX or (y2 - y1) < VISION_MIN_PX:
            continue
        kind = str(reg.get("kind") or "region").strip().lower()
        conf = reg.get("confidence")
        raw_nodes.append({
            "ref": "",
            "id": str(reg.get("id") or ""),
            "kind": kind,
            "cls": f"Vision:{kind}",
            "text": " ".join(str(reg.get("text") or "").split()),
            "label": " ".join(str(reg.get("label") or "").split()),
            "confidence": float(conf) if isinstance(conf, (int, float)) else None,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "bounds": f"[{x1},{y1}][{x2},{y2}]",
            "center": [(x1 + x2) // 2, (y1 + y2) // 2],
            "size": [x2 - x1, y2 - y1],
        })

    if not merge:
        raw_nodes.sort(key=lambda n: (n["y1"] // ROW_BAND_PX, n["x1"]))
        for idx, n in enumerate(raw_nodes, 1):
            n["ref"] = f"@v{idx}"
        return raw_nodes

    # 1. Deduplicate overlapping icon boxes (IoU >= 0.80), keeping higher confidence
    icons = sorted(
        (n for n in raw_nodes if n["kind"] == "icon"),
        key=lambda n: -(n["confidence"] if n["confidence"] is not None else -1.0),
    )
    kept_icons: list[dict] = []
    for ic in icons:
        if any(_iou(ic, prev) >= ICON_IOU_DEDUP for prev in kept_icons):
            continue
        kept_icons.append(ic)

    texts = [n for n in raw_nodes if n["kind"] != "icon"]

    # 2. Absorb OCR text boxes inside button-sized icon boxes (avoid huge card containers)
    absorbed_ids: set[int] = set()
    bucket: dict[int, list[dict]] = {id(ic): [] for ic in kept_icons}
    for tx in texts:
        t_area = _rect_area(tx)
        if t_area <= 0 or not tx["text"]:
            continue
        best_icon = None
        best_area = None
        for ic in kept_icons:
            i_area = _rect_area(ic)
            if i_area > 16 * t_area or i_area > MAX_CARD_RATIO * screen_area:
                continue
            if _inter_area(tx, ic) >= TEXT_ABSORB_RATIO * t_area:
                if best_area is None or i_area < best_area:
                    best_icon = ic
                    best_area = i_area
        if best_icon is not None:
            bucket[id(best_icon)].append(tx)
            absorbed_ids.add(id(tx))

    for ic in kept_icons:
        inner = bucket[id(ic)]
        if inner:
            inner.sort(key=lambda n: (n["y1"] // MERGE_BAND_PX, n["x1"]))
            ic["text"] = " ".join(n["text"] for n in inner if n["text"])

    merged = kept_icons + [tx for tx in texts if id(tx) not in absorbed_ids]
    merged.sort(key=lambda n: (n["y1"] // ROW_BAND_PX, n["x1"]))
    for idx, node in enumerate(merged, 1):
        node["ref"] = f"@v{idx}"
    return merged


class SnapshotTracker:
    """Minimal generation guard (`cid`) binding @eN / @vN refs to a specific screen state."""

    def __init__(self) -> None:
        self._seq = 0
        self._current_cid: str | None = None
        self._coords: dict[str, tuple[int, int]] = {}

    def issue(self, nodes: list[dict], header_meta: str = "") -> tuple[str, str]:
        """Issue a new `cid` and format nodes as plain text ready for `ui_focus.focus_text`."""
        self._seq += 1
        cid = f"c{self._seq:04d}"
        self._current_cid = cid
        self._coords = {}

        lines = [f"[cid={cid}]"]
        if header_meta:
            lines.append(header_meta)
        for n in nodes:
            ref = str(n["ref"])
            cx, cy = n["center"]
            self._coords[ref.lower()] = (int(cx), int(cy))
            parts = [f"[{ref}]", n.get("cls") or "Node"]
            if n.get("text"):
                parts.append(f'"{n["text"]}"')
            elif n.get("label"):
                parts.append(f'label="{n["label"]}"')
            parts.append(f"center=({cx},{cy})")
            parts.append(f"bounds={n.get('bounds')}")
            lines.append(" ".join(parts))
        return cid, "\n".join(lines)

    def resolve_click(self, ref: str, cid: str) -> tuple[int, int]:
        """Resolve (`ref`, `cid`) -> `(x, y)` or raise ValueError if snapshot is stale."""
        if cid != self._current_cid:
            raise ValueError(f"stale_cid: expected {self._current_cid}, got {cid} (re-perceive before clicking)")
        norm = ref.strip().lstrip("[").rstrip("]").lower()
        if not norm.startswith("@"):
            norm = "@" + norm
        if norm not in self._coords:
            raise KeyError(f"unknown_ref: {ref}")
        return self._coords[norm]
