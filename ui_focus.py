#!/usr/bin/env python3
"""ui_focus.py — Read-only semantic focus filter for UI snapshots (@eN / @vN).

Zero third-party dependencies (Python 3.9+ stdlib only).
Reads a plain-text UI snapshot (browser ARIA tree, mobile accessibility dump,
or fused OCR/OmniParser @vN lines), queries Gemini Decisions API (`gemini-decision-1`)
for a normalized probability distribution over candidate nodes, and keeps the
Top-P (default 0.90, clamped to [5, 15]) lines annotated with scores.

Design invariants:
1. Read-only filter: never mutates ref tokens (@e12 / @v28) and never triggers actions.
2. Fail-Open: missing key, timeout, HTTP error, or malformed JSON returns the
   unmodified input snapshot with a single diagnostic header line.
3. Zero key leakage: API keys live only in memory/request headers and never appear
   in output text, logs, or error codes.

CLI usage (Unix pipe or file):
    browser-snapshot | python3 ui_focus.py -f "open account billing settings"
    python3 ui_focus.py snapshot.txt -f "click the checkout button"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"
MODEL = "gemini-decision-1"
QUESTION_KEY = "target"

DEFAULT_KEY_FILE = "~/.config/ui-focus/gemini-api.key"
KEY_ENV_ORDER = ("UI_FOCUS_GEMINI_KEY", "GEMINI_API_KEY")
KEY_FILE_ENV = "UI_FOCUS_KEY_FILE"

TOP_P = 0.90
MIN_K = 5
MAX_K = 15
BATCH_LIMIT = 200
API_MAX_OPTIONS = 255
FINALISTS_PER_BATCH = 30
MAX_WORKERS = 4
TIMEOUT_S = 4.5

STAR_AT = 0.20
FIRST_STAR_AT = 0.15
LOW_MATCH_AT = 0.15
CRITERIA_MAX_CHARS = 240
SKIP_NOTE = "[ui_focus: skipped (<2 nodes)]"

REF_RE = re.compile(r"@([eEvV]\d{1,6})")
SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9_]+$")
_CTX_RE = re.compile(r"\[ctx:[^\n]*")
_NOISE_RE = re.compile(r"\b(?:bounds|center|size|merged|conf)=\S+")
_WS_RE = re.compile(r"\s+")

_KEY_LOCK = threading.Lock()
_KEY_CACHE: dict[str, object] = {"loaded": False, "value": None}


def reset_key_cache() -> None:
    with _KEY_LOCK:
        _KEY_CACHE["loaded"] = False
        _KEY_CACHE["value"] = None


def _read_key_file(path: str) -> str | None:
    try:
        text = Path(os.path.expanduser(path)).read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return None


def load_key(env: dict[str, str] | None = None, refresh: bool = False) -> str | None:
    mapping = os.environ if env is None else env
    if not refresh:
        with _KEY_LOCK:
            if _KEY_CACHE["loaded"]:
                return _KEY_CACHE["value"]  # type: ignore[return-value]
    value = None
    for name in KEY_ENV_ORDER:
        raw = str(mapping.get(name) or "").strip()
        if raw:
            value = raw
            break
    if value is None:
        explicit = str(mapping.get(KEY_FILE_ENV) or "").strip()
        for path in ([explicit] if explicit else [DEFAULT_KEY_FILE]):
            value = _read_key_file(path)
            if value:
                break
    with _KEY_LOCK:
        _KEY_CACHE["loaded"] = True
        _KEY_CACHE["value"] = value
    return value


def safe_option_key(ref: str) -> str | None:
    """Convert '@e12' / '[@v28]' -> 'e12' / 'v28' (Decisions API requires ^[A-Za-z0-9_]+$)."""
    raw = str(ref or "").strip()
    if not raw:
        return None
    match = REF_RE.search(raw)
    if match:
        token = match.group(1)
    elif re.match(r"^[eEvV]\d{1,6}$", raw):
        token = raw
    else:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", token)
    if not cleaned or not SAFE_KEY_RE.match(cleaned):
        return None
    return cleaned[:32]


def describe_line(line: str) -> str:
    """Strip coordinates and structural noise so criteria hold concise semantics."""
    body = REF_RE.sub(" ", str(line or ""))
    body = _CTX_RE.sub(" ", body)
    body = _NOISE_RE.sub(" ", body)
    body = body.replace("[", " ").replace("]", " ")
    body = _WS_RE.sub(" ", body).strip().strip("-*•|> \t").strip()
    return body[:CRITERIA_MAX_CHARS] if body else "(unlabeled node)"


def extract_candidates(text: str) -> tuple[list[str], list[dict], list[str]]:
    """Split snapshot text into (head_lines, candidate_items, tail_lines)."""
    lines = str(text or "").split("\n")
    head: list[str] = []
    candidates: list[dict] = []
    tail: list[str] = []
    seen: set[str] = set()
    for index, line in enumerate(lines):
        match = REF_RE.search(line)
        if match:
            key = safe_option_key(match.group(0))
            if key:
                if key not in seen:
                    seen.add(key)
                    candidates.append({
                        "index": index,
                        "line": line,
                        "key": key,
                        "ref": "@" + match.group(1),
                        "label": describe_line(line),
                    })
                continue
        if candidates:
            tail.append(line)
        else:
            head.append(line)
    return head, candidates, tail


def _urlopen(request: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(request, timeout=timeout)


def _build_body(focus: str, candidates: list[dict]) -> dict:
    criteria = {}
    lines = []
    for item in candidates:
        key = item.get("key")
        if key and SAFE_KEY_RE.match(key):
            label = item.get("label") or "(unlabeled node)"
            criteria[key] = label
            lines.append(f"{key}: {label}")
    return {
        "model": MODEL,
        "input": "\n".join(lines),
        "questions": {
            QUESTION_KEY: {
                "type": "choice",
                "instructions": f"Target user intent: {focus}. Pick the UI node that best matches.",
                "criteria": criteria,
            }
        },
    }


def _http_error_code(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_{exc.code}"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "timeout"
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason or exc).lower():
            return "timeout"
        return "network"
    if isinstance(exc, OSError):
        return "network"
    return "transport_error"


def _parse_probabilities(payload: object, keys: list[str]) -> tuple[dict[str, float], str]:
    known = set(keys)
    if not isinstance(payload, dict):
        return {}, "bad_json"
    answers = payload.get("answers")
    if not isinstance(answers, dict) or not answers:
        return {}, "bad_json"
    answer = answers.get(QUESTION_KEY)
    if not isinstance(answer, dict):
        only = list(answers.values())
        answer = only[0] if len(only) == 1 and isinstance(only[0], dict) else None
    if not isinstance(answer, dict):
        return {}, "bad_json"

    raw = answer.get("probabilities")
    probs: dict[str, float] = {}
    if isinstance(raw, dict) and raw:
        for key, value in raw.items():
            if not isinstance(key, str):
                continue
            norm = safe_option_key(key) or key
            if norm in known and isinstance(value, (int, float)) and not isinstance(value, bool):
                probs[norm] = max(0.0, float(value))
    if not probs:
        choice = answer.get("choice")
        conf = answer.get("confidence")
        if isinstance(choice, str) and choice in known:
            top = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else 1.0
            top = max(0.0, min(1.0, top))
            others = [k for k in known if k != choice]
            share = (max(0.0, 1.0 - top) / len(others)) if others else 0.0
            probs = {k: (top if k == choice else share) for k in known}
    total = sum(probs.values())
    if total <= 0.0:
        return {}, "empty_result"
    if abs(total - 1.0) > 1e-6:
        probs = {k: v / total for k, v in probs.items()}
    return probs, ""


def post_decision(focus: str, candidates: list[dict], key: str | None, timeout_s: float | None = None) -> tuple[dict[str, float], str]:
    if not candidates:
        return {}, "no_candidates"
    if len(candidates) < 2:
        return {}, "too_few"
    if len(candidates) > API_MAX_OPTIONS:
        return {}, "too_many_options"
    api_key = key if key else load_key()
    if not api_key:
        return {}, "no_key"
    timeout = TIMEOUT_S if timeout_s is None else float(timeout_s)
    data = json.dumps(_build_body(focus, candidates), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=data,
        method="POST",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
    )
    try:
        with _urlopen(req, timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            raw = resp.read()
    except Exception as exc:
        return {}, _http_error_code(exc)
    if int(status) != 200:
        return {}, f"http_{int(status)}"
    try:
        payload = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return {}, "bad_json"
    return _parse_probabilities(payload, [c["key"] for c in candidates])


def rank_candidates(
    candidates: list[dict],
    focus: str,
    *,
    key: str | None = None,
    timeout_s: float | None = None,
    batch_limit: int = BATCH_LIMIT,
    finalists_per_batch: int = FINALISTS_PER_BATCH,
    max_workers: int = MAX_WORKERS,
) -> dict:
    """Score candidates; shards >200 nodes concurrently and runs a unified finalist pass."""
    items = [c for c in (candidates or []) if isinstance(c, dict) and c.get("key")]
    if len(items) < 2:
        return {"scores": {}, "error": "too_few" if items else "no_candidates", "batches": 0, "final": False, "ms": 0}
    api_key = key if key else load_key()
    if not api_key:
        return {"scores": {}, "error": "no_key", "batches": 0, "final": False, "ms": 0}

    limit = max(2, int(batch_limit))
    batches = [items[i:i + limit] for i in range(0, len(items), limit)]
    started = time.perf_counter()

    if len(batches) == 1:
        probs, err = post_decision(focus, batches[0], api_key, timeout_s)
        ms = int(round((time.perf_counter() - started) * 1000))
        return {"scores": probs, "error": err, "batches": 1, "final": False, "ms": ms}

    def run_batch(batch: list[dict]):
        probs, err = post_decision(focus, batch, api_key, timeout_s)
        return batch, probs, err

    finalists: dict[str, tuple[dict, float]] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(max(1, max_workers), len(batches))) as pool:
        for batch, probs, err in pool.map(run_batch, batches):
            if err:
                failures.append(err)
                continue
            ordered = sorted(
                (c for c in batch if c["key"] in probs),
                key=lambda c: (-probs[c["key"]], c.get("index", 0)),
            )
            for c in ordered[:max(1, finalists_per_batch)]:
                p = float(probs[c["key"]])
                prev = finalists.get(c["key"])
                if prev is None or p > prev[1]:
                    finalists[c["key"]] = (c, p)

    if len(finalists) < 2:
        ms = int(round((time.perf_counter() - started) * 1000))
        return {"scores": {}, "error": failures[0] if failures else "empty_result", "batches": len(batches), "final": False, "ms": ms}

    picked = [item for item, _ in sorted(finalists.values(), key=lambda x: (-x[1], x[0].get("index", 0)))[:limit]]
    probs, err = post_decision(focus, picked, api_key, timeout_s)
    ms = int(round((time.perf_counter() - started) * 1000))
    return {"scores": probs if not err else {}, "error": err, "batches": len(batches), "final": True, "ms": ms}


def select_top_k(
    scores: dict[str, float],
    total: int,
    *,
    top_p: float = TOP_P,
    min_k: int = MIN_K,
    max_k: int = MAX_K,
    order_hint: dict[str, int] | None = None,
) -> tuple[list[str], float]:
    hint = order_hint or {}
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], hint.get(kv[0], 0), kv[0]))
    if not ordered:
        return [], 0.0
    lower = max(1, min(int(min_k), total or len(ordered)))
    upper = max(lower, min(int(max_k), total or len(ordered)))

    count = 0
    acc = 0.0
    for _, prob in ordered:
        count += 1
        acc += prob
        if acc >= float(top_p) - 1e-9:
            break
    boundary = ordered[count - 1][1]
    while count < len(ordered) and abs(ordered[count][1] - boundary) <= 1e-9:
        count += 1
    count = max(lower, min(upper, count))
    return [k for k, _ in ordered[:count]], ordered[0][1]


def focus_text(
    text: str,
    focus: str,
    *,
    top_p: float = TOP_P,
    min_k: int = MIN_K,
    max_k: int = MAX_K,
    timeout_s: float | None = None,
    key: str | None = None,
) -> dict:
    """Filter a multi-line UI snapshot down to Top-P intent-matched nodes."""
    base = str(text or "")
    all_lines = base.split("\n")
    intent = str(focus or "").strip()
    if not intent:
        return {"text": base, "status": "off", "code": "", "kept": 0, "total": 0, "top1": 0.0, "ms": 0}

    head, candidates, tail = extract_candidates(base)
    total = len(candidates)
    rest = all_lines[len(head):]
    if total < 2:
        return {
            "text": "\n".join(head + [SKIP_NOTE] + rest),
            "status": "skipped",
            "code": "lt2",
            "kept": total,
            "total": total,
            "top1": 0.0,
            "ms": 0,
        }

    ranked = rank_candidates(candidates, intent, key=key, timeout_s=timeout_s)
    scores = ranked["scores"]
    err = ranked["error"]
    ms = ranked["ms"]
    if err or not scores:
        note = f"[ui_focus_fallback: {err or 'empty_result'}, returning raw {total} nodes]"
        return {
            "text": "\n".join(head + [note] + rest),
            "status": "fallback",
            "code": err or "empty_result",
            "kept": total,
            "total": total,
            "top1": 0.0,
            "ms": ms,
        }

    by_key = {c["key"]: c for c in candidates}
    chosen, top1 = select_top_k(
        scores,
        total,
        top_p=top_p,
        min_k=min_k,
        max_k=max_k,
        order_hint={c["key"]: c["index"] for c in candidates},
    )
    body = []
    for rank, k in enumerate(chosen, 1):
        item = by_key[k]
        prob = float(scores[k])
        raw = item["line"]
        indent = raw[:len(raw) - len(raw.lstrip())]
        star = "★ " if (prob >= STAR_AT or (rank == 1 and prob >= FIRST_STAR_AT)) else "  "
        body.append(f"{indent}{star}[{prob:.2f}] {raw.strip()}")

    low = " LOW_MATCH" if top1 < LOW_MATCH_AT else ""
    summary = f'[ui_focus="{intent}" | kept {len(body)}/{total} | top1={top1:.2f}{low} | {ms}ms]'
    kept_head = [ln for ln in head if not (ln and ln[0].isspace())]
    kept_tail = [ln for ln in tail if ln and not ln[0].isspace()]
    rebuilt = kept_head + [summary] + body + kept_tail
    if base.endswith("\n") and rebuilt and rebuilt[-1] != "":
        rebuilt.append("")
    return {
        "text": "\n".join(rebuilt),
        "status": "ok",
        "code": "",
        "kept": len(body),
        "total": total,
        "top1": top1,
        "ms": ms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Semantic focus filter for @eN / @vN UI snapshots.")
    parser.add_argument("file", nargs="?", help="Snapshot text file (defaults to stdin)")
    parser.add_argument("-f", "--focus", required=True, help="Current target intent")
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--min-k", type=int, default=MIN_K)
    parser.add_argument("--max-k", type=int, default=MAX_K)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = parser.parse_args(argv)

    raw = Path(args.file).read_text(encoding="utf-8", errors="replace") if args.file else sys.stdin.read()
    res = focus_text(raw, args.focus, top_p=args.top_p, min_k=args.min_k, max_k=args.max_k, timeout_s=args.timeout)
    sys.stdout.write(res["text"])
    if not res["text"].endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
