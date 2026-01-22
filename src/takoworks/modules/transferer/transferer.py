#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASS text transfer (source -> timed target) with fallback merge for unmatched source lines.

Features requested for TakoWorks:
- Transfers Text from source to timed target WITHOUT changing target Start/End.
- Ignores events where:
    - Name/Actor == "CARTEL"   (case-insensitive)
    - Style starts with "Cart_" (case-insensitive)
- Normal phase: each target Dialogue picks its best-matching source Dialogue by time overlap.
- Fallback phase: for every source line that was NOT used in the normal phase:
    - Find the best target line by time.
    - Decide prepend vs append based on which is closer:
        abs(src_start - dst_start) vs abs(src_end - dst_end)
        (tie => append)
    - Merge text into that target line:
        - If actors end up being different in the merged line:
            - Actor field becomes "Actor1; Actor2; ..."
            - Text becomes dialogue format:
                "-Line1\\N-Line2" (order respects prepend/append)
        - If actors are the same (or effectively only one actor):
            - Text becomes simple concatenation with a space.

Optional:
- Writes a TSV report of which source lines needed fallback, and where they were merged.

Usage:
  python ass_transfer_takoworks.py --src bad.ass --dst timed.ass --out merged.ass --report fallback.tsv
"""

import argparse
import bisect
import re
import statistics
from typing import Dict, List, Optional, Tuple


TIME_RE = re.compile(r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})[.](?P<cs>\d{2})")


def time_to_ms(t: str) -> int:
    m = TIME_RE.match(t.strip())
    if not m:
        raise ValueError(f"Bad ASS time: {t!r}")
    h = int(m.group("h"))
    mi = int(m.group("m"))
    s = int(m.group("s"))
    cs = int(m.group("cs"))
    return ((h * 3600 + mi * 60 + s) * 1000) + cs * 10


def interval_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return (inter / union) if union > 0 else 0.0


def parse_ass(path: str):
    """
    Minimal ASS parser for [Events] lines.

    Returns:
      - all file lines (for reconstruction)
      - parsed event dicts with:
        prefix (Dialogue/Comment), parts (csv fields), indices for Start/End/Style/Name/Text, ms times.
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    in_events = False
    fmt: Optional[List[str]] = None
    events = []

    for i, line in enumerate(lines):
        line_ending = "\n" if line.endswith("\n") else ""
        line_body = line[:-1] if line_ending else line

        stripped = line_body.lstrip()
        leading = line_body[: len(line_body) - len(stripped)]

        if stripped.startswith("["):
            in_events = stripped.strip().lower() == "[events]"
            continue

        if in_events and stripped.startswith("Format:"):
            fmt = [x.strip() for x in stripped[len("Format:") :].strip().split(",")]
            continue

        if in_events and fmt and (stripped.startswith("Dialogue:") or stripped.startswith("Comment:")):
            prefix, rest = stripped.split(":", 1)
            rest = rest.lstrip()

            parts = rest.split(",", maxsplit=len(fmt) - 1)
            if len(parts) < len(fmt):
                parts += [""] * (len(fmt) - len(parts))

            def idx_of(field: str) -> Optional[int]:
                return fmt.index(field) if field in fmt else None

            sidx = idx_of("Start")
            eidx = idx_of("End")
            stidx = idx_of("Style")
            nidx = idx_of("Name")
            tidx = idx_of("Text")

            start = parts[sidx].strip() if sidx is not None else ""
            end = parts[eidx].strip() if eidx is not None else ""

            try:
                start_ms = time_to_ms(start) if start else None
                end_ms = time_to_ms(end) if end else None
            except Exception:
                start_ms = None
                end_ms = None

            events.append(
                {
                    "line_index": i,
                    "leading": leading,
                    "prefix": prefix,  # Dialogue / Comment
                    "fmt": fmt,
                    "parts": parts,
                    "line_ending": line_ending,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "style_idx": stidx,
                    "name_idx": nidx,
                    "text_idx": tidx,
                }
            )

    return lines, events


def get_field(ev: dict, idx: Optional[int]) -> str:
    if idx is None:
        return ""
    if idx < 0 or idx >= len(ev["parts"]):
        return ""
    return ev["parts"][idx] or ""


def is_ignored(ev: dict) -> bool:
    name = get_field(ev, ev.get("name_idx")).strip()
    style = get_field(ev, ev.get("style_idx")).strip()
    if name.lower() == "cartel":
        return True
    if style.lower().startswith("cart_"):
        return True
    return False


def estimate_offset_ms(src_events: List[dict], dst_events: List[dict], sample_n: int = 80) -> int:
    """
    Estimate small offset between tracks (ms) for better matching.
    NOTE: This NEVER changes output timings; it is used only for matching comparisons.
    """
    src = [e for e in src_events if e.get("start_ms") is not None]
    dst = [e for e in dst_events if e.get("start_ms") is not None]
    if not src or not dst:
        return 0

    src_sorted = sorted(src, key=lambda e: e["start_ms"])
    dst_sorted = sorted(dst, key=lambda e: e["start_ms"])
    src_starts = [e["start_ms"] for e in src_sorted]

    diffs = []
    for d in dst_sorted[:sample_n]:
        x = d["start_ms"]
        j = bisect.bisect_left(src_starts, x)
        cand = []
        for k in (j - 1, j, j + 1):
            if 0 <= k < len(src_starts):
                cand.append(src_starts[k] - x)
        if cand:
            diffs.append(min(cand, key=abs))

    return int(statistics.median(diffs)) if diffs else 0


def match_best_src_for_each_dst(
    src_events: List[dict],
    dst_events: List[dict],
    *,
    offset_ms: int = 0,
    max_start_diff: int = 800,
    max_end_diff: int = 1000,
    min_iou: float = 0.10,
    window: int = 40,
) -> List[Optional[int]]:
    """
    For each dst line, pick the best src line (or None) based on time overlap + diff penalties.
    """
    src_sorted = sorted(list(enumerate(src_events)), key=lambda t: (t[1]["start_ms"] or 0))
    src_starts = [(e["start_ms"] or 0) + offset_ms for _, e in src_sorted]

    mapping: List[Optional[int]] = [None] * len(dst_events)

    for di, d in enumerate(dst_events):
        ds, de = d.get("start_ms"), d.get("end_ms")
        if ds is None or de is None:
            continue

        j = bisect.bisect_left(src_starts, ds)
        k0 = max(0, j - window)
        k1 = min(len(src_sorted), j + window)

        best_si = None
        best_score = -10**18

        for k in range(k0, k1):
            si, s = src_sorted[k]
            ss, se = s.get("start_ms"), s.get("end_ms")
            if ss is None or se is None:
                continue

            ss += offset_ms
            se += offset_ms

            iou = interval_iou(ss, se, ds, de)
            sd = abs(ss - ds)
            ed = abs(se - de)

            if iou < min_iou and (sd > max_start_diff or ed > max_end_diff):
                continue

            score = iou * 1000 - (sd + ed)
            if score > best_score:
                best_score = score
                best_si = si

        mapping[di] = best_si

    return mapping


def best_dst_for_src(
    src_ev: dict,
    dst_events: List[dict],
    *,
    offset_ms: int = 0,
    window: int = 60,
) -> Optional[int]:
    """
    Find the best dst index for a given src line (used for fallback).
    """
    ss, se = src_ev.get("start_ms"), src_ev.get("end_ms")
    if ss is None or se is None:
        return None
    ss += offset_ms
    se += offset_ms

    dst_sorted = sorted(list(enumerate(dst_events)), key=lambda t: (t[1]["start_ms"] or 0))
    dst_starts = [(e["start_ms"] or 0) for _, e in dst_sorted]

    j = bisect.bisect_left(dst_starts, ss)
    k0 = max(0, j - window)
    k1 = min(len(dst_sorted), j + window)

    best_di = None
    best_score = -10**18

    for k in range(k0, k1):
        di, d = dst_sorted[k]
        ds, de = d.get("start_ms"), d.get("end_ms")
        if ds is None or de is None:
            continue
        iou = interval_iou(ss, se, ds, de)
        sd = abs(ss - ds)
        ed = abs(se - de)
        score = iou * 1000 - (sd + ed)
        if score > best_score:
            best_score = score
            best_di = di

    return best_di


def normalize_actor_list(actor_str: str) -> List[str]:
    """
    Turns "A; B;C" into ["A","B","C"] unique-preserving order.
    """
    items = [x.strip() for x in actor_str.split(";") if x.strip()]
    out: List[str] = []
    for it in items:
        if it not in out:
            out.append(it)
    return out


def format_dialogue_lines(texts: List[str]) -> str:
    # "-Line1\N-Line2"
    cleaned = [t.strip() for t in texts if t and t.strip()]
    return r"\N".join(["-" + t for t in cleaned])


def build_merged_ass(
    src_path: str,
    dst_path: str,
    out_path: str,
    *,
    include_comments: bool = False,
    no_auto_offset: bool = False,
    report_tsv: Optional[str] = None,
    max_start_diff_ms: int = 800,
    max_end_diff_ms: int = 1000,
    min_iou: float = 0.10,
    window: int = 40,
) -> Tuple[int, int, int, int]:
    """
    Returns stats:
      (src_useful, dst_useful, direct_used_src_unique, fallback_src_count)
    """
    src_lines, src_events_all = parse_ass(src_path)
    dst_lines, dst_events_all = parse_ass(dst_path)

    def pick(events: List[dict]) -> List[dict]:
        if include_comments:
            out = [e for e in events if e["prefix"].lower() in ("dialogue", "comment")]
        else:
            out = [e for e in events if e["prefix"].lower() == "dialogue"]
        return [e for e in out if not is_ignored(e)]

    src_events = pick(src_events_all)
    dst_events = pick(dst_events_all)

    offset = 0 if no_auto_offset else estimate_offset_ms(src_events, dst_events)

    mapping = match_best_src_for_each_dst(
        src_events,
        dst_events,
        offset_ms=offset,
        max_start_diff=max_start_diff_ms,
        max_end_diff=max_end_diff_ms,
        min_iou=min_iou,
        window=window,
    )

    # Track which src lines were used directly (unique)
    direct_used_src = {si for si in mapping if si is not None}

    # We’ll accumulate "segments" per dst line:
    #   each segment is (actor, text) and order matters.
    # The FIRST segment uses the dst actor if present (keeps your translated actor names),
    # otherwise falls back to the src actor.
    segments: Dict[int, List[Tuple[str, str]]] = {di: [] for di in range(len(dst_events))}

    # DIRECT TRANSFER
    for di, si in enumerate(mapping):
        if si is None:
            continue

        s = src_events[si]
        d = dst_events[di]

        src_text = get_field(s, s.get("text_idx")).strip()
        src_actor = get_field(s, s.get("name_idx")).strip()

        # Keep target actor if already filled; else copy from src
        dst_actor_existing = get_field(d, d.get("name_idx")).strip()
        first_actor = dst_actor_existing if dst_actor_existing else src_actor

        # Store the segment actor for later merge-format decisions
        segments[di].append((first_actor, src_text))

    # FALLBACK MERGE for "unmatched source lines"
    fallback_rows = []  # for report
    fallback_src_count = 0

    for si, s in enumerate(src_events):
        if si in direct_used_src:
            continue  # this one already transferred as primary match

        src_text = get_field(s, s.get("text_idx")).strip()
        if not src_text:
            continue

        src_actor = get_field(s, s.get("name_idx")).strip()

        best_di = best_dst_for_src(s, dst_events, offset_ms=offset)
        if best_di is None:
            continue

        d = dst_events[best_di]

        # Decide prepend/append by comparing closeness to start vs end
        ss = (s.get("start_ms") or 0) + offset
        se = (s.get("end_ms") or 0) + offset
        ds = d.get("start_ms") or 0
        de = d.get("end_ms") or 0

        start_diff = abs(ss - ds)
        end_diff = abs(se - de)
        prepend = start_diff < end_diff  # tie => append

        if prepend:
            segments[best_di].insert(0, (src_actor, src_text))
            where = "prepend"
        else:
            segments[best_di].append((src_actor, src_text))
            where = "append"

        fallback_src_count += 1
        fallback_rows.append(
            {
                "src_index": si,
                "src_start_ms": s.get("start_ms"),
                "src_end_ms": s.get("end_ms"),
                "src_actor": src_actor,
                "src_text": src_text,
                "dst_index": best_di,
                "dst_start_ms": d.get("start_ms"),
                "dst_end_ms": d.get("end_ms"),
                "where": where,
                "start_diff_ms": start_diff,
                "end_diff_ms": end_diff,
            }
        )

    # FINALIZE: write merged actor/text into dst_events (timings untouched)
    for di, d in enumerate(dst_events):
        if not segments[di]:
            continue

        # Build ordered unique actor list
        actor_order: List[str] = []
        for actor, _ in segments[di]:
            for a in normalize_actor_list(actor.strip()):
                if a and a not in actor_order:
                    actor_order.append(a)

        multi_actor = len(actor_order) > 1

        if multi_actor:
            # dialogue format
            merged_text = format_dialogue_lines([t for _, t in segments[di]])
            merged_actor = "; ".join(actor_order)
        else:
            merged_text = " ".join([t.strip() for _, t in segments[di] if t and t.strip()])
            merged_actor = actor_order[0] if actor_order else ""

        if d.get("name_idx") is not None and merged_actor:
            d["parts"][d["name_idx"]] = merged_actor
        if d.get("text_idx") is not None:
            d["parts"][d["text_idx"]] = merged_text

    # Rebuild file lines, WITHOUT modifying times
    out_lines = list(dst_lines)
    for d in dst_events:
        i = d["line_index"]
        out_lines[i] = f"{d['leading']}{d['prefix']}: " + ",".join(d["parts"]) + d["line_ending"]

    with open(out_path, "w", encoding="utf-8-sig", errors="replace") as f:
        f.writelines(out_lines)

    # Optional TSV report
    if report_tsv:
        def ms_to_ass(ms: Optional[int]) -> str:
            if ms is None:
                return ""
            cs = (ms % 1000) // 10
            total_s = ms // 1000
            s = total_s % 60
            total_m = total_s // 60
            m = total_m % 60
            h = total_m // 60
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        with open(report_tsv, "w", encoding="utf-8", errors="replace") as f:
            f.write(
                "\t".join(
                    [
                        "src_index",
                        "src_start",
                        "src_end",
                        "src_actor",
                        "src_text",
                        "dst_index",
                        "dst_start",
                        "dst_end",
                        "where",
                        "start_diff_ms",
                        "end_diff_ms",
                    ]
                )
                + "\n"
            )
            for r in fallback_rows:
                f.write(
                    "\t".join(
                        [
                            str(r["src_index"]),
                            ms_to_ass(r["src_start_ms"]),
                            ms_to_ass(r["src_end_ms"]),
                            r["src_actor"].replace("\t", " "),
                            r["src_text"].replace("\t", " "),
                            str(r["dst_index"]),
                            ms_to_ass(r["dst_start_ms"]),
                            ms_to_ass(r["dst_end_ms"]),
                            r["where"],
                            str(r["start_diff_ms"]),
                            str(r["end_diff_ms"]),
                        ]
                    )
                    + "\n"
                )

    return (len(src_events), len(dst_events), len(direct_used_src), fallback_src_count)


def main():
    ap = argparse.ArgumentParser(description="ASS transfer for TakoWorks (with fallback prepend/append merges).")
    ap.add_argument("--src", required=True, help="Source ASS (with text).")
    ap.add_argument("--dst", required=True, help="Timed target ASS.")
    ap.add_argument("--out", required=True, help="Output merged ASS.")
    ap.add_argument("--report", default=None, help="Optional TSV report for fallback-merged source lines.")
    ap.add_argument("--include-comments", action="store_true", help="Also process Comment: lines (default: only Dialogue:).")
    ap.add_argument("--no-auto-offset", action="store_true", help="Disable offset estimation (matching only).")
    ap.add_argument("--max-start-diff-ms", type=int, default=800)
    ap.add_argument("--max-end-diff-ms", type=int, default=1000)
    ap.add_argument("--min-iou", type=float, default=0.10)
    ap.add_argument("--window", type=int, default=40)
    args = ap.parse_args()

    src_n, dst_n, direct_unique, fallback_n = build_merged_ass(
        args.src,
        args.dst,
        args.out,
        include_comments=args.include_comments,
        no_auto_offset=args.no_auto_offset,
        report_tsv=args.report,
        max_start_diff_ms=args.max_start_diff_ms,
        max_end_diff_ms=args.max_end_diff_ms,
        min_iou=args.min_iou,
        window=args.window,
    )

    print(f"[ok] wrote: {args.out}")
    print(f"[stats] src_useful={src_n} dst_useful={dst_n} direct_used_unique={direct_unique} fallback_merged={fallback_n}")
    print("[note] Target timings (Start/End) are NOT modified.")
    print('[note] Ignored: Name=="CARTEL" or Style startswith "Cart_".')


if __name__ == "__main__":
    main()
