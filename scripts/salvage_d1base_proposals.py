#!/usr/bin/env python3
"""Salvage the d1base arm's byte-level-BPE-garbled proposals (loss-free de-tokenization).

The d1base checkpoint was served under a vLLM detokenizer that emitted raw byte-level-BPE
glyphs (Ġ = space, Ċ = newline, ...) instead of applying the ByteLevel byte-decoder. The stored
text is *exactly* the byte-level token representation, so inverting GPT-2's bytes_to_unicode
mapping (glyph -> byte -> UTF-8) recovers the original text with no loss.

Only fields that still contain the tell-tale Ġ/Ċ glyphs are transformed; already-clean records
and other arms are never touched. Backs up the original to <file>.garbled.bak before rewriting.

Usage:
  python scripts/salvage_d1base_proposals.py
  python scripts/salvage_d1base_proposals.py --file <path> --dry-run
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
DEFAULT = HARNESS / "runs/researcher_cot/mls_lite_eval/proposals/proposals_d1base.jsonl"
FIELDS = ("proposal", "idea")


def byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2 bytes_to_unicode: {display-glyph: byte}."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


DEC = byte_decoder()


def is_garbled(s) -> bool:
    return isinstance(s, str) and ("Ġ" in s or "Ċ" in s)


def degarble(s: str) -> str:
    """Byte-level-BPE glyph stream -> real text. Returns input unchanged if not decodable."""
    if not is_garbled(s):
        return s
    try:
        return bytes(DEC[ch] for ch in s).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    p = Path(args.file)
    if not p.exists():
        sys.exit(f"not found: {p}")

    recs = [json.loads(l) for l in p.open() if l.strip()]
    changed = 0
    for r in recs:
        for f in FIELDS:
            if is_garbled(r.get(f)):
                fixed = degarble(r[f])
                if fixed != r[f]:
                    r[f] = fixed
                    changed += 1
    print(f"[salvage] {p.name}: {len(recs)} records, {changed}/{len(recs)*len(FIELDS)} fields de-tokenized")
    if changed and recs:
        print(f"[sample] {recs[0].get('task')}: {recs[0].get('proposal','')[:140]!r}")
    if args.dry_run:
        print("[dry-run] no files written"); return
    if not changed:
        print("[salvage] nothing to fix (already clean)"); return

    bak = p.with_suffix(p.suffix + ".garbled.bak")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"[salvage] backed up original -> {bak.name}")
    with p.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[salvage] rewrote {p}")


if __name__ == "__main__":
    main()
