#!/usr/bin/env python3
"""Push the rendered CoT RedDoc pages to the Agentic Training space via the `hi` CLI.

Order: create the overview page (space root) -> create each subpage under it (raw_prompts + 69 CoTs)
-> append a linked index to the overview. Idempotent: a local created.json records every page's
shortcutId/url, so a re-run resumes without duplicating. Page bodies are read from files and piped to
`hi docs:create` (contents never enter the model context).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = "4709bcf060916d61278c76e8a1b668e0"  # Agentic Training


def hi_json(args: list[str], stdin: str | None = None) -> dict[str, Any]:
    p = subprocess.run(["hi", *args], input=stdin, capture_output=True, text=True, timeout=180)
    out = p.stdout.strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        s = out.find("{")
        if s >= 0:
            return json.loads(out[s:])
        raise RuntimeError(f"hi {' '.join(args[:2])} -> {out[:200]} / err {p.stderr[:200]}")


def op_code() -> str:
    r = hi_json(["utils:generate-operate-code"])
    return r["operateCode"]


def create_page(title: str, body: str, parent_id: str | None, space_id: str | None,
                retries: int = 3) -> dict[str, str]:
    args = ["docs:create", "--title", title, "--content", "-", "--operate-code", op_code()]
    if parent_id:
        args += ["--parent-id", parent_id]
    else:
        args += ["--space-id", space_id]
    last = ""
    for attempt in range(retries):
        try:
            r = hi_json(args, stdin=body)
            if r.get("shortcutId"):
                return {"shortcutId": r["shortcutId"], "url": r.get("url", "")}
            last = str(r)
        except Exception as e:
            last = str(e)
        time.sleep(3 * (attempt + 1))
        args[args.index("--operate-code") + 1] = op_code()  # fresh code per retry
    raise RuntimeError(f"create failed for {title!r}: {last}")


def doc_hash(shortcut_id: str) -> str:
    r = hi_json(["docs:get", "--shortcut-id", shortcut_id, "--mode", "common"])
    return r["hash"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages-dir", type=Path, default=ROOT / "runs" / "researcher_cot" / "redoc_pages")
    ap.add_argument("--space-id", default=SPACE_ID)
    ap.add_argument("--limit", type=int, default=None, help="cap number of subpages (smoke)")
    args = ap.parse_args()

    manifest = json.loads((args.pages_dir / "pages_manifest.json").read_text())
    created_path = args.pages_dir / "created.json"
    created: dict[str, dict[str, str]] = json.loads(created_path.read_text()) if created_path.exists() else {}

    def save():
        created_path.write_text(json.dumps(created, ensure_ascii=False, indent=2))

    overview = next(m for m in manifest if m["kind"] == "overview")
    subpages = [m for m in manifest if m["kind"] != "overview"]
    if args.limit:
        subpages = subpages[: args.limit]

    # 1) overview
    if overview["file"] not in created:
        body = (args.pages_dir / overview["file"]).read_text()
        created[overview["file"]] = create_page(overview["title"], body, None, args.space_id)
        save()
        print(f"[push] overview -> {created[overview['file']]['url']}", flush=True)
    ov_id = created[overview["file"]]["shortcutId"]

    # 2) subpages under overview
    for i, m in enumerate(subpages, 1):
        if m["file"] in created:
            continue
        body = (args.pages_dir / m["file"]).read_text()
        created[m["file"]] = create_page(m["title"], body, ov_id, None)
        save()
        print(f"[push] {i}/{len(subpages)} {m['title'][:40]} -> {created[m['file']]['url']}", flush=True)

    # 3) append linked index to overview (once)
    if not created.get("_index_appended"):
        lines = ["", "## Pages (links)", ""]
        for m in manifest:
            if m["kind"] == "overview":
                continue
            c = created.get(m["file"])
            if c:
                lines.append(f"- [{m['title']}]({c['url']})")
        idx_md = "\n".join(lines)
        h = doc_hash(ov_id)
        hi_json(["docs:edit", "--shortcut-id", ov_id, "--hash", h, "--append", "-"], stdin=idx_md)
        created["_index_appended"] = {"done": "1"}
        save()
        print("[push] appended index to overview", flush=True)

    print(json.dumps({"status": "ok", "overview_url": created[overview["file"]]["url"],
                      "subpages": len([k for k in created if k.endswith('.md') and k != overview['file']])},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
