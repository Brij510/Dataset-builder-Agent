#!/usr/bin/env python3
"""Monitor an Unsloth Studio training run over CDP and log progress.

Usage:
  python monitor.py --url https://host.trycloudflare.com/studio --interval 60
  python monitor.py --once

Writes a rolling status line to stdout and appends history to monitor.log.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

JS_SUMMARY = r"""
(() => {
  const t = document.body.innerText || '';
  const grab = (label) => {
    const i = t.indexOf(label);
    if (i < 0) return null;
    const rest = t.slice(i + label.length, i + label.length + 40);
    const m = rest.match(/([0-9][0-9.,]*\s*[A-Za-z%]*)/);
    return m ? m[1].trim() : null;
  };
  const grabLine = (label) => {
    const i = t.indexOf(label);
    if (i < 0) return null;
    const line = t.slice(i, i + 120).split('\n');
    return line.slice(0, 4).join(' ').replace(/\s+/g, ' ').trim();
  };
  return {
    url: location.href,
    title: document.title,
    state: (t.match(/Training (in progress|stopped|complete)[^\n]*/) || [])[0] || null,
    step: grabLine('Step '),
    epoch: grabLine('Epoch '),
    loss: grabLine('Loss'),
    lr: grabLine('LR'),
    elapsed: grabLine('Elapsed:'),
    eta: grabLine('ETA:'),
    sps: grabLine('steps/s'),
    vram: grabLine('VRAM'),
    util: grabLine('Utilization'),
    temp: grabLine('Temperature'),
    saved: (t.match(/Model saved to ([^\n]+)/) || [])[1] || null,
    error: (t.match(/blocked[^\n]*/i) || [])[0] || null,
    raw_len: t.length
  };
})()
"""


async def snapshot(url):
    try:
        from cdp import with_cdp
    except ImportError:
        return None
    async def run(s, t):
        if url:
            await s.send("Page.navigate", url=url)
            await asyncio.sleep(8)
        r = await s.send("Runtime.evaluate", expression=JS_SUMMARY, returnByValue=True)
        return r["result"].get("value")
    try:
        return await asyncio.wait_for(with_cdp(run), timeout=90)
    except Exception as e:
        return {"error": str(e)}


def fmt(d):
    if not d:
        return "no snapshot"
    if d.get("error"):
        return f"ERROR: {d['error']}"
    parts = [
        d.get("state") or "unknown",
        d.get("step") or "",
        d.get("loss") or "",
        f"ETA {d.get('eta')}" if d.get("eta") else "",
        f"{d.get('sps')}" if d.get("sps") else "",
        f"VRAM {d.get('vram')}" if d.get("vram") else "",
    ]
    if d.get("saved"):
        parts.append(f"SAVED {d['saved']}")
    if d.get("error"):
        parts.append(f"!! {d['error']}")
    return " | ".join(p for p in parts if p)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--log", default="monitor.log")
    ap.add_argument("--max-polls", type=int, default=0)
    args = ap.parse_args()

    polls = 0
    while True:
        d = await snapshot(args.url)
        line = f"{time.strftime('%H:%M:%S')}  {fmt(d)}"
        print(line, flush=True)
        try:
            with open(args.log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "line": line, "data": d}) + "\n")
        except Exception:
            pass
        polls += 1
        if args.once:
            break
        if args.max_polls and polls >= args.max_polls:
            break
        await asyncio.sleep(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
