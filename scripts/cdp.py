#!/usr/bin/env python3
"""Minimal Chrome DevTools Protocol driver (no MCP needed).

Usage:
  python cdp.py nav <url>
  python cdp.py text
  python cdp.py eval <js>
  python cdp.py shot <out.png>
  python cdp.py click-text <label>
  python cdp.py fill <selector> <value>
  python cdp.py upload <selector> <file1> [file2 ...]
  python cdp.py waittext <text> [timeout_s]
"""
import asyncio, json, sys, os, base64, urllib.request
import websockets

CDP = "http://127.0.0.1:9223"

def page_ws():
    pages = json.load(urllib.request.urlopen(f"{CDP}/json/list", timeout=10))
    best = None
    for p in pages:
        if p.get("type") == "page" and p.get("url", "").startswith("http"):
            best = p
            break
    if best is None:
        for p in pages:
            if p.get("type") == "page":
                best = p
                break
    if best is None:
        raise SystemExit("no page target found")
    return best["webSocketDebuggerUrl"], best

class CDPSession:
    def __init__(self, ws):
        self.ws = ws
        self.i = 0
    async def send(self, method, **params):
        self.i += 1
        msg = {"id": self.i, "method": method, "params": params}
        await self.ws.send(json.dumps(msg))
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=120)
            data = json.loads(raw)
            if data.get("id") == self.i:
                if "error" in data:
                    raise RuntimeError(f"{method}: {data['error']}")
                return data.get("result", {})

async def with_cdp(fn):
    url, target = page_ws()
    async with websockets.connect(url, max_size=200*1024*1024) as ws:
        s = CDPSession(ws)
        await s.send("Page.enable")
        await s.send("Runtime.enable")
        await s.send("DOM.enable")
        return await fn(s, target)

# ---------------- JS helpers ----------------
JS_TEXT = """
(() => {
  const walk = document.body ? document.body.innerText : '';
  return walk;
})()
"""

JS_CLICK_TEXT = r"""
(args) => {
  const label = args[0].toLowerCase();
  const els = Array.from(document.querySelectorAll('button,a,div,span,label,input[type=submit],input[type=button]'));
  // exact-ish match first
  let hit = els.find(e => (e.innerText||e.value||'').trim().toLowerCase() === label);
  if (!hit) hit = els.find(e => (e.innerText||e.value||'').trim().toLowerCase().includes(label));
  if (!hit) return 'NOT_FOUND';
  hit.scrollIntoView({block:'center'});
  hit.click();
  return 'CLICKED:' + (hit.innerText||hit.value||'').trim().slice(0,60);
}
"""

JS_FILL = r"""
(args) => {
  const [sel, val] = args;
  const el = document.querySelector(sel);
  if (!el) return 'NOT_FOUND:' + sel;
  const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
  setter.call(el, val);
  el.dispatchEvent(new Event('input', {bubbles:true}));
  el.dispatchEvent(new Event('change', {bubbles:true}));
  return 'FILLED';
}
"""

JS_SUMMARY = r"""
(() => {
  const inputs = Array.from(document.querySelectorAll('input')).map(i => ({
    type: i.type, name: i.name, id: i.id, ph: i.placeholder, val: i.type==='password'?'***':i.value
  }));
  const btns = Array.from(document.querySelectorAll('button')).map(b => (b.innerText||'').trim()).filter(Boolean);
  const links = Array.from(document.querySelectorAll('a')).map(a => (a.innerText||'').trim()).filter(Boolean);
  return {url: location.href, title: document.title, inputs, buttons: btns, links: links.slice(0,40)};
})()
"""

async def cmd_nav(s, t, url):
    return await s.send("Page.navigate", url=url)

async def cmd_text(s, t):
    r = await s.send("Runtime.evaluate", expression=JS_TEXT, returnByValue=True)
    return r["result"].get("value", "")

async def cmd_eval(s, t, js):
    r = await s.send("Runtime.evaluate", expression=js, returnByValue=True, awaitPromise=True)
    v = r["result"].get("value")
    if isinstance(v, (dict, list)):
        return json.dumps(v, indent=2, ensure_ascii=False)
    return str(v)

async def cmd_shot(s, t, path):
    r = await s.send("Page.captureScreenshot", format="png")
    with open(path, "wb") as f:
        f.write(base64.b64decode(r["data"]))
    return f"saved {path}"

async def cmd_click_text(s, t, label):
    r = await s.send("Runtime.evaluate", expression=JS_CLICK_TEXT, returnByValue=True, arguments=[label])
    return str(r["result"].get("value"))

async def cmd_fill(s, t, sel, val):
    r = await s.send("Runtime.evaluate", expression=JS_FILL, returnByValue=True, arguments=[sel, val])
    return str(r["result"].get("value"))

async def cmd_summary(s, t):
    r = await s.send("Runtime.evaluate", expression=JS_SUMMARY, returnByValue=True)
    return json.dumps(r["result"].get("value"), indent=2, ensure_ascii=False)

async def cmd_upload(s, t, sel, files):
    doc = await s.send("DOM.getDocument", depth=-1)
    root = doc["root"]["nodeId"]
    q = await s.send("DOM.querySelector", nodeId=root, selector=sel)
    if not q.get("nodeId"):
        return f"NOT_FOUND:{sel}"
    await s.send("DOM.setFileInputFiles", files=files, nodeId=q["nodeId"])
    return f"UPLOADED:{files}"

async def cmd_clickel(s, t, sel, nth=0):
    """Real mouse click at the element's centre (trusted event)."""
    r = await s.send("Runtime.evaluate", returnByValue=True, expression=f"""
      (()=>{{const els=Array.from(document.querySelectorAll({json.dumps(sel)}));
        const e=els[{int(nth)}]; if(!e) return null;
        e.scrollIntoView({{block:'center'}});
        const r=e.getBoundingClientRect();
        return {{x:r.left+r.width/2, y:r.top+r.height/2, w:r.width, h:r.height, text:(e.innerText||'').trim().slice(0,50)}};
      }})()""")
    box = r["result"].get("value")
    if not box:
        return f"NOT_FOUND:{sel}[{nth}]"
    x, y = box["x"], box["y"]
    for et in ("mousePressed", "mouseReleased"):
        await s.send("Input.dispatchMouseEvent", type=et, x=x, y=y, button="left",
                     clickCount=1, buttons=1 if et == "mousePressed" else 0)
    return f"CLICKED {box['text']!r} at ({x:.0f},{y:.0f})"

async def cmd_clicktext(s, t, label, exact="true"):
    """Real mouse click on an element whose text matches label."""
    r = await s.send("Runtime.evaluate", returnByValue=True, expression=f"""
      (()=>{{
        const want={json.dumps(label)}, exact={json.dumps(exact)}==='true';
        const els=Array.from(document.querySelectorAll('button,a,[role=button],[role=option],div,span,label'));
        const m=els.filter(e=>{{const t=(e.innerText||'').trim();
          return exact ? t===want : t.toLowerCase().includes(want.toLowerCase());}});
        const cand=m.filter(e=>e.getBoundingClientRect().height>0);
        if(!cand.length) return null;
        const e=cand[cand.length-1];
        e.scrollIntoView({{block:'center'}});
        const r=e.getBoundingClientRect();
        return {{x:r.left+r.width/2,y:r.top+r.height/2,text:(e.innerText||'').trim().slice(0,60)}};
      }})()""")
    box = r["result"].get("value")
    if not box:
        return f"NOT_FOUND:{label}"
    x, y = box["x"], box["y"]
    for et in ("mousePressed", "mouseReleased"):
        await s.send("Input.dispatchMouseEvent", type=et, x=x, y=y, button="left",
                     clickCount=1, buttons=1 if et == "mousePressed" else 0)
    return f"CLICKED {box['text']!r} at ({x:.0f},{y:.0f})"

async def cmd_setval(s, t, sel, val):
    """Set a React-controlled input/select value reliably."""
    r = await s.send("Runtime.evaluate", returnByValue=True, expression=f"""
      (()=>{{
        const e=document.querySelector({json.dumps(sel)});
        if(!e) return 'NOT_FOUND';
        const proto = e.tagName==='SELECT'?HTMLSelectElement.prototype:
                      (e instanceof HTMLTextAreaElement?HTMLTextAreaElement.prototype:HTMLInputElement.prototype);
        const d=Object.getOwnPropertyDescriptor(proto,'value');
        d.set.call(e, {json.dumps(val)});
        e.dispatchEvent(new Event('input',{{bubbles:true}}));
        e.dispatchEvent(new Event('change',{{bubbles:true}}));
        e.dispatchEvent(new Event('blur',{{bubbles:true}}));
        return 'SET:'+e.value;
      }})()""")
    return str(r["result"].get("value"))

async def cmd_dump(s, t, sel):
    r = await s.send("Runtime.evaluate", returnByValue=True, expression=f"""
      Array.from(document.querySelectorAll({json.dumps(sel)})).map(e=>({{
        tag:e.tagName, id:e.id, name:e.name, type:e.type, val:e.value,
        ph:e.placeholder, aria:e.getAttribute('aria-label'),
        txt:(e.innerText||'').trim().slice(0,40)
      }}))""")
    return json.dumps(r["result"].get("value"), indent=1, ensure_ascii=False)

async def cmd_waittext(s, t, needle, timeout=60):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = await s.send("Runtime.evaluate", expression=JS_TEXT, returnByValue=True)
        body = r["result"].get("value", "") or ""
        if needle.lower() in body.lower():
            return f"FOUND '{needle}'"
        await asyncio.sleep(2)
    return f"TIMEOUT waiting for '{needle}'"

def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    c = sys.argv[1]
    async def run(s, t):
        if c == "nav":
            return await cmd_nav(s, t, sys.argv[2])
        if c == "text":
            return await cmd_text(s, t)
        if c == "eval":
            return await cmd_eval(s, t, sys.argv[2])
        if c == "shot":
            return await cmd_shot(s, t, sys.argv[2])
        if c == "click-text":
            return await cmd_click_text(s, t, sys.argv[2])
        if c == "fill":
            return await cmd_fill(s, t, sys.argv[2], sys.argv[3])
        if c == "summary":
            return await cmd_summary(s, t)
        if c == "upload":
            return await cmd_upload(s, t, sys.argv[2], sys.argv[3:])
        if c == "waittext":
            return await cmd_waittext(s, t, sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 60)
        if c == "clickel":
            return await cmd_clickel(s, t, sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
        if c == "click-text2":
            return await cmd_clicktext(s, t, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "true")
        if c == "setval":
            return await cmd_setval(s, t, sys.argv[2], sys.argv[3])
        if c == "dump":
            return await cmd_dump(s, t, sys.argv[2])
        return "unknown cmd"
    out = asyncio.run(with_cdp(run))
    print(out if isinstance(out, str) else json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
