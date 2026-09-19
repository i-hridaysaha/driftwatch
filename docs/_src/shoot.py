"""Screenshots of the running dashboard, taken with headless Chrome over CDP.

Streamlit renders over a websocket, so `chrome --screenshot` alone captures
the loading skeleton; this drives the DevTools protocol instead, waits for
the app to settle, forces the light theme and hides the Streamlit toolbar.

    uv run streamlit run src/driftwatch/dashboard/app.py --server.headless true
    uv run python docs/_src/shoot.py docs/shots \\
        "segment_view:1800=http://localhost:8501/?model=demo-segment-isolated&view=segment_view"

Each argument is NAME[:WIDTH]=URL and lands at OUT_DIR/NAME.png, trimmed of
Streamlit's chrome. Needs a local Chrome; `websockets` and `Pillow` come in
with uvicorn[standard] and streamlit.
"""

import asyncio
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websockets
from PIL import Image

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333
HIDE_CHROME = (
    '[data-testid="stToolbar"],[data-testid="stDecoration"],'
    '[data-testid="stStatusWidget"],.stAppDeployButton{display:none!important}'
    ' section[data-testid="stSidebar"]{transform:none!important;transition:none!important;'
    "width:300px!important;min-width:300px!important;max-width:300px!important}"
    ' section[data-testid="stSidebar"] > div{width:300px!important;min-width:300px!important}'
)


async def shoot(ws_url: str, url: str, out: Path, width: int, settle: float = 9.0) -> None:
    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        mid = 0

        async def call(method: str, **params: object) -> dict:
            nonlocal mid
            mid += 1
            await ws.send(json.dumps({"id": mid, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == mid:
                    return msg.get("result", {})

        await call(
            "Emulation.setEmulatedMedia",
            features=[{"name": "prefers-color-scheme", "value": "light"}],
        )
        await call(
            "Emulation.setDeviceMetricsOverride",
            width=width, height=1000, deviceScaleFactor=2, mobile=False,
        )
        await call("Page.navigate", url=url)
        await asyncio.sleep(settle)
        await call(
            "Runtime.evaluate",
            expression=(
                "(() => { const st = document.createElement('style');"
                f" st.textContent = {json.dumps(HIDE_CHROME)};"
                " document.head.appendChild(st); return 1; })()"
            ),
        )
        metrics = await call("Page.getLayoutMetrics")
        height = int(min(metrics["cssContentSize"]["height"], 4000))
        await call(
            "Emulation.setDeviceMetricsOverride",
            width=width, height=height, deviceScaleFactor=2, mobile=False,
        )
        await asyncio.sleep(3.0)
        shot = await call(
            "Page.captureScreenshot",
            format="png", captureBeyondViewport=True,
            clip={"x": 0, "y": 0, "width": width, "height": height, "scale": 1},
        )
        out.write_bytes(base64.b64decode(shot["data"]))
        trim(out)
        print(f"wrote {out} ({width}x{height} css px)")


def trim(path: Path, scale: float = 0.75) -> None:
    """Drop Streamlit's empty header band and the blank tail below the last
    widget, then downscale: 2x is sharper than the page needs."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    px = im.load()
    main_x0 = int(w * 0.32)  # right of the sidebar
    last = 0
    for y in range(h - 1, 0, -1):
        if any(px[x, y] != (255, 255, 255) for x in range(main_x0, w, 12)):
            last = y
            break
    crop = im.crop((0, 100, w, min(h, last + 60)))
    crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
    crop.save(path, optimize=True)


def devtools(path: str, method: str = "GET") -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/json{path}", method=method)
    with urllib.request.urlopen(req) as r:
        body = r.read()
    return json.loads(body) if body else {}


async def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [arg.split("=", 1) for arg in sys.argv[2:]]
    proc = subprocess.Popen(
        [
            CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            f"--remote-debugging-port={PORT}", "--no-first-run", "--no-default-browser-check",
            "--user-data-dir=/tmp/driftwatch-shoot-profile", "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            try:
                devtools("/version")
                break
            except OSError:
                time.sleep(0.2)
        for name, url in jobs:  # one target per job, so each page starts cold
            width = 1440
            if ":" in name:
                name, w = name.split(":", 1)
                width = int(w)
            target = devtools("/new?about:blank", method="PUT")
            await shoot(target["webSocketDebuggerUrl"], url, out_dir / f"{name}.png", width)
            devtools(f"/close/{target['id']}")
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())
