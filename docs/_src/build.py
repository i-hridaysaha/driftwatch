"""Assemble docs/index.html from the template, the extracted run data and the
verifier's output for that run.

    uv run python docs/_src/build.py

`data.json` comes from extract.py and `verify.log` is the concatenated
`driftwatch demo-verify <scenario>` output of the same run; both are checked
in, so the page rebuilds without a database.
"""

import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent


def render_verify(log: str) -> dict[str, str]:
    """Split the log into one block per scenario, with the command and the
    PASS markers wrapped for styling. The scenario-loaded check's long
    remediation hint is dimmed, not dropped."""
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in log.splitlines():
        match = re.match(r"\$ driftwatch demo-verify (\w+)", line)
        if match:
            current = match.group(1)
            blocks[current] = [f'<span class="cmd">driftwatch demo-verify {current}</span>']
            continue
        if not line.strip() or current is None:
            continue
        if line.startswith("[PASS]"):
            rest = re.sub(r"( -- run `.*)$", r'<span class="dim">\1</span>', html.escape(line[6:]))
            blocks[current].append(f'<span class="pass">[PASS]</span>{rest}')
        else:
            blocks[current].append(html.escape(line))
    return {name: "\n".join(lines) for name, lines in blocks.items()}


def main() -> None:
    template = (HERE / "template.html").read_text()
    data = (HERE / "data.json").read_text()
    assert "</" not in data, "data would close the script tag"
    verify = render_verify((HERE / "verify.log").read_text())
    page = template.replace("__DATA__", data).replace("__VERIFY__", json.dumps(verify))
    out = DOCS / "index.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
