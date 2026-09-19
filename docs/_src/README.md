# Demo page source

The interactive demo at https://i-hridaysaha.github.io/driftwatch/ is
`docs/index.html`, served by GitHub Pages from this directory's parent. It is
built from one real run, and the run's inputs are checked in beside the page
so it rebuilds without a database.

    uv run driftwatch demo clean                          # then the other four with --no-reset
    uv run driftwatch demo-verify <scenario> > ...        # concatenated into verify.log
    uv run python docs/_src/extract.py > docs/_src/data.json
    uv run python docs/_src/shoot.py docs/shots "NAME[:WIDTH]=URL" ...   # dashboard screenshots
    uv run python docs/_src/build.py                      # template + data + verify -> docs/index.html

| File | Job |
|---|---|
| `template.html` | the page: copy, styles, and the chart / lifecycle-strip / replay code that draws `data.json` in the browser, no library |
| `extract.py` | reads the per-window drift and performance traces and every alert row out of Postgres after the five scenarios are loaded |
| `shoot.py` | headless-Chrome screenshots of the running Streamlit dashboard over the DevTools protocol, light theme, toolbar hidden, trimmed |
| `build.py` | substitutes `data.json` and `verify.log` into the template and writes `docs/index.html` |
| `data.json` | the run's traces and alerts, as `extract.py` wrote them |
| `verify.log` | the run's `demo-verify` output, one block per scenario, shown verbatim on the page |

Everything the page states as a number comes from `data.json` or `verify.log`;
the scenario copy (what was injected, what must hold) is written in
`template.html` and mirrors `configs/scenarios/*.yaml`.
