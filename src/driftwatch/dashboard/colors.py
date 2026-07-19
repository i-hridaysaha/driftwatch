import hashlib
from collections.abc import Iterable

import altair as alt

_PALETTE = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#B279A2",  # purple
    "#E45756",  # red
    "#72B7B2",  # teal
    "#EECA3B",  # yellow
    "#FF9DA6",  # pink
    "#9D755D",  # brown
    "#BAB0AC",  # grey
]
"""Fixed, light-theme-legible categorical palette (Vega's "tableau10",
hardcoded rather than referenced by scheme name so the exact hex values --
and therefore the rendered pixels -- can never silently change with a
library upgrade)."""


def stable_color(name: str) -> str:
    """Maps `name` (a feature name or segment value) to one of _PALETTE's
    colors, independent of query result order, independent of what other
    names happen to appear alongside it in a given chart, and stable across
    process restarts. Deliberately hashlib.md5, not Python's builtin
    hash(): str hashing is randomized per-process (PYTHONHASHSEED) unless
    explicitly disabled, which would silently break "same data produces a
    pixel-identical chart on every run" -- md5 has no such randomization.
    Collisions across many distinct names are an accepted simplification
    for this project's scale (a handful of features/segments per model)."""
    digest = hashlib.md5(name.encode("utf-8")).digest()
    return _PALETTE[digest[0] % len(_PALETTE)]


def color_scale(names: Iterable[str]) -> alt.Scale:
    """An Altair color scale over `names` with each name's color coming
    from stable_color -- domain is sorted only for a readable legend order,
    never for correctness (correctness is the hash, not the position)."""
    domain = sorted(set(names))
    return alt.Scale(domain=domain, range=[stable_color(name) for name in domain])
