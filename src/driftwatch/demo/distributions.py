"""Pure, seeded sampling helpers. Every function here takes a
numpy.random.Generator explicitly (never creates or seeds its own) and
draws from it in a single call per invocation -- determinism comes from
callers always drawing in the same order from the same Generator, not from
anything in this module itself. No wall-clock reads, no global state.
"""

import numpy as np

from driftwatch.demo.schema import DistributionSpec


def sample_distribution(rng: np.random.Generator, spec: DistributionSpec, n: int) -> np.ndarray:
    """n draws from `spec`, all in one call to keep the RNG's call sequence
    (and therefore reproducibility) independent of `n`."""
    if spec.type == "normal":
        assert spec.mean is not None and spec.std is not None
        return rng.normal(spec.mean, spec.std, n)
    if spec.type == "lognormal":
        assert spec.mean is not None and spec.sigma is not None
        return rng.lognormal(spec.mean, spec.sigma, n)
    if spec.type == "beta":
        assert spec.a is not None and spec.b is not None
        return spec.loc + spec.scale * rng.beta(spec.a, spec.b, n)
    if spec.type == "uniform":
        assert spec.low is not None and spec.high is not None
        return rng.uniform(spec.low, spec.high, n)
    raise ValueError(f"unknown distribution type: {spec.type!r}")


def sample_categories(
    rng: np.random.Generator, categories: list[str], weights: list[float] | None, n: int
) -> np.ndarray:
    """n draws from `categories`, weighted by `weights` (uniform if None).
    `categories` and `weights` must already be in a fixed, deterministic
    order -- callers own that, this just samples."""
    p = None if weights is None else np.asarray(weights) / sum(weights)
    return rng.choice(np.asarray(categories, dtype=object), size=n, p=p)


def sample_high_cardinality_ids(rng: np.random.Generator, cardinality: int, n: int) -> np.ndarray:
    """n draws, uniformly, from `cardinality` synthetic id strings
    ("id-0000".."id-{cardinality-1}") -- deterministic naming so the same
    cardinality always produces the same label set, and uniform sampling
    so no id dominates (a high-cardinality categorical exercising bin
    dedup differently than the low-cardinality one)."""
    ids = [f"id-{i:04d}" for i in range(cardinality)]
    return rng.choice(np.asarray(ids, dtype=object), size=n)


def sample_missing_mask(rng: np.random.Generator, missing_rate: float, n: int) -> np.ndarray:
    """Boolean mask, True where a value should be dropped to None."""
    if missing_rate <= 0:
        return np.zeros(n, dtype=bool)
    return rng.random(n) < missing_rate


def sample_bernoulli(rng: np.random.Generator, p: np.ndarray) -> np.ndarray:
    """One Bernoulli(p_i) draw per element of `p` (elementwise probabilities,
    e.g. per-prediction label-positive probability derived from that
    prediction's own score)."""
    return rng.random(len(p)) < p
