import subprocess
import sys

from driftwatch.dashboard.colors import color_scale, stable_color


def test_same_name_gets_same_color_repeatedly() -> None:
    assert stable_color("age") == stable_color("age")


def test_color_is_independent_of_what_else_is_in_the_chart() -> None:
    # "age"'s color must not depend on whether "income" is also present --
    # otherwise the same feature could get a different color across two
    # charts that plot different subsets of features.
    with_income = stable_color("age")
    color_scale(["age", "income", "region"])
    assert stable_color("age") == with_income


def test_color_scale_domain_is_sorted_and_matches_stable_color() -> None:
    scale = color_scale(["region", "age", "income"])

    assert scale.domain == ["age", "income", "region"]
    assert scale.range == [stable_color(name) for name in scale.domain]


def test_stable_color_is_deterministic_across_process_restarts() -> None:
    """Python's builtin hash() for str is randomized per-process
    (PYTHONHASHSEED) unless disabled -- this proves stable_color does NOT
    rely on it, by checking two entirely separate interpreter processes
    agree, without pinning PYTHONHASHSEED for either."""
    script = (
        "from driftwatch.dashboard.colors import stable_color; "
        "print(stable_color('age'), stable_color('income'), stable_color('region'))"
    )
    first = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    second = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
