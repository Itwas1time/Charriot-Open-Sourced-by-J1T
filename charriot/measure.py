"""Turn frame intervals into a verdict.

The headline is the 1% low: the 99th percentile frame time, which is the worst
1% of frames. Mean frame rate is reported too, and deliberately not headlined,
because it hides exactly the degradation partitioning exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
import json


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. No interpolation, so the number is a real frame."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
    return ordered[index]


@dataclass(frozen=True)
class Stats:
    frames: int
    median: float
    p99: float          # the 1% low
    p999: float
    worst: float
    mean: float
    work_median: float
    work_p99: float

    @property
    def fps_mean(self) -> float:
        return 1000.0 / self.mean if self.mean else float("nan")


def summarise(record: dict) -> Stats:
    intervals = [float(v) for v in record.get("intervals_ms", [])]
    work = [float(v) for v in record.get("work_ms", [])]
    return Stats(
        frames=len(intervals),
        median=percentile(intervals, 0.50),
        p99=percentile(intervals, 0.99),
        p999=percentile(intervals, 0.999),
        worst=max(intervals) if intervals else float("nan"),
        mean=sum(intervals) / len(intervals) if intervals else float("nan"),
        work_median=percentile(work, 0.50),
        work_p99=percentile(work, 0.99),
    )


def load(path: str) -> Stats:
    with open(path) as handle:
        return summarise(json.load(handle))


def change(baseline: float, candidate: float) -> float:
    """Percent improvement of candidate over baseline. Positive is better."""
    if not baseline or baseline != baseline:
        return float("nan")
    return (baseline - candidate) / baseline * 100.0


def spread(values: list[float]) -> tuple[float, float, float]:
    """Median, min, max. Three numbers is enough to see whether to believe one."""
    if not values:
        nan = float("nan")
        return nan, nan, nan
    ordered = sorted(values)
    middle = ordered[len(ordered) // 2]
    return middle, ordered[0], ordered[-1]


def report_repeats(pairs: list[tuple[Stats, Stats]]) -> str:
    """Summarise several runs of the same comparison.

    A single sample cannot distinguish a real effect from a lucky scheduler, so
    the headline here is the median across runs and the range it moved in.
    """
    if not pairs:
        return "no runs"

    changes = [change(before.p99, after.p99) for before, after in pairs]
    lines = [
        f"{'run':<6}{'unpartitioned':>15}{'partitioned':>14}{'1% low change':>16}",
        "-" * 51,
    ]
    for index, ((before, after), delta) in enumerate(zip(pairs, changes), start=1):
        lines.append(
            f"{index:<6}{before.p99:>13.2f}ms{after.p99:>12.2f}ms{delta:>14.1f}%"
        )

    middle, low, high = spread(changes)
    lines += [
        "-" * 51,
        f"{'median':<6}{'':>15}{'':>14}{middle:>14.1f}%",
        f"{'range':<6}{'':>15}{'':>14}{f'{low:.1f}..{high:.1f}%':>15}",
        "",
    ]

    if middle != middle:
        lines.append("VERDICT  no usable data")
    elif low > 10:
        lines.append(
            f"VERDICT  partitioning improves the 1% low by {middle:.1f}% "
            f"(every run above 10%)"
        )
    elif middle >= 10:
        lines.append(
            f"VERDICT  partitioning improves the 1% low by {middle:.1f}% at the median, "
            f"but runs ranged {low:.1f}..{high:.1f}% - repeat before relying on it"
        )
    elif high < -10:
        lines.append(f"VERDICT  partitioning is consistently WORSE ({middle:.1f}%)")
    else:
        lines.append(
            f"VERDICT  no reliable difference (median {middle:+.1f}%, "
            f"range {low:.1f}..{high:.1f}%)"
        )
    return "\n".join(lines)


def report(baseline: Stats, partitioned: Stats) -> str:
    rows = (
        ("1% low (p99)", baseline.p99, partitioned.p99),
        ("0.1% low (p99.9)", baseline.p999, partitioned.p999),
        ("worst frame", baseline.worst, partitioned.worst),
        ("median", baseline.median, partitioned.median),
        ("mean", baseline.mean, partitioned.mean),
        ("work slice p99", baseline.work_p99, partitioned.work_p99),
    )

    lines = [
        f"{'metric':<18}{'unpartitioned':>15}{'partitioned':>14}{'change':>12}",
        "-" * 59,
    ]
    for name, before, after in rows:
        delta = change(before, after)
        arrow = "better" if delta > 0 else "worse"
        lines.append(
            f"{name:<18}{before:>13.2f}ms{after:>12.2f}ms"
            f"{abs(delta):>10.1f}% {arrow}"
        )

    lines.append("")
    lines.append(
        f"frames: {baseline.frames} unpartitioned, {partitioned.frames} partitioned"
    )

    headline = change(baseline.p99, partitioned.p99)
    lines.append("")
    if headline != headline:
        lines.append("VERDICT  no usable data")
    elif headline >= 10:
        lines.append(
            f"VERDICT  partitioning improves the 1% low by {headline:.1f}% on this machine"
        )
    elif headline <= -10:
        lines.append(
            f"VERDICT  partitioning makes the 1% low {abs(headline):.1f}% WORSE on this machine"
        )
    else:
        lines.append(
            f"VERDICT  no meaningful difference ({headline:+.1f}%); "
            "partitioning is not earning its keep here"
        )
    return "\n".join(lines)
