"""Synthetic workloads, so Charriot can be measured without a game.

``frameloop`` imitates the shape of a game's main loop: a fixed slice of work
per frame, then sleep until the next frame boundary. What it measures is
scheduling latency — how long the work slice actually took, and how far frame
starts drifted from their target. That is the quantity partitioning is supposed
to protect.

``saturate`` is the antagonist: CPU-bound children that will happily consume
every core they are allowed to touch.

Neither renders anything. This measures the CPU side of frame pacing only, and
a real game adds GPU submission, driver, and compositor behaviour on top.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time


# How long before the frame boundary to stop sleeping and start spinning.
SPIN_MARGIN = 0.002


def _work(rounds: int) -> float:
    """A deterministic slice of arithmetic. Same work every frame, every run."""
    total = 0.0
    for i in range(rounds):
        total += (i * i) ** 0.5
    return total


def calibrate(target_ms: float) -> int:
    """Find the round count that takes roughly ``target_ms`` at full clock.

    Two things make the naive version badly wrong, and both were shipped once:

    * A cold CPU is a slow CPU. Measuring during frequency ramp-up
      overestimates the cost per round by an order of magnitude and returns a
      round count that then does almost no work at steady clock.
    * A single sample can be preempted. Taking the minimum of several is the
      standard defence: the fastest observed run is the one least disturbed.
    """
    # Spin until the governor has ramped. Nothing is measured here.
    warm_until = time.perf_counter() + 0.4
    while time.perf_counter() < warm_until:
        _work(20_000)

    def cost(rounds: int) -> float:
        """Milliseconds per run, least-disturbed of five attempts."""
        best = float("inf")
        for _ in range(5):
            start = time.perf_counter()
            _work(rounds)
            best = min(best, (time.perf_counter() - start) * 1000)
        return best

    rounds = 20_000
    for _ in range(28):
        elapsed = cost(rounds)
        if elapsed >= target_ms * 0.5:
            return max(1, int(rounds * target_ms / elapsed))
        rounds *= 2
    return rounds


def frameloop(
    fps: float, seconds: float, work_ms: float, out: str | None, rounds: int = 0
) -> int:
    budget = 1.0 / fps
    # Calibrating here would be a trap: the two benchmark conditions start
    # under different contention, so each would settle on a different round
    # count and then be compared as though they had done the same work. The
    # caller calibrates once, on an idle machine, and passes the result in.
    rounds = rounds or calibrate(work_ms)

    intervals: list[float] = []
    work_times: list[float] = []

    start = time.perf_counter()
    deadline = start + seconds
    previous = time.perf_counter()
    next_frame = previous + budget

    while time.perf_counter() < deadline:
        work_start = time.perf_counter()
        _work(rounds)
        work_end = time.perf_counter()
        work_times.append((work_end - work_start) * 1000)

        # Frame interval is measured start-to-start, which is what a player
        # perceives as pacing.
        intervals.append((work_start - previous) * 1000)
        previous = work_start

        # Sleep the bulk of the wait, then spin the last slice. Sleeping all
        # the way to the boundary makes the loop's own wake-up latency the
        # dominant term — measured at 29ms p99 on a completely idle machine,
        # which is larger than any effect worth measuring. Real engines busy-
        # wait the tail for the same reason.
        remaining = next_frame - time.perf_counter()
        if remaining > SPIN_MARGIN:
            time.sleep(remaining - SPIN_MARGIN)
        while time.perf_counter() < next_frame:
            pass

        if time.perf_counter() - next_frame > budget:
            # Fell more than a whole frame behind: resynchronise rather than
            # spiral trying to catch up.
            next_frame = time.perf_counter() + budget
        else:
            next_frame += budget

    # The first interval is start-up noise, not a frame.
    record = {
        "fps_target": fps,
        "work_ms_target": work_ms,
        "rounds": rounds,
        "frames": len(intervals) - 1,
        "intervals_ms": intervals[1:],
        "work_ms": work_times[1:],
        "cpus": sorted(os.sched_getaffinity(0)),
    }
    if out:
        with open(out, "w") as handle:
            json.dump(record, handle)
    else:
        json.dump(record, sys.stdout)
    return 0


def saturate(workers: int) -> int:
    """Fork CPU-bound children, which inherit our affinity mask."""
    allowed = len(os.sched_getaffinity(0))
    workers = workers or allowed

    children: list[int] = []

    def stop(_signum=None, _frame=None):
        # SIGTERM's default action would kill this process outright, so the
        # cleanup below would never run and every child would be orphaned.
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for _ in range(workers):
        pid = os.fork()
        if pid == 0:
            try:
                while True:
                    _work(1_000_000)
            except KeyboardInterrupt:
                os._exit(0)
            os._exit(0)
        children.append(pid)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for pid in children:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="charriot.workloads")
    sub = parser.add_subparsers(dest="kind", required=True)

    loop = sub.add_parser("frameloop", help="a synthetic frame loop that records pacing")
    loop.add_argument("--fps", type=float, default=60.0)
    loop.add_argument("--seconds", type=float, default=20.0)
    loop.add_argument("--work-ms", type=float, default=6.0)
    loop.add_argument("--out")
    loop.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="fixed work per frame; 0 calibrates locally (never do this mid-comparison)",
    )

    sub.add_parser("calibrate", help="print the round count for --work-ms").add_argument(
        "--work-ms", type=float, default=6.0
    )

    load = sub.add_parser("saturate", help="consume every core this process may touch")
    load.add_argument("--workers", type=int, default=0)

    args = parser.parse_args(argv)
    if args.kind == "calibrate":
        print(calibrate(args.work_ms))
        return 0
    if args.kind == "frameloop":
        return frameloop(args.fps, args.seconds, args.work_ms, args.out, args.rounds)
    return saturate(args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
