"""Charriot command line."""

from __future__ import annotations

import argparse
import json
import sys

from . import plan as planning
from . import topology as topo
from .apply import _fmt
from .config import ConfigError, load
from .run import RunError, run


def _human_bytes(value: int | None) -> str:
    if not value:
        return "unknown"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return "unknown"


def show_topology(machine: topo.Topology) -> None:
    kinds = {}
    for core in machine.cores:
        kinds[core.kind] = kinds.get(core.kind, 0) + 1
    detail = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()) if k != "unknown")

    print("CPU")
    print(f"  physical cores   {machine.physical_cores}{f'  ({detail})' if detail else ''}")
    print(f"  logical CPUs     {machine.logical_cpus}")
    print(f"  SMT              {'yes' if any(c.smt for c in machine.cores) else 'no'}")
    print(f"  available to us  {_fmt(machine.affinity)}")
    if machine.flags:
        print(f"  features         {' '.join(machine.flags)}")

    print(f"\nMemory\n  total            {_human_bytes(machine.memory_bytes)}")

    print("\nGPU")
    if not machine.gpus:
        print("  none detected")
    for gpu in machine.gpus:
        kind = {True: "discrete", False: "integrated", None: "unknown"}[gpu.discrete]
        vram = f", {_human_bytes(gpu.vram_bytes)} VRAM" if gpu.vram_bytes else ""
        primary = ", drives the display" if gpu.boot_vga else ""
        print(f"  {gpu.card:<8} {gpu.vendor} ({gpu.driver or 'no driver'}) {kind}{vram}{primary}")

    print("\nControl")
    print(f"  cgroup v2        {'yes' if machine.cgroup2 else 'no'}")
    print(f"  delegated        {' '.join(machine.delegated_controllers) or 'none'}")

    if machine.warnings:
        print("\nNotes")
        for warning in machine.warnings:
            print(f"  - {warning}")


def show_plan(result: planning.Plan, machine: topo.Topology) -> None:
    print("partitioned      " + ("yes" if result.partitioned else "no"))
    if result.reserved_cpus:
        print(f"reserved CPUs    {_fmt(result.reserved_cpus)}")
    print()
    for assignment in result.assignments:
        print(assignment.name)
        print(f"  CPUs           {_fmt(assignment.cpus)}  ({assignment.cores} physical cores)")
        if assignment.memory_max:
            print(f"  memory ceiling {_human_bytes(assignment.memory_max)}")
        gpu = assignment.gpu or "none"
        print(f"  GPU            {gpu}" + (f"  ({assignment.gpu_reason})" if assignment.gpu_reason else ""))
        if assignment.env:
            for key in sorted(assignment.env):
                print(f"  env            {key}={assignment.env[key]}")
        for note in assignment.notes:
            print(f"  note           {note}")
        print()
    if result.notes:
        print("Notes")
        for note in result.notes:
            print(f"  - {note}")


def doctor(machine: topo.Topology) -> int:
    checks: list[tuple[bool, str, str]] = []
    checks.append(
        (
            bool(machine.cores),
            "CPU topology readable",
            "cannot read /sys/devices/system/cpu; pinning will not work",
        )
    )
    checks.append(
        (
            hasattr(__import__("os"), "sched_setaffinity"),
            "sched_setaffinity available",
            "this platform cannot pin processes; Charriot needs Linux",
        )
    )
    checks.append(
        (
            len(machine.affinity) == len(machine.online_cpus),
            "full machine available",
            "something already restricted this process; plans will be smaller",
        )
    )
    checks.append(
        (
            "memory" in machine.delegated_controllers,
            "cgroup memory delegated",
            "memory ceilings will be skipped",
        )
    )
    checks.append(
        (
            machine.physical_cores >= planning.MIN_CORES_TO_PARTITION,
            f"enough cores to partition (>= {planning.MIN_CORES_TO_PARTITION})",
            "too few physical cores; partitioning would starve both sides",
        )
    )

    failed = 0
    for ok, good, bad in checks:
        print(f"  {'PASS' if ok else 'WARN'}  {good if ok else bad}")
        failed += 0 if ok else 1
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 0


BENCH_CONFIG = """
[charriot]
reserve_cores = {reserve}
partition = {partition}

[[workload]]
name = "frameloop"
role = "latency"
command = ["{python}", "-m", "charriot.workloads", "frameloop",
           "--fps", "{fps}", "--seconds", "{seconds}",
           "--work-ms", "{work_ms}", "--rounds", "{rounds}", "--out", "{out}"]
cores = {cores}
gpu = "none"

[[workload]]
name = "load"
role = "throughput"
command = ["{python}", "-m", "charriot.workloads", "saturate",
           "--workers", "{workers}"]
gpu = "none"
"""


def bench(args, machine: topo.Topology) -> int:
    """Measure whether partitioning protects frame pacing on this machine."""
    import tempfile
    from pathlib import Path

    from . import measure
    from .run import run as run_workloads

    if machine.physical_cores < planning.MIN_CORES_TO_PARTITION:
        print(
            f"charriot: {machine.physical_cores} physical cores is too few to "
            "partition; there is nothing to measure here.",
            file=sys.stderr,
        )
        return 2

    # Calibrate once, here, while the machine is still idle. Letting each
    # condition calibrate itself would give them different work per frame and
    # the comparison would be meaningless.
    from .workloads import calibrate

    rounds = calibrate(args.work_ms)
    print(f"calibrated: {rounds} rounds per frame (~{args.work_ms:g}ms idle)")

    pairs: list[tuple[measure.Stats, measure.Stats]] = []
    with tempfile.TemporaryDirectory(prefix="charriot-bench-") as workspace:
        for attempt in range(1, args.repeat + 1):
            results: dict[str, measure.Stats] = {}
            for label, partitioned in (("unpartitioned", False), ("partitioned", True)):
                out = Path(workspace) / f"{label}-{attempt}.json"
                config = load_config_text(
                    BENCH_CONFIG.format(
                        reserve=args.reserve,
                        partition="true" if partitioned else "false",
                        python=sys.executable,
                        fps=args.fps,
                        seconds=args.duration,
                        work_ms=args.work_ms,
                        out=out,
                        cores=args.cores,
                        workers=args.load_workers,
                        rounds=rounds,
                    )
                )
                result = planning.plan(machine, config)
                frame = result.for_name("frameloop")
                load = result.for_name("load")
                suffix = f" (run {attempt}/{args.repeat})" if args.repeat > 1 else ""
                print(
                    f"\n=== {label}{suffix} ===\n"
                    f"  frameloop  {_fmt(frame.cpus) if frame.cpus else 'all CPUs'}\n"
                    f"  load       {_fmt(load.cpus) if load.cpus else 'all CPUs'}\n"
                    f"  running {args.duration:g}s at {args.fps:g} fps target"
                )
                run_workloads(config, result, log=lambda _message: None)

                if not out.exists():
                    print("charriot: the frame loop produced no data", file=sys.stderr)
                    return 1
                results[label] = measure.load(str(out))
            pairs.append((results["unpartitioned"], results["partitioned"]))

    print("\n" + measure.report(pairs[-1][0], pairs[-1][1]))
    if args.repeat > 1:
        print("\n" + measure.report_repeats(pairs))
    print(
        "\nThis measures CPU-side frame pacing with a synthetic loop. A real game "
        "adds\nGPU submission, driver, and compositor behaviour on top."
    )
    return 0


def load_config_text(text: str):
    from .config import loads

    return loads(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="charriot",
        description="Give two workloads the keys to a machine, on purpose.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_topology = sub.add_parser("topology", help="describe this machine")
    p_topology.add_argument("--json", action="store_true")

    p_plan = sub.add_parser("plan", help="show how a config would divide this machine")
    p_plan.add_argument("config")
    p_plan.add_argument("--json", action="store_true")

    p_run = sub.add_parser("run", help="apply a plan and run the workloads")
    p_run.add_argument("config")

    sub.add_parser("doctor", help="check whether this host can be governed")

    p_bench = sub.add_parser(
        "bench", help="measure whether partitioning helps on this machine"
    )
    p_bench.add_argument("--duration", type=float, default=20.0, help="seconds per condition")
    p_bench.add_argument("--fps", type=float, default=60.0)
    p_bench.add_argument("--work-ms", type=float, default=6.0, help="work per frame")
    p_bench.add_argument("--cores", type=int, default=4, help="cores for the frame loop")
    p_bench.add_argument("--reserve", type=int, default=1)
    p_bench.add_argument(
        "--repeat", type=int, default=1, help="run the comparison N times and report spread"
    )
    p_bench.add_argument(
        "--load-workers",
        type=int,
        default=0,
        help="CPU burners in the throughput workload (0 = one per core it may use)",
    )

    args = parser.parse_args(argv)
    machine = topo.read()

    if args.command == "bench":
        return bench(args, machine)

    if args.command == "topology":
        if args.json:
            print(json.dumps(machine.to_dict(), indent=2, sort_keys=True))
        else:
            show_topology(machine)
        return 0

    if args.command == "doctor":
        return doctor(machine)

    try:
        config = load(args.config)
    except ConfigError as error:
        print(f"charriot: {error}", file=sys.stderr)
        return 2

    result = planning.plan(machine, config)

    if args.command == "plan":
        if args.json:
            from dataclasses import asdict

            print(json.dumps(asdict(result), indent=2, sort_keys=True))
        else:
            show_plan(result, machine)
        return 0

    try:
        return run(config, result)
    except RunError as error:
        print(f"charriot: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
