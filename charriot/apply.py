"""Enforce a plan on real processes.

Two mechanisms, in order of preference:

* ``sched_setaffinity`` — needs no privileges, works on every Linux, and is
  applied in the child between fork and exec so the process never runs a single
  instruction on a CPU it was not given.
* cgroup v2 — used only for memory ceilings, and only when the controller is
  actually delegated to us.

``RLIMIT_AS`` is deliberately never used. It caps address space rather than
resident memory, so it breaks any program that memory-maps a large file, which
is exactly how large model weights are loaded.
"""

from __future__ import annotations

from pathlib import Path
import errno
import os
import re

CGROUP_ROOT = Path("/sys/fs/cgroup")


def child_preexec(cpus: tuple[int, ...], cgroup: Path | None = None):
    """Build the between-fork-and-exec hook that confines a child.

    Both the pin and the cgroup are applied here rather than from the parent
    after ``Popen`` returns. That closes a real window: a process moved into a
    memory cgroup after it has already started can allocate past the ceiling
    before the ceiling exists, and a model loader mapping weights does exactly
    that within milliseconds of starting.
    """
    if not cpus and cgroup is None:
        return None

    def setup() -> None:  # pragma: no cover - runs only in the child
        if cgroup is not None:
            try:
                (cgroup / "cgroup.procs").write_text(str(os.getpid()))
            except OSError:
                # The parent re-checks membership and reports a discrepancy.
                pass
        if cpus:
            try:
                os.sched_setaffinity(0, cpus)
            except OSError:
                # Losing the pin must not stop the workload from starting; the
                # parent verifies afterwards and reports it.
                pass

    return setup


def environment(assignment, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.update(assignment.env)
    return env


def read_affinity(pid: int) -> tuple[int, ...]:
    """Read a process's actual CPU mask from /proc, not from what we asked for."""
    try:
        status = (Path("/proc") / str(pid) / "status").read_text()
    except OSError:
        return ()
    match = re.search(r"^Cpus_allowed_list:\s*(.+)$", status, re.MULTILINE)
    if not match:
        return ()
    from .topology import parse_cpu_list

    return parse_cpu_list(match.group(1))


def verify(pid: int, expected: tuple[int, ...]) -> tuple[bool, str]:
    """Confirm the kernel agrees with the plan."""
    if not expected:
        return True, "unpinned by design"
    actual = read_affinity(pid)
    if not actual:
        return False, "could not read the process CPU mask"
    if set(actual) != set(expected):
        return False, f"expected CPUs {_fmt(expected)}, kernel reports {_fmt(actual)}"
    return True, f"pinned to CPUs {_fmt(actual)}"


def _fmt(cpus: tuple[int, ...]) -> str:
    """Render 0,1,2,3,8 as 0-3,8."""
    if not cpus:
        return "none"
    parts: list[str] = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(parts)


def own_cgroup() -> Path | None:
    try:
        own = (Path("/proc/self/cgroup")).read_text().rpartition("::")[2].strip()
    except OSError:
        return None
    if not own:
        return None
    path = CGROUP_ROOT / own.lstrip("/")
    return path if path.is_dir() else None


MAIN_LEAF = "charriot-main"

_ADVICE = (
    "re-run under a scope of Charriot's own:\n"
    "        systemd-run --user --scope -- charriot run <config>"
)


def prepare_cgroups() -> tuple[bool, str]:
    """Make our cgroup able to hold controlled children.

    cgroup v2 forbids a cgroup from both containing processes and enabling
    controllers for its children ("no internal processes"). Charriot therefore
    steps sideways into a leaf of its own first, which leaves the parent empty
    and free to delegate. Children we spawn inherit the leaf and are moved from
    there into their own.
    """
    scope = own_cgroup()
    if scope is None:
        return False, "cgroup v2 scope not found"
    try:
        if "memory" not in (scope / "cgroup.controllers").read_text().split():
            return False, "memory controller is not delegated"
    except OSError:
        return False, "cannot read delegated controllers"

    subtree = scope / "cgroup.subtree_control"
    try:
        if "memory" in (subtree.read_text() if subtree.exists() else ""):
            return True, "already delegating"
    except OSError:
        return False, "cannot read subtree control"

    own = scope / MAIN_LEAF
    try:
        own.mkdir(exist_ok=True)
        (own / "cgroup.procs").write_text(str(os.getpid()))
        subtree.write_text("+memory")
    except OSError as error:
        if error.errno == errno.EBUSY:
            return False, f"scope is shared with other processes; {_ADVICE}"
        return False, f"could not delegate memory control: {error}"
    return True, "delegated"


def make_ceiling(name: str, limit: int | None) -> tuple[Path | None, str]:
    """Create an empty cgroup carrying a memory ceiling, ready for a child.

    Called *before* the process exists, so the child can join it from
    ``preexec`` and is bounded from its first instruction.
    """
    if limit is None:
        return None, "no ceiling requested"
    scope = own_cgroup()
    if scope is None:
        return None, "cgroup v2 scope not found"
    # own_cgroup() follows us into charriot-main, so climb back to the parent.
    if scope.name == MAIN_LEAF:
        scope = scope.parent

    leaf = scope / f"charriot-{name}"
    try:
        leaf.mkdir(exist_ok=True)
        (leaf / "memory.max").write_text(str(limit))
    except OSError as error:
        if error.errno == errno.EBUSY:
            return None, f"memory control is not delegated here; {_ADVICE}"
        return None, f"could not create ceiling: {error}"
    return leaf, f"memory.max = {limit >> 20} MiB"


def in_cgroup(cgroup: Path, pid: int) -> bool:
    """Confirm the child actually joined, rather than trusting that it did."""
    try:
        members = (cgroup / "cgroup.procs").read_text().split()
    except OSError:
        return False
    return str(pid) in members
