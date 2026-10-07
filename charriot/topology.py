"""Read the machine.

This module observes and never decides. Everything it returns is either a
measured fact or ``None``; it does not guess, and it does not fall back to a
plausible-sounding default when the host will not answer. `plan.py` is where
judgement happens, and it can only be trusted if what it reads is true.

Nothing here requires privileges.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import os
import re

SYS_CPU = Path("/sys/devices/system/cpu")
SYS_NODE = Path("/sys/devices/system/node")
SYS_DRM = Path("/sys/class/drm")

# PCI vendor IDs we can name. Anything else is reported by raw ID.
VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x8086: "intel"}

# CPU flags worth reporting: they decide which kernels a workload may use.
FLAGS_OF_INTEREST = (
    "avx", "avx2", "avx512f", "avx512bw", "avx512vnni", "avx_vnni",
    "f16c", "fma", "amx_int8", "amx_bf16", "sse4_2",
)


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    raw = _read(path)
    if raw is None:
        return None
    try:
        return int(raw, 0)
    except ValueError:
        return None


def parse_cpu_list(raw: str | None) -> tuple[int, ...]:
    """Expand a Linux cpulist such as ``0-3,8,10-11`` into explicit CPU ids."""
    if not raw:
        return ()
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            try:
                out.extend(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        else:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return tuple(sorted(set(out)))


@dataclass(frozen=True)
class Core:
    """One physical core and the logical CPUs that share it."""

    core_id: int
    cpus: tuple[int, ...]
    kind: str = "unknown"  # "performance" | "efficiency" | "unknown"
    node: int | None = None

    @property
    def smt(self) -> bool:
        return len(self.cpus) > 1


@dataclass(frozen=True)
class Gpu:
    card: str
    vendor: str
    vendor_id: int | None = None
    device_id: int | None = None
    driver: str | None = None
    boot_vga: bool = False
    vram_bytes: int | None = None
    discrete: bool | None = None  # None = could not be determined


@dataclass(frozen=True)
class Topology:
    cores: tuple[Core, ...] = ()
    online_cpus: tuple[int, ...] = ()
    affinity: tuple[int, ...] = ()
    gpus: tuple[Gpu, ...] = ()
    memory_bytes: int | None = None
    flags: tuple[str, ...] = ()
    cgroup2: bool = False
    delegated_controllers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default=())

    @property
    def physical_cores(self) -> int:
        return len(self.cores)

    @property
    def logical_cpus(self) -> int:
        return len(self.online_cpus)

    @property
    def hybrid(self) -> bool:
        return len({c.kind for c in self.cores} - {"unknown"}) > 1

    def cores_of(self, kind: str) -> tuple[Core, ...]:
        return tuple(c for c in self.cores if c.kind == kind)

    def to_dict(self) -> dict:
        return asdict(self)


def _core_kinds() -> dict[int, str]:
    """Map logical CPU -> core kind on Intel hybrid parts.

    Newer kernels expose ``/sys/devices/system/cpu/types/{intel_core,
    intel_atom}*/cpulist``. Where that is absent every core reads "unknown"
    rather than being guessed from frequency, which is unreliable under
    thermal and firmware variation.
    """
    kinds: dict[int, str] = {}
    types = SYS_CPU / "types"
    if not types.is_dir():
        return kinds
    for entry in sorted(types.iterdir()):
        name = entry.name
        if name.startswith("intel_core"):
            kind = "performance"
        elif name.startswith("intel_atom"):
            kind = "efficiency"
        else:
            continue
        for cpu in parse_cpu_list(_read(entry / "cpulist")):
            kinds[cpu] = kind
    return kinds


def _numa() -> dict[int, int]:
    nodes: dict[int, int] = {}
    if not SYS_NODE.is_dir():
        return nodes
    for entry in sorted(SYS_NODE.glob("node[0-9]*")):
        try:
            node = int(entry.name.removeprefix("node"))
        except ValueError:
            continue
        for cpu in parse_cpu_list(_read(entry / "cpulist")):
            nodes[cpu] = node
    return nodes


def read_cores(warnings: list[str]) -> tuple[tuple[Core, ...], tuple[int, ...]]:
    online = parse_cpu_list(_read(SYS_CPU / "online"))
    if not online:
        count = os.cpu_count() or 0
        online = tuple(range(count))
        if count:
            warnings.append("cpu/online unreadable; assuming a dense CPU range")

    kinds = _core_kinds()
    nodes = _numa()

    # Group logical CPUs by their SMT sibling set. Reading siblings rather
    # than core_id keeps this correct when core ids repeat across sockets.
    groups: dict[tuple[int, ...], list[int]] = {}
    for cpu in online:
        base = SYS_CPU / f"cpu{cpu}" / "topology"
        siblings = parse_cpu_list(_read(base / "thread_siblings_list"))
        key = siblings or (cpu,)
        groups.setdefault(key, []).append(cpu)

    readable = any(
        (SYS_CPU / f"cpu{cpu}" / "topology" / "thread_siblings_list").exists()
        for cpu in online
    )
    if not readable:
        warnings.append("SMT topology unavailable; treating every CPU as a core")

    cores: list[Core] = []
    for index, (key, members) in enumerate(sorted(groups.items())):
        cpus = tuple(sorted(set(key) & set(online)) or sorted(members))
        kind = {kinds.get(c, "unknown") for c in cpus}
        cores.append(
            Core(
                core_id=index,
                cpus=cpus,
                kind=kind.pop() if len(kind) == 1 else "unknown",
                node=nodes.get(cpus[0]),
            )
        )
    return tuple(cores), online


def read_gpus(warnings: list[str]) -> tuple[Gpu, ...]:
    if not SYS_DRM.is_dir():
        return ()
    found: list[Gpu] = []
    for card in sorted(SYS_DRM.glob("card[0-9]*")):
        if "-" in card.name:  # connectors such as card0-HDMI-A-1
            continue
        device = card / "device"
        vendor_id = _read_int(device / "vendor")
        device_id = _read_int(device / "device")
        driver = None
        try:
            driver = (device / "driver").resolve().name
        except OSError:
            pass

        vram = _read_int(device / "mem_info_vram_total")

        # Classification order matters, and PCI bus position is NOT a usable
        # signal: AMD APUs put their integrated GPU behind an internal bridge
        # (0000:00:08.1 -> 0000:05:00.0), so "on the root bus" reads false for
        # the most common integrated part there is.
        #
        # What does hold: NVIDIA ships no PC integrated graphics, i915 is only
        # ever integrated, and an integrated GPU's "VRAM" is a small firmware
        # carve-out of system RAM while its GTT aperture is huge.
        discrete: bool | None = None
        if driver in {"nvidia", "nouveau"}:
            discrete = True
        elif driver == "i915":
            discrete = False
        elif vram is not None and vram < (2 << 30):
            discrete = False
        elif vram is not None and vram >= (8 << 30):
            discrete = True
        else:
            # Ambiguous: an Intel Arc on xe, or an APU with a generous
            # carve-out. Say so rather than guess; planning treats an unknown
            # GPU as one worth protecting.
            warnings.append(f"{card.name}: could not determine discrete vs integrated")

        found.append(
            Gpu(
                card=card.name,
                vendor=VENDORS.get(vendor_id or -1, f"0x{vendor_id:04x}" if vendor_id else "unknown"),
                vendor_id=vendor_id,
                device_id=device_id,
                driver=driver,
                boot_vga=_read_int(device / "boot_vga") == 1,
                vram_bytes=vram,
                discrete=discrete,
            )
        )
    return tuple(found)


def read_flags() -> tuple[str, ...]:
    raw = _read(Path("/proc/cpuinfo")) or ""
    match = re.search(r"^flags\s*:\s*(.+)$", raw, re.MULTILINE)
    if not match:
        return ()
    present = set(match.group(1).split())
    return tuple(f for f in FLAGS_OF_INTEREST if f in present)


def read_memory() -> int | None:
    raw = _read(Path("/proc/meminfo")) or ""
    match = re.search(r"^MemTotal:\s+(\d+) kB$", raw, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def read_cgroup2(warnings: list[str]) -> tuple[bool, tuple[str, ...]]:
    """Report cgroup v2 and which controllers this process may actually use.

    Delegation matters more than presence: ``cpuset`` in particular is often
    absent from a user slice, and memory ceilings are unavailable without it.
    """
    root = Path("/sys/fs/cgroup")
    if not (root / "cgroup.controllers").exists():
        return False, ()

    own = _read(Path("/proc/self/cgroup")) or ""
    rel = own.rpartition("::")[2].strip().lstrip("/")
    scope = root / rel if rel else root
    delegated = _read(scope / "cgroup.controllers")
    if delegated is None:
        # Fall back to the parent, which is what a delegated leaf inherits.
        delegated = _read(scope.parent / "cgroup.controllers") or ""
    controllers = tuple(sorted(delegated.split()))
    if "cpuset" not in controllers:
        warnings.append("cgroup cpuset not delegated; affinity will use sched_setaffinity")
    if "memory" not in controllers:
        warnings.append("cgroup memory not delegated; memory ceilings unavailable")
    return True, controllers


def read() -> Topology:
    """Inspect this machine."""
    warnings: list[str] = []
    cores, online = read_cores(warnings)
    cgroup2, controllers = read_cgroup2(warnings)
    try:
        affinity = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        affinity = online
        warnings.append("sched_getaffinity unavailable; assuming all CPUs")

    if len(affinity) < len(online):
        warnings.append(
            f"this process is already restricted to {len(affinity)} of {len(online)} CPUs; "
            "plans are computed against the restricted set"
        )

    return Topology(
        cores=cores,
        online_cpus=online,
        affinity=affinity,
        gpus=read_gpus(warnings),
        memory_bytes=read_memory(),
        flags=read_flags(),
        cgroup2=cgroup2,
        delegated_controllers=controllers,
        warnings=tuple(warnings),
    )
