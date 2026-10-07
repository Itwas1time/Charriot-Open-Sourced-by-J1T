"""Decide how to divide the machine.

This is a pure function: ``plan(topology, config) -> Plan``. It touches
nothing, so every policy decision Charriot makes is testable against a
recorded machine description without owning that machine.

The two rules that shape everything here:

* **Partition by physical core, never by logical CPU.** SMT siblings share
  execution resources, so handing one sibling to a latency workload and the
  other to a throughput workload gives the interference the partition exists
  to prevent.
* **Charriot sets bounds, not policy.** A workload is told how much machine it
  may use. How to use it is the workload's decision, and a program that
  autotunes will beat any number Charriot could guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config, Workload
from .topology import Core, Topology

# Below this many physical cores, carving the machine up starves both sides.
MIN_CORES_TO_PARTITION = 4


@dataclass(frozen=True)
class Assignment:
    name: str
    cpus: tuple[int, ...]
    cores: int
    memory_max: int | None = None
    env: dict[str, str] = field(default_factory=dict)
    gpu: str | None = None
    gpu_reason: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    assignments: tuple[Assignment, ...]
    partitioned: bool
    reserved_cpus: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()

    def for_name(self, name: str) -> Assignment | None:
        for assignment in self.assignments:
            if assignment.name == name:
                return assignment
        return None


def _usable_cores(topology: Topology) -> list[Core]:
    """Physical cores fully contained in this process's affinity mask.

    A core is only usable if every one of its logical CPUs is available;
    a half-visible core cannot be reasoned about.
    """
    allowed = set(topology.affinity or topology.online_cpus)
    return [core for core in topology.cores if set(core.cpus) <= allowed and core.cpus]


def _order_for(role: str, cores: list[Core]) -> list[Core]:
    """Prefer performance cores for latency work, efficiency for throughput."""
    if role == "latency":
        rank = {"performance": 0, "unknown": 1, "efficiency": 2}
    else:
        rank = {"efficiency": 0, "unknown": 1, "performance": 2}
    return sorted(cores, key=lambda c: (rank.get(c.kind, 1), c.core_id))


def _flatten(cores: list[Core]) -> tuple[int, ...]:
    return tuple(sorted(cpu for core in cores for cpu in core.cpus))


def _spare(gpus: list) -> list:
    """GPUs not driving the display, and so free for background work."""
    return [gpu for gpu in gpus if not gpu.boot_vga]


def _assign_gpus(
    workloads: tuple[Workload, ...], topology: Topology
) -> dict[str, tuple[str | None, str]]:
    """Choose a GPU per workload.

    Only standard, vendor-neutral environment variables are emitted here.
    Program-specific switches belong in that workload's own ``env``.
    """
    out: dict[str, tuple[str | None, str]] = {}
    gpus = list(topology.gpus)
    discrete = [g for g in gpus if g.discrete]
    integrated = [g for g in gpus if g.discrete is False]

    wants = {w.name: w.gpu for w in workloads if w.gpu != "none"}
    latency_wants = [w for w in workloads if w.role == "latency" and w.name in wants]
    throughput_wants = [w for w in workloads if w.role == "throughput" and w.name in wants]
    split = bool(discrete) and bool(integrated) and latency_wants and throughput_wants

    for workload in workloads:
        if workload.gpu == "none":
            out[workload.name] = (None, "configured off")
            continue
        if not gpus:
            out[workload.name] = (None, "no GPU detected")
            continue

        if split and workload.role == "latency":
            card = integrated[0]
            out[workload.name] = (card.card, "integrated GPU, leaving the discrete card free")
        elif split and workload.role == "throughput":
            card = discrete[0]
            out[workload.name] = (card.card, "discrete GPU, away from the display")
        elif workload.role == "throughput" and latency_wants and not _spare(gpus):
            # Keyed off which GPU drives the display rather than off
            # discrete/integrated: a GPU of unknown type is treated as one
            # worth protecting, which is the safe way to be wrong.
            out[workload.name] = (
                None,
                "the only GPU is driving the display; staying on CPU to protect it",
            )
        else:
            card = (discrete or integrated or gpus)[0]
            out[workload.name] = (card.card, "only GPU available")
    return out


def _gpu_env(card: str | None, topology: Topology) -> dict[str, str]:
    if card is None:
        # Standard way to hide every CUDA device from a process.
        return {"CUDA_VISIBLE_DEVICES": ""}
    env: dict[str, str] = {}
    for index, gpu in enumerate(topology.gpus):
        if gpu.card != card:
            continue
        if gpu.vendor == "nvidia":
            env["CUDA_VISIBLE_DEVICES"] = str(
                sum(1 for g in topology.gpus[:index] if g.vendor == "nvidia")
            )
            if not gpu.boot_vga:
                env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
                env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
        else:
            # DRI_PRIME selects default (0) or offload (1). It is not a card
            # index: enumerating card0/card1 says nothing about which one the
            # display is on, so keying off the index picks the wrong GPU
            # whenever the discrete card enumerates first.
            env["DRI_PRIME"] = "0" if gpu.boot_vga else "1"
    return env


def plan(topology: Topology, config: Config) -> Plan:
    notes: list[str] = []
    cores = _usable_cores(topology)
    total = len(cores)

    if not cores:
        notes.append("no usable physical cores detected; running unpinned")
        return Plan(
            assignments=tuple(
                Assignment(name=w.name, cpus=(), cores=0, env=dict(w.env)) for w in config.workloads
            ),
            partitioned=False,
            notes=tuple(notes),
        )

    gpu_choice = _assign_gpus(config.workloads, topology)
    memory_ok = "memory" in topology.delegated_controllers
    if not memory_ok and any(w.memory_max for w in config.workloads):
        notes.append(
            "cgroup memory controller is not delegated; memory ceilings are advisory only. "
            "RLIMIT_AS is deliberately not used because it breaks large mmap loads."
        )

    reserved: list[Core] = []
    if config.reserve_cores:
        reserve = min(config.reserve_cores, max(0, total - 1))
        if reserve:
            reserved = cores[-reserve:]
            cores = cores[:-reserve]
            notes.append(f"reserved {reserve} physical core(s) for the rest of the system")

    partition = config.partition and len(cores) >= MIN_CORES_TO_PARTITION
    if config.partition and not partition:
        notes.append(
            f"only {len(cores)} usable physical core(s); partitioning disabled "
            f"(needs {MIN_CORES_TO_PARTITION}). Every workload sees the whole machine."
        )

    def finish(name: str, chosen: list[Core], extra: list[str]) -> Assignment:
        workload = config.by_name(name)
        if workload is None:  # pragma: no cover - guarded by config validation
            raise ValueError(f"no workload named {name!r}")
        card, reason = gpu_choice.get(name, (None, ""))
        env = dict(workload.env)
        env.update(_gpu_env(card, topology))
        if workload.threads_env:
            # Physical cores, not logical CPUs: SMT siblings contend.
            env[workload.threads_env] = str(max(1, len(chosen)))
        return Assignment(
            name=name,
            cpus=_flatten(chosen),
            cores=len(chosen),
            memory_max=workload.memory_max if memory_ok else None,
            env=env,
            gpu=card,
            gpu_reason=reason,
            notes=tuple(extra),
        )

    if not partition:
        return Plan(
            assignments=tuple(
                finish(w.name, cores, []) for w in config.workloads
            ),
            partitioned=False,
            reserved_cpus=_flatten(reserved),
            notes=tuple(notes),
        )

    latency = [w for w in config.workloads if w.role == "latency"]
    throughput = [w for w in config.workloads if w.role == "throughput"]
    shared = [w for w in config.workloads if w.role == "shared"]

    pool = list(cores)
    assignments: list[Assignment] = []

    # Latency workloads are served first: they are the reason to partition.
    for workload in latency:
        want = workload.cores or max(2, len(cores) // 2)
        extra: list[str] = []
        # Always leave at least one core for everything else.
        cap = max(1, len(pool) - (1 if (throughput or shared) else 0))
        take = min(want, cap)
        if take < want:
            extra.append(f"requested {want} core(s), granted {take}")
        ordered = _order_for("latency", pool)[:take]
        pool = [core for core in pool if core not in ordered]
        assignments.append(finish(workload.name, ordered, extra))

    remainder = pool or cores
    if not pool and (throughput or shared):
        notes.append("no cores left after latency workloads; sharing the full set")

    for workload in throughput:
        want = workload.cores
        ordered = _order_for("throughput", remainder)
        chosen = ordered[:want] if want else ordered
        extra = []
        if want and len(chosen) < want:
            extra.append(f"requested {want} core(s), granted {len(chosen)}")
        assignments.append(finish(workload.name, chosen, extra))

    for workload in shared:
        assignments.append(
            finish(workload.name, list(cores), ["shared role: not partitioned"])
        )

    order = {w.name: i for i, w in enumerate(config.workloads)}
    assignments.sort(key=lambda a: order[a.name])
    return Plan(
        assignments=tuple(assignments),
        partitioned=True,
        reserved_cpus=_flatten(reserved),
        notes=tuple(notes),
    )
