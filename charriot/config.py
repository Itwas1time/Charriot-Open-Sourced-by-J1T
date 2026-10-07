"""Workload configuration.

Charriot knows nothing about any particular program. A workload is a command,
a role, and some optional hints. Which program that command starts is the
user's business, and changing it must never require changing Charriot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

ROLES = ("latency", "throughput", "shared")
GPU_MODES = ("prefer", "require", "none", "auto")

_SUFFIX = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}


class ConfigError(ValueError):
    pass


def parse_bytes(value: object, context: str) -> int | None:
    """Accept 12884901888, "12G", "512M". Reject everything else."""
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ConfigError(f"{context}: must be positive")
        return value
    if not isinstance(value, str):
        raise ConfigError(f"{context}: must be a size string or integer")
    text = value.strip().lower().removesuffix("b")
    if not text:
        raise ConfigError(f"{context}: empty size")
    multiplier = _SUFFIX.get(text[-1], 1)
    if multiplier != 1:
        text = text[:-1]
    try:
        amount = float(text)
    except ValueError:
        raise ConfigError(f"{context}: {value!r} is not a size") from None
    if amount <= 0:
        raise ConfigError(f"{context}: must be positive")
    return int(amount * multiplier)


@dataclass(frozen=True)
class Workload:
    name: str
    command: tuple[str, ...]
    role: str = "shared"
    cwd: str | None = None
    cores: int | None = None            # requested physical cores
    gpu: str = "auto"
    memory_max: int | None = None
    env: dict[str, str] = field(default_factory=dict)
    threads_env: str | None = None      # Charriot writes its core budget here
    provides_endpoint: str | None = None
    endpoint_env: str | None = None     # receives another workload's endpoint
    needs: str | None = None            # name of the workload it depends on
    ready_url: str | None = None
    ready_timeout: float = 60.0
    restart_max: int = 0


@dataclass(frozen=True)
class Config:
    workloads: tuple[Workload, ...]
    reserve_cores: int = 0
    partition: bool = True

    def by_name(self, name: str) -> Workload | None:
        for workload in self.workloads:
            if workload.name == name:
                return workload
        return None


def _require(table: dict, key: str, context: str):
    if key not in table:
        raise ConfigError(f"{context}: missing {key!r}")
    return table[key]


def _workload(raw: dict, index: int) -> Workload:
    context = f"workload[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{context}: must be a table")

    name = _require(raw, "name", context)
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{context}: name must be a non-empty string")
    context = f"workload {name!r}"

    command = _require(raw, "command", context)
    if isinstance(command, str):
        raise ConfigError(
            f"{context}: command must be a list of arguments, not a string. "
            "Charriot never runs a shell."
        )
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) for part in command
    ):
        raise ConfigError(f"{context}: command must be a non-empty list of strings")

    role = raw.get("role", "shared")
    if role not in ROLES:
        raise ConfigError(f"{context}: role must be one of {', '.join(ROLES)}")

    gpu = raw.get("gpu", "auto")
    if gpu not in GPU_MODES:
        raise ConfigError(f"{context}: gpu must be one of {', '.join(GPU_MODES)}")

    cores = raw.get("cores")
    if cores is not None and (not isinstance(cores, int) or cores < 1):
        raise ConfigError(f"{context}: cores must be a positive integer")

    env = raw.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise ConfigError(f"{context}: env must be a table of strings")

    restart_max = raw.get("restart_max", 0)
    if not isinstance(restart_max, int) or restart_max < 0:
        raise ConfigError(f"{context}: restart_max must be a non-negative integer")

    timeout = raw.get("ready_timeout", 60.0)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ConfigError(f"{context}: ready_timeout must be positive")

    def text(key: str) -> str | None:
        value = raw.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{context}: {key} must be a non-empty string")
        return value

    endpoint_env = text("endpoint_env")
    needs = text("needs")
    if endpoint_env and not needs:
        # Otherwise the variable is silently never set and the workload starts
        # with no idea where its provider is.
        raise ConfigError(
            f"{context}: endpoint_env is set but needs is not, so nothing would "
            "be injected. Name the workload this one depends on."
        )

    return Workload(
        name=name,
        command=tuple(command),
        role=role,
        cwd=text("cwd"),
        cores=cores,
        gpu=gpu,
        memory_max=parse_bytes(raw.get("memory_max"), f"{context}.memory_max"),
        env=dict(env),
        threads_env=text("threads_env"),
        provides_endpoint=text("provides_endpoint"),
        endpoint_env=endpoint_env,
        needs=needs,
        ready_url=text("ready_url"),
        ready_timeout=float(timeout),
        restart_max=restart_max,
    )


def loads(text: str) -> Config:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML: {error}") from None

    entries = raw.get("workload")
    if not entries:
        raise ConfigError("no [[workload]] entries")
    if not isinstance(entries, list):
        raise ConfigError("[[workload]] must be an array of tables")

    workloads = tuple(_workload(entry, i) for i, entry in enumerate(entries))

    names = [w.name for w in workloads]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ConfigError(f"duplicate workload names: {', '.join(sorted(duplicates))}")

    for workload in workloads:
        if workload.needs and workload.needs not in names:
            raise ConfigError(
                f"workload {workload.name!r} needs unknown workload {workload.needs!r}"
            )
        if workload.needs == workload.name:
            raise ConfigError(f"workload {workload.name!r} cannot need itself")

    section = raw.get("charriot", {})
    reserve = section.get("reserve_cores", 0)
    if not isinstance(reserve, int) or reserve < 0:
        raise ConfigError("charriot.reserve_cores must be a non-negative integer")

    return Config(
        workloads=workloads,
        reserve_cores=reserve,
        partition=bool(section.get("partition", True)),
    )


def load(path: str | Path) -> Config:
    try:
        text = Path(path).read_text()
    except OSError as error:
        raise ConfigError(f"cannot read {path}: {error}") from None
    return loads(text)
