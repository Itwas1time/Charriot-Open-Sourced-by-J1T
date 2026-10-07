"""Launch workloads under a plan, bind them together, and tear them down.

Ordering is derived from ``needs``. A workload that provides an endpoint is
started and proven ready before anything that depends on it, and the endpoint
is injected into the dependant's environment.

There is no automatic restart loop. ``restart_max`` defaults to zero, and a
workload that exhausts its budget stays down and is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.error import URLError
from urllib.request import urlopen
import os
import signal
import subprocess
import time

from .apply import child_preexec, environment, in_cgroup, make_ceiling, prepare_cgroups, verify
from .config import Config, Workload
from .plan import Assignment, Plan


@dataclass
class Running:
    workload: Workload
    assignment: Assignment
    process: subprocess.Popen
    injected: dict[str, str] = field(default_factory=dict)
    restarts: int = 0
    notes: list[str] = field(default_factory=list)


class RunError(RuntimeError):
    pass


def order(config: Config) -> tuple[Workload, ...]:
    """Topologically order workloads by their ``needs`` edges."""
    remaining = list(config.workloads)
    placed: list[Workload] = []
    names: set[str] = set()
    while remaining:
        progressed = False
        for workload in list(remaining):
            if workload.needs is None or workload.needs in names:
                placed.append(workload)
                names.add(workload.name)
                remaining.remove(workload)
                progressed = True
        if not progressed:
            cycle = ", ".join(w.name for w in remaining)
            raise RunError(f"circular needs between: {cycle}")
    return tuple(placed)


def wait_ready(url: str, timeout: float, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout
    last = "no response"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RunError(f"exited with status {process.returncode} before becoming ready")
        try:
            with urlopen(url, timeout=2) as response:  # noqa: S310 - operator-supplied
                if 200 <= response.status < 300:
                    return
                last = f"HTTP {response.status}"
        except URLError as error:
            last = str(error.reason)
        except OSError as error:
            last = str(error)
        time.sleep(0.25)
    raise RunError(f"not ready within {timeout:g}s: {last}")


def _launch(workload: Workload, assignment: Assignment, extra_env: dict[str, str], log) -> Running:
    env = environment(assignment)
    env.update(extra_env)

    # The ceiling is created before the process exists so the child can join it
    # from preexec and is bounded from its first instruction.
    cgroup, detail = make_ceiling(workload.name, assignment.memory_max)
    if assignment.memory_max:
        log(f"  {workload.name}: {detail}")

    try:
        process = subprocess.Popen(
            list(workload.command),
            cwd=workload.cwd,
            env=env,
            # Its own session, so teardown can signal the whole process group.
            # Signalling only the direct child orphans anything it forked, and
            # an orphaned CPU-bound worker runs until the machine reboots.
            start_new_session=True,
            preexec_fn=child_preexec(assignment.cpus, cgroup),  # noqa: PLW1509 - intentional
        )
    except (OSError, ValueError) as error:
        raise RunError(f"{workload.name}: could not start: {error}") from None

    running = Running(
        workload=workload,
        assignment=assignment,
        process=process,
        injected=dict(extra_env),
    )

    ok, detail = verify(process.pid, assignment.cpus)
    running.notes.append(detail)
    log(f"  {workload.name}: pid {process.pid}, {detail}")
    if not ok:
        log(f"  {workload.name}: WARNING - {detail}")

    if cgroup is not None and not in_cgroup(cgroup, process.pid):
        log(f"  {workload.name}: WARNING - did not join its memory cgroup")
        running.notes.append("memory ceiling not applied")

    return running


def run(config: Config, plan_: Plan, *, log=lambda message: print(message, flush=True)) -> int:
    """Start everything, wait for the first exit, then stop the rest."""
    started: list[Running] = []
    endpoints: dict[str, str] = {}

    if any(a.memory_max for a in plan_.assignments):
        ok, detail = prepare_cgroups()
        if not ok:
            log(f"memory ceilings unavailable: {detail}")

    try:
        for workload in order(config):
            assignment = plan_.for_name(workload.name)
            if assignment is None:
                raise RunError(f"{workload.name}: no assignment in plan")

            extra: dict[str, str] = {}
            if workload.endpoint_env and workload.needs:
                endpoint = endpoints.get(workload.needs)
                if endpoint is None:
                    raise RunError(
                        f"{workload.name}: {workload.needs!r} provides no endpoint to inject"
                    )
                extra[workload.endpoint_env] = endpoint

            log(f"starting {workload.name}")
            running = _launch(workload, assignment, extra, log)
            started.append(running)

            if workload.ready_url:
                log(f"  {workload.name}: waiting for {workload.ready_url}")
                try:
                    wait_ready(workload.ready_url, workload.ready_timeout, running.process)
                except RunError as error:
                    raise RunError(f"{workload.name}: {error}") from None
                log(f"  {workload.name}: ready")

            if workload.provides_endpoint:
                endpoints[workload.name] = workload.provides_endpoint

        if not started:
            return 0

        log("all workloads running; ctrl-c to stop")
        return _supervise(started, log)
    finally:
        _teardown(started, log)


def _supervise(started: list[Running], log) -> int:
    """Wait until something exits, restarting only within its budget."""
    while True:
        try:
            time.sleep(0.4)
        except KeyboardInterrupt:
            log("interrupted")
            return 130

        for running in started:
            status = running.process.poll()
            if status is None:
                continue

            name = running.workload.name
            if status == 0:
                log(f"{name} exited cleanly")
                return 0

            if running.restarts >= running.workload.restart_max:
                log(f"{name} exited with status {status}; not restarting")
                return status

            running.restarts += 1
            log(
                f"{name} exited with status {status}; "
                f"restart {running.restarts}/{running.workload.restart_max}"
            )
            # Re-inject what it was started with. Passing an empty environment
            # here would silently strip an injected endpoint on restart, so the
            # replacement would come back unable to reach its provider.
            replacement = _launch(
                running.workload, running.assignment, running.injected, log
            )
            running.process = replacement.process


def _signal_group(process: subprocess.Popen, sig: int) -> None:
    """Signal the whole process group, falling back to the child alone.

    Each workload gets its own session, so its group id is its pid. Anything it
    forked is in that group and is signalled with it.
    """
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (OSError, ProcessLookupError):
        try:
            process.send_signal(sig)
        except OSError:
            pass


def _teardown(started: list[Running], log) -> None:
    for running in reversed(started):
        process = running.process
        if process.poll() is not None:
            # Reaped, but anything it forked may still be running.
            _signal_group(process, signal.SIGKILL)
            continue
        name = running.workload.name
        log(f"stopping {name}")
        _signal_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log(f"  {name} did not stop; killing")
            _signal_group(process, signal.SIGKILL)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log(f"  {name} could not be killed")
        # Sweep the group even after a clean exit: the leader can exit while
        # its children keep running.
        _signal_group(process, signal.SIGKILL)
