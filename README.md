# Charriot

Charriot launches latency-sensitive and throughput-heavy workloads on separate
physical CPU cores and applies available memory and GPU bounds. It is for
personal Linux machines; Python 3.11 or newer, Linux CPU-affinity support,
and cgroup v2 are required for its full controls.

## Install

```sh
git clone https://github.com/Itwas1time/Charriot.git
cd Charriot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Configure and run

Start with `charriot.example.toml`. Put your own commands, environment values,
and paths in a private local copy. Commands are argument lists.

```sh
charriot topology
charriot doctor
charriot plan mine.toml
charriot run mine.toml
```

Inspect the plan before launching. For delegated memory controls, run inside
your own systemd user scope:

```sh
systemd-run --user --scope -- charriot run mine.toml
```

Charriot starts providers before their consumers, checks readiness, applies
CPU-affinity bounds, and supervises the configured processes. Missing host
capabilities are reported by `doctor`. GPU choices are communicated through
standard environment variables; application-specific limits belong in the
workload configuration. Restart budgets default to zero.

To measure partitioning on your own machine:

```sh
systemd-run --user --scope -- charriot bench --duration 60
```

This launches synthetic CPU workloads and can heavily load the machine.
Partitioning benefits depend on contention and hardware; performance is not
guaranteed. Use `charriot --help` for command options.

## Security

Charriot executes the programs you configure with your user privileges. CPU
and memory bounds are resource controls, not a security sandbox. Run only
trusted programs, keep private workload files out of Git, and avoid running
as root. Workload endpoints inherit the security of those applications;
review their bindings before allowing network access.
See [SECURITY.md](SECURITY.md). Software is licensed under [MIT](LICENSE).
