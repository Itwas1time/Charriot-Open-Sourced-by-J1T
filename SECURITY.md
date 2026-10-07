# Security and workload data

Charriot launches commands that you configure, with your user's privileges.
CPU affinity, GPU selection, and cgroup resource limits are not a security
sandbox. Use only workloads and configuration files you trust. Do not run
untrusted configurations or elevate the launcher merely to run an example.

Keep your real workload definitions, private paths, environment credentials,
logs, and measurement artifacts out of Git. Copy charriot.example.toml to
a local file such as workloads.local.toml and supply your own paths. Avoid
putting secrets directly in command arguments or publishing them in reports.

Provider endpoints in the example use loopback. Charriot does not add
authentication or encryption to an application's API. Keep local providers
on loopback; do not expose them through port forwarding or public tunnels.

Security reports should use GitHub private vulnerability reporting. Remove
private paths, workload details, and credentials from reports or public issues.
