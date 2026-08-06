"""Inspect secure containers, data/key mounts, limits, and image contents."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "phase5"
COMPOSE = PHASE5 / "docker-compose.yml"


def docker_executable() -> str:
    found = shutil.which("docker")
    if found:
        return found
    fallback = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/DockerDesktop/resources/bin/docker.exe"
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("Docker CLI not found")


def run(*arguments: str) -> str:
    docker = docker_executable()
    environment = os.environ.copy()
    environment["PATH"] = str(Path(docker).parent) + os.pathsep + environment.get("PATH", "")
    return subprocess.run(
        [docker, *arguments], check=True, capture_output=True, text=True, env=environment
    ).stdout.strip()


def memory_bytes(value: str) -> int:
    return int(float(value[:-1]) * {"m": 1024**2, "g": 1024**3}[value[-1].lower()])


def main() -> None:
    demo = json.loads((PHASE5 / "config/demo_config.json").read_text(encoding="utf-8"))
    expected = {item["client_id"]: item for item in demo["clients"]}
    ids = run("compose", "-f", str(COMPOSE), "ps", "-a", "-q").splitlines()
    assert len(ids) == 6
    containers = json.loads(run("inspect", *ids))
    services = {container["Config"]["Labels"]["com.docker.compose.service"]: container for container in containers}
    assert set(services) == {"control-center", *expected}

    checks = {}
    for service, container in services.items():
        host = container["HostConfig"]
        assert host["ReadonlyRootfs"] is True
        assert "ALL" in host["CapDrop"]
        assert "no-new-privileges:true" in host["SecurityOpt"]
        mounts = {mount["Destination"]: mount for mount in container["Mounts"]}
        assert mounts["/security/trust_store.json"]["RW"] is False
        assert mounts["/security/sign_secret.key"]["RW"] is False
        secret_source = Path(mounts["/security/sign_secret.key"]["Source"])
        if service == "control-center":
            assert secret_source.parts[-2:] == ("server", "sign_secret.key")
            assert "/data/train.csv" not in mounts
            assert mounts["/data/validation.csv"]["RW"] is False
            expected_cpus = float(demo["server"]["cpus"])
            expected_memory = memory_bytes(demo["server"]["memory"])
            assert container["State"]["Running"] is True
            assert container["State"]["Health"]["Status"] == "healthy"
        else:
            record = expected[service]
            assert secret_source.parts[-3:] == ("clients", service, "sign_secret.key")
            assert mounts["/data/train.csv"]["RW"] is False
            assert Path(mounts["/data/train.csv"]["Source"]).name == record["partition_filename"]
            assert "/data/validation.csv" not in mounts
            expected_cpus = float(record["cpus"])
            expected_memory = memory_bytes(record["memory"])
            assert container["State"]["Status"] == "exited"
            assert container["State"]["ExitCode"] == 0
        assert host["NanoCpus"] == int(expected_cpus * 1_000_000_000)
        assert host["Memory"] == expected_memory
        checks[service] = {
            "read_only_root": True,
            "cpus": expected_cpus,
            "memory_bytes": expected_memory,
            "data_mounts": sorted(path for path in mounts if path.startswith("/data/")),
            "secret_mounts": ["/security/sign_secret.key"],
            "exit_code": container["State"].get("ExitCode"),
        }

    server_id = services["control-center"]["Id"]
    embedded = run(
        "exec",
        server_id,
        "python",
        "-c",
        "import pathlib; print(sum(1 for p in pathlib.Path('/app').rglob('*') if p.suffix in {'.csv','.key','.pem'}))",
    )
    assert embedded == "0"
    result = {
        "verified": True,
        "services": checks,
        "server_training_partitions_visible": 0,
        "client_training_partitions_visible_each": 1,
        "private_keys_visible_per_service": 1,
        "sensitive_files_embedded_in_image": 0,
    }
    output = PHASE5 / "runtime" / "latest_isolation_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
