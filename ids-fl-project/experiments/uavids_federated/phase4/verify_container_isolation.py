"""Inspect the live Compose deployment for mounts, limits, and exit states."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT / "phase4"
COMPOSE = PHASE4 / "docker-compose.yml"


def docker_executable() -> str:
    found = shutil.which("docker")
    if found:
        return found
    fallback = Path.home() / "AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe"
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("Docker CLI not found")


def run(*arguments: str) -> str:
    docker = docker_executable()
    environment = os.environ.copy()
    environment["PATH"] = str(Path(docker).parent) + os.pathsep + environment.get("PATH", "")
    completed = subprocess.run(
        [docker, *arguments], check=True, capture_output=True, text=True, env=environment
    )
    return completed.stdout.strip()


def memory_bytes(value: str) -> int:
    units = {"m": 1024**2, "g": 1024**3}
    return int(float(value[:-1]) * units[value[-1].lower()])


def main() -> None:
    demo = json.loads((PHASE4 / "config/demo_config.json").read_text(encoding="utf-8"))
    expected = {item["client_id"]: item for item in demo["clients"]}
    ids = run("compose", "-f", str(COMPOSE), "ps", "-a", "-q").splitlines()
    assert len(ids) == 6
    containers = json.loads(run("inspect", *ids))
    services = {
        container["Config"]["Labels"]["com.docker.compose.service"]: container
        for container in containers
    }
    assert set(services) == {"control-center", *expected}

    checks = {}
    for service, container in services.items():
        host = container["HostConfig"]
        assert host["ReadonlyRootfs"] is True
        assert "ALL" in host["CapDrop"]
        assert "no-new-privileges:true" in host["SecurityOpt"]
        mounts = {mount["Destination"]: mount for mount in container["Mounts"]}
        if service == "control-center":
            assert "/data/train.csv" not in mounts
            assert mounts["/data/validation.csv"]["RW"] is False
            assert all("train_uav_client_" not in mount["Source"] for mount in mounts.values())
            expected_cpus = float(demo["server"]["cpus"])
            expected_memory = memory_bytes(demo["server"]["memory"])
            assert container["State"]["Running"] is True
            assert container["State"]["Health"]["Status"] == "healthy"
        else:
            record = expected[service]
            assert mounts["/data/train.csv"]["RW"] is False
            assert Path(mounts["/data/train.csv"]["Source"]).name == record["partition_filename"]
            data_mounts = [destination for destination in mounts if destination.startswith("/data/")]
            assert data_mounts == ["/data/train.csv"]
            assert all(mount["RW"] is False for mount in mounts.values())
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
            "data_mounts": sorted(destination for destination in mounts if destination.startswith("/data/")),
            "exit_code": container["State"].get("ExitCode"),
        }

    server_id = services["control-center"]["Id"]
    embedded_csv = run("exec", server_id, "python", "-c", "import pathlib; print(len(list(pathlib.Path('/app').rglob('*.csv'))))")
    assert embedded_csv == "0"
    result = {
        "verified": True,
        "services": checks,
        "server_training_partitions_visible": 0,
        "client_training_partitions_visible_each": 1,
        "training_csv_files_embedded_in_image": 0,
    }
    output = PHASE4 / "runtime" / "latest_isolation_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
