#!/usr/bin/env python3
"""RunPod control CLI for a CPU pod (US region) — sample-rate probing job.

Provisions / inspects / tears down a single CPU pod via the RunPod REST API
(https://rest.runpod.io/v1). CPU pod (computeType=CPU), constrained to US data
centers, container disk only (no /workspace network volume). SSH via injected
PUBLIC_KEY. Reads RUNPOD_API_KEY from the environment or a .env (searched in this
dir, then parent repos). stdlib only.

Commands: launch | status | ssh | terminate   (pod metadata cached in pod.json)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
POD_JSON = HERE / "pod.json"
API_BASE = os.environ.get("RUNPOD_API_BASE", "https://rest.runpod.io/v1").rstrip("/")

# Stock RunPod image: its start.sh applies $PUBLIC_KEY + starts sshd (works on CPU
# pods too). Heavier than needed but proven for SSH; deps are installed on top.
DEFAULT_IMAGE = os.environ.get("RUNPOD_IMAGE", "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")
# CPU flavors to accept (priority=availability lets RunPod pick an available one).
# Broad list for availability; container disk capped at 160 GB (gen-3 limit) so all
# of them are valid (gen-5 allow up to 240 GB but are often unavailable).
DEFAULT_CPU_FLAVORS = ["cpu5c", "cpu3c", "cpu5g", "cpu3g"]
# US-only data centers (subset of the API's allowed list).
US_DATACENTERS = ["US-IL-1", "US-TX-3", "US-KS-2", "US-GA-2", "US-WA-1", "US-TX-1",
                  "US-TX-4", "US-CA-2", "US-NC-1", "US-DE-1", "US-KS-3", "US-GA-1", "US-MD-1"]


def load_dotenv() -> None:
    for cand in (HERE / ".env", HERE.parent / ".env", HERE.parent.parent / ".env",
                 Path("/home/husein/ssd3/Sidon/.env")):
        if cand.exists():
            for line in cand.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY missing (set it in .env or the environment)")
    return key


def api(method: str, path: str, body: dict | None = None) -> dict:
    url = API_BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {_api_key()}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"RunPod API {method} {path} failed: {e.code} {e.read().decode()[:500]}")
    except urllib.error.URLError as e:
        sys.exit(f"RunPod API {method} {path} network error: {e}")


def extract_ssh(pod: dict):
    public_ip = pod.get("publicIp") or pod.get("public_ip")
    pms = pod.get("portMappings")
    if isinstance(pms, dict):
        for k, v in pms.items():
            try:
                if int(k) == 22 and v:
                    return public_ip, int(v)
            except (TypeError, ValueError):
                continue
    runtime = pod.get("runtime") or {}
    for p in runtime.get("ports") or []:
        if isinstance(p, dict) and p.get("privatePort") == 22 and p.get("publicPort"):
            return p.get("ip") or public_ip, int(p["publicPort"])
    return public_ip, None


def read_pubkey(path: str) -> str:
    p = Path(os.path.expanduser(path))
    if not p.exists():
        sys.exit(f"SSH public key not found: {p}")
    return p.read_text().strip()


def save_pod(meta: dict) -> None:
    POD_JSON.write_text(json.dumps(meta, indent=2))


def load_pod() -> dict:
    if not POD_JSON.exists():
        sys.exit(f"no pod metadata at {POD_JSON} — run `launch` first")
    return json.loads(POD_JSON.read_text())


def cmd_launch(a: argparse.Namespace) -> None:
    body = {
        "name": a.name,
        "imageName": a.image,
        "computeType": "CPU",
        "cpuFlavorIds": DEFAULT_CPU_FLAVORS,
        "cpuFlavorPriority": "availability",
        "vcpuCount": a.vcpu,
        "cloudType": "SECURE",
        "dataCenterIds": US_DATACENTERS,
        "dataCenterPriority": "availability",
        "containerDiskInGb": a.disk_gb,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": read_pubkey(a.pubkey)},
    }
    print(f"[launch] CPU pod {a.name!r}: vcpu={a.vcpu}, flavors={DEFAULT_CPU_FLAVORS}, "
          f"US datacenters, {a.disk_gb}GB disk, image={a.image}")
    resp = api("POST", "/pods", body)
    pod_id = resp.get("id")
    if not pod_id:
        sys.exit(f"provision response missing id: {resp}")
    meta = {"pod_id": pod_id, "name": a.name, "cost_per_hr": resp.get("costPerHr"),
            "ssh_key": os.path.expanduser(a.pubkey).replace(".pub", "")}
    save_pod(meta)
    print(f"[launch] pod_id={pod_id} cost={resp.get('costPerHr')}/hr — waiting for RUNNING + SSH …")
    if not a.no_wait:
        wait_for_ssh(pod_id, meta, a.timeout)


def wait_for_ssh(pod_id: str, meta: dict, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = api("GET", f"/pods/{pod_id}")
        status = pod.get("desiredStatus") or pod.get("status")
        ip, port = extract_ssh(pod)
        if status == "RUNNING" and ip and port:
            meta.update({"ip": ip, "ssh_port": port,
                         "cost_per_hr": pod.get("costPerHr", meta.get("cost_per_hr"))})
            save_pod(meta)
            key = meta.get("ssh_key", "~/.ssh/id_rsa")
            print(f"[launch] RUNNING ip={ip} port={port}")
            print(f"[launch] ssh -p {port} -i {key} -o StrictHostKeyChecking=no "
                  f"-o UserKnownHostsFile=/dev/null root@{ip}")
            return
        print(f"  … status={status} ip={ip} port={port} ({int(deadline - time.time())}s left)")
        time.sleep(10)
    sys.exit("[launch] timed out waiting for RUNNING + SSH")


def cmd_status(a: argparse.Namespace) -> None:
    pod_id = a.pod_id or load_pod()["pod_id"]
    pod = api("GET", f"/pods/{pod_id}")
    ip, port = extract_ssh(pod)
    print(json.dumps({"pod_id": pod_id, "name": pod.get("name"),
                      "desiredStatus": pod.get("desiredStatus"), "ip": ip, "ssh_port": port,
                      "costPerHr": pod.get("costPerHr"),
                      "machine": (pod.get("machine") or {}).get("dataCenterId")}, indent=2))


def cmd_ssh(a: argparse.Namespace) -> None:
    meta = load_pod()
    pod = api("GET", f"/pods/{a.pod_id or meta['pod_id']}")
    ip, port = extract_ssh(pod)
    if not (ip and port):
        sys.exit("SSH endpoint not ready")
    print(f"ssh -p {port} -i {meta.get('ssh_key', '~/.ssh/id_rsa')} "
          f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@{ip}")


def cmd_terminate(a: argparse.Namespace) -> None:
    pod_id = a.pod_id or load_pod()["pod_id"]
    api("DELETE", f"/pods/{pod_id}")
    print(f"[terminate] pod {pod_id} deleted")
    if POD_JSON.exists() and not a.pod_id:
        POD_JSON.unlink()


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--name", default="sr-probe-cpu")
    p.add_argument("--vcpu", type=int, default=16)
    p.add_argument("--disk-gb", type=int, default=60)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--pubkey", default="~/.ssh/id_rsa.pub")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--no-wait", action="store_true")
    p.set_defaults(func=cmd_launch)
    for name, fn in (("status", cmd_status), ("ssh", cmd_ssh), ("terminate", cmd_terminate)):
        q = sub.add_parser(name)
        q.add_argument("--pod-id", default=None)
        q.set_defaults(func=fn)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
