#!/usr/bin/env python3
"""Process many source datasets through prepare.py in parallel, with GPU
partitioning so the concurrent jobs don't oversubscribe the GPUs.

Each job gets a disjoint slice of GPUs via CUDA_VISIBLE_DEVICES (embedding.py and
convert_neucodec.py both honor it). A free slice is handed to the next dataset as
soon as a job finishes (dynamic scheduling). Datasets with a checkpoint are
skipped; each prepare.py self-cleans its local artifacts after pushing.

Usage:
  python3 run_prepare_pool.py --list tasks.json --workers 5 --gpus 0,1,2,3,4,5,6,7 \
      --workdir /share [--limit N] [--max-samples M]

tasks.json: [{"repo": "owner/name", "name": "ConfigName"}, ...]
(name optional -> derived from repo's last path segment)
"""
import os, sys, re, json, time, signal, subprocess
from collections import deque
import click

HERE = os.path.dirname(os.path.abspath(__file__))


def slug(repo):
    return re.sub(r"[^0-9A-Za-z_.-]", "_", repo.split("/")[-1])


def split_gpus(gpus, n):
    """Split gpu id list into n near-even contiguous buckets."""
    k, m = divmod(len(gpus), n)
    out, i = [], 0
    for w in range(n):
        sz = k + (1 if w < m else 0)
        out.append(gpus[i:i + sz])
        i += sz
    return [b for b in out if b]  # drop empty buckets if n > len(gpus)


def gpu_free_mb(ids):
    """{gpu_id: free_MiB} for the given gpu ids, via nvidia-smi (co-tenant aware)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"], text=True, timeout=30)
    except Exception:
        return {}
    free = {}
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0] in ids:
            try:
                free[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return free


def record_failure(workdir, name, repo, gpu, rc, timed_out, logpath):
    """Append a failure record to <workdir>/failures.jsonl for later reprocessing,
    flagging OOM specifically."""
    reason, oom = "unknown", False
    try:
        tail = open(logpath, errors="ignore").read()[-6000:]
        low = tail.lower()
        oom = ("out of memory" in low) or ("cufft_internal" in low) or ("cuda error" in low)
        fl = [l for l in tail.splitlines() if l.startswith("[fail]")]
        if fl:
            reason = fl[-1][:200]
        elif timed_out:
            reason = "timeout"
        elif oom:
            reason = "oom"
    except Exception:
        pass
    rec = {"name": name, "repo": repo, "gpu": gpu, "rc": rc,
           "timed_out": timed_out, "oom": oom, "reason": reason, "t": time.time()}
    try:
        with open(os.path.join(workdir, "failures.jsonl"), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


@click.command()
@click.option("--list", "list_file", required=True, help="tasks json [{repo,name}]")
@click.option("--workers", default=5, type=int)
@click.option("--gpus", default=None, help="comma gpu ids (default: all visible)")
@click.option("--workdir", default="/share")
@click.option("--limit", default=None, type=int, help="cap number of datasets")
@click.option("--max-samples", default=None, type=int, help="passthrough (testing)")
@click.option("--cluster-threshold", default=0.1, type=float)
@click.option("--job-timeout", default=5400, type=int,
              help="kill a single dataset job after this many seconds (hang guard)")
@click.option("--min-free-gb", default=30.0, type=float,
              help="only schedule a job onto a GPU with at least this much free VRAM")
@click.option("--extract-cores", default=16, type=int,
              help="parallel mp3-extract worker processes per dataset job")
def main(list_file, workers, gpus, workdir, limit, max_samples, cluster_threshold,
         job_timeout, min_free_gb, extract_cores):
    if gpus is None:
        import torch
        gpus = [str(i) for i in range(torch.cuda.device_count())]
    else:
        gpus = [g.strip() for g in gpus.split(",") if g.strip()]
    max_concurrent = min(workers, len(gpus)) if workers else len(gpus)
    min_free_mb = int(min_free_gb * 1024)
    print(f"[pool] up to {max_concurrent} concurrent | candidate gpus={gpus} | "
          f"min_free={min_free_gb}GB | DYNAMIC gpu assignment (co-tenant aware)")

    tasks = json.load(open(list_file))
    if limit:
        tasks = tasks[:limit]
    queue = deque()
    ckpt = os.path.join(workdir, "checkpoints")
    os.makedirs(ckpt, exist_ok=True)
    logdir = os.path.join(workdir, "logs")
    os.makedirs(logdir, exist_ok=True)
    skipped = 0
    for t in tasks:
        repo = t["repo"]
        name = t.get("name") or slug(repo)
        if os.path.exists(os.path.join(ckpt, f"{name}.done")):
            skipped += 1
            continue
        queue.append((repo, name))
    print(f"[pool] {len(queue)} to process, {skipped} already done")

    busy = set()   # gpu ids currently assigned to one of my running jobs
    running = {}   # popen -> (gpu, name, t0, logpath, lf)
    done = failed = 0
    total = len(queue)

    def launch(repo, name, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        cmd = [sys.executable, "-u", os.path.join(HERE, "prepare.py"),
               "--repo", repo, "--name", name, "--workdir", workdir,
               "--cluster-threshold", str(cluster_threshold),
               "--extract-cores", str(extract_cores)]
        if max_samples:
            cmd += ["--max-samples", str(max_samples)]
        logpath = os.path.join(logdir, f"{name}.log")
        lf = open(logpath, "w")
        # own session/process-group so a timeout can kill the whole job tree
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                             start_new_session=True)
        running[p] = (gpu, name, time.time(), logpath, lf, repo)
        busy.add(gpu)
        print(f"[pool] START {name}  gpu={gpu}  (repo={repo})")

    def pick_gpu():
        """A GPU not already running my job, with >= min_free_mb free now."""
        free = gpu_free_mb(gpus)
        cand = sorted(((mb, g) for g, mb in free.items()
                       if g not in busy and mb >= min_free_mb), reverse=True)
        return cand[0][1] if cand else None

    while queue or running:
        while queue and len(running) < max_concurrent:
            gpu = pick_gpu()
            if gpu is None:
                break  # no GPU free enough right now; try again next tick
            repo, name = queue.popleft()
            launch(repo, name, gpu)
        for p in list(running):
            gpu, name, t0, logpath, lf, repo = running[p]
            rc = p.poll()
            timed_out = False
            if rc is None:
                if time.time() - t0 <= job_timeout:
                    continue
                timed_out = True
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
                try:
                    p.wait(timeout=15)
                except Exception:
                    pass
            running.pop(p)
            lf.close()
            busy.discard(gpu)
            ok = os.path.exists(os.path.join(ckpt, f"{name}.done"))
            done += ok
            failed += (not ok)
            dt = time.time() - t0
            tag = "TIMEOUT" if timed_out else ("DONE" if ok else "FAIL")
            print(f"[pool] {tag} {name} in {dt:.0f}s (rc={p.returncode})  "
                  f"[{done} ok / {failed} fail / {total} total]")
            if not ok:
                print(f"       see {logpath}")
                record_failure(workdir, name, repo, gpu, p.returncode, timed_out, logpath)
        time.sleep(3)

    print(f"[pool] FINISHED: {done} ok, {failed} failed of {total}")


if __name__ == "__main__":
    main()
