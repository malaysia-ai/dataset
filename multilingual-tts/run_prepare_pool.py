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
def main(list_file, workers, gpus, workdir, limit, max_samples, cluster_threshold,
         job_timeout):
    if gpus is None:
        import torch
        gpus = [str(i) for i in range(torch.cuda.device_count())]
    else:
        gpus = [g.strip() for g in gpus.split(",") if g.strip()]
    workers = min(workers, len(gpus))
    buckets = split_gpus(gpus, workers)
    print(f"[pool] {workers} workers, gpu buckets: {buckets}")

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

    free = list(buckets)
    running = {}   # popen -> (bucket, name, t0, logpath)
    done = failed = 0
    total = len(queue)

    def launch(repo, name, bucket):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(bucket))
        cmd = [sys.executable, os.path.join(HERE, "prepare.py"),
               "--repo", repo, "--name", name, "--workdir", workdir,
               "--cluster-threshold", str(cluster_threshold)]
        if max_samples:
            cmd += ["--max-samples", str(max_samples)]
        logpath = os.path.join(logdir, f"{name}.log")
        lf = open(logpath, "w")
        # own session/process-group so a timeout can kill the whole job tree
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                             start_new_session=True)
        running[p] = (bucket, name, time.time(), logpath, lf)
        print(f"[pool] START {name}  gpus={','.join(bucket)}  (repo={repo})")

    while queue or running:
        while queue and free:
            repo, name = queue.popleft()
            launch(repo, name, free.pop())
        for p in list(running):
            bucket, name, t0, logpath, lf = running[p]
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
            free.append(bucket)
            ok = os.path.exists(os.path.join(ckpt, f"{name}.done"))
            done += ok
            failed += (not ok)
            dt = time.time() - t0
            tag = "TIMEOUT" if timed_out else ("DONE" if ok else "FAIL")
            print(f"[pool] {tag} {name} in {dt:.0f}s (rc={p.returncode})  "
                  f"[{done} ok / {failed} fail / {total} total]")
            if not ok:
                print(f"       see {logpath}")
        time.sleep(2)

    print(f"[pool] FINISHED: {done} ok, {failed} failed of {total}")


if __name__ == "__main__":
    main()
