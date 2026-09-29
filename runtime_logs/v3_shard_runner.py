# -*- coding: utf-8 -*-
"""
OS-level shard runner for the V3 confirmatory run.

NOT frozen code and NOT part of the experiment: it only calls the frozen
`v3_cellrunner.py` once per cell, in a fresh subprocess, exactly like the bash
loops it replaces. Same commit, same manifest, same shards, same output root --
only the thing that keeps the process alive changes (Windows Task Scheduler
instead of the Claude Code background task manager).

Lives in runtime_logs/ (gitignored) so the repo stays clean.

Usage: python runtime_logs/v3_shard_runner.py <shard 0|1>
"""
import os
import subprocess
import sys
import time

ROOT = r"E:\work\freshman"
PY = sys.executable


def main():
    shard = sys.argv[1]
    os.chdir(ROOT)
    lst = os.path.join(ROOT, "runtime_logs", f"worker{shard}_cells.txt")
    cells = [c.strip() for c in open(lst, encoding="utf-8") if c.strip()]
    logpath = os.path.join(ROOT, "runtime_logs", f"w{shard}_os.log")
    log = open(logpath, "a", encoding="utf-8", buffering=1)
    print(f"[w{shard}] OS runner start {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"cells={len(cells)} pid={os.getpid()}", file=log)
    done = 0
    for i, c in enumerate(cells, 1):
        t0 = time.time()
        p = subprocess.run(
            [PY, os.path.join("experiments", "model_v0", "v3_cellrunner.py"),
             "--cell", c], capture_output=True, text=True)
        out = [l for l in (p.stdout or "").splitlines() if l.strip()]
        last = out[-1].strip() if out else "(no stdout)"
        el = time.time() - t0
        if "done (ok)" in last:
            done += 1
        print(f"[w{shard} {i}/{len(cells)}] {c} :: {last}  [{el:.0f}s]",
              file=log)
        if p.returncode != 0:
            print(f"[w{shard}] rc={p.returncode} stderr tail: "
                  f"{(p.stderr or '')[-400:]}", file=log)
    print(f"[w{shard}] OS runner END {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"newly_done={done}", file=log)
    log.close()


if __name__ == "__main__":
    main()
