# -*- coding: utf-8 -*-
"""Parallel download of AD2 'can' category parts from ModelScope, then join."""
import concurrent.futures as cf
import os
import subprocess
import sys

REPO = "AIinspect/MVTec_AD_2"
OUT = r"E:\work\freshman\data\mvtec_ad2_can_parts"
N_PARTS = 86  # part_0000 .. part_0085
BASE = f"https://www.modelscope.cn/api/v1/datasets/{REPO}/repo?Revision=master&FilePath=can.tar.gz.parts/part_"

os.makedirs(OUT, exist_ok=True)

def fetch(i):
    name = f"part_{i:04d}"
    dst = os.path.join(OUT, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 30_000_000:
        return name, True
    url = BASE + f"{i:04d}"
    r = subprocess.run(["curl.exe", "-L", "-s", "--retry", "3", "-o", dst, url],
                       capture_output=True, timeout=600)
    ok = r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 1_000_000
    return name, ok

with cf.ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(fetch, range(N_PARTS)))

ok = sum(1 for _, s in results if s)
print(f"downloaded {ok}/{N_PARTS} parts")
if ok == N_PARTS:
    # join parts into can.tar.gz
    out_file = r"E:\work\freshman\data\mvtec_ad2_can.tar.gz"
    with open(out_file, "wb") as out:
        for i in range(N_PARTS):
            with open(os.path.join(OUT, f"part_{i:04d}"), "rb") as fh:
                out.write(fh.read())
    print("joined ->", out_file, os.path.getsize(out_file) / 1e9, "GB")
else:
    print("MISSING:", [n for n, s in results if not s][:10])
    sys.exit(1)
