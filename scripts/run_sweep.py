#!/usr/bin/env python3
"""runs every benchmark and drops the csvs plus a meta.json into results/<tag>/"""
import argparse
import json
import os
import platform
import subprocess
import sys


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def cpu_model():
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def compiler_from_cache(build):
    cache = os.path.join(build, "CMakeCache.txt")
    if not os.path.exists(cache):
        return "unknown"
    for line in open(cache):
        if line.startswith("CMAKE_CXX_COMPILER:"):
            path = line.split("=", 1)[1].strip()
            return sh(f"{path} --version").splitlines()[0]
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wsl", help="folder name under results/, like wsl or native")
    ap.add_argument("--build", default="build")
    ap.add_argument("--cpu", default="2")
    ap.add_argument("--cpus", default="2,4")
    ap.add_argument("--quick", action="store_true", help="tiny sizes, finishes in seconds")
    ap.add_argument("--counters", action="store_true")
    args = ap.parse_args()

    exe = os.path.join(args.build, "cpuprobe")
    if not os.path.exists(exe):
        sys.exit(f"cant find {exe}, build first")

    out = os.path.join("results", args.tag)
    os.makedirs(out, exist_ok=True)

    meta = {
        "cpu": cpu_model(),
        "kernel": platform.release(),
        "compiler": compiler_from_cache(args.build),
        "git_commit": sh("git rev-parse --short HEAD"),
        "governor": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or "n/a",
        "perf_event_paranoid": sh("cat /proc/sys/kernel/perf_event_paranoid"),
        "quick": args.quick,
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

    quick = ["--quick"] if args.quick else []
    counters = ["--counters"] if args.counters else []

    runs = [
        [exe, "latency", "--cpu", args.cpu, "--csv", os.path.join(out, "latency.csv"), *quick],
        [exe, "kernel", "all", "--cpu", args.cpu, "--cpus", args.cpus,
         "--csv", os.path.join(out, "kernels.csv"), *quick, *counters],
    ]
    for cmd in runs:
        print("\n$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    print(f"\ndone, everything is in {out}/")


if __name__ == "__main__":
    main()
