#!/usr/bin/env python3
"""turns the latency csv into a png and draws lines where /sys says L1/L2/L3 end"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_size(s: str) -> int:
    s = s.strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return int(s[:-1]) * mult[s[-1]] if s[-1] in mult else int(s)


def sys_caches(cpu=0):
    caches = {}
    for d in sorted(glob.glob(f"/sys/devices/system/cpu/cpu{cpu}/cache/index*")):
        typ = open(f"{d}/type").read().strip()
        if typ == "Instruction":
            continue
        level = int(open(f"{d}/level").read())
        caches[f"L{level}"] = parse_size(open(f"{d}/size").read())
    return caches


def human(b):
    for unit, n in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if b >= n:
            return f"{b / n:g}{unit}"
    return f"{b}B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="results/latency.png")
    ap.add_argument("--title", default="Memory latency vs working-set size")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["bytes"], df["ns_per_access"], marker="o", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Working-set size")
    ax.set_ylabel("ns per access (median)")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.3)

    ticks = [b for b in df["bytes"] if (b & (b - 1)) == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([human(t) for t in ticks], rotation=45)

    colors = {"L1": "tab:green", "L2": "tab:orange", "L3": "tab:red"}
    for name, size in sys_caches().items():
        ax.axvline(size, linestyle="--", color=colors.get(name, "gray"),
                   label=f"{name} = {human(size)} (/sys)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
