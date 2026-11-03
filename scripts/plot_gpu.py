#!/usr/bin/env python3
"""charts for the gpu side.

  plot_gpu.py latency results/gpu/latency.csv -o results/gpu/latency.png
  plot_gpu.py bench   results/gpu/bench.csv   -o results/gpu
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def read_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            k, _, v = line[1:].strip().partition("=")
            meta[k.strip()] = v.strip()
    return meta


def human(b):
    for unit, n in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if b >= n:
            return f"{b / n:g}{unit}"
    return f"{b}B"


def plot_latency(csv, out):
    meta = read_meta(csv)
    df = pd.read_csv(csv, comment="#")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["bytes"], df["ns_per_access"], marker="o", linewidth=2, color="tab:green")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Working-set size")
    ax.set_ylabel("ns per access (median)")
    ax.set_title(f"GPU memory latency, single-thread pointer chase ({meta.get('gpu', 'GPU')})")
    ax.grid(True, which="both", alpha=0.3)
    ticks = [b for b in df["bytes"] if (b & (b - 1)) == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([human(t) for t in ticks], rotation=45)
    if "l2_bytes" in meta:
        l2 = int(meta["l2_bytes"])
        ax.axvline(l2, linestyle="--", color="tab:red", label=f"L2 = {human(l2)} (cudaDeviceProp)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def plot_bench(csv, outdir):
    meta = read_meta(csv)
    df = pd.read_csv(csv, comment="#")
    os.makedirs(outdir, exist_ok=True)
    for bench, col, unit in (("transpose", "gbps", "GB/s"), ("sgemm", "gflops", "GFLOP/s")):
        d = df[df["benchmark"] == bench]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(d["variant"], d[col], color="tab:green")
        ax.bar_label(bars, fmt="%.0f")
        ax.set_ylabel(unit)
        n = int(d["size"].iloc[0])
        ax.set_title(f"{bench} {n}x{n} fp32 on {meta.get('gpu', 'GPU')} (median of {int(d['reps'].iloc[0])})")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out = os.path.join(outdir, f"{bench}.png")
        fig.savefig(out, dpi=150)
        print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["latency", "bench"])
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    if a.kind == "latency":
        plot_latency(a.csv, a.out)
    else:
        plot_bench(a.csv, a.out)


if __name__ == "__main__":
    main()
