#!/usr/bin/env python3
"""bar charts for the good vs bad kernels: time, and ipc + misses when counters were on"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def bar_panel(ax, df, col, title, fmt="{:.1f}"):
    labels = [f"{k}\n{v}" for k, v in zip(df["kernel"], df["variant"])]
    bars = ax.bar(range(len(df)), df[col], color="tab:blue")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, df[col]):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--outdir", default="results")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.outdir, exist_ok=True)
    has_counters = df["cycles"].sum() > 0

    # one row per kernel, time normalized to the fastest variant so they share a scale
    kernels = list(dict.fromkeys(df["kernel"]))
    fig, axes = plt.subplots(1, len(kernels), figsize=(4 * len(kernels), 4))
    if len(kernels) == 1:
        axes = [axes]
    for ax, k in zip(axes, kernels):
        sub = df[df["kernel"] == k].copy()
        sub["rel"] = sub["ms"] / sub["ms"].min()
        bar_panel(ax, sub, "rel", f"{k}: time vs fastest", "{:.1f}x")
    fig.tight_layout()
    path = os.path.join(args.outdir, "kernels_time.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")

    if has_counters:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        bar_panel(axes[0], df, "ipc", "instructions per cycle", "{:.2f}")
        df = df.assign(l1d_m=df["l1d_misses"] / 1e6, br_m=df["branch_misses"] / 1e6)
        bar_panel(axes[1], df, "l1d_m", "l1d misses (millions)")
        bar_panel(axes[2], df, "br_m", "branch misses (millions)")
        fig.tight_layout()
        path = os.path.join(args.outdir, "kernels_counters.png")
        fig.savefig(path, dpi=150)
        print(f"wrote {path}")
    else:
        print("no counter data in this csv, skipped the counters chart")


if __name__ == "__main__":
    main()
