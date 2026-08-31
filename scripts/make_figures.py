"""Figures for REPORT.md, mirroring the paper's own.

Fig 2a  playing strength against move-prediction accuracy (claim C1)
Fig 2b  evaluation MSE by stage of game, value net vs rollouts (claim C4)
Fig 4b  Elo of the component ablations (claim C5)
plus    the C3 train/test gap for the two value-net data schemes
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PALETTE = ["#2f6fb2", "#c0562f", "#4a9e5c", "#8a6bbf", "#b9982f", "#777777"]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_c1(runs, out):
    """Accuracy vs strength, the paper's Fig. 2a."""
    path = os.path.join(runs, "c1_strength.json")
    if not os.path.exists(path):
        print("[fig] no c1_strength.json, skipping Fig 2a")
        return
    d = json.load(open(path))
    pts = d["points"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for h in d.get("histories", []):
        axes[0].plot([x["step"] for x in h["history"]],
                     [100 * x["test_acc"] for x in h["history"]],
                     label=f"k={h['filters']}", linewidth=1.4)
    if d.get("ceiling"):
        axes[0].axhline(100 * d["ceiling"], color="#999", linestyle="--",
                        linewidth=1,
                        label=f"teacher self-agreement {100*d['ceiling']:.0f}%")
    _style(axes[0], "a  supervised learning curves", "SGD step",
           "test move-prediction accuracy (%)")
    axes[0].legend(fontsize=7, frameon=False)

    acc = [100 * p["test_acc"] for p in pts]
    wr = [100 * p["win_rate"] for p in pts]
    err = [[100 * (p["win_rate"] - p["ci"][0]) for p in pts],
           [100 * (p["ci"][1] - p["win_rate"]) for p in pts]]
    axes[1].errorbar(acc, wr, yerr=err, fmt="o", color=PALETTE[0],
                     capsize=3, markersize=6, linewidth=1)
    for p in pts:
        axes[1].annotate(p["label"], (100 * p["test_acc"], 100 * p["win_rate"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=7)
    axes[1].axhline(50, color="#999", linewidth=0.8, linestyle=":")
    _style(axes[1], "b  strength follows accuracy (C1)",
           "test move-prediction accuracy (%)",
           "win rate vs fixed reference (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_c1_accuracy_strength.png"), dpi=160)
    print("[fig] wrote fig_c1_accuracy_strength.png")


def fig_c3(runs, out):
    path = os.path.join(runs, "value_results.json")
    if not os.path.exists(path):
        print("[fig] no value_results.json, skipping C3")
        return
    d = json.load(open(path))["results"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for i, (name, r) in enumerate(d.items()):
        h = r["history"]
        axes[0].plot([x["step"] for x in h], [x["train_mse"] for x in h],
                     color=PALETTE[i], linewidth=1.4, label=f"{name} train")
        axes[0].plot([x["step"] for x in h], [x["test_mse"] for x in h],
                     color=PALETTE[i], linewidth=1.4, linestyle="--",
                     label=f"{name} test")
    axes[0].axhline(list(d.values())[0]["floor_common"], color="#999",
                    linestyle=":", linewidth=1, label="constant-predictor floor")
    _style(axes[0], "a  value-net MSE (C3)", "SGD step", "MSE")
    axes[0].legend(fontsize=7, frameon=False)

    names = list(d)
    gaps = [d[n]["best_gap"] for n in names]
    axes[1].bar(names, gaps, color=[PALETTE[0], PALETTE[1]], width=0.5)
    axes[1].axhline(0, color="#333", linewidth=0.8)
    for i, gname in enumerate(names):
        axes[1].annotate(f"{gaps[i]:+.3f}", (i, gaps[i]),
                         ha="center", va="bottom", fontsize=8)
    _style(axes[1], "b  test - train gap (lower = less memorised)",
           "training-data scheme", "MSE gap")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_c3_value_overfit.png"), dpi=160)
    print("[fig] wrote fig_c3_value_overfit.png")


def fig_c4(runs, out):
    path = os.path.join(runs, "c4_value_vs_rollouts.json")
    if not os.path.exists(path):
        print("[fig] no c4 json, skipping Fig 2b")
        return
    d = json.load(open(path))
    rows = d["by_bin"]
    if not rows:
        return
    x = [0.5 * (r["lo"] + r["hi"]) for r in rows]
    keys = [k for k in d["overall"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for i, k in enumerate(keys):
        ax.plot(x, [r[k] for r in rows], marker="o", markersize=4,
                color=PALETTE[i % len(PALETTE)], linewidth=1.4,
                label=k.replace("rollout_", "rollouts: ").replace("value_net",
                                                                  "value net"))
    ax.plot(x, [r["floor"] for r in rows], color="#999", linestyle=":",
            linewidth=1.2, label="constant-predictor floor")
    _style(ax, "Fig 2b  evaluation accuracy by stage of game (C4)",
           "move number", "MSE against the true outcome")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_c4_value_vs_rollouts.png"), dpi=160)
    print("[fig] wrote fig_c4_value_vs_rollouts.png")


def fig_c5(runs, out):
    path = os.path.join(runs, "tourney", "elo.json")
    if not os.path.exists(path):
        print("[fig] no elo.json, skipping Fig 4b")
        return
    d = json.load(open(path))["elo"]
    order = sorted(d, key=lambda n: d[n][0])
    vals = [d[n][0] for n in order]
    lo = [d[n][0] - d[n][1] for n in order]
    hi = [d[n][2] - d[n][0] for n in order]
    colours = ["#c0562f" if n in ("a_rvp",) else
               "#2f6fb2" if n.startswith("a_") else "#888888" for n in order]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(order, vals, xerr=[lo, hi], color=colours, height=0.6,
            error_kw=dict(lw=1, capsize=3, ecolor="#444"))
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}", (v, i), textcoords="offset points",
                    xytext=(6, -3), fontsize=8)
    _style(ax, "Elo of the component ablations (C5, C6, C7)",
           "Elo (random play anchored at 0)", "")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_c5_elo.png"), dpi=160)
    print("[fig] wrote fig_c5_elo.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/content/runs")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_c1(a.runs, a.out)
    fig_c3(a.runs, a.out)
    fig_c4(a.runs, a.out)
    fig_c5(a.runs, a.out)


if __name__ == "__main__":
    main()
