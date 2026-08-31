"""Claim C1: does move-prediction accuracy translate into playing strength?

    "Plot showing the playing strength of policy networks as a function of their
     training accuracy.  Policy networks with 128, 192, 256 and 384
     convolutional filters per layer were evaluated periodically during
     training; the plot shows the winning rate of AlphaGo using that policy
     network against the match version of AlphaGo."  (Fig. 2a)

    "Small improvements in accuracy led to large improvements in playing
     strength."

The paper varies width and reads accuracy off during training.  Same here, with
k = 32/64/128 instead of 128/192/256/384, and each checkpoint played **with no
search at all** against one fixed, non-learning reference opponent.  Using a
fixed reference rather than "the match version" matters: the paper's y-axis
moves as its own reference improves, whereas a frozen opponent gives a scale
that means the same thing at every point.

The accuracy axis is reported against the teacher's self-agreement, measured by
check_eval.py.  Without that ceiling "we reached X%" is not interpretable --
the teacher is stochastic, so no policy can predict it perfectly.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ag import nets
from ag.arena import match
from ag.players import MCTSPlayer, PolicyNetPlayer
from ag.rollout import RolloutPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/content/runs")
    ap.add_argument("--rollout", default="/content/runs/rollout.npz")
    ap.add_argument("--ref-sims", type=int, default=30)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--gate", default="/content/runs/eval_gate.json")
    ap.add_argument("--out", default="/content/runs/c1_strength.json")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rp = RolloutPolicy.load(a.rollout) if os.path.exists(a.rollout) \
        else RolloutPolicy()

    # One frozen opponent for every measurement, so the y-axis is a fixed ruler.
    ref = MCTSPlayer(name="ref", n_sims=a.ref_sims, lmbda=1.0, rollout=rp,
                     rng=np.random.default_rng(99))

    ceiling = None
    if os.path.exists(a.gate):
        g = json.load(open(a.gate))
        ceiling = g.get("teacher_self_agreement", {}).get("self_agreement")

    histories = []
    for hpath in sorted(glob.glob(os.path.join(a.runs, "sl_k*_history.json"))):
        h = json.load(open(hpath))
        histories.append(dict(filters=h["args"]["filters"], tag=h["tag"],
                              n_params=h["n_params"], history=h["history"]))

    points = []
    ckpts = sorted(glob.glob(os.path.join(a.runs, "sl_k*_step*.pt")))
    # Keep a spread: every checkpoint of every width would be hundreds of games.
    chosen = {}
    for c in ckpts:
        m = re.search(r"sl_k(\d+)_step(\d+)\.pt$", os.path.basename(c))
        if not m:
            continue
        chosen.setdefault(int(m.group(1)), []).append((int(m.group(2)), c))
    todo = []
    for k, lst in sorted(chosen.items()):
        lst.sort()
        pick = [lst[len(lst) // 4], lst[len(lst) // 2], lst[-1]] if len(lst) >= 4 \
            else lst
        for step, path in pick:
            todo.append((k, step, path))

    print(f"[C1] {len(todo)} checkpoints x {a.games} games vs "
          f"MCTS-{a.ref_sims} reference", flush=True)
    for k, step, path in todo:
        net, ck = nets.load(path, map_location=device)
        acc = ck["history"][-1]["test_acc"]
        ev = nets.PolicyEvaluator(net, device=device, symmetry="none")
        pl = PolicyNetPlayer(ev, name=f"k{k}s{step}", temperature=0.0)
        r = match(pl, ref, n_games=a.games)
        points.append(dict(label=f"k{k}", filters=k, step=step,
                           test_acc=acc, n_params=ck["n_params"],
                           win_rate=r["rate"], ci=list(r["ci"]),
                           games=r["games"], wins=r["wins"]))
        print(f"[C1] k={k:4d} step {step:6d}  acc {acc:.4f}  "
              f"win rate vs ref {r['rate']:5.1%} "
              f"CI[{r['ci'][0]:.0%},{r['ci'][1]:.0%}]", flush=True)

    # Does strength rise with accuracy?  Report the rank correlation, which is
    # the actual claim ("small improvements in accuracy -> large improvements
    # in strength") without assuming the relationship is linear.
    if len(points) >= 3:
        acc = np.array([p["test_acc"] for p in points])
        wr = np.array([p["win_rate"] for p in points])
        ra, rw = acc.argsort().argsort(), wr.argsort().argsort()
        rho = float(np.corrcoef(ra, rw)[0, 1])
    else:
        rho = float("nan")

    out = dict(points=points, histories=histories, ceiling=ceiling,
               spearman_acc_vs_winrate=rho, ref_sims=a.ref_sims,
               games_per_point=a.games)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[C1] Spearman(accuracy, win rate) = {rho:+.3f}"
          f"{'' if ceiling is None else f'   (accuracy ceiling {ceiling:.1%})'}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
