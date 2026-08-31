"""Figure 2b: value network versus Monte-Carlo rollouts (claim C4).

    "Comparison of evaluation accuracy between the value network and rollouts
     with different policies.  Positions and outcomes were sampled from human
     expert games.  Each position was evaluated by a single forward pass of the
     value network v_theta, or by the mean outcome of 100 rollouts, played out
     using either uniform random rollouts, the fast rollout policy p_pi, the SL
     policy network p_sigma or the RL policy network p_rho.  The mean squared
     error between the predicted value and the actual game outcome is plotted
     against the stage of the game."

    "the value function was consistently more accurate.  A single evaluation of
     v_theta(s) also approached the accuracy of Monte Carlo rollouts using the
     RL policy network p_rho, but using 15,000 times less computation."

The comparison is between five estimators of the same quantity, so the honest
way to read it is against the **constant-predictor floor**, which is printed for
every move bin.  A rollout estimator that scores 0.95 where the floor is 0.99 is
barely doing anything, and a network that scores 0.55 there is doing a great
deal; the raw numbers do not say that on their own.

The compute cost of each estimator is also recorded, because the claim is a
cost/accuracy claim and not an accuracy claim.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ag import data, features as feat, go, nets, selfplay as sp
from ag.go import BLACK, PASS
from ag.rollout import RolloutPolicy


def net_rollout_values(positions, policy, n_roll, rng, chunk=4096):
    """Mean outcome of ``n_roll`` net-policy playouts from each position.

    Every playout of every position is one game in a single lockstep batch, so
    the network sees a few hundred large forward passes rather than millions of
    tiny ones.
    """
    out = np.zeros(len(positions))
    jobs = [(i, p) for i, p in enumerate(positions) for _ in range(n_roll)]
    acc = np.zeros(len(positions))
    for s in range(0, len(jobs), chunk):
        part = jobs[s:s + chunk]
        games = sp.run_games(len(part), lambda c, i, g: policy, rng=rng,
                             record=False,
                             init_positions=[p for _, p in part])
        for (i, _), g in zip(part, games):
            acc[i] += g.z_black
    out = acc / n_roll
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/expert_*.npz")
    ap.add_argument("--value", default="/content/runs/value_uncorrelated.pt")
    ap.add_argument("--rollout", default="/content/runs/rollout.npz")
    ap.add_argument("--sl", default="/content/runs/sl_k64_final.pt")
    ap.add_argument("--rl", default="/content/runs/rl_final.pt")
    ap.add_argument("--positions", type=int, default=150)
    ap.add_argument("--n-roll", type=int, default=100)
    ap.add_argument("--out", default="/content/runs/c4_value_vs_rollouts.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    ds = data.load(sorted(glob.glob(a.data)))

    # Sample positions spread across the game, as Fig. 2b bins them.
    mv = ds["move_no"].astype(int)
    bins = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 80), (80, 200)]
    idx = []
    per = max(1, a.positions // len(bins))
    for lo, hi in bins:
        cand = np.flatnonzero((mv >= lo) & (mv < hi))
        if len(cand):
            idx += rng.choice(cand, size=min(per, len(cand)),
                              replace=False).tolist()
    idx = np.array(sorted(set(idx)))
    positions = [data.decode(ds, int(i)) for i in idx]
    # z in the frame of the player to move, exactly what v_theta predicts
    zb = ds["z"][idx].astype(np.float64)
    z = np.where(ds["to_play"][idx] == BLACK, zb, -zb)
    move_no = mv[idx]
    print(f"[C4] {len(idx)} positions, E[z]={z.mean():+.3f}", flush=True)

    est, cost = {}, {}

    # -- value network: one forward pass ---------------------------------
    val_net, _ = nets.load(a.value, map_location=device)
    ev = nets.ValueEvaluator(val_net, device=device, with_colour=True,
                             symmetry="none", cache=False)
    t0 = time.time()
    est["value_net"] = np.array([ev(p) for p in positions])
    cost["value_net"] = (time.time() - t0) / len(positions)

    # -- uniform random rollouts -----------------------------------------
    uni = RolloutPolicy()                      # all-zero weights = uniform
    t0 = time.time()
    est["rollout_uniform"] = np.array([
        np.mean([1.0 if uni.rollout(p) > 0 else -1.0 for _ in range(a.n_roll)])
        * (1 if p.to_play == BLACK else -1) for p in positions])
    cost["rollout_uniform"] = (time.time() - t0) / len(positions)

    # -- fast rollout policy p_pi ----------------------------------------
    rp = RolloutPolicy.load(a.rollout)
    t0 = time.time()
    est["rollout_p_pi"] = np.array([
        np.mean([1.0 if rp.rollout(p) > 0 else -1.0 for _ in range(a.n_roll)])
        * (1 if p.to_play == BLACK else -1) for p in positions])
    cost["rollout_p_pi"] = (time.time() - t0) / len(positions)

    # -- rollouts with the policy networks -------------------------------
    for tag, path in (("rollout_p_sl", a.sl), ("rollout_p_rl", a.rl)):
        if not os.path.exists(path):
            print(f"[C4] skipping {tag}: {path} missing", flush=True)
            continue
        net, _ = nets.load(path, map_location=device)
        pol = sp.BatchPolicy(net, device, temperature=1.0, rng=rng)
        t0 = time.time()
        vb = net_rollout_values(positions, pol, a.n_roll, rng)
        est[tag] = vb * np.where(ds["to_play"][idx] == BLACK, 1, -1)
        cost[tag] = (time.time() - t0) / len(positions)

    # -- MSE overall and by move bin, always against the floor -----------
    report = {"n_positions": int(len(idx)), "n_roll": a.n_roll,
              "mean_z": float(z.mean()), "cost_seconds_per_position": cost,
              "overall": {}, "by_bin": []}
    floor = float(np.mean((z - z.mean()) ** 2))
    report["floor_overall"] = floor
    print(f"\n[C4] constant-predictor floor = {floor:.4f}")
    print(f"{'estimator':<18}{'MSE':>8}{'  vs floor':>11}{'  sec/pos':>11}")
    for k, v in est.items():
        m = float(np.mean((v - z) ** 2))
        report["overall"][k] = m
        print(f"{k:<18}{m:>8.4f}{m/floor:>11.3f}{cost[k]:>11.4f}")

    for lo, hi in bins:
        sel = (move_no >= lo) & (move_no < hi)
        if sel.sum() < 5:
            continue
        zb_ = z[sel]
        row = {"lo": lo, "hi": hi, "n": int(sel.sum()),
               "floor": float(np.mean((zb_ - zb_.mean()) ** 2))}
        for k, v in est.items():
            row[k] = float(np.mean((v[sel] - zb_) ** 2))
        report["by_bin"].append(row)

    print(f"\n{'moves':<10}{'n':>5}{'floor':>8}" +
          "".join(f"{k.replace('rollout_',''):>12}" for k in est))
    for row in report["by_bin"]:
        print(f"{row['lo']:>3}-{row['hi']:<6}{row['n']:>5}{row['floor']:>8.3f}" +
              "".join(f"{row[k]:>12.3f}" for k in est))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
