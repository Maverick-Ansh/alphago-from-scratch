"""The evaluation gate.  Run this BEFORE spending GPU hours on the sweep.

On a resized reproduction the evaluation breaks more often than the model does,
and a broken instrument produces numbers that look like results.  Every claim in
CLAIMS.md is a comparison, and a comparison is only meaningful inside a known
range, so this script measures the range first:

1. **Accuracy ceiling for C1.**  The expert is a stochastic MCTS teacher, so
   perfect imitation is impossible.  Running the teacher twice on the same
   position with different seeds and measuring how often it agrees with itself
   gives the ceiling: no policy network can exceed the teacher's own
   reproducibility, and reporting "we reached 40% where the paper reached 57%"
   without that number is meaningless.

2. **Value-MSE floor for C3/C4.**  For outcomes z in {-1,+1} the best constant
   predictor scores 1 - E[z]^2, near 1.0.  Every MSE is reported against it.

3. **Elo bracket for C5/C7.**  A uniform-random player is the floor and the
   teacher is the reference.  Any rating outside that bracket means the
   tournament is broken, not that a player is strong.

4. **Degenerate shortcuts.**  "Always pass" and "always play the lowest-index
   point" are scored explicitly.  If either is competitive, the benchmark has a
   hole in it and nothing measured above it means anything.

5. **Colour balance.**  Komi 7.5 on 9x9 is a big handicap; if Black's win rate
   under symmetric play is far from 50%, colour assignment would dominate any
   unpaired match.

It prints a VERDICT and exits non-zero if a bracket is too narrow to measure in.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ag import arena, data, go
from ag.arena import match, agresti_coull
from ag.mcts import MCTS
from ag.players import (RandomPlayer, PassPlayer, FirstPointPlayer,
                        MCTSPlayer, RolloutPolicyPlayer)
from ag.rollout import RolloutPolicy
from ag.go import BLACK, PASS


def teacher_self_agreement(ds, rollout, n_positions, n_sims, seed=0):
    """How often two independent runs of the teacher pick the same move.

    This is the ceiling on C1's move-prediction accuracy.  It is also a direct
    read on how noisy the supervised labels are.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds["action"]), size=min(n_positions, len(ds["action"])),
                     replace=False)
    agree_self = 0
    agree_label = 0
    n = 0
    t0 = time.time()
    for j, i in enumerate(idx):
        pos = data.decode(ds, int(i))
        if pos.is_over():
            continue
        a1, _ = MCTS(n_sims=n_sims, lmbda=1.0, rollout=rollout,
                     rng=np.random.default_rng(seed * 7919 + j)).choose(pos)
        a2, _ = MCTS(n_sims=n_sims, lmbda=1.0, rollout=rollout,
                     rng=np.random.default_rng(seed * 104729 + j)).choose(pos)
        agree_self += (a1 == a2)
        agree_label += (a1 == int(ds["action"][i]))
        n += 1
    return dict(n=n, self_agreement=agree_self / max(n, 1),
                agreement_with_stored_label=agree_label / max(n, 1),
                seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/expert_*.npz")
    ap.add_argument("--rollout", default=None, help="trained p_pi .npz")
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--agree-positions", type=int, default=120)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--out", default="/content/runs/eval_gate.json")
    a = ap.parse_args()

    report = {}
    rp = RolloutPolicy.load(a.rollout) if a.rollout else RolloutPolicy()
    report["rollout_policy"] = a.rollout or "untrained (uniform weights)"

    # ---- 2. value floor ------------------------------------------------
    paths = sorted(glob.glob(a.data))
    ds = data.load(paths) if paths else None
    if ds is not None:
        zb = ds["z"].astype(np.float32)
        z_player = np.where(ds["to_play"] == BLACK, zb, -zb)
        floor = float(np.mean((z_player - z_player.mean()) ** 2))
        report["value_mse_floor"] = dict(
            floor=floor, mean_z=float(z_player.mean()),
            black_win_rate=float((zb > 0).mean()),
            n_positions=int(len(zb)),
            n_games=int(len(np.unique(ds["game_id"]))))
        print(f"[floor] constant-predictor value MSE = {floor:.4f} "
              f"(E[z]={z_player.mean():+.3f}); Black wins "
              f"{(zb > 0).mean():.1%} of teacher games", flush=True)

    # ---- 1. accuracy ceiling -------------------------------------------
    if ds is not None:
        print(f"[ceiling] measuring teacher self-agreement over "
              f"{a.agree_positions} positions at {a.sims} sims ...", flush=True)
        ag_ = teacher_self_agreement(ds, rp, a.agree_positions, a.sims)
        report["teacher_self_agreement"] = ag_
        print(f"[ceiling] teacher agrees with itself {ag_['self_agreement']:.1%} "
              f"of the time; with the stored label "
              f"{ag_['agreement_with_stored_label']:.1%}  "
              f"[{ag_['seconds']:.0f}s]", flush=True)

    # ---- 3/4/5. tournament brackets ------------------------------------
    rng = np.random.default_rng(0)
    teacher = MCTSPlayer(name="teacher", n_sims=a.sims, lmbda=1.0, rollout=rp,
                         rng=np.random.default_rng(1))
    tests = [
        ("colour balance (random vs random)",
         RandomPlayer(np.random.default_rng(2)), RandomPlayer(np.random.default_rng(3))),
        ("degenerate: random vs always-pass", RandomPlayer(rng), PassPlayer()),
        ("degenerate: random vs first-point", RandomPlayer(rng), FirstPointPlayer()),
        ("ceiling: teacher vs random", teacher, RandomPlayer(rng)),
        ("p_pi vs random", RolloutPolicyPlayer(rp), RandomPlayer(rng)),
    ]
    report["matches"] = []
    for label, p1, p2 in tests:
        r = match(p1, p2, n_games=a.games)
        r["label"] = label
        report["matches"].append(r)
        print(f"[bracket] {label:38s} {r['wins']:3d}/{r['games']} = "
              f"{r['rate']:5.1%}  CI[{r['ci'][0]:.0%},{r['ci'][1]:.0%}]  "
              f"asB {r['wins_as_black']} asW {r['wins_as_white']}", flush=True)

    # ---- verdict --------------------------------------------------------
    problems = []
    by_label = {m["label"]: m for m in report["matches"]}
    bal = by_label["colour balance (random vs random)"]
    black_rate = (bal["wins_as_black"] + (bal["games"] // 2 -
                                          bal["wins_as_white"])) / bal["games"]
    report["black_win_rate_random"] = black_rate
    if not 0.30 <= black_rate <= 0.70:
        problems.append(f"colour imbalance: Black wins {black_rate:.0%} under "
                        f"random play; komi is badly wrong for this board size")
    if by_label["degenerate: random vs always-pass"]["rate"] < 0.9:
        problems.append("always-pass is competitive against random play; "
                        "scoring or komi is broken")
    if by_label["ceiling: teacher vs random"]["rate"] < 0.9:
        problems.append("the teacher barely beats random play; the Elo bracket "
                        "is too narrow to separate anything inside it")
    if ds is not None and report["teacher_self_agreement"]["self_agreement"] < 0.25:
        problems.append("the teacher agrees with itself <25% of the time; the "
                        "supervised labels are mostly noise and C1 is not "
                        "measurable at this simulation count")

    print(f"\nBlack wins {black_rate:.1%} of random-vs-random games "
          f"(komi {go.KOMI} on {go.N}x{go.N})")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    report["problems"] = problems
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)

    if problems:
        print("\nVERDICT: FAIL -- do not run the sweep")
        for p in problems:
            print("  * " + p)
        sys.exit(1)
    print("\nVERDICT: PASS -- the instrument has range; the sweep can proceed")


if __name__ == "__main__":
    main()
