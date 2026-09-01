"""Claim C7: how much search is the raw policy network worth?

    "Remarkably, AlphaGo without any rollouts (alpha_vp, 2177 Elo) already
     exceeded the state of the art [...] The version solely using the policy
     network does not perform any search."  (Ext. Data Table 7: alpha_p = 1517
     Elo, above Fuego at 1148 and GnuGo at 431.)

The claim is that one forward pass of the SL policy plays at the level of an
MCTS program running thousands of rollouts.  The round-robin in
``tournament.py`` cannot answer that: every search player in it gets the same
fixed ``--sims`` budget (100), so it measures where the raw network sits in one
particular field, not how much search the network is *worth*.

So the search opponent is swept instead of fixed.  ``p_sl`` -- one forward pass
per move, no tree -- plays a paired-colour match against the paper's
``alpha_r`` (MCTS with p_pi rollouts, lambda=1, uniform prior, no network
anywhere) at a ladder of simulation counts.  The reported number is the largest
budget at which the raw network still holds its own, which is a quantity, not a
verdict: "the SL policy net is worth about N rollouts a move".

Two things make the answer readable rather than merely true:

* the ladder is anchored at the bottom by the same opponent at a small budget,
  so a flat curve (the raw net beating every budget equally) is distinguishable
  from a real trade-off;
* ``p_rl`` is measured on the same ladder when a checkpoint exists.  C2 says RL
  is the stronger player and C6 says SL is the better prior; C7's ladder is
  where those two facts stop being about search at all.

Rollouts are CPU work and each doubling of the budget doubles the wall clock,
so the ladder is deliberately short and the game counts small.  The confidence
intervals are reported and are wide; a crossover read off a single 20-game
match is a location, not a decimal.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ag import nets
from ag.arena import match
from ag.players import MCTSPlayer, PolicyNetPlayer
from ag.rollout import RolloutPolicy


def crossover(rows):
    """Largest simulation budget at which the network's win rate is >= 50%.

    Reported as an interval, not a point: the last budget it beat and the first
    budget it lost to.  With 20-game matches the CI on a single rate is about
    +-20 points, so anything finer than "between these two rungs" would be
    reading noise.
    """
    beat = [r["sims"] for r in rows if r["rate"] >= 0.5]
    lost = [r["sims"] for r in rows if r["rate"] < 0.5]
    return dict(last_beaten=max(beat) if beat else None,
                first_lost_to=min(lost) if lost else None,
                beat_every_rung=len(lost) == 0,
                lost_every_rung=len(beat) == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", default="/content/runs/rollout.npz")
    ap.add_argument("--sl", default="/content/runs/sl_k64_final.pt")
    ap.add_argument("--rl", default="/content/runs/rl_final.pt")
    ap.add_argument("--ladder", default="50,100,300,1000")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--out", default="/content/runs/c7_search_free.json")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rp = RolloutPolicy.load(a.rollout)
    ladder = [int(x) for x in a.ladder.split(",")]

    nets_to_test = []
    sl_net, sl_meta = nets.load(a.sl, map_location=device)
    nets_to_test.append(("p_sl", PolicyNetPlayer(
        nets.PolicyEvaluator(sl_net, device=device, symmetry="random"),
        name="p_sl", temperature=0.0)))
    if a.rl and os.path.exists(a.rl):
        rl_net, _ = nets.load(a.rl, map_location=device)
        nets_to_test.append(("p_rl", PolicyNetPlayer(
            nets.PolicyEvaluator(rl_net, device=device, symmetry="random"),
            name="p_rl", temperature=0.0)))

    # Resume support: this is the most expensive single measurement in the
    # pipeline and the box it runs on is ephemeral.
    have = {}
    if os.path.exists(a.out):
        for r in json.load(open(a.out)).get("rows", []):
            have[(r["net"], r["sims"])] = r

    rows = list(have.values())
    t0 = time.time()
    for net_name, player in nets_to_test:
        for sims in ladder:
            if (net_name, sims) in have:
                print(f"[c7] skip {net_name} vs a_r@{sims} (done)", flush=True)
                continue
            opp = MCTSPlayer(name=f"a_r{sims}", n_sims=sims, lmbda=1.0,
                             rollout=rp, rng=np.random.default_rng(1000 + sims))
            r = match(player, opp, n_games=a.games)
            r = dict(r, net=net_name, sims=sims)
            rows.append(r)
            print(f"[c7] {net_name:>5s} vs a_r@{sims:<5d} "
                  f"{r['wins']:2d}/{r['games']} = {r['rate']:5.1%} "
                  f"CI[{r['ci'][0]:.0%},{r['ci'][1]:.0%}] "
                  f"asB {r['wins_as_black']} asW {r['wins_as_white']} "
                  f"len {r['mean_len']:.0f} | {time.time()-t0:.0f}s", flush=True)
            with open(a.out, "w") as f:
                json.dump(dict(args=vars(a), rows=rows), f, indent=2)

    summary = {}
    for net_name, _ in nets_to_test:
        mine = sorted([r for r in rows if r["net"] == net_name],
                      key=lambda r: r["sims"])
        summary[net_name] = crossover(mine)

    with open(a.out, "w") as f:
        json.dump(dict(args=vars(a), rows=rows, summary=summary,
                       sl_test_acc=(sl_meta.get("history") or [{}])[-1]
                       .get("test_acc")), f, indent=2)

    print("\n=== C7: the raw policy network against a search ladder ===")
    print(f"{'net':<7}{'sims':>7}{'win rate':>11}{'  95% CI':>18}")
    for net_name, _ in nets_to_test:
        for r in sorted([x for x in rows if x["net"] == net_name],
                        key=lambda x: x["sims"]):
            print(f"{net_name:<7}{r['sims']:>7}{r['rate']:>10.1%}   "
                  f"[{r['ci'][0]:>5.0%},{r['ci'][1]:>5.0%}]")
        s = summary[net_name]
        if s["beat_every_rung"]:
            print(f"  -> {net_name} beat every rung up to {max(ladder)} sims")
        elif s["lost_every_rung"]:
            print(f"  -> {net_name} lost to every rung, from {min(ladder)} sims up")
        else:
            print(f"  -> {net_name} is worth between {s['last_beaten']} and "
                  f"{s['first_lost_to']} rollouts a move")
    print("paper: alpha_p 1517 Elo, above Fuego (1148) and GnuGo (431)")


if __name__ == "__main__":
    main()
