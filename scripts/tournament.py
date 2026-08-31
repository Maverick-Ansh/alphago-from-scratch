"""Round-robin tournament: claims C5, C6 and C7 (Extended Data Table 7).

The paper's ablation table is one program with parts switched off:

    alpha_rvp   p_sigma   v_theta   p_pi   lambda=0.5   2890 Elo
    alpha_vp    p_sigma   v_theta    -     lambda=0     2177
    alpha_rp    p_sigma      -      p_pi   lambda=1     2416
    alpha_rv   [p_tau]    v_theta   p_pi   lambda=0.5   2077
    alpha_v    [p_tau]    v_theta    -     lambda=0     1655
    alpha_r    [p_tau]       -      p_pi   lambda=1     1457
    alpha_p     p_sigma      -       -        -         1517

Reproduced here with the same names, plus two extra entries the paper describes
in prose but never tabulates:

* ``rvp_rl`` -- the full player with the **RL** policy as its prior instead of
  the SL policy.  "It is worth noting that the SL policy network performed
  better in AlphaGo than the stronger RL policy network, presumably because
  humans select a diverse beam of promising moves, whereas RL optimizes for the
  single best move."  That sentence carries no number anywhere in the paper.
  Claim C6 puts one on it.
* ``p_rl`` -- the RL policy playing alone, so that C6's two halves can be
  checked together: RL must be the *stronger player* and the *worse prior*.

``random`` anchors the Elo scale at 0 and brackets it from below.

Sharding: pairs are split across processes with ``--shard k --nshards K`` so the
round robin uses every core; results are merged by ``--merge``.
"""

import argparse
import glob
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ag import arena, nets
from ag.arena import match, elo_bootstrap
from ag.players import (MCTSPlayer, PolicyNetPlayer, RandomPlayer,
                        RolloutPolicyPlayer)
from ag.rollout import RolloutPolicy


def build_players(args, device):
    rp = RolloutPolicy.load(args.rollout)
    sl_net, _ = nets.load(args.sl, map_location=device)
    sl_eval = nets.PolicyEvaluator(sl_net, device=device, symmetry="random")
    val_net, _ = nets.load(args.value, map_location=device)
    val_eval = nets.ValueEvaluator(val_net, device=device, with_colour=True,
                                  symmetry="random")

    rl_eval = None
    if args.rl and os.path.exists(args.rl):
        rl_net, _ = nets.load(args.rl, map_location=device)
        rl_eval = nets.PolicyEvaluator(rl_net, device=device, symmetry="random")

    S = args.sims
    P = {}
    P["random"] = RandomPlayer(np.random.default_rng(0))
    P["p_pi"] = RolloutPolicyPlayer(rp)
    # alpha_p: "The version solely using the policy network does not perform
    # any search."
    P["p_sl"] = PolicyNetPlayer(sl_eval, name="p_sl", temperature=0.0)
    # alpha_r / alpha_rp / alpha_vp / alpha_rvp
    P["a_r"] = MCTSPlayer(name="a_r", n_sims=S, lmbda=1.0, rollout=rp,
                          rng=np.random.default_rng(1))
    P["a_rp"] = MCTSPlayer(name="a_rp", n_sims=S, lmbda=1.0, rollout=rp,
                           prior_fn=sl_eval, rng=np.random.default_rng(2))
    P["a_vp"] = MCTSPlayer(name="a_vp", n_sims=S, lmbda=0.0,
                           value_fn=val_eval, prior_fn=sl_eval,
                           rng=np.random.default_rng(3))
    P["a_rvp"] = MCTSPlayer(name="a_rvp", n_sims=S, lmbda=0.5, rollout=rp,
                            value_fn=val_eval, prior_fn=sl_eval,
                            rng=np.random.default_rng(4))
    P["a_v"] = MCTSPlayer(name="a_v", n_sims=S, lmbda=0.0, value_fn=val_eval,
                          rng=np.random.default_rng(5))
    if rl_eval is not None:
        P["p_rl"] = PolicyNetPlayer(rl_eval, name="p_rl", temperature=0.0)
        P["rvp_rl"] = MCTSPlayer(name="rvp_rl", n_sims=S, lmbda=0.5,
                                 rollout=rp, value_fn=val_eval,
                                 prior_fn=rl_eval, rng=np.random.default_rng(6))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", default="/content/runs/rollout.npz")
    ap.add_argument("--sl", default="/content/runs/sl_k64_final.pt")
    ap.add_argument("--rl", default="/content/runs/rl_final.pt")
    ap.add_argument("--value", default="/content/runs/value_uncorrelated.pt")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--out", default="/content/runs/tourney")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    if a.merge:
        results = []
        for f in sorted(glob.glob(os.path.join(a.out, "shard_*.json"))):
            results += json.load(open(f))["results"]
        pairs = {}
        for r in results:
            pairs[(r["p1"], r["p2"])] = r
        triples = [(r["p1"], r["p2"], r["wins"], r["games"])
                   for r in pairs.values()]
        elo = elo_bootstrap(triples, anchor="random", anchor_elo=0.0)
        order = sorted(elo, key=lambda n: -elo[n][0])
        print(f"\n{'player':<10}{'Elo':>8}{'  95% CI':>18}")
        for n in order:
            e, lo, hi = elo[n]
            print(f"{n:<10}{e:>8.0f}   [{lo:>6.0f},{hi:>6.0f}]")
        with open(os.path.join(a.out, "elo.json"), "w") as f:
            json.dump(dict(elo={k: list(v) for k, v in elo.items()},
                           results=list(pairs.values())), f, indent=2)
        print(f"\nwrote {os.path.join(a.out, 'elo.json')}")
        return

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    P = build_players(a, device)
    names = list(P)
    all_pairs = list(itertools.combinations(names, 2))
    mine = [p for i, p in enumerate(all_pairs) if i % a.nshards == a.shard]
    print(f"[shard {a.shard}/{a.nshards}] {len(mine)}/{len(all_pairs)} pairs, "
          f"{a.games} games each, {a.sims} sims, device={device}", flush=True)

    out_path = os.path.join(a.out, f"shard_{a.shard}.json")
    done = {}
    if os.path.exists(out_path):
        for r in json.load(open(out_path))["results"]:
            done[(r["p1"], r["p2"])] = r

    results = list(done.values())
    t0 = time.time()
    for i, (n1, n2) in enumerate(mine):
        if (n1, n2) in done:
            print(f"[shard {a.shard}] skip {n1} vs {n2} (done)", flush=True)
            continue
        r = match(P[n1], P[n2], n_games=a.games)
        results.append(r)
        print(f"[shard {a.shard}] {i+1}/{len(mine)} {n1:>7s} vs {n2:<7s} "
              f"{r['wins']:2d}/{r['games']} = {r['rate']:5.1%} "
              f"CI[{r['ci'][0]:.0%},{r['ci'][1]:.0%}] "
              f"asB {r['wins_as_black']} asW {r['wins_as_white']} "
              f"| {time.time()-t0:.0f}s", flush=True)
        with open(out_path, "w") as f:
            json.dump(dict(args=vars(a), results=results), f, indent=2)
    print(f"[shard {a.shard}] DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
