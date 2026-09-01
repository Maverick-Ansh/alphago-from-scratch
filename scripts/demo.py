"""Watch the finished system play, and see what the search was thinking.

Prints the board after every move, and for the move actually chosen shows the
three quantities Fig. 5 of the paper visualises: the policy network's prior,
the search's visit distribution, and the value estimate at the root.

    python scripts/demo.py --runs /content/runs
    python scripts/demo.py --runs /content/runs --black a_rvp --white a_r
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ag import go, nets
from ag.go import N, NN, PASS, BLACK, WHITE
from ag.mcts import MCTS
from ag.players import (MCTSPlayer, PolicyNetPlayer, RandomPlayer,
                        RolloutPolicyPlayer)
from ag.rollout import RolloutPolicy

COLS = "ABCDEFGHJKLMNOPQRST"


def name_of(a):
    if a == PASS:
        return "pass"
    return f"{COLS[a % N]}{N - a // N}"


def build(runs, device):
    rp = RolloutPolicy.load(os.path.join(runs, "rollout.npz"))
    sl, _ = nets.load(os.path.join(runs, "sl_k64_final.pt"), map_location=device)
    sl_eval = nets.PolicyEvaluator(sl, device=device, symmetry="random")
    val, _ = nets.load(os.path.join(runs, "value_uncorrelated.pt"),
                       map_location=device)
    val_eval = nets.ValueEvaluator(val, device=device, with_colour=True,
                                   symmetry="random")
    rl_path = os.path.join(runs, "rl_final.pt")
    rl_eval = None
    if os.path.exists(rl_path):
        rl, _ = nets.load(rl_path, map_location=device)
        rl_eval = nets.PolicyEvaluator(rl, device=device, symmetry="random")
    return rp, sl_eval, val_eval, rl_eval


def make(kind, rp, sl_eval, val_eval, rl_eval, sims, seed):
    rng = np.random.default_rng(seed)
    if kind == "random":
        return RandomPlayer(rng)
    if kind == "p_pi":
        return RolloutPolicyPlayer(rp)
    if kind == "p_sl":
        return PolicyNetPlayer(sl_eval, name="p_sl", temperature=0.0)
    if kind == "p_rl":
        return PolicyNetPlayer(rl_eval, name="p_rl", temperature=0.0)
    if kind == "a_r":
        return MCTSPlayer(name="a_r", n_sims=sims, lmbda=1.0, rollout=rp, rng=rng)
    if kind == "a_rp":
        return MCTSPlayer(name="a_rp", n_sims=sims, lmbda=1.0, rollout=rp,
                          prior_fn=sl_eval, rng=rng)
    if kind == "a_vp":
        return MCTSPlayer(name="a_vp", n_sims=sims, lmbda=0.0,
                          value_fn=val_eval, prior_fn=sl_eval, rng=rng)
    if kind == "a_rvp":
        return MCTSPlayer(name="a_rvp", n_sims=sims, lmbda=0.5, rollout=rp,
                          value_fn=val_eval, prior_fn=sl_eval, rng=rng)
    if kind == "a_v":
        return MCTSPlayer(name="a_v", n_sims=sims, lmbda=0.0,
                          value_fn=val_eval, rng=rng)
    # The strongest player measured here (1524 Elo): the full program with the
    # RL policy as its prior instead of the SL policy.  It was missing, which
    # meant the demo could not show the thing the tournament says is best.
    if kind == "rvp_rl":
        if rl_eval is None:
            raise SystemExit("rvp_rl needs rl_final.pt")
        return MCTSPlayer(name="rvp_rl", n_sims=sims, lmbda=0.5, rollout=rp,
                          value_fn=val_eval, prior_fn=rl_eval, rng=rng)
    raise SystemExit(
        f"unknown player {kind!r}. choices: random p_pi p_sl p_rl "
        f"a_r a_rp a_vp a_v a_rvp rvp_rl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/content/runs")
    ap.add_argument("--black", default="a_rvp")
    ap.add_argument("--white", default="a_r")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="only the final board")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rp, sl_eval, val_eval, rl_eval = build(a.runs, device)
    B = make(a.black, rp, sl_eval, val_eval, rl_eval, a.sims, a.seed)
    W = make(a.white, rp, sl_eval, val_eval, rl_eval, a.sims, a.seed + 1)
    print(f"black = {B.name}   white = {W.name}   {a.sims} sims   "
          f"komi {go.KOMI}\n")

    pos = go.Position()
    while not pos.is_over():
        player = B if pos.to_play == BLACK else W
        if isinstance(player, MCTSPlayer):
            root = player.mcts.search(pos)
            counts = np.where(root.legal, player.mcts.visit_counts(root), 0.0)
            act = int(np.argmax(counts)) if counts.sum() > 0 else PASS
            if not a.quiet:
                order = np.argsort(-counts)[:4]
                top = "  ".join(
                    f"{name_of(int(i))} {counts[i]/max(counts.sum(),1):.0%}"
                    for i in order if counts[i] > 0)
                prior = "  ".join(
                    f"{name_of(int(i))} {root.P[i]:.0%}"
                    for i in np.argsort(-root.P)[:3] if root.P[i] > 0)
                v = player.mcts.root_value(root)
                print(f"move {pos.move_no+1:3d}  {'B' if pos.to_play==BLACK else 'W'}"
                      f" plays {name_of(act):5s} | value {v:+.2f} | "
                      f"visits: {top} | prior: {prior}")
        else:
            act = player.move(pos)
            if not a.quiet:
                print(f"move {pos.move_no+1:3d}  "
                      f"{'B' if pos.to_play==BLACK else 'W'} plays "
                      f"{name_of(act)}")
        pos.play(act)

    print()
    print(pos)
    s = pos.score()
    print(f"\nfinal score {s:+.1f} (Black's view, komi {go.KOMI} deducted) -> "
          f"{'BLACK' if s > 0 else 'WHITE'} wins by {abs(s):.1f}")
    print(f"{pos.move_no} moves, "
          f"{int((pos.board == BLACK).sum())} black stones, "
          f"{int((pos.board == WHITE).sum())} white stones")


if __name__ == "__main__":
    main()
