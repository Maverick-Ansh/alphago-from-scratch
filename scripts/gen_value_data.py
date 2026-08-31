"""Stage 3a: the self-play data set for the value network (claim C3, arm B).

    "The naive approach of predicting game outcomes from data consisting of
     complete games leads to overfitting.  The problem is that successive
     positions are strongly correlated, differing by just one stone, but the
     regression target is shared for the entire game. [...] To mitigate this
     problem, we generated a new self-play data set consisting of 30 million
     distinct positions, each sampled from a separate game.  Each game was
     generated in three phases by randomly sampling a time step
     U ~ unif{1, 450}, and sampling the first t = 1,...U-1 moves from the SL
     policy network, a_t ~ p_sigma(.|s_t); then sampling one move uniformly at
     random from available moves, a_U ~ unif{1, 361} (repeatedly until a_U is
     legal); then sampling the remaining sequence of moves until the game
     terminates, t = U+1,...T, from the RL policy network, a_t ~ p_rho(.|s_t).
     [...] Only a single training example (s_{U+1}, z_{U+1}) is added to the
     data set from each game."

Three phases, three different move sources, and **one** position kept per game.
Every part of that is deliberate:

* the SL prefix puts the position on the manifold of plausible play;
* the single uniform-random move at U breaks the correlation with the prefix
  and guarantees the recorded position is not one the policy would have chosen,
  which is what makes the sample unbiased for the value function;
* the RL suffix means the outcome z is the outcome under the *strong* policy,
  so the target is v^{p_rho}, which is what the paper wants to approximate;
* keeping one position per game is the entire point -- 100 positions from one
  game share a label and are 99 near-duplicates.

Resize: U ~ unif{1, 2*N*N} = unif{1, 162}, the same "up to the maximum possible
game length" rule the paper's 450 encodes for 19x19.  When a game happens to end
before move U the sample is discarded; the discard rate is reported.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ag import data, go, nets, selfplay as sp
from ag.go import BLACK, PASS, NN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl", required=True)
    ap.add_argument("--rl", default=None, help="defaults to --sl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    sl_net, _ = nets.load(a.sl, map_location=device)
    rl_net, _ = nets.load(a.rl or a.sl, map_location=device)
    # "sampling ... from" the networks: stochastic, not argmax.
    sl_pol = sp.BatchPolicy(sl_net, device, temperature=1.0, rng=rng)
    rl_pol = sp.BatchPolicy(rl_net, device, temperature=1.0, rng=rng)
    rnd_pol = sp.RandomBatchPolicy(rng=rng)

    records = []
    n_done = n_discard = 0
    t0 = time.time()
    while n_done < a.games:
        n = min(a.batch, a.games - n_done)
        U = rng.integers(1, go.MAX_MOVES + 1, size=n)

        def policy_for(colour, i, g):
            m = g.pos.move_no                 # moves already played
            if m <= U[i] - 2:
                return sl_pol                 # moves 1..U-1
            if m == U[i] - 1:
                return rnd_pol                # move U, uniformly at random
            return rl_pol                     # moves U+1..T

        def on_step(games):
            for i, g in enumerate(games):
                if "sample" not in g.meta and g.pos.move_no == U[i]:
                    g.meta["sample"] = g.pos.copy()

        games = sp.run_games(n, policy_for, rng=rng, record=False,
                             on_step=on_step)

        for i, g in enumerate(games):
            s = g.meta.get("sample")
            if s is None:
                n_discard += 1                # game ended before move U
                continue
            # z from the perspective of the player to move at s_{U+1}
            z_black = 1 if g.pos.winner() == BLACK else -1
            records.append(data.encode(s, PASS, z_black,
                                       game_id=n_done + i))
        n_done += n
        el = time.time() - t0
        print(f"[value-data] {n_done}/{a.games} games, {len(records)} kept, "
              f"{n_discard} discarded ({n_discard/max(n_done,1):.1%}), "
              f"{el:.0f}s ({el/max(n_done,1)*1000:.0f} ms/game)", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    data.save(a.out, records)
    print(f"[value-data] DONE {len(records)} positions (one per game) -> "
          f"{a.out}; discarded {n_discard/a.games:.1%} of games because they "
          f"ended before move U", flush=True)


if __name__ == "__main__":
    main()
