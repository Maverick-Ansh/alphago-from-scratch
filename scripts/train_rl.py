"""Stage 2: policy-gradient reinforcement learning (claim C2).

    "Each iteration consisted of a mini-batch of n games played in parallel,
     between the current policy network p_rho that is being trained, and an
     opponent p_rho^- that uses parameters rho^- from a previous iteration,
     randomly sampled from a pool of opponents, so as to increase the stability
     of training.  Weights were initialized to rho = rho^- = sigma.  Every 500
     iterations, we added the current parameters rho to the opponent pool. [...]
     using the REINFORCE algorithm with baseline v(s_t) for variance reduction.
     On the first pass through the training pipeline, the baseline was set to
     zero."

The update, verbatim from the paper:

    delta_rho = (alpha/n) sum_i sum_t  d log p_rho(a_t^i | s_t^i) / d rho
                                       * (z_t^i - v(s_t^i))

Note it is a **sum over time steps** and a **mean over games**, not a mean over
moves.  A game contributes gradient in proportion to its length, which is the
correct REINFORCE estimator for an episodic return and is why the learning rate
here is much smaller than the supervised one.

Faithful: the opponent pool and its refresh cadence (scaled to keep the paper's
ratio of 20 snapshots over training), REINFORCE with a zero baseline as on the
paper's first pass, the learner playing both colours, and initialisation from
the SL weights.

**The paper does not state the RL learning rate.**  It gives alpha = 0.003 for
supervised learning and says nothing for this stage.  It is exposed as a flag
and the value used is reported in REPORT.md; this is a genuine gap in the
paper, not an omission here.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ag import nets, selfplay as sp
from ag.go import BLACK, WHITE, NN, PASS


def collect(learner_pol, opp_pol, n_games, rng, learner_black_first=True):
    """Play n_games; learner takes black in half of them.

    Returns (planes, actions, returns) over the learner's moves only, plus the
    learner's win rate.
    """
    half = n_games // 2
    learner_colour = np.array([BLACK] * half + [WHITE] * (n_games - half))

    def policy_for(colour, i, g):
        return learner_pol if colour == learner_colour[i] else opp_pol

    games = sp.run_games(n_games, policy_for, rng=rng, record=True)

    P, A, R = [], [], []
    wins = 0
    for i, g in enumerate(games):
        c = learner_colour[i]
        z = sp.outcomes_for(g, c)
        wins += z > 0
        for planes, a, mover in zip(g.planes, g.actions, g.movers):
            if mover != c or a == PASS:
                continue
            P.append(planes)
            A.append(a)
            R.append(z)
    if not P:
        # Every game ended with the learner making no non-pass move at all.
        # Possible only if something upstream is broken; say so rather than
        # dying inside np.stack with an unrelated message.
        raise RuntimeError("no learner moves collected -- check colour "
                           "assignment or the pass rule")
    return (np.stack(P), np.array(A, dtype=np.int64),
            np.array(R, dtype=np.float32), wins / n_games, games)


def evaluate_vs(learner_net, ref_net, device, n_games, rng, temperature=1.0):
    """Head-to-head win rate of the learner against a reference network."""
    lp = sp.BatchPolicy(learner_net, device, temperature=temperature, rng=rng)
    rp = sp.BatchPolicy(ref_net, device, temperature=temperature, rng=rng)
    half = n_games // 2
    colour = np.array([BLACK] * half + [WHITE] * (n_games - half))
    games = sp.run_games(n_games, lambda c, i, g: lp if c == colour[i] else rp,
                         rng=rng, record=False)
    return float(np.mean([sp.outcomes_for(g, colour[i]) > 0
                          for i, g in enumerate(games)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl", required=True, help="SL checkpoint to initialise from")
    ap.add_argument("--out", default="/content/runs")
    ap.add_argument("--tag", default="rl")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--games", type=int, default=32, help="games per iteration")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--pool-every", type=int, default=15)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    net, _ = nets.load(a.sl, map_location=device)
    net = net.to(device)
    sl_frozen, _ = nets.load(a.sl, map_location=device)      # rho = sigma
    sl_frozen = sl_frozen.to(device).eval()

    # "Weights were initialized to rho = rho^- = sigma."
    pool = [copy.deepcopy(net.state_dict())]
    opp_net, _ = nets.load(a.sl, map_location=device)
    opp_net = opp_net.to(device).eval()

    opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    learner_pol = sp.BatchPolicy(net, device, temperature=a.temperature, rng=rng)
    opp_pol = sp.BatchPolicy(opp_net, device, temperature=a.temperature, rng=rng)

    hist = []
    t0 = time.time()
    base_rate = evaluate_vs(net, sl_frozen, device, a.eval_games, rng,
                            a.temperature)
    print(f"[{a.tag}] sanity: RL(=SL) vs SL before any update = "
          f"{base_rate:.3f} (should be ~0.5)", flush=True)

    for it in range(1, a.iters + 1):
        # "an opponent that uses parameters from a previous iteration,
        #  randomly sampled from a pool of opponents"
        opp_net.load_state_dict(pool[int(rng.integers(len(pool)))])
        net.train()
        P, A, R, winrate, _ = collect(learner_pol, opp_pol, a.games, rng)

        x = torch.from_numpy(P).to(device).float()
        y = torch.from_numpy(A).to(device)
        z = torch.from_numpy(R).to(device)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=(device == "cuda")):
            logp = Fn.log_softmax(net(x).float(), dim=1)
            chosen = logp.gather(1, y[:, None]).squeeze(1)
            # sum over time steps, mean over games (baseline v(s) = 0)
            loss = -(chosen * z).sum() / a.games
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        net.eval()

        if it % a.pool_every == 0:
            pool.append(copy.deepcopy(net.state_dict()))

        if it % a.eval_every == 0 or it == a.iters:
            wr = evaluate_vs(net, sl_frozen, device, a.eval_games, rng,
                             a.temperature)
            el = time.time() - t0
            print(f"[{a.tag}] iter {it:4d}/{a.iters} loss {loss.item():8.2f} "
                  f"selfplay_wr {winrate:.2f} pool {len(pool):2d} | "
                  f"vs SL {wr:.3f} ({a.eval_games} games) | {el:.0f}s",
                  flush=True)
            hist.append(dict(iter=it, loss=float(loss.item()),
                             selfplay_winrate=winrate, vs_sl=wr,
                             pool=len(pool), seconds=el))
            # Periodic saves go to _latest, never to _final.  A file called
            # "final" that appears at iteration 25 is a trap: the pipeline
            # treats it as this stage's output and skips the stage on resume,
            # and anything downstream that loads it gets a policy that is
            # partly trained without any indication that it is.
            nets.save(net, os.path.join(a.out, f"{a.tag}_latest.pt"),
                      history=hist, args=vars(a), complete=False)

    nets.save(net, os.path.join(a.out, f"{a.tag}_final.pt"),
              history=hist, args=vars(a), complete=True)
    with open(os.path.join(a.out, f"{a.tag}_history.json"), "w") as f:
        json.dump(dict(tag=a.tag, args=vars(a), baseline_check=base_rate,
                       history=hist), f, indent=2)
    print(f"[{a.tag}] DONE vs SL = {hist[-1]['vs_sl']:.3f} "
          f"(paper: RL beat SL in >80% of games)", flush=True)


if __name__ == "__main__":
    main()
