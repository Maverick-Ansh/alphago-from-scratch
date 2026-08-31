"""Stage 0: generate the expert game record that the SL policy learns from.

The paper's expert is a human:

    "This data set contains 29.4 million positions from 160,000 games played by
     KGS 6 to 9 dan human players"

There is no 9x9 equivalent of the KGS archive, so the expert here is the thing
that was state of the art in computer Go *before* AlphaGo: Monte-Carlo tree
search with random rollouts and no neural network at all (the paper's Pachi,
Fuego, and its own ``alpha_r`` row).  That keeps the structure of the claim
intact -- a slow, strong teacher; a fast network distilled from it; then RL and
search on top -- while removing the dependence on a human game archive.

Diversity, which in the paper comes free from 160,000 different human games, has
to be manufactured here:

* a random opening of 0-4 uniformly sampled sensible moves;
* the teacher's own first ``--temp-moves`` moves are *played* by sampling from
  the visit distribution rather than taking the argmax.

In both cases the **label is always the teacher's argmax move**, never the
randomised one, so the supervised targets stay clean while the positions spread
out.  Opening moves are not labelled at all.

Each position is stored with the final outcome of its game, which is also
exactly the correlated "whole games" dataset that claim C3 needs as its
negative arm.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ag import data, go
from ag.go import BLACK, PASS
from ag.mcts import MCTS
from ag.rollout import RolloutPolicy


def gen_shard(n_games, n_sims, seed, out_path, temp_moves=8, temp=1.0,
              max_open=4, log_every=25, chunk=40):
    """Generate games, writing a partial shard every ``chunk`` games.

    The first version of this wrote once at the end.  Four workers then died
    of heap corruption 35 minutes in (see REPORT.md) and every game was lost,
    because nothing had been written yet.  Checkpointing costs a fraction of a
    second per chunk and bounds the worst case to ``chunk`` games.
    """
    rng = np.random.default_rng(seed)
    rp = RolloutPolicy()
    np.random.seed(seed % (2 ** 31))          # numba's RNG lives here
    teacher = MCTS(n_sims=n_sims, lmbda=1.0, rollout=rp, rng=rng)

    base = out_path[:-4] if out_path.endswith(".npz") else out_path
    records = []
    n_part = 0
    n_total = 0
    t0 = time.time()

    def flush():
        nonlocal records, n_part, n_total
        if not records:
            return
        part = f"{base}_part{n_part}.npz"
        data.save(part, records)
        n_total += len(records)
        print(f"[{os.path.basename(base)}] wrote {part} "
              f"({len(records)} positions)", flush=True)
        records = []
        n_part += 1

    for gi in range(n_games):
        pos = go.Position()
        # random opening
        for _ in range(int(rng.integers(0, max_open + 1))):
            idx = np.flatnonzero(pos.legal_actions(exclude_eyes=True))
            idx = idx[idx != PASS]
            if len(idx) == 0:
                break
            pos.play(int(rng.choice(idx)))

        pending = []
        n_teacher = 0
        while not pos.is_over():
            root = teacher.search(pos)
            counts = np.where(root.legal, teacher.visit_counts(root), 0.0)
            if counts.sum() <= 0:
                label = PASS
                played = PASS
            else:
                label = int(np.argmax(counts))
                if n_teacher < temp_moves:
                    p = counts ** (1.0 / temp)
                    played = int(rng.choice(len(p), p=p / p.sum()))
                else:
                    played = label
            pending.append((pos.copy(), label))
            pos.play(played)
            n_teacher += 1

        z_black = 1 if pos.winner() == BLACK else -1
        for p, label in pending:
            records.append(data.encode(p, label, z_black, game_id=gi))

        if (gi + 1) % log_every == 0:
            el = time.time() - t0
            print(f"[{os.path.basename(base)}] {gi+1}/{n_games} games, "
                  f"{n_total + len(records)} positions, {el:.0f}s "
                  f"({el/(gi+1):.1f}s/game)", flush=True)
        if (gi + 1) % chunk == 0:
            flush()

    flush()
    with open(base + ".done", "w") as f:
        f.write(str(n_total) + "\n")
    print(f"[{os.path.basename(base)}] DONE {n_total} positions in "
          f"{time.time()-t0:.0f}s", flush=True)
    return n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=250)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--temp-moves", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=40,
                    help="write a partial shard every N games")
    a = ap.parse_args()
    base = a.out[:-4] if a.out.endswith(".npz") else a.out
    if os.path.exists(base + ".done"):
        # Completed workers are skippable so a killed sweep resumes.
        print(f"[skip] {base}.done already exists", flush=True)
        return
    gen_shard(a.games, a.sims, a.seed, a.out,
              temp_moves=a.temp_moves, temp=a.temp, chunk=a.chunk)


if __name__ == "__main__":
    main()
