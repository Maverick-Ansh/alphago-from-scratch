"""Stage 1b: fit the fast rollout policy p_pi by softmax regression.

    "Similar to the policy network, the weights pi of the rollout policy are
     trained from 8 million positions from human games on the Tygem server to
     maximize log likelihood by stochastic gradient descent. [...] this achieved
     an accuracy of 24.2%, using just 2 micro-seconds to select an action,
     rather than 3 ms for the policy network."

The model is linear, so the gradient of the log-likelihood is the difference
between the features of the move actually played and their expectation under
the current policy:

    d/dw log p(a|s) = phi(a) - sum_b p(b|s) phi(b)

which for one-hot features means "add the learning rate to the played move's
pattern weight, and subtract p(b) from every legal move's pattern weight".  No
autodiff needed, and it runs in numba at the same speed as the rollout itself.

Note what p_pi is *for*.  It is not meant to be a good player -- the paper's
gets 24.2% where the network gets 57.0%.  It is meant to be a good *rollout*:
fast enough that MCTS can afford thousands of them, and just biased enough
towards local shape that the games it plays out are not noise.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ag import data, rollout as R
from ag.go import (NN, PASS, BLACK, EMPTY, NBRS, DIAGS, NDIAG,
                   legal_mask, is_legal)


@njit(cache=True)
def _epoch(boards, to_plays, kos, last_moves, actions, order,
           nbrs, diags, ndiag, surr, canon_b, canon_w,
           w_pat, w_resp, w_tac, lr, train,
           board, buf, seen, tagbox, scratch, tacbuf, scores, pat, mask,
           probs):
    """One pass over ``order``.  Returns (sum_nll, n_correct, n)."""
    nll = 0.0
    ncorrect = 0
    n = 0
    for oi in range(order.shape[0]):
        i = order[oi]
        for p in range(NN):
            board[p] = boards[i, p]
        color = to_plays[i]
        ko = kos[i]
        lm = last_moves[i]
        a = actions[i]
        if a < 0 or a >= NN:
            continue

        canon = canon_b if color == BLACK else canon_w
        R.build_pat(board, surr, pat)
        R.move_scores(board, nbrs, surr, pat, canon, color, lm,
                      w_pat, w_resp, w_tac, buf, seen, tagbox, scratch,
                      tacbuf, scores)
        legal_mask(board, nbrs, diags, ndiag, color, ko,
                   buf, seen, tagbox, mask, True)

        # softmax over the legal, non-eye-filling moves only
        smax = -1e30
        for p in range(NN):
            if mask[p] and scores[p] > smax:
                smax = scores[p]
        if smax < -1e29:
            continue
        tot = 0.0
        for p in range(NN):
            if mask[p]:
                probs[p] = np.exp(scores[p] - smax)
                tot += probs[p]
            else:
                probs[p] = 0.0
        if tot <= 0.0:
            continue
        best = -1
        bestp = -1.0
        for p in range(NN):
            probs[p] /= tot
            if probs[p] > bestp:
                bestp = probs[p]
                best = p

        if not mask[a]:
            # The teacher played a move our eye-filter calls senseless; it
            # carries no gradient for a policy that can never propose it.
            continue
        nll -= np.log(max(probs[a], 1e-300))
        if best == a:
            ncorrect += 1
        n += 1
        if not train:
            continue

        # gradient: + features(a), - E_p[features(b)]
        ca = canon[pat[a]]
        ra = R.response_offset(surr, a, lm)
        w_pat[ca] += lr
        w_resp[ra] += lr
        if ra < 8:
            R.tactical_features(board, nbrs, a, color, buf, seen, tagbox,
                                scratch, tacbuf)
            for j in range(w_tac.shape[0]):
                w_tac[j] += lr * tacbuf[j]
        for b in range(NN):
            pb = probs[b]
            if pb <= 1e-12:
                continue
            w_pat[canon[pat[b]]] -= lr * pb
            rb = R.response_offset(surr, b, lm)
            w_resp[rb] -= lr * pb
            if rb < 8:
                R.tactical_features(board, nbrs, b, color, buf, seen, tagbox,
                                    scratch, tacbuf)
                for j in range(w_tac.shape[0]):
                    w_tac[j] -= lr * pb * tacbuf[j]
    return nll, ncorrect, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/expert_*.npz")
    ap.add_argument("--out", default="/content/runs/rollout.npz")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lr-decay", type=float, default=0.7)
    ap.add_argument("--max-train", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ds = data.load(sorted(glob.glob(a.data)))
    keep = ds["action"] != PASS
    ds = {k: v[keep] for k, v in ds.items()}

    # split by game, same reasoning as the policy network
    gid = ds["game_id"]
    games = np.unique(gid)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(games)
    test_games = set(games[:max(1, len(games) // 10)].tolist())
    is_test = np.array([g in test_games for g in gid])
    tr = np.flatnonzero(~is_test)
    te = np.flatnonzero(is_test)
    if len(tr) > a.max_train:
        tr = rng.choice(tr, a.max_train, replace=False)
    print(f"[p_pi] train {len(tr)} / test {len(te)} positions "
          f"({len(games)} games)", flush=True)

    boards = np.ascontiguousarray(ds["boards"].astype(np.int8))
    to_plays = ds["to_play"].astype(np.int8)
    kos = ds["ko"].astype(np.int32)
    last_moves = ds["last_move"].astype(np.int32)
    actions = ds["action"].astype(np.int32)

    rp = R.RolloutPolicy()
    board = np.zeros(NN, dtype=np.int8)
    mask = np.zeros(NN + 1, dtype=np.bool_)
    probs = np.zeros(NN, dtype=np.float64)

    def run(idx, lr, train):
        return _epoch(boards, to_plays, kos, last_moves, actions,
                      idx.astype(np.int64), NBRS, DIAGS, NDIAG, R.SURR,
                      R.CANON_B, R.CANON_W, rp.w_pat, rp.w_resp, rp.w_tac,
                      lr, train, board, rp.buf, rp.seen, rp.tagbox,
                      rp.scratch, rp.tacbuf, rp.scores, rp.pat, mask, probs)

    t0 = time.time()
    lr = a.lr
    for ep in range(a.epochs):
        order = rng.permutation(tr)
        nll, nc, n = run(order, lr, True)
        vnll, vnc, vn = run(te.astype(np.int64), 0.0, False)
        print(f"[p_pi] epoch {ep+1}/{a.epochs} lr {lr:.4f} | "
              f"train nll {nll/max(n,1):.4f} acc {nc/max(n,1):.4f} | "
              f"test nll {vnll/max(vn,1):.4f} acc {vnc/max(vn,1):.4f} "
              f"| {time.time()-t0:.0f}s", flush=True)
        lr *= a.lr_decay

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rp.save(a.out)
    vnll, vnc, vn = run(te.astype(np.int64), 0.0, False)
    print(f"[p_pi] saved {a.out}; final test accuracy {vnc/max(vn,1):.4f} "
          f"(paper: 24.2% on 19x19 vs human experts)", flush=True)
    print(f"[p_pi] |w_pat| mean {np.abs(rp.w_pat).mean():.4f}  "
          f"w_resp {np.round(rp.w_resp, 3)}  w_tac {np.round(rp.w_tac, 3)}",
          flush=True)


if __name__ == "__main__":
    main()
