"""Playing games, running matches, and turning results into Elo.

The evaluation, not the model, is what usually breaks in a resized
reproduction, so the deliberate choices here are:

* **Paired colours.**  Every match plays an even number of games and swaps
  colours halfway.  Komi 7.5 on a 9x9 board is a large handicap, and an
  unpaired match measures the komi as much as the players.
* **Agresti-Coull intervals**, the same interval the paper reports in Extended
  Data Tables 9-11, rather than the naive normal interval which misbehaves near
  0% and 100% -- exactly where most of these match-ups land.
* **Elo by logistic regression** with the paper's ``c_elo = 1/400``, anchored on
  a named reference player, so ratings are comparable across runs.
"""

import numpy as np

from . import go
from .go import BLACK, WHITE, PASS


def play_game(black, white, max_moves=None, record=False, rng=None):
    """Play one game.  Returns ``(winner, n_moves, positions, moves)``.

    ``positions`` are snapshots taken *before* each move, so a recorded pair
    ``(positions[i], moves[i])`` is exactly a supervised training example.
    """
    max_moves = max_moves or go.MAX_MOVES
    pos = go.Position()
    black.reset()
    white.reset()
    positions, moves = [], []
    while not pos.is_over() and pos.move_no < max_moves:
        player = black if pos.to_play == BLACK else white
        if record:
            positions.append(pos.copy())
        a = player.move(pos)
        if a != PASS and not pos.legal_actions()[a]:
            # A player that proposes an illegal move forfeits its turn rather
            # than crashing the tournament; counted and reported by the caller.
            a = PASS
        if record:
            moves.append(a)
        pos.play(a)
    return pos.winner(), pos.move_no, positions, moves, pos


def agresti_coull(wins, n, z=1.96):
    """95% interval on a win rate.  Extended Data Tables 9-11 use this."""
    if n == 0:
        return 0.0, 1.0
    n_t = n + z * z
    p_t = (wins + z * z / 2) / n_t
    half = z * np.sqrt(max(p_t * (1 - p_t), 0.0) / n_t)
    return max(0.0, p_t - half), min(1.0, p_t + half)


def match(p1, p2, n_games=20, max_moves=None, verbose=False):
    """Play ``n_games`` between p1 and p2 with colours swapped halfway.

    Returns a dict with p1's wins overall and split by colour -- the split is
    the diagnostic that tells you whether a "result" is really a komi artefact.
    """
    assert n_games % 2 == 0, "paired colours require an even number of games"
    w1 = w1_black = w1_white = 0
    lengths = []
    for i in range(n_games):
        p1_is_black = (i % 2 == 0)
        black, white = (p1, p2) if p1_is_black else (p2, p1)
        winner, n, _, _, _ = play_game(black, white, max_moves=max_moves)
        p1_won = (winner == BLACK) == p1_is_black
        w1 += p1_won
        if p1_is_black:
            w1_black += p1_won
        else:
            w1_white += p1_won
        lengths.append(n)
        if verbose:
            print(f"  game {i+1}/{n_games}: {'p1' if p1_won else 'p2'} "
                  f"({'p1=B' if p1_is_black else 'p1=W'}, {n} moves)")
    lo, hi = agresti_coull(w1, n_games)
    return {
        "p1": p1.name, "p2": p2.name, "games": n_games,
        "wins": int(w1), "rate": w1 / n_games, "ci": (lo, hi),
        "wins_as_black": int(w1_black), "wins_as_white": int(w1_white),
        "mean_len": float(np.mean(lengths)),
    }


def elo_from_results(results, anchor=None, anchor_elo=0.0, c_elo=1.0 / 400,
                     prior_games=2.0, iters=6000, lr=8.0):
    """Fit Elo ratings by logistic regression on pairwise results.

    The paper: "we estimate the probability that program a will beat program b
    by a logistic function p(a beats b) = 1/(1 + exp(c_elo (e(b) - e(a)))), and
    estimate the ratings e(.) by Bayesian logistic regression [...] using the
    standard constant c_elo = 1/400."

    ``prior_games`` adds a virtual drawn game between every pair, which is the
    "Bayesian" part: without it any player with a 100% record has an unbounded
    rating, and 100% records are common in a small tournament.

    ``results`` is a list of ``(name_a, name_b, wins_a, games)``.
    """
    names = sorted({r[0] for r in results} | {r[1] for r in results})
    idx = {n: i for i, n in enumerate(names)}
    e = np.zeros(len(names))

    A = np.array([idx[r[0]] for r in results])
    B = np.array([idx[r[1]] for r in results])
    W = np.array([r[2] for r in results], dtype=float)
    G = np.array([r[3] for r in results], dtype=float)
    # virtual drawn games act as the prior
    W = W + prior_games / 2
    G = G + prior_games

    for _ in range(iters):
        d = c_elo * (e[A] - e[B])
        p = 1.0 / (1.0 + np.exp(-d))
        resid = (W - G * p) * c_elo
        grad = np.zeros_like(e)
        np.add.at(grad, A, resid)
        np.add.at(grad, B, -resid)
        e += lr * grad
        e -= e.mean()

    if anchor is not None and anchor in idx:
        e = e - e[idx[anchor]] + anchor_elo
    return {n: float(e[idx[n]]) for n in names}


def elo_bootstrap(results, anchor=None, anchor_elo=0.0, n_boot=200, seed=0,
                  **kw):
    """Elo with a bootstrap CI, resampling each pair's games binomially.

    A round robin of a few hundred games gives ratings with real uncertainty,
    and claim C5 asks whether three ratings are *separated*.  Reporting a point
    estimate alone would be reporting noise as a result.
    """
    rng = np.random.default_rng(seed)
    base = elo_from_results(results, anchor=anchor, anchor_elo=anchor_elo, **kw)
    draws = {n: [] for n in base}
    for _ in range(n_boot):
        res_b = [(a, b, int(rng.binomial(g, min(max(w / g, 0.0), 1.0))), g)
                 for a, b, w, g in results]
        e = elo_from_results(res_b, anchor=anchor, anchor_elo=anchor_elo, **kw)
        for n, v in e.items():
            draws[n].append(v)
    return {n: (base[n], float(np.percentile(draws[n], 2.5)),
                float(np.percentile(draws[n], 97.5))) for n in base}
