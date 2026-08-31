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


def play_game(black, white, max_moves=None, record=False, rng=None):                  # +-- PLAYING ONE GAME -----------------------------------------
    """Play one game.  Returns ``(winner, n_moves, positions, moves)``.

    ``positions`` are snapshots taken *before* each move, so a recorded pair
    ``(positions[i], moves[i])`` is exactly a supervised training example.
    """
    max_moves = max_moves or go.MAX_MOVES                                             # | Positions are snapshotted before the move is made, so a
    pos = go.Position()                                                               # | recorded pair is directly a supervised training example with
    black.reset()                                                                     # | no off-by-one to get wrong. A player that proposes an
    white.reset()                                                                     # | illegal move forfeits its turn instead of stopping the
    positions, moves = [], []                                                         # | tournament: a crash in game two hundred of a round robin
    while not pos.is_over() and pos.move_no < max_moves:                              # | would lose every result before it, and a player that cannot
        player = black if pos.to_play == BLACK else white                             # | produce legal moves will lose on the board anyway.
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


def agresti_coull(wins, n, z=1.96):                                                   # +-- A CONFIDENCE INTERVAL THAT SURVIVES 100 PERCENT ----------
    """95% interval on a win rate.  Extended Data Tables 9-11 use this."""            # | The interval the paper reports in its cross-tables. The
    if n == 0:                                                                        # | naive normal interval collapses to zero width at a perfect
        return 0.0, 1.0                                                               # | record and claims no uncertainty at all from twenty games.
    n_t = n + z * z                                                                   # | Most match-ups in a small tournament land at or near 0 and
    p_t = (wins + z * z / 2) / n_t                                                    # | 100 percent, which is exactly where an interval is needed
    half = z * np.sqrt(max(p_t * (1 - p_t), 0.0) / n_t)                               # | and exactly where the naive one is worthless. Adding a
    return max(0.0, p_t - half), min(1.0, p_t + half)                                 # | couple of imagined games of each result first keeps it
                                                                                      # | finite.

def match(p1, p2, n_games=20, max_moves=None, verbose=False):                         # +-- COLOURS MUST BE PAIRED -----------------------------------
    """Play ``n_games`` between p1 and p2 with colours swapped halfway.

    Returns a dict with p1's wins overall and split by colour -- the split is
    the diagnostic that tells you whether a "result" is really a komi artefact.
    """
    assert n_games % 2 == 0, "paired colours require an even number of games"         # | Half the games are played each way round. Komi of seven and
    w1 = w1_black = w1_white = 0                                                      # | a half on a nine by nine board is a large fixed handicap, so
    lengths = []                                                                      # | an unpaired match measures the komi as much as it measures
    for i in range(n_games):                                                          # | the players, and which side got black would dominate any
        p1_is_black = (i % 2 == 0)                                                    # | small sample. The per-colour split is kept and reported,
        black, white = (p1, p2) if p1_is_black else (p2, p1)                          # | because a result that is lopsided by colour is the signature
        winner, n, _, _, _ = play_game(black, white, max_moves=max_moves)             # | of a broken comparison rather than a strong player.
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


def elo_from_results(results, anchor=None, anchor_elo=0.0, c_elo=1.0 / 400,           # +-- TURNING WIN COUNTS INTO RATINGS --------------------------
                     prior_games=2.0, iters=6000, lr=8.0):                            # | Elo assumes the chance of one player beating another depends
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
    names = sorted({r[0] for r in results} | {r[1] for r in results})                 # | only on the gap between their ratings, through a logistic
    idx = {n: i for i, n in enumerate(names)}                                         # | curve. Finding the ratings is then just logistic regression
    e = np.zeros(len(names))                                                          # | on the pairwise records, which is what the loop does:
                                                                                      # | predict each pairing from the current gap, compare with what
    A = np.array([idx[r[0]] for r in results])                                        # | happened, and push both ratings apart in proportion to the
    B = np.array([idx[r[1]] for r in results])                                        # | error. Two imagined drawn games are added to every pair
    W = np.array([r[2] for r in results], dtype=float)                                # | before fitting, and that is the whole Bayesian part. Without
    G = np.array([r[3] for r in results], dtype=float)                                # | it a player with a perfect record has no finite rating at
    # virtual drawn games act as the prior                                            # | all, because the fit keeps gaining by pushing it further up
    W = W + prior_games / 2                                                           # | forever, and perfect records are common when a tournament
    G = G + prior_games                                                               # | has only a dozen games per pair. Ratings are recentred each
                                                                                      # | step because only differences mean anything; the anchor at
    for _ in range(iters):                                                            # | the end pins the scale to one named player so numbers can be
        d = c_elo * (e[A] - e[B])                                                     # | compared across runs.
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


def elo_bootstrap(results, anchor=None, anchor_elo=0.0, n_boot=200, seed=0,           # +-- HOW MUCH OF THAT RATING IS NOISE -------------------------
                  **kw):                                                              # | A dozen games per pair gives ratings with real uncertainty,
    """Elo with a bootstrap CI, resampling each pair's games binomially.

    A round robin of a few hundred games gives ratings with real uncertainty,
    and claim C5 asks whether three ratings are *separated*.  Reporting a point
    estimate alone would be reporting noise as a result.
    """
    rng = np.random.default_rng(seed)                                                 # | and claim C5 asks whether three ratings are actually
    base = elo_from_results(results, anchor=anchor, anchor_elo=anchor_elo, **kw)      # | separated. So each pair's record is redrawn from a coin
    draws = {n: [] for n in base}                                                     # | weighted by its observed win rate, the whole fit is redone,
    for _ in range(n_boot):                                                           # | and the spread of the answers is reported. Any ordering that
        res_b = [(a, b, int(rng.binomial(g, min(max(w / g, 0.0), 1.0))), g)           # | does not survive this is noise being read as a result.
                 for a, b, w, g in results]
        e = elo_from_results(res_b, anchor=anchor, anchor_elo=anchor_elo, **kw)
        for n, v in e.items():
            draws[n].append(v)
    return {n: (base[n], float(np.percentile(draws[n], 2.5)),
                float(np.percentile(draws[n], 97.5))) for n in base}
