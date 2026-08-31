"""The fast rollout policy p_pi (Extended Data Table 4), resized.

The paper:

    "The rollout policy p_pi(a|s) is a linear softmax policy based on fast,
     incrementally computed, local pattern-based features consisting of both
     'response' patterns around the previous move that led to state s, and
     'non-response' patterns around the candidate move a in state s. [...]
     the weights pi of the rollout policy are trained from 8 million positions
     from human games on the Tygem server to maximize log likelihood by
     stochastic gradient descent."

Note "incrementally computed" -- that phrase is doing real work.  The paper
quotes 2 microseconds per action for p_pi against 3 ms for the policy network,
a factor of 1500, and you cannot get there by rebuilding every feature every
move.  Everything below is shaped by that constraint.

Feature set (resized; deviations recorded in REPORT.md)
-------------------------------------------------------
1. **Non-response 3x3 pattern** centred on the candidate move `a`.  The paper
   keys these on colour *and* liberty count (3 x 3 = 9 states per point,
   9^8 = 43M patterns, hashed).  We key on colour alone -- {off-board, empty,
   black, white} -- giving 4^8 = 65,536 patterns, which fits in a dense table
   with **no hash collisions at all**.  Liberty information is not lost; it
   moves into the tactical features below, where it is exact rather than
   truncated at "three or more".
   Patterns are canonicalised over the dihedral group of 8, so a shape and its
   mirror share one weight: an 8x multiplier on effective training data, free.

2. **Response offset.** Which of the 8 neighbours of the *previous* move the
   candidate sits on, or "not adjacent to the previous move".  This is the
   cheap stand-in for the paper's 12-point diamond response pattern.  It is
   what makes rollouts answer a local threat instead of wandering off.

3. **Tactical features**, computed exactly by playing the move on a scratch
   board: captures 1, captures 2+, saves a chain from atari, self-atari.  These
   are evaluated only for candidates inside the previous move's
   8-neighbourhood -- the locality restriction every fast Go engine uses, and
   where the paper's own response features live.  A capture on the far side of
   the board therefore gets no bonus.  Documented deviation.

Two representation choices make this fast
-----------------------------------------
*Incremental codes.*  Pattern codes are stored **absolutely** (black/white, not
own/opponent) in ``pat[NN]``.  When the colour at a point q changes, only the
codes of q's 8 ring neighbours change, and each changes in exactly one base-4
digit -- because the ring is antisymmetric, q occupies slot ``7-k`` of the
neighbour at q's slot ``k``.  So a move costs ~8 digit patches instead of 648
table reads.  Captures patch the same way, one vacated point at a time.

*Two canon tables.*  Storing codes absolutely means the own/opponent relabelling
cannot be baked into the code, so it is baked into the lookup instead:
``CANON_B`` and ``CANON_W`` are the same canonicalisation with the two stone
digits swapped.  Choosing the table by colour costs nothing at run time.

Sampling
--------
A rollout must not build the full legal-move mask: suicide detection costs a
flood fill per point, and doing 60 of them per move would dominate the program.
Instead we sample by inverse-CDF over unnormalised softmax weights, and test
legality only on the candidate actually drawn.  A candidate that turns out to be
illegal, or to fill our own eye, has its weight zeroed and removed from the
running total, and we redraw -- which is exactly sampling from the softmax
conditioned on the accepted set, while paying for the expensive legality test
once or twice per move rather than sixty times.
"""

import numpy as np
from numba import njit

from . import go
from .go import (N, NN, PASS, EMPTY, BLACK, WHITE, NBRS, DIAGS, NDIAG,
                 group_libs, is_legal, is_simple_eye, place_stone,
                 score_tromp_taylor, _tag)

# 3x3 ring around a point, in raster order:
#   0 1 2
#   3 . 4
#   5 6 7
# The ordering matters: slot k and slot 7-k are opposite offsets, which is what
# makes the incremental digit patch a one-liner.
_RING = ((-1, -1), (-1, 0), (-1, 1),
         (0, -1),           (0, 1),
         (1, -1),  (1, 0),  (1, 1))

N_PAT_RAW = 4 ** 8          # 65536 colour patterns
N_RESP = 9                  # 8 neighbour offsets + "not a response"
N_TAC = 4                   # capture1, capture2plus, save_atari, self_atari

POW4 = (4 ** np.arange(8)).astype(np.int32)

# base-4 digits: 0 off-board, 1 empty, 2 black, 3 white
_D_OFF, _D_EMPTY = 0, 1


def _build_surr(n):
    """SURR[p, k] = the k-th ring neighbour of p, or -1 if off the board."""
    nn = n * n
    surr = np.full((nn, 8), -1, dtype=np.int32)
    for r in range(n):
        for c in range(n):
            p = r * n + c
            for k, (dr, dc) in enumerate(_RING):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    surr[p, k] = rr * n + cc
    return surr


def _ring_symmetries():
    """The 8 dihedral transforms, as permutations of the ring's 8 slots."""
    perms = []
    for flip in (False, True):
        for rot in range(4):
            perm = np.empty(8, dtype=np.int64)
            for k, (dr, dc) in enumerate(_RING):
                r, c = dr, dc
                if flip:
                    c = -c
                for _ in range(rot):          # rotate 90 degrees: (r,c)->(c,-r)
                    r, c = c, -r
                perm[k] = _RING.index((r, c))
            perms.append(perm)
    return np.stack(perms)


def _build_canon_tables():
    """Canonical class id for every raw code, for each colour to move.

    Canonical form = the numerically smallest code over the 8 dihedral images.
    ``CANON_B`` reads a code as-is (black to play, so digit 2 = own);
    ``CANON_W`` first swaps digits 2 and 3, so white's shapes map onto the same
    classes.  One shared weight vector then serves both colours, exactly as the
    paper computes its features "relative to the current colour to play".
    """
    perms = _ring_symmetries()
    codes = np.arange(N_PAT_RAW, dtype=np.int64)
    digits = np.empty((N_PAT_RAW, 8), dtype=np.int64)
    for k in range(8):
        digits[:, k] = (codes // (4 ** k)) % 4

    pow4 = 4 ** np.arange(8, dtype=np.int64)
    best = np.full(N_PAT_RAW, np.iinfo(np.int64).max, dtype=np.int64)
    for perm in perms:
        np.minimum(best, digits[:, perm] @ pow4, out=best)
    uniq, canon = np.unique(best, return_inverse=True)
    canon_b = canon.astype(np.int32)

    # swap the two stone digits (2 <-> 3) to get white's view
    swapped = digits.copy()
    swapped[digits == 2] = 3
    swapped[digits == 3] = 2
    canon_w = canon_b[(swapped @ pow4).astype(np.int64)]
    return canon_b, canon_w.astype(np.int32), len(uniq)


SURR = _build_surr(N)
CANON_B, CANON_W, N_PAT = _build_canon_tables()


# --------------------------------------------------------------------------
# incremental pattern codes
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _digit(v):
    if v == EMPTY:
        return 1
    elif v == BLACK:
        return 2
    return 3


@njit(cache=True)
def build_pat(board, surr, pat):
    """Full rebuild of the absolute pattern codes.  O(8*NN); once per rollout."""
    nn = board.shape[0]
    for p in range(nn):
        code = 0
        for k in range(8):
            q = surr[p, k]
            d = _D_OFF if q < 0 else _digit(board[q])
            code += d * POW4[k]
        pat[p] = code


@njit(cache=True, inline="always")
def patch_pat(board, surr, pat, q):
    """The colour at ``q`` just changed: repair the codes that mention it.

    Only q's 8 ring neighbours contain q in their own ring, and q sits in slot
    ``7-k`` of the neighbour found at q's slot ``k``.  So this is 8 single-digit
    patches, not 8 recomputations.
    """
    d = _digit(board[q])
    for k in range(8):
        r = surr[q, k]
        if r < 0:
            continue
        pw = POW4[7 - k]
        old = (pat[r] // pw) % 4
        pat[r] += (d - old) * pw


@njit(cache=True)
def place_stone_track(board, nbrs, pt, color, buf, seen, tagbox, caps_out):
    """``go.place_stone`` but also reporting *which* points were vacated.

    The rollout needs the captured points to patch their pattern codes; the
    plain version throws them away.
    """
    board[pt] = color
    opp = 3 - color
    n_captured = 0
    last_captured = -1
    for k in range(4):
        q = nbrs[pt, k]
        if q < 0 or board[q] != opp:
            continue
        ng, nl = group_libs(board, nbrs, q, buf, seen, _tag(tagbox))
        if nl == 0:
            for i in range(ng):
                pnt = buf[i]
                last_captured = pnt
                board[pnt] = EMPTY
                caps_out[n_captured + i] = pnt
            n_captured += ng
    ko = -1
    if n_captured == 1:
        ng, nl = group_libs(board, nbrs, pt, buf, seen, _tag(tagbox))
        if ng == 1 and nl == 1:
            ko = last_captured
    return ko, n_captured


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def response_offset(surr, p, last_move):
    """Which ring slot of ``last_move`` the candidate ``p`` occupies (else 8)."""
    if last_move < 0:
        return 8
    for k in range(8):
        if surr[last_move, k] == p:
            return k
    return 8


@njit(cache=True)
def tactical_features(board, nbrs, p, color, buf, seen, tagbox, scratch, out):
    """Exact capture / atari features for playing ``color`` at ``p``.

    Computed by actually playing the move on a scratch copy of the board.  An
    incremental liberty-tracking scheme would be faster, but this runs for at
    most 8 candidates per move and being obviously correct is worth more here
    than being clever.
    """
    out[0] = 0.0
    out[1] = 0.0
    out[2] = 0.0
    out[3] = 0.0

    # Was any friendly neighbouring chain in atari before the move?
    in_atari_before = False
    for k in range(4):
        q = nbrs[p, k]
        if q < 0 or board[q] != color:
            continue
        ng, nl = group_libs(board, nbrs, q, buf, seen, _tag(tagbox))
        if nl == 1:
            in_atari_before = True

    scratch[:] = board
    _, ncap = place_stone(scratch, nbrs, p, color, buf, seen, tagbox)
    ng, nl = group_libs(scratch, nbrs, p, buf, seen, _tag(tagbox))

    if ncap >= 1:
        out[0] = 1.0
    if ncap >= 2:
        out[1] = 1.0
    if in_atari_before and nl > 1:
        out[2] = 1.0                    # rescued a chain that was in atari
    if nl == 1 and ncap == 0:
        out[3] = 1.0                    # self-atari (usually a blunder)
    return ncap


@njit(cache=True)
def move_scores(board, nbrs, surr, pat, canon, color, last_move,
                w_pat, w_resp, w_tac, buf, seen, tagbox, scratch, tacbuf,
                out):
    """Linear score for every point on the board (-inf on occupied points).

    Legality is deliberately *not* checked here -- that is the expensive part,
    and the sampler checks it lazily on the few candidates it actually draws.

    Structured as "cheap pass over everything, then fix up the few special
    points" rather than "ask every point whether it is special".  At most 8
    points can be a response to the previous move, so scanning the previous
    move's ring *for each* of 81 candidates would cost 648 comparisons to
    discover 8 facts.  Writing the response term only where it differs from the
    default costs 8.  The tactical features ride along in the same fix-up.
    """
    nn = board.shape[0]
    base_resp = w_resp[8]                    # "not adjacent to the last move"
    for p in range(nn):
        if board[p] != EMPTY:
            out[p] = -1e30
        else:
            out[p] = w_pat[canon[pat[p]]] + base_resp
    if last_move >= 0 and last_move < nn:
        for k in range(8):
            p = surr[last_move, k]
            if p < 0 or board[p] != EMPTY:
                continue
            s = out[p] - base_resp + w_resp[k]
            tactical_features(board, nbrs, p, color, buf, seen, tagbox,
                              scratch, tacbuf)
            for j in range(w_tac.shape[0]):
                s += w_tac[j] * tacbuf[j]
            out[p] = s
    return out


# --------------------------------------------------------------------------
# sampling and playout
# --------------------------------------------------------------------------
@njit(cache=True)
def sample_move(board, nbrs, diags, ndiag, surr, pat, canon, color, ko,
                last_move, w_pat, w_resp, w_tac,
                buf, seen, tagbox, scratch, tacbuf, scores, wbuf, temp):
    """Draw one move from softmax(scores/temp) restricted to sensible moves.

    Sampling is by inverse-CDF over unnormalised weights rather than by adding
    Gumbel noise.  Both draw from the same distribution, but Gumbel costs two
    logarithms *per point* (81 points, 162 transcendentals per move) while this
    costs one exponential per point and a single uniform.  Transcendental
    functions dominate everything else here, so halving their count roughly
    halves the cost of a rollout move.

    Rejection is exact, not approximate: a candidate that turns out to be
    illegal (or to fill our own eye) has its weight zeroed and removed from the
    running total, and we redraw.  That is precisely sampling from the softmax
    conditioned on the accepted set -- while paying the flood-fill legality
    test only on the one or two candidates we actually draw.
    """
    nn = board.shape[0]
    move_scores(board, nbrs, surr, pat, canon, color, last_move,
                w_pat, w_resp, w_tac, buf, seen, tagbox, scratch, tacbuf,
                scores)

    # Shift by the max before exponentiating: the weights are unnormalised, so
    # a common offset is free, and it keeps exp() away from overflow.
    smax = -1e29
    for p in range(nn):
        if scores[p] > smax:
            smax = scores[p]
    total = 0.0
    for p in range(nn):
        if scores[p] > -1e29:
            w = np.exp((scores[p] - smax) / temp)
            wbuf[p] = w
            total += w
        else:
            wbuf[p] = 0.0

    while total > 1e-300:
        u = np.random.random() * total
        acc = 0.0
        pick = -1
        for p in range(nn):
            if wbuf[p] > 0.0:
                acc += wbuf[p]
                if acc >= u:
                    pick = p
                    break
                pick = p        # floating-point guard: keep the last live index
        if pick < 0:
            return PASS
        total -= wbuf[pick]
        wbuf[pick] = 0.0
        if not is_legal(board, nbrs, pick, color, ko, buf, seen, tagbox):
            continue
        if is_simple_eye(board, nbrs, diags, ndiag, pick, color):
            continue            # never fill your own eye: rollouts would
        return pick             # otherwise never terminate
    return PASS


@njit(cache=True)
def playout(board, ko, to_play, last_move, n_passes, move_no,
            nbrs, diags, ndiag, surr, canon_b, canon_w,
            w_pat, w_resp, w_tac, komi, max_moves,
            buf, seen, tagbox, scratch, tacbuf, scores, wbuf, pat, caps, temp):
    """Play ``board`` to the end with the rollout policy; return Black's score.

    ``board`` is mutated -- callers pass a scratch copy.  This is the whole
    "Evaluation" half of Fig. 3c: the rollout that produces z_L.
    """
    build_pat(board, surr, pat)
    while n_passes < 2 and move_no < max_moves:
        canon = canon_b if to_play == BLACK else canon_w
        a = sample_move(board, nbrs, diags, ndiag, surr, pat, canon,
                        to_play, ko, last_move, w_pat, w_resp, w_tac,
                        buf, seen, tagbox, scratch, tacbuf, scores, wbuf, temp)
        if a == PASS:
            n_passes += 1
            ko = -1
        else:
            ko, ncap = place_stone_track(board, nbrs, a, to_play,
                                         buf, seen, tagbox, caps)
            patch_pat(board, surr, pat, a)
            for i in range(ncap):
                patch_pat(board, surr, pat, caps[i])
            n_passes = 0
        last_move = a
        move_no += 1
        to_play = 3 - to_play
    return score_tromp_taylor(board, nbrs, komi)


# --------------------------------------------------------------------------
# Python-side owner of the weights
# --------------------------------------------------------------------------
class RolloutPolicy:
    """Weights + scratch for p_pi.  Not thread-safe (the scratch is shared)."""

    def __init__(self, temp=1.0):
        self.w_pat = np.zeros(N_PAT, dtype=np.float64)
        self.w_resp = np.zeros(N_RESP, dtype=np.float64)
        self.w_tac = np.zeros(N_TAC, dtype=np.float64)
        self.temp = float(temp)
        self._alloc()

    def _alloc(self):
        self.buf = np.empty(NN, dtype=np.int32)
        self.seen = np.zeros(NN, dtype=np.int32)
        self.tagbox = np.zeros(1, dtype=np.int64)
        self.scratch = np.zeros(NN, dtype=np.int8)
        self.tacbuf = np.zeros(N_TAC, dtype=np.float64)
        self.scores = np.zeros(NN, dtype=np.float64)
        self.wbuf = np.zeros(NN, dtype=np.float64)
        self.pat = np.zeros(NN, dtype=np.int32)
        self.caps = np.zeros(NN, dtype=np.int32)

    def _canon(self, color):
        return CANON_B if color == BLACK else CANON_W

    # -- inference --------------------------------------------------------
    def logits(self, pos):
        """Raw linear scores over the NN points for the player to move."""
        build_pat(pos.board, SURR, self.pat)
        move_scores(pos.board, NBRS, SURR, self.pat, self._canon(pos.to_play),
                    pos.to_play, pos.last_move, self.w_pat, self.w_resp,
                    self.w_tac, self.buf, self.seen, self.tagbox,
                    self.scratch, self.tacbuf, self.scores)
        return self.scores.copy()

    def sample(self, pos):
        build_pat(pos.board, SURR, self.pat)
        return sample_move(pos.board, NBRS, DIAGS, NDIAG, SURR, self.pat,
                           self._canon(pos.to_play), pos.to_play, pos.ko,
                           pos.last_move, self.w_pat, self.w_resp, self.w_tac,
                           self.buf, self.seen, self.tagbox, self.scratch,
                           self.tacbuf, self.scores, self.wbuf, self.temp)

    def rollout(self, pos):
        """Score (Black's view) of playing ``pos`` out to the end."""
        return playout(pos.board.copy(), pos.ko, pos.to_play, pos.last_move,
                       pos.n_passes, pos.move_no,
                       NBRS, DIAGS, NDIAG, SURR, CANON_B, CANON_W,
                       self.w_pat, self.w_resp, self.w_tac,
                       go.KOMI, go.MAX_MOVES,
                       self.buf, self.seen, self.tagbox, self.scratch,
                       self.tacbuf, self.scores, self.wbuf, self.pat,
                       self.caps, self.temp)

    # -- persistence ------------------------------------------------------
    def save(self, path):
        np.savez(path, w_pat=self.w_pat, w_resp=self.w_resp,
                 w_tac=self.w_tac, temp=self.temp)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        r = cls(temp=float(z["temp"]))
        r.w_pat[:] = z["w_pat"]
        r.w_resp[:] = z["w_resp"]
        r.w_tac[:] = z["w_tac"]
        return r
