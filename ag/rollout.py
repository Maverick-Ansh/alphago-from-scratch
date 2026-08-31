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

# 3x3 ring around a point, in raster order:                                         # +-- THE RING, AND WHY ITS ORDER MATTERS ------------------------
#   0 1 2                                                                           # | A pattern is the eight points around a candidate move, read in
#   3 . 4                                                                           # | reading order. The ordering is not arbitrary: slot k and slot
#   5 6 7                                                                           # | 7-k are always opposite offsets, so if point q sits in slot k
# The ordering matters: slot k and slot 7-k are opposite offsets, which is what     # | of point r, then r sits in slot 7-k of q. That one fact is
# makes the incremental digit patch a one-liner.                                    # | what turns updating a pattern after a move into a single
_RING = ((-1, -1), (-1, 0), (-1, 1),                                                # | arithmetic patch instead of a recomputation. Each of the eight
         (0, -1),           (0, 1),                                                 # | slots holds one base-4 digit, so a whole pattern is one
         (1, -1),  (1, 0),  (1, 1))                                                 # | integer below 65536, small enough for a dense table with no
                                                                                    # | hashing and therefore no collisions at all.
N_PAT_RAW = 4 ** 8          # 65536 colour patterns
N_RESP = 9                  # 8 neighbour offsets + "not a response"
N_TAC = 4                   # capture1, capture2plus, save_atari, self_atari

POW4 = (4 ** np.arange(8)).astype(np.int32)

# base-4 digits: 0 off-board, 1 empty, 2 black, 3 white
_D_OFF, _D_EMPTY = 0, 1


def _build_surr(n):                                                                 # +-- PRECOMPUTED RING NEIGHBOURS --------------------------------
    """SURR[p, k] = the k-th ring neighbour of p, or -1 if off the board."""        # | Which board point sits in each of the eight slots around each
    nn = n * n                                                                      # | point, worked out once at import. Off-board slots hold -1.
    surr = np.full((nn, 8), -1, dtype=np.int32)                                     # | Doing this ahead of time removes all row and column
    for r in range(n):                                                              # | arithmetic, and all edge-of-board tests, from the inner loop
        for c in range(n):                                                          # | that runs millions of times per second.
            p = r * n + c
            for k, (dr, dc) in enumerate(_RING):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    surr[p, k] = rr * n + cc
    return surr


def _ring_symmetries():                                                             # +-- THE EIGHT SYMMETRIES AS SLOT PERMUTATIONS ------------------
    """The 8 dihedral transforms, as permutations of the ring's 8 slots."""         # | A Go shape and its mirror image are the same shape. To use
    perms = []                                                                      # | that, each of the eight rotations and reflections is written
    for flip in (False, True):                                                      # | down as a rearrangement of the eight ring slots: apply the
        for rot in range(4):                                                        # | geometric transform to a slot's offset, then look up which
            perm = np.empty(8, dtype=np.int64)                                      # | slot the result lands in. These permutations are the only
            for k, (dr, dc) in enumerate(_RING):                                    # | geometry in the file; everything after works on integers.
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
    perms = _ring_symmetries()                                                      # +-- ONE WEIGHT PER SHAPE, NOT PER ORIENTATION ------------------
    codes = np.arange(N_PAT_RAW, dtype=np.int64)                                    # | Every one of the 65536 raw patterns is transformed all eight
    digits = np.empty((N_PAT_RAW, 8), dtype=np.int64)                               # | ways, and the smallest resulting number becomes its class. Two
    for k in range(8):                                                              # | patterns that are rotations of each other get the same class
        digits[:, k] = (codes // (4 ** k)) % 4                                      # | and therefore share a single learned weight, which cuts the
                                                                                    # | parameters to 8740 and multiplies the effective training data
    pow4 = 4 ** np.arange(8, dtype=np.int64)                                        # | by eight. Two tables are built rather than one because
    best = np.full(N_PAT_RAW, np.iinfo(np.int64).max, dtype=np.int64)               # | patterns are stored in absolute colours: reading a code as-is
    for perm in perms:                                                              # | gives black's view, and swapping the two stone digits first
        np.minimum(best, digits[:, perm] @ pow4, out=best)                          # | gives white's. Choosing the table by whose turn it is costs
    uniq, canon = np.unique(best, return_inverse=True)                              # | nothing at run time and is what lets one weight vector serve
    canon_b = canon.astype(np.int32)                                                # | both players, which is the paper's rule that features are
                                                                                    # | computed relative to the colour to play.
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
@njit(cache=True, inline="always")                                                  # +-- BUILDING ALL THE CODES FROM SCRATCH ------------------------
def _digit(v):                                                                      # | Colours become digits: 0 for off the board, 1 for empty, 2 for
    if v == EMPTY:                                                                  # | black, 3 for white. Absolute colours, not own and opponent,
        return 1                                                                    # | because own and opponent swap every single move and would
    elif v == BLACK:                                                                # | invalidate every stored code each turn. This full rebuild
        return 2                                                                    # | costs eight reads per point and runs once at the start of a
    return 3                                                                        # | rollout; after that the codes are repaired in place.


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


@njit(cache=True, inline="always")                                                  # +-- REPAIRING CODES AFTER A MOVE -------------------------------
def patch_pat(board, surr, pat, q):                                                 # | When the colour at one point changes, the only codes that
    """The colour at ``q`` just changed: repair the codes that mention it.

    Only q's 8 ring neighbours contain q in their own ring, and q sits in slot
    ``7-k`` of the neighbour found at q's slot ``k``.  So this is 8 single-digit
    patches, not 8 recomputations.
    """
    d = _digit(board[q])                                                            # | become wrong are those of its eight ring neighbours, because
    for k in range(8):                                                              # | those are exactly the points that have it in their own ring.
        r = surr[q, k]                                                              # | Each of those codes is wrong in exactly one digit, and the
        if r < 0:                                                                   # | antisymmetry of the ring ordering says which one: if the
            continue                                                                # | changed point sits at slot k of the neighbour's view, it
        pw = POW4[7 - k]                                                            # | occupies slot 7-k. So the fix is to subtract the old digit's
        old = (pat[r] // pw) % 4                                                    # | contribution and add the new one, eight times. That is the
        pat[r] += (d - old) * pw                                                    # | difference between eight operations per move and six hundred
                                                                                    # | and forty-eight.

@njit(cache=True)                                                                   # +-- PLACING A STONE AND REMEMBERING WHAT VANISHED --------------
def place_stone_track(board, nbrs, pt, color, buf, seen, tagbox, caps_out):         # | The same capture logic as the rules engine, with one addition:
    """``go.place_stone`` but also reporting *which* points were vacated.

    The rollout needs the captured points to patch their pattern codes; the
    plain version throws them away.
    """
    board[pt] = color                                                               # | the points of every captured stone are written out. The plain
    opp = 3 - color                                                                 # | version throws them away because nothing else needs them, but
    n_captured = 0                                                                  # | pattern codes do -- a captured stone changes colour to empty
    last_captured = -1                                                              # | exactly like a played stone changes colour, and each vacated
    for k in range(4):                                                              # | point must have its neighbours' patterns repaired too. Missing
        q = nbrs[pt, k]                                                             # | that would leave the rollout scoring moves against a board
        if q < 0 or board[q] != opp:                                                # | that no longer exists.
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
@njit(cache=True, inline="always")                                                  # +-- IS THIS MOVE A REPLY TO THE LAST ONE -----------------------
def response_offset(surr, p, last_move):                                            # | Which of the eight slots around the previous move the
    """Which ring slot of ``last_move`` the candidate ``p`` occupies (else 8)."""   # | candidate sits in, or 8 for anywhere else. This single feature
    if last_move < 0:                                                               # | is what stops rollouts from wandering: without it a rollout
        return 8                                                                    # | answers a threat on one side of the board by playing on the
    for k in range(8):                                                              # | other, and the game it plays out says nothing about the
        if surr[last_move, k] == p:                                                 # | position it started from.
            return k
    return 8


@njit(cache=True)                                                                   # +-- CAPTURES AND ATARI, COMPUTED EXACTLY -----------------------
def tactical_features(board, nbrs, p, color, buf, seen, tagbox, scratch, out):      # | Whether a move captures, and whether it leaves a chain with
    """Exact capture / atari features for playing ``color`` at ``p``.

    Computed by actually playing the move on a scratch copy of the board.  An
    incremental liberty-tracking scheme would be faster, but this runs for at
    most 8 candidates per move and being obviously correct is worth more here
    than being clever.
    """
    out[0] = 0.0                                                                    # | one liberty, cannot be read off the board without working out
    out[1] = 0.0                                                                    # | what the board would become. So the move is played on a
    out[2] = 0.0                                                                    # | scratch copy and the result measured. Incremental liberty
    out[3] = 0.0                                                                    # | tracking would be faster, but this runs for at most eight
                                                                                    # | candidates per move, and being obviously correct is worth more
    # Was any friendly neighbouring chain in atari before the move?                 # | here than being clever. Note the atari check has to happen
    in_atari_before = False                                                         # | before the move as well as after: rescuing a chain means it
    for k in range(4):                                                              # | had one liberty and now has more, which needs both readings.
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


@njit(cache=True)                                                                   # +-- SCORE EVERY POINT, CHEAPLY ---------------------------------
def move_scores(board, nbrs, surr, pat, canon, color, last_move,                    # | A flat pass writes the pattern weight plus the default
                w_pat, w_resp, w_tac, buf, seen, tagbox, scratch, tacbuf,           # | response weight onto every empty point, then a second short
                out):                                                               # | pass fixes up only the points that are special. At most eight
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
    nn = board.shape[0]                                                             # | candidates can be a reply to the last move, so asking all
    base_resp = w_resp[8]                    # "not adjacent to the last move"      # | eighty-one points whether they are special would cost six
    for p in range(nn):                                                             # | hundred and forty-eight comparisons to learn eight facts.
        if board[p] != EMPTY:                                                       # | Writing the answer only where it differs costs eight. Legality
            out[p] = -1e30                                                          # | is deliberately not checked here, because detecting suicide
        else:                                                                       # | needs a flood fill per point and doing sixty of those per move
            out[p] = w_pat[canon[pat[p]]] + base_resp                               # | would cost more than everything else in this file put
    if last_move >= 0 and last_move < nn:                                           # | together.
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
@njit(cache=True)                                                                   # +-- TURNING SCORES INTO A DRAW ---------------------------------
def sample_move(board, nbrs, diags, ndiag, surr, pat, canon, color, ko,             # | Exponentiating gives unnormalised probabilities. The maximum
                last_move, w_pat, w_resp, w_tac,                                    # | is subtracted first, which changes nothing about the
                buf, seen, tagbox, scratch, tacbuf, scores, wbuf, temp):            # | distribution because the weights are unnormalised anyway, and
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
    nn = board.shape[0]                                                             # | keeps the exponential away from overflow. This is done instead
    move_scores(board, nbrs, surr, pat, canon, color, last_move,                    # | of adding Gumbel noise and taking the largest, which samples
                w_pat, w_resp, w_tac, buf, seen, tagbox, scratch, tacbuf,           # | from the same distribution but costs two logarithms per point
                scores)                                                             # | rather than one exponential. Transcendental functions dominate
                                                                                    # | the cost of a rollout move, so halving their count roughly
    # Shift by the max before exponentiating: the weights are unnormalised, so      # | halves the cost of the whole rollout.
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

    while total > 1e-300:                                                           # +-- REJECTING WITHOUT DISTORTING -------------------------------
        u = np.random.random() * total                                              # | A drawn candidate may turn out to be illegal, or to fill one
        acc = 0.0                                                                   # | of our own eyes. Rather than pre-filtering every point, its
        pick = -1                                                                   # | weight is zeroed, removed from the running total, and another
        for p in range(nn):                                                         # | draw is taken. That is exactly sampling from the original
            if wbuf[p] > 0.0:                                                       # | distribution restricted to the moves that survive, so nothing
                acc += wbuf[p]                                                      # | is biased, and the expensive legality test is paid for only on
                if acc >= u:                                                        # | the one or two candidates actually drawn. The eye rule is not
                    pick = p                                                        # | a style preference: a policy allowed to fill its own eyes will
                    break                                                           # | destroy its own living groups and the rollout will never reach
                pick = p        # floating-point guard: keep the last live index    # | a position where both sides want to stop.
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


@njit(cache=True)                                                                   # +-- PLAYING THE GAME OUT ---------------------------------------
def playout(board, ko, to_play, last_move, n_passes, move_no,                       # | The whole leaf evaluation of the search, in one loop. Codes
            nbrs, diags, ndiag, surr, canon_b, canon_w,                             # | are built once here and then only patched, which is the entire
            w_pat, w_resp, w_tac, komi, max_moves,                                  # | reason for the machinery above. The colour to move selects
            buf, seen, tagbox, scratch, tacbuf, scores, wbuf, pat, caps, temp):     # | which of the two canonical tables is read, so one weight
    """Play ``board`` to the end with the rollout policy; return Black's score.

    ``board`` is mutated -- callers pass a scratch copy.  This is the whole
    "Evaluation" half of Fig. 3c: the rollout that produces z_L.
    """
    build_pat(board, surr, pat)                                                     # | vector serves both players. The board passed in is destroyed,
    while n_passes < 2 and move_no < max_moves:                                     # | so callers hand over a copy. The loop ends on two consecutive
        canon = canon_b if to_play == BLACK else canon_w                            # | passes, or on a move cap that guarantees termination even if a
        a = sample_move(board, nbrs, diags, ndiag, surr, pat, canon,                # | repetition the simple-ko rule cannot see were to arise.
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
class RolloutPolicy:                                                                # +-- WEIGHTS PLUS SCRATCH ---------------------------------------
    """Weights + scratch for p_pi.  Not thread-safe (the scratch is shared)."""     # | The learned weights are three small arrays: one per pattern
                                                                                    # | class, one per response slot, and one per tactical feature.
    def __init__(self, temp=1.0):                                                   # | Everything else here is scratch space, allocated once and
        self.w_pat = np.zeros(N_PAT, dtype=np.float64)                              # | reused, because a rollout runs hundreds of thousands of times
        self.w_resp = np.zeros(N_RESP, dtype=np.float64)                            # | and allocating inside it would cost more than the arithmetic.
        self.w_tac = np.zeros(N_TAC, dtype=np.float64)                              # | Sharing that scratch is what makes this object unsafe to use
        self.temp = float(temp)                                                     # | from two threads at once.
        self._alloc()

    def _alloc(self):
        self.buf = np.empty(NN, dtype=np.int32)
        self.seen = np.zeros(NN, dtype=np.int64)
        self.tagbox = np.zeros(1, dtype=np.int64)
        self.scratch = np.zeros(NN, dtype=np.int8)
        self.tacbuf = np.zeros(N_TAC, dtype=np.float64)
        self.scores = np.zeros(NN, dtype=np.float64)
        self.wbuf = np.zeros(NN, dtype=np.float64)
        self.pat = np.zeros(NN, dtype=np.int32)
        self.caps = np.zeros(NN, dtype=np.int32)

    def _canon(self, color):
        return CANON_B if color == BLACK else CANON_W

    # -- inference --------------------------------------------------------         # +-- THE PYTHON-FACING SURFACE ----------------------------------
    def logits(self, pos):                                                          # | Three ways to use the same weights: read the raw scores for
        """Raw linear scores over the NN points for the player to move."""          # | every point, draw a single move, or play a position out to the
        build_pat(pos.board, SURR, self.pat)                                        # | end and return the score. The first two rebuild the pattern
        move_scores(pos.board, NBRS, SURR, self.pat, self._canon(pos.to_play),      # | codes because they are called on positions arriving from
                    pos.to_play, pos.last_move, self.w_pat, self.w_resp,            # | outside with no history; the third rebuilds once and then
                    self.w_tac, self.buf, self.seen, self.tagbox,                   # | patches for the rest of the game. Saving keeps only the
                    self.scratch, self.tacbuf, self.scores)                         # | weights and the temperature, since everything else is derived
        return self.scores.copy()                                                   # | from the board size at import.

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
