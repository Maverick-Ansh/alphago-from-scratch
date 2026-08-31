"""Go rules engine (Tromp-Taylor area scoring, simple ko).

This is the *substrate* of the whole reproduction.  Silver et al. 2016 assume a
correct 19x19 Go implementation and never discuss one; we need our own, and it
has to be fast enough that Monte-Carlo rollouts are not the bottleneck.

Design notes
------------
* Board is a flat ``int8`` array of length ``N*N``: 0 empty, 1 black, 2 white.
  Flat indexing keeps the numba kernels branch-light; a precomputed neighbour
  table ``NBRS[p, k]`` (-1 = off-board) removes all bounds arithmetic from the
  inner loop.
* Chains are found by flood fill rather than union-find.  On a 9x9 board the
  mean chain is a handful of stones, so a fill costs ~30 array reads -- cheaper
  than maintaining incremental liberty counts, and far easier to verify.
* The flood fill needs a "visited" array.  Zeroing 81 entries per call would
  cost more than the fill itself, so instead ``seen`` is ``int32`` and each call
  stamps it with a monotonically increasing *tag* drawn from ``tagbox`` (a
  1-element array so numba can mutate it).  A stale entry can never collide
  with the current tag.
* Ko: we implement **simple ko** (an immediate single-stone recapture is
  banned), not positional superko.  This is what fast Go engines use inside
  rollouts.  The deviation from the paper is recorded in REPORT.md.

Terminal conditions: two consecutive passes, or a move cap (so that long
superko cycles cannot loop forever).
"""

import os
import numpy as np
from numba import njit

# --------------------------------------------------------------------------
# Board geometry.  AG_BOARD_SIZE lets the whole pipeline be re-run at 7x7 if
# the compute budget bites; 9 is the default and what the report uses.
# --------------------------------------------------------------------------
N = int(os.environ.get("AG_BOARD_SIZE", "9"))
NN = N * N
PASS = NN                 # action index reserved for the pass move
N_ACTIONS = NN + 1

EMPTY, BLACK, WHITE = 0, 1, 2

# Komi 7.5, exactly as in the paper ("games were scored using Chinese rules
# with a komi of 7.5 points").  The half point also makes draws impossible.
KOMI = float(os.environ.get("AG_KOMI", "7.5"))

# Rollouts and self-play games are capped so no game can run forever.
MAX_MOVES = int(os.environ.get("AG_MAX_MOVES", str(2 * NN)))


def _build_tables(n):
    nn = n * n
    nbrs = np.full((nn, 4), -1, dtype=np.int32)
    diags = np.full((nn, 4), -1, dtype=np.int32)
    ndiag = np.zeros(nn, dtype=np.int32)
    for r in range(n):
        for c in range(n):
            p = r * n + c
            k = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    nbrs[p, k] = rr * n + cc
                    k += 1
            k = 0
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    diags[p, k] = rr * n + cc
                    k += 1
            ndiag[p] = k
    return nbrs, diags, ndiag


NBRS, DIAGS, NDIAG = _build_tables(N)


# --------------------------------------------------------------------------
# numba kernels
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _tag(tagbox):
    tagbox[0] += 1
    return tagbox[0]


@njit(cache=True)
def group_libs(board, nbrs, pt, buf, seen, tag):
    """Flood-fill the chain containing ``pt``.

    Returns ``(n_group, n_libs)``; the chain's points land in ``buf[:n_group]``.
    ``seen`` is stamped with ``tag`` for both chain members and liberties, so
    each liberty is counted exactly once.
    """
    color = board[pt]
    buf[0] = pt
    seen[pt] = tag
    n_group = 1
    n_libs = 0
    head = 0
    while head < n_group:
        p = buf[head]
        head += 1
        for k in range(4):
            q = nbrs[p, k]
            if q < 0:
                continue
            if seen[q] == tag:
                continue
            v = board[q]
            if v == EMPTY:
                seen[q] = tag
                n_libs += 1
            elif v == color:
                seen[q] = tag
                buf[n_group] = q
                n_group += 1
    return n_group, n_libs


@njit(cache=True)
def is_legal(board, nbrs, pt, color, ko, buf, seen, tagbox):
    """Legality of playing ``color`` at point ``pt`` -- no suicide, no ko.

    Decided without mutating the board: a move is legal iff it has an immediate
    liberty, OR joins a friendly chain that still has a liberty to spare, OR
    captures an enemy chain that is down to its last liberty.
    """
    if board[pt] != EMPTY:
        return False
    if pt == ko:
        return False
    opp = 3 - color
    for k in range(4):
        q = nbrs[pt, k]
        if q >= 0 and board[q] == EMPTY:
            return True
    for k in range(4):
        q = nbrs[pt, k]
        if q < 0:
            continue
        v = board[q]
        ng, nl = group_libs(board, nbrs, q, buf, seen, _tag(tagbox))
        if v == color and nl > 1:
            return True
        if v == opp and nl == 1:
            return True
    return False


@njit(cache=True)
def place_stone(board, nbrs, pt, color, buf, seen, tagbox):
    """Place an (already-verified-legal) stone and resolve captures.

    Returns ``(ko_point, n_captured)``.  ``ko_point`` is the simple-ko ban for
    the opponent's next move: set only when exactly one stone was captured and
    the stone just played is itself a lone stone with a single liberty -- the
    textbook ko shape.
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
                last_captured = buf[i]
                board[buf[i]] = EMPTY
            n_captured += ng
    ko = -1
    if n_captured == 1:
        ng, nl = group_libs(board, nbrs, pt, buf, seen, _tag(tagbox))
        if ng == 1 and nl == 1:
            ko = last_captured
    return ko, n_captured


@njit(cache=True)
def is_simple_eye(board, nbrs, diags, ndiag, pt, color):
    """True if ``pt`` is a one-point eye owned by ``color``.

    Used for the paper's "sensibleness" feature ("whether a move is legal and
    does not fill its own eyes", Extended Data Table 2) and to stop rollouts
    from filling their own eyes, which is what makes them terminate.
    Definition: all orthogonal neighbours are ``color``, and the opponent holds
    no diagonal in the corner/edge case, at most one in the interior.
    """
    if board[pt] != EMPTY:
        return False
    for k in range(4):
        q = nbrs[pt, k]
        if q >= 0 and board[q] != color:
            return False
    opp = 3 - color
    n_bad = 0
    nd = ndiag[pt]
    for k in range(nd):
        if board[diags[pt, k]] == opp:
            n_bad += 1
    if nd < 4:
        return n_bad == 0
    return n_bad <= 1


@njit(cache=True)
def legal_mask(board, nbrs, diags, ndiag, color, ko, buf, seen, tagbox,
               out, exclude_eyes):
    """Fill ``out`` (length NN+1) with the legal-move mask; pass is always legal.

    ``exclude_eyes`` additionally forbids filling one's own eyes -- the
    "sensible move" set that rollouts sample from.
    """
    nn = board.shape[0]
    n = 0
    for p in range(nn):
        ok = board[p] == EMPTY and p != ko
        if ok:
            ok = is_legal(board, nbrs, p, color, ko, buf, seen, tagbox)
        if ok and exclude_eyes:
            if is_simple_eye(board, nbrs, diags, ndiag, p, color):
                ok = False
        out[p] = ok
        if ok:
            n += 1
    out[nn] = True
    return n


@njit(cache=True)
def score_tromp_taylor(board, nbrs, komi):
    """Area score from Black's point of view: stones + solely-reachable empties.

    Chinese-style scoring, so a rollout that plays until both sides pass can be
    scored mechanically with no notion of "dead" stones -- the reason MCTS Go
    engines all use it.
    """
    nn = board.shape[0]
    seen = np.zeros(nn, np.uint8)
    stack = np.empty(nn, np.int32)
    black = 0
    white = 0
    for p in range(nn):
        if board[p] == BLACK:
            black += 1
        elif board[p] == WHITE:
            white += 1
    for p in range(nn):
        if board[p] != EMPTY or seen[p]:
            continue
        stack[0] = p
        seen[p] = 1
        ns = 1
        head = 0
        n_region = 0
        touch_b = False
        touch_w = False
        while head < ns:
            q = stack[head]
            head += 1
            n_region += 1
            for k in range(4):
                r = nbrs[q, k]
                if r < 0:
                    continue
                v = board[r]
                if v == BLACK:
                    touch_b = True
                elif v == WHITE:
                    touch_w = True
                elif seen[r] == 0:
                    seen[r] = 1
                    stack[ns] = r
                    ns += 1
        if touch_b and not touch_w:
            black += n_region
        elif touch_w and not touch_b:
            white += n_region
    return float(black) - float(white) - komi


# --------------------------------------------------------------------------
# Python-level game state.
#
# The scratch buffers are module-global on purpose: they are pure transients,
# so keeping them out of Position makes Position.copy() -- which MCTS calls
# once per simulation step -- cost only the ~250 bytes of real game state.
# --------------------------------------------------------------------------
_BUF = np.empty(NN, dtype=np.int32)
_SEEN = np.zeros(NN, dtype=np.int32)
_TAGBOX = np.zeros(1, dtype=np.int64)
_MASK = np.zeros(N_ACTIONS, dtype=np.bool_)


class Position:
    """One Go position plus the small amount of history the features need."""

    __slots__ = ("board", "to_play", "ko", "n_passes", "move_no",
                 "move_age", "last_move", "caps")

    def __init__(self):
        self.board = np.zeros(NN, dtype=np.int8)
        self.to_play = BLACK
        self.ko = -1
        self.n_passes = 0
        self.move_no = 0
        # "Turns since a move was played" (Extended Data Table 2). -1 = empty.
        self.move_age = np.full(NN, -1, dtype=np.int16)
        self.last_move = -1
        self.caps = [0, 0, 0]          # stones captured by [_, black, white]

    def copy(self):
        p = Position.__new__(Position)
        p.board = self.board.copy()
        p.to_play = self.to_play
        p.ko = self.ko
        p.n_passes = self.n_passes
        p.move_no = self.move_no
        p.move_age = self.move_age.copy()
        p.last_move = self.last_move
        p.caps = list(self.caps)
        return p

    # -- queries ----------------------------------------------------------
    def legal_actions(self, exclude_eyes=False):
        """Boolean mask over NN+1 actions.  Returns a *view* of shared scratch,
        so copy it if you need to keep it past the next call."""
        legal_mask(self.board, NBRS, DIAGS, NDIAG, self.to_play, self.ko,
                   _BUF, _SEEN, _TAGBOX, _MASK, exclude_eyes)
        return _MASK

    def is_over(self):
        return self.n_passes >= 2 or self.move_no >= MAX_MOVES

    def score(self):
        """Area score from Black's perspective, komi already subtracted."""
        return score_tromp_taylor(self.board, NBRS, KOMI)

    def winner(self):
        """BLACK or WHITE.  Komi is a half-integer so there are no draws."""
        return BLACK if self.score() > 0 else WHITE

    def result_for(self, color):
        """+1 if ``color`` won, -1 otherwise -- the paper's z_t = +-r(s_T)."""
        return 1.0 if self.winner() == color else -1.0

    # -- transition -------------------------------------------------------
    def play(self, action):
        """Apply ``action`` (a point index, or PASS) for the player to move."""
        if action == PASS:
            self.n_passes += 1
            self.ko = -1
        else:
            ko, ncap = place_stone(self.board, NBRS, action, self.to_play,
                                   _BUF, _SEEN, _TAGBOX)
            self.ko = ko
            self.caps[self.to_play] += ncap
            self.n_passes = 0
            # move_age is bumped for every stone still on the board; stones
            # that were just captured are reset to -1.
            np.add(self.move_age, 1, out=self.move_age,
                   where=(self.move_age >= 0))
            self.move_age[self.board == EMPTY] = -1
            self.move_age[action] = 0
        self.last_move = action
        self.move_no += 1
        self.to_play = 3 - self.to_play
        return self

    def play_checked(self, action):
        if action != PASS and not self.legal_actions()[action]:
            raise ValueError(f"illegal move {action} for player {self.to_play}")
        return self.play(action)

    # -- display ----------------------------------------------------------
    def __str__(self):
        sym = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        rows = []
        for r in range(N):
            rows.append("%2d " % (N - r) + " ".join(
                sym[self.board[r * N + c]] for c in range(N)))
        rows.append("   " + " ".join("ABCDEFGHJKLMNOPQRST"[c] for c in range(N)))
        rows.append(f"   to_play={'X' if self.to_play == BLACK else 'O'} "
                    f"ko={self.ko} move={self.move_no} passes={self.n_passes}")
        return "\n".join(rows)


def coord(s):
    """'D4' -> flat index, for writing readable tests."""
    col = "ABCDEFGHJKLMNOPQRST".index(s[0].upper())
    row = N - int(s[1:])
    return row * N + col
