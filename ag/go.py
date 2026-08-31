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
  cost more than the fill itself, so instead ``seen`` stores a monotonically
  increasing *tag* drawn from ``tagbox`` (a 1-element array so numba can mutate
  it).  A stale entry can never collide with the current tag.
  ``seen`` MUST be int64, the same width as the tag.  It was int32 once, and
  past 2**31 the store silently truncated, so ``seen[q] == tag`` could never be
  true again, the fill stopped marking anything visited, and it re-appended the
  same stones without bound -- straight off the end of ``buf`` and into the
  heap.  That took about 35 minutes of continuous play to reach.  See REPORT.md.
* Ko: we implement **simple ko** (an immediate single-stone recapture is
  banned), not positional superko.  This is what fast Go engines use inside
  rollouts.  The deviation from the paper is recorded in REPORT.md.

Terminal conditions: two consecutive passes, or a move cap (so that long
superko cycles cannot loop forever).
"""

import os                                                                             # +-- BOARD SIZE, KOMI, AND A HARD STOP ------------------------
import numpy as np                                                                    # | Board size is read from the environment so the whole
from numba import njit                                                                # | pipeline can be re-run at 7x7 without editing anything. Komi
                                                                                      # | is 7.5, the paper's value: the half point makes draws
# --------------------------------------------------------------------------          # | impossible, so every game has a winner and the outcome is
# Board geometry.  AG_BOARD_SIZE lets the whole pipeline be re-run at 7x7 if          # | always exactly +1 or -1. The move cap exists because simple
# the compute budget bites; 9 is the default and what the report uses.                # | ko forbids only an immediate recapture, not a longer
# --------------------------------------------------------------------------          # | repeating cycle; without a cap a rollout could in principle
N = int(os.environ.get("AG_BOARD_SIZE", "9"))                                         # | loop forever, and a rollout that never returns stops the
NN = N * N                                                                            # | whole search.
PASS = NN                 # action index reserved for the pass move
N_ACTIONS = NN + 1

EMPTY, BLACK, WHITE = 0, 1, 2

# Komi 7.5, exactly as in the paper ("games were scored using Chinese rules
# with a komi of 7.5 points").  The half point also makes draws impossible.
KOMI = float(os.environ.get("AG_KOMI", "7.5"))

# Rollouts and self-play games are capped so no game can run forever.
MAX_MOVES = int(os.environ.get("AG_MAX_MOVES", str(2 * NN)))


def _build_tables(n):                                                                 # +-- WHO IS NEXT TO WHOM, WORKED OUT ONCE ---------------------
    nn = n * n                                                                        # | For every point, its up-to-four orthogonal neighbours and
    nbrs = np.full((nn, 4), -1, dtype=np.int32)                                       # | up-to-four diagonal ones, with -1 marking positions off the
    diags = np.full((nn, 4), -1, dtype=np.int32)                                      # | board. Storing this once removes every row and column
    ndiag = np.zeros(nn, dtype=np.int32)                                              # | calculation and every edge test from code that runs tens of
    for r in range(n):                                                                # | millions of times a second. Orthogonal neighbours decide
        for c in range(n):                                                            # | liberties and captures; diagonals only matter for deciding
            p = r * n + c                                                             # | whether an empty point is an eye.
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
@njit(cache=True, inline="always")                                                    # +-- FINDING A CHAIN AND COUNTING ITS AIR ---------------------
def _tag(tagbox):                                                                     # | A chain is all the same-coloured stones connected
    tagbox[0] += 1                                                                    # | orthogonally, and it lives as long as it touches at least
    return tagbox[0]                                                                  # | one empty point. This walks outward from one stone,
                                                                                      # | collecting the chain into a buffer and counting the distinct
                                                                                      # | empty points it touches. The visited array is stamped with
@njit(cache=True)                                                                     # | an increasing tag rather than cleared, because clearing
def group_libs(board, nbrs, pt, buf, seen, tag):                                      # | eighty-one entries would cost more than the walk itself, and
    """Flood-fill the chain containing ``pt``.

    Returns ``(n_group, n_libs)``; the chain's points land in ``buf[:n_group]``.
    ``seen`` is stamped with ``tag`` for both chain members and liberties, so
    each liberty is counted exactly once.
    """
    color = board[pt]                                                                 # | a stamp from an earlier call can never be mistaken for the
    buf[0] = pt                                                                       # | current one. Liberties are stamped too, which is what stops
    seen[pt] = tag                                                                    # | the same empty point being counted once for every stone that
    n_group = 1                                                                       # | touches it.
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
                if n_group >= buf.shape[0]:
                    # Unreachable while tags are wider than the stores that
                    # hold them: a chain cannot contain more points than the
                    # board has.  Kept as a hard stop so that a future width
                    # mismatch degrades into a wrong answer instead of heap
                    # corruption, which is far harder to trace back.
                    return n_group, n_libs
                buf[n_group] = q
                n_group += 1
    return n_group, n_libs


@njit(cache=True)                                                                     # +-- IS THIS MOVE ALLOWED -------------------------------------
def is_legal(board, nbrs, pt, color, ko, buf, seen, tagbox):                          # | A move is illegal if it would leave its own stone with no
    """Legality of playing ``color`` at point ``pt`` -- no suicide, no ko.

    Decided without mutating the board: a move is legal iff it has an immediate
    liberty, OR joins a friendly chain that still has a liberty to spare, OR
    captures an enemy chain that is down to its last liberty.
    """
    if board[pt] != EMPTY:                                                            # | air, unless it takes the opponent's last air first. This
        return False                                                                  # | decides that without ever touching the board: the move is
    if pt == ko:                                                                      # | fine if it has an empty point beside it, or if it joins a
        return False                                                                  # | friendly chain that has air to spare, or if it takes the
    opp = 3 - color                                                                   # | final liberty of an enemy chain. The ko point is refused
    for k in range(4):                                                                # | outright. Deciding rather than simulating matters because
        q = nbrs[pt, k]                                                               # | this runs for every candidate move of every position.
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


@njit(cache=True)                                                                     # +-- PLAYING A STONE, AND SPOTTING A KO -----------------------
def place_stone(board, nbrs, pt, color, buf, seen, tagbox):                           # | The stone goes down, then each adjacent enemy chain is
    """Place an (already-verified-legal) stone and resolve captures.

    Returns ``(ko_point, n_captured)``.  ``ko_point`` is the simple-ko ban for
    the opponent's next move: set only when exactly one stone was captured and
    the stone just played is itself a lone stone with a single liberty -- the
    textbook ko shape.
    """
    board[pt] = color                                                                 # | checked and removed if it has no air left. Order matters:
    opp = 3 - color                                                                   # | enemies are resolved before the new stone's own liberties
    n_captured = 0                                                                    # | are looked at, which is what makes a capturing move legal
    last_captured = -1                                                                # | even when the stone would otherwise be surrounded. A ko is
    for k in range(4):                                                                # | recognised by its shape rather than by comparing board
        q = nbrs[pt, k]                                                               # | positions: exactly one stone was taken, and the stone just
        if q < 0 or board[q] != opp:                                                  # | played is itself alone with exactly one liberty. Only then
            continue                                                                  # | could the opponent take straight back and return the board
        ng, nl = group_libs(board, nbrs, q, buf, seen, _tag(tagbox))                  # | to what it was, so only then is the recapture banned.
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


@njit(cache=True)                                                                     # +-- WHAT COUNTS AS AN EYE ------------------------------------
def is_simple_eye(board, nbrs, diags, ndiag, pt, color):                              # | An empty point surrounded on all four sides by one colour,
    """True if ``pt`` is a one-point eye owned by ``color``.

    Used for the paper's "sensibleness" feature ("whether a move is legal and
    does not fill its own eyes", Extended Data Table 2) and to stop rollouts
    from filling their own eyes, which is what makes them terminate.
    Definition: all orthogonal neighbours are ``color``, and the opponent holds
    no diagonal in the corner/edge case, at most one in the interior.
    """
    if board[pt] != EMPTY:                                                            # | whose diagonals that colour also mostly holds. The diagonal
        return False                                                                  # | rule is what separates a real eye from a point that merely
    for k in range(4):                                                                # | looks enclosed: in the middle of the board one hostile
        q = nbrs[pt, k]                                                               # | diagonal can be tolerated because the intrusion can be
        if q >= 0 and board[q] != color:                                              # | captured, but on an edge or in a corner there are fewer
            return False                                                              # | diagonals and a single hostile one is enough to break the
    opp = 3 - color                                                                   # | eye. This matters far beyond style. A player allowed to fill
    n_bad = 0                                                                         # | its own eyes kills its own living groups, and a rollout
    nd = ndiag[pt]                                                                    # | doing that never reaches a position where both sides are
    for k in range(nd):                                                               # | content to stop.
        if board[diags[pt, k]] == opp:
            n_bad += 1
    if nd < 4:
        return n_bad == 0
    return n_bad <= 1


@njit(cache=True)                                                                     # +-- ALL THE LEGAL MOVES AT ONCE ------------------------------
def legal_mask(board, nbrs, diags, ndiag, color, ko, buf, seen, tagbox,               # | A flag for every point plus pass, which is always allowed.
               out, exclude_eyes):                                                    # | With eyes excluded this becomes the set of sensible moves:
    """Fill ``out`` (length NN+1) with the legal-move mask; pass is always legal.

    ``exclude_eyes`` additionally forbids filling one's own eyes -- the
    "sensible move" set that rollouts sample from.
    """
    nn = board.shape[0]                                                               # | what self-play samples from, and the set the policy
    n = 0                                                                             # | network's output is masked down to. The network is never
    for p in range(nn):                                                               # | told which moves are legal and will happily put probability
        ok = board[p] == EMPTY and p != ko                                            # | on occupied points, so this mask is the only thing standing
        if ok:                                                                        # | between it and an illegal move.
            ok = is_legal(board, nbrs, p, color, ko, buf, seen, tagbox)
        if ok and exclude_eyes:
            if is_simple_eye(board, nbrs, diags, ndiag, p, color):
                ok = False
        out[p] = ok
        if ok:
            n += 1
    out[nn] = True
    return n


@njit(cache=True)                                                                     # +-- COUNTING THE SCORE WITHOUT JUDGING LIFE ------------------
def score_tromp_taylor(board, nbrs, komi):                                            # | Area scoring: each side gets its stones plus every empty
    """Area score from Black's point of view: stones + solely-reachable empties.

    Chinese-style scoring, so a rollout that plays until both sides pass can be
    scored mechanically with no notion of "dead" stones -- the reason MCTS Go
    engines all use it.
    """
    nn = board.shape[0]                                                               # | region that touches only its own colour. A region touching
    seen = np.zeros(nn, np.uint8)                                                     # | both belongs to neither. The important property is that no
    stack = np.empty(nn, np.int32)                                                    # | judgement of which groups are alive is needed anywhere - a
    black = 0                                                                         # | game played until both sides pass can be scored purely
    white = 0                                                                         # | mechanically, which is why every Monte-Carlo Go program uses
    for p in range(nn):                                                               # | this rule and not territory scoring. Regions are found by
        if board[p] == BLACK:                                                         # | flooding outward from each unvisited empty point and
            black += 1                                                                # | recording which colours the flood ran into.
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


# --------------------------------------------------------------------------          # +-- SCRATCH LIVES OUTSIDE THE POSITION -----------------------
# Python-level game state.                                                            # | These buffers are pure working space with no meaning between
#                                                                                     # | calls, so they are shared at module level instead of held
# The scratch buffers are module-global on purpose: they are pure transients,         # | per position. The search copies a position once per
# so keeping them out of Position makes Position.copy() -- which MCTS calls           # | simulation step, and this choice is what keeps that copy
# once per simulation step -- cost only the ~250 bytes of real game state.            # | down to the few hundred bytes of real game state rather than
# --------------------------------------------------------------------------          # | several kilobytes of scratch that would be overwritten
_BUF = np.empty(NN, dtype=np.int32)                                                   # | immediately anyway.
_SEEN = np.zeros(NN, dtype=np.int64)   # must match the tag width
_TAGBOX = np.zeros(1, dtype=np.int64)
_MASK = np.zeros(N_ACTIONS, dtype=np.bool_)


class Position:                                                                       # +-- WHAT A POSITION HAS TO REMEMBER --------------------------
    """One Go position plus the small amount of history the features need."""         # | The stones, whose turn it is, the ko ban, and how many
                                                                                      # | passes have just happened. Beyond the rules there is one
    __slots__ = ("board", "to_play", "ko", "n_passes", "move_no",                     # | extra: the age of every stone, which the network features
                 "move_age", "last_move", "caps")                                     # | need in order to see the recent history of the fight rather
                                                                                      # | than a static picture. Slots are declared explicitly so a
    def __init__(self):                                                               # | position carries no dictionary, because millions of these
        self.board = np.zeros(NN, dtype=np.int8)                                      # | get created and copied.
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
    def legal_actions(self, exclude_eyes=False):                                      # +-- READING THE POSITION -------------------------------------
        """Boolean mask over NN+1 actions.  Returns a *view* of shared scratch,
        so copy it if you need to keep it past the next call."""
        legal_mask(self.board, NBRS, DIAGS, NDIAG, self.to_play, self.ko,             # | The legal-move mask hands back shared scratch, so anything
                   _BUF, _SEEN, _TAGBOX, _MASK, exclude_eyes)                         # | that needs to keep it must copy it first. The score is
        return _MASK                                                                  # | always from black's point of view with komi already taken
                                                                                      # | off, and because komi ends in a half point the sign of that
    def is_over(self):                                                                # | number alone decides the game. Reporting a result for a
        return self.n_passes >= 2 or self.move_no >= MAX_MOVES                        # | given colour is the paper's z: +1 for a win, -1 for a loss,
                                                                                      # | with no draw and no notion of margin.
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
    def play(self, action):                                                           # +-- MAKING A MOVE --------------------------------------------
        """Apply ``action`` (a point index, or PASS) for the player to move."""       # | A pass clears the ko ban and counts toward ending the game;
        if action == PASS:                                                            # | a stone resolves captures and may create a new ban. Stone
            self.n_passes += 1                                                        # | ages advance for everything still standing, and any point
            self.ko = -1                                                              # | that just became empty is reset, so a captured stone leaves
        else:                                                                         # | no trace in the history planes. Note the ages are bumped
            ko, ncap = place_stone(self.board, NBRS, action, self.to_play,            # | before the new stone is set to zero, which is what makes the
                                   _BUF, _SEEN, _TAGBOX)                              # | freshest stone distinguishable from a one-move-old one. The
            self.ko = ko                                                              # | unchecked version is used inside search where legality was
            self.caps[self.to_play] += ncap                                           # | already established; the checked version exists so a bug in
            self.n_passes = 0                                                         # | a caller fails loudly instead of quietly corrupting a game.
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
    def __str__(self):                                                                # +-- PRINTING AND TYPING COORDINATES --------------------------
        sym = {EMPTY: ".", BLACK: "X", WHITE: "O"}                                    # | Rows numbered from the bottom and columns lettered with I
        rows = []                                                                     # | skipped, the standard Go convention. It exists so that tests
        for r in range(N):                                                            # | and debugging sessions can name a point the way a Go player
            rows.append("%2d " % (N - r) + " ".join(                                  # | would, which is the difference between a test that reads as
                sym[self.board[r * N + c]] for c in range(N)))                        # | a board position and one that reads as a list of integers.
        rows.append("   " + " ".join("ABCDEFGHJKLMNOPQRST"[c] for c in range(N)))
        rows.append(f"   to_play={'X' if self.to_play == BLACK else 'O'} "
                    f"ko={self.ko} move={self.move_no} passes={self.n_passes}")
        return "\n".join(rows)


def coord(s):
    """'D4' -> flat index, for writing readable tests."""
    col = "ABCDEFGHJKLMNOPQRST".index(s[0].upper())
    row = N - int(s[1:])
    return row * N + col
