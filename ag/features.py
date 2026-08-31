"""Input feature planes -- Extended Data Table 2.

    | Stone colour        | 3 | Player stone / opponent stone / empty
    | Ones                | 1 | A constant plane filled with 1
    | Turns since         | 8 | How many turns since a move was played
    | Liberties           | 8 | Number of liberties (empty adjacent points)
    | Capture size        | 8 | How many opponent stones would be captured
    | Self-atari size     | 8 | How many of own stones would be captured
    | Liberties after move| 8 | Number of liberties after this move is played
    | Ladder capture      | 1 | Whether a move at this point is a successful ladder capture
    | Ladder escape       | 1 | Whether a move at this point is a successful ladder escape
    | Sensibleness        | 1 | Whether a move is legal and does not fill its own eyes
    | Zeros               | 1 | A constant plane filled with 0
    | Player color        | 1 | Whether current player is black   [value network only]

48 planes for the policy network, 49 for the value network.  And:

    "All features were computed relative to the current colour to play; for
     example, the stone colour at each intersection was represented as either
     player or opponent rather than black or white.  Each integer feature value
     is split into multiple 19 x 19 planes of binary values (one-hot encoding)."

Deviation: the two **ladder** planes are not implemented, leaving 46 planes for
the policy network and 47 for the value network.  A ladder is a running capture
sequence that resolves only when it reaches the edge of the board; on 9x9 every
ladder terminates within about four moves, and its first move is already
visible to the network through *capture size*, *self-atari size* and
*liberties after move*, which are implemented exactly.  Recorded in REPORT.md.

Everything else is faithful, including the one-hot bucketing at 8 ("separate
binary feature planes are used to represent whether an intersection has 1
liberty, 2 liberties, ..., >= 8 liberties").

Cost note: capture size, self-atari size and liberties-after-move each require
knowing the result of playing at every empty point, so the kernel plays each
candidate on a scratch board.  That is ~81 simulated moves per position, which
is why features are extracted once and cached rather than recomputed per epoch.
"""

import numpy as np
from numba import njit

from . import go
from .go import (N, NN, EMPTY, BLACK, WHITE, NBRS, DIAGS, NDIAG,
                 group_libs, is_legal, is_simple_eye, place_stone, _tag)

# plane layout
P_OWN, P_OPP, P_EMPTY, P_ONES = 0, 1, 2, 3
P_AGE = 4           # 8 planes
P_LIB = 12          # 8
P_CAP = 20          # 8
P_SELFATARI = 28    # 8
P_LIBAFTER = 36     # 8
P_SENSIBLE = 44
P_ZEROS = 45
P_COLOUR = 46       # value network only

N_PLANES_POLICY = 46
N_PLANES_VALUE = 47


@njit(cache=True, inline="always")
def _bucket(v):
    """One-hot bucket index for a count: 1,2,...,>=8 -> 0..7."""
    if v <= 1:
        return 0
    if v >= 8:
        return 7
    return v - 1


@njit(cache=True)
def extract(board, to_play, ko, move_age, nbrs, diags, ndiag,
            buf, seen, tagbox, scratch, out, with_colour):
    """Fill ``out`` (n_planes, NN) with the feature planes for this position."""
    nn = board.shape[0]
    opp = 3 - to_play
    out[:, :] = 0.0

    for p in range(nn):
        v = board[p]
        if v == to_play:
            out[P_OWN, p] = 1.0
        elif v == opp:
            out[P_OPP, p] = 1.0
        else:
            out[P_EMPTY, p] = 1.0
        out[P_ONES, p] = 1.0

        # "Turns since a move was played" -- only meaningful where a stone is.
        a = move_age[p]
        if v != EMPTY and a >= 0:
            k = a
            if k > 7:
                k = 7
            out[P_AGE + k, p] = 1.0

        # Liberties of the chain occupying p.
        if v != EMPTY:
            ng, nl = group_libs(board, nbrs, p, buf, seen, _tag(tagbox))
            out[P_LIB + _bucket(nl), p] = 1.0

    # The move-dependent planes: what happens if the player to move plays here.
    for p in range(nn):
        if board[p] != EMPTY:
            continue
        if not is_legal(board, nbrs, p, to_play, ko, buf, seen, tagbox):
            continue
        scratch[:] = board
        _, ncap = place_stone(scratch, nbrs, p, to_play, buf, seen, tagbox)
        ng, nl = group_libs(scratch, nbrs, p, buf, seen, _tag(tagbox))

        if ncap > 0:
            out[P_CAP + _bucket(ncap), p] = 1.0
        # "Self-atari size: how many of own stones would be captured" -- i.e.
        # the size of the chain we would leave on a single liberty.
        if nl == 1:
            out[P_SELFATARI + _bucket(ng), p] = 1.0
        out[P_LIBAFTER + _bucket(nl), p] = 1.0

        if not is_simple_eye(board, nbrs, diags, ndiag, p, to_play):
            out[P_SENSIBLE, p] = 1.0

    # P_ZEROS stays zero, by construction.
    if with_colour:
        c = 1.0 if to_play == BLACK else 0.0
        for p in range(nn):
            out[P_COLOUR, p] = c
    return out


class FeatureExtractor:
    """Owns the scratch buffers; produces (C, N, N) float32 planes."""

    def __init__(self, with_colour=False):
        self.with_colour = with_colour
        self.n_planes = N_PLANES_VALUE if with_colour else N_PLANES_POLICY
        self.buf = np.empty(NN, dtype=np.int32)
        self.seen = np.zeros(NN, dtype=np.int32)
        self.tagbox = np.zeros(1, dtype=np.int64)
        self.scratch = np.zeros(NN, dtype=np.int8)
        self.out = np.zeros((self.n_planes, NN), dtype=np.float32)

    def __call__(self, pos):
        extract(pos.board, pos.to_play, pos.ko, pos.move_age,
                NBRS, DIAGS, NDIAG, self.buf, self.seen, self.tagbox,
                self.scratch, self.out, self.with_colour)
        return self.out.reshape(self.n_planes, N, N).copy()

    def batch(self, positions):
        return np.stack([self(p) for p in positions])


def from_dataset(ds, idx, with_colour=False):
    """Extract planes for rows ``idx`` of a loaded dataset."""
    from . import data
    fx = FeatureExtractor(with_colour=with_colour)
    out = np.empty((len(idx), fx.n_planes, N, N), dtype=np.float32)
    for i, j in enumerate(idx):
        out[i] = fx(data.decode(ds, j))
    return out


# --------------------------------------------------------------------------
# dihedral symmetry
# --------------------------------------------------------------------------
# "we exploit symmetries at run-time by dynamically transforming each position
#  s using the dihedral group of eight reflections and rotations".
# Used two ways here: to augment the supervised training set 8x, and as the
# implicit symmetry ensemble at evaluation time.

def transform_planes(x, k):
    """Apply dihedral element ``k`` (0..7) to a (..., N, N) array."""
    r, f = k % 4, k // 4
    y = np.rot90(x, r, axes=(-2, -1))
    if f:
        y = np.flip(y, axis=-1)
    return np.ascontiguousarray(y)


def transform_actions(a, k):
    """Map action indices through the same dihedral element.

    Built by transforming an index grid rather than by deriving the formula:
    the point of the array below is that it is *the same* transform the planes
    get, so the policy target can never fall out of step with the input.
    """
    grid = np.arange(NN, dtype=np.int64).reshape(N, N)
    moved = transform_planes(grid, k).reshape(-1)
    # moved[i] = index in the ORIGINAL board that lands at position i
    inv = np.empty(NN, dtype=np.int64)
    inv[moved] = np.arange(NN)
    a = np.asarray(a)
    out = np.where(a < NN, inv[np.clip(a, 0, NN - 1)], NN)
    return out
