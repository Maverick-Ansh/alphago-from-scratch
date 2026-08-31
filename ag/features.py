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

# plane layout                                                                        # +-- WHERE EACH FEATURE LIVES ---------------------------------
P_OWN, P_OPP, P_EMPTY, P_ONES = 0, 1, 2, 3                                            # | Offsets into the plane stack. Six of the twelve feature
P_AGE = 4           # 8 planes                                                        # | groups are eight planes wide, because the paper turns every
P_LIB = 12          # 8                                                               # | count into a one-hot bank rather than feeding a number: a
P_CAP = 20          # 8                                                               # | point with three liberties gets a 1 in the third plane and 0
P_SELFATARI = 28    # 8                                                               # | everywhere else. A convolution over a raw count would have
P_LIBAFTER = 36     # 8                                                               # | to learn that four is between three and five and that the
P_SENSIBLE = 44                                                                       # | difference between one and two matters far more than between
P_ZEROS = 45                                                                          # | seven and eight; one-hot lets it learn a separate response
P_COLOUR = 46       # value network only                                              # | to each case and costs only planes, which are cheap.

N_PLANES_POLICY = 46
N_PLANES_VALUE = 47


@njit(cache=True, inline="always")                                                    # +-- BUCKETING COUNTS AT EIGHT --------------------------------
def _bucket(v):                                                                       # | Counts of one through seven get their own plane and
    """One-hot bucket index for a count: 1,2,...,>=8 -> 0..7."""                      # | everything from eight upward shares the last one. Beyond
    if v <= 1:                                                                        # | eight liberties a chain is simply safe and the exact number
        return 0                                                                      # | carries no tactical meaning, so spending planes to
    if v >= 8:                                                                        # | distinguish nine from ten would add parameters and no
        return 7                                                                      # | information.
    return v - 1


@njit(cache=True)                                                                     # +-- THE STATIC PLANES ----------------------------------------
def extract(board, to_play, ko, move_age, nbrs, diags, ndiag,                         # | One pass writing what is true of the board as it stands.
            buf, seen, tagbox, scratch, out, with_colour):                            # | Stone colours are recorded as own and opponent rather than
    """Fill ``out`` (n_planes, NN) with the feature planes for this position."""      # | black and white, so the same learned filter works for both
    nn = board.shape[0]                                                               # | players and the network never has to learn one set of shapes
    opp = 3 - to_play                                                                 # | twice. The constant plane of ones gives the convolution a
    out[:, :] = 0.0                                                                   # | way to tell a real board edge from the zero padding around
                                                                                      # | it, which is otherwise invisible to it. Stone age is only
    for p in range(nn):                                                               # | meaningful where a stone stands, and the liberty count is
        v = board[p]                                                                  # | read by flooding each chain, which repeats work for every
        if v == to_play:                                                              # | stone in a chain but keeps the code obviously right.
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
            out[P_LIB + _bucket(nl), p] = 1.0                                         # +-- THE PLANES THAT ASK WHAT WOULD HAPPEN --------------------
                                                                                      # | Capture size, self-atari size and liberties-after-move
    # The move-dependent planes: what happens if the player to move plays here.       # | cannot be read off the board, because each asks about a
    for p in range(nn):                                                               # | board that does not exist yet. So the move is played on a
        if board[p] != EMPTY:                                                         # | scratch copy of every empty legal point in turn and the
            continue                                                                  # | result measured. This is why extracting features costs about
        if not is_legal(board, nbrs, p, to_play, ko, buf, seen, tagbox):              # | eighty simulated moves, and why they are computed once and
            continue                                                                  # | cached rather than recomputed every epoch. Self-atari is
        scratch[:] = board                                                            # | recorded only when the resulting chain would have exactly
        _, ncap = place_stone(scratch, nbrs, p, to_play, buf, seen, tagbox)           # | one liberty, and its size is the number of stones that would
        ng, nl = group_libs(scratch, nbrs, p, buf, seen, _tag(tagbox))                # | then be at risk, which is what makes the difference between
                                                                                      # | a harmless sacrifice and losing a group.
        if ncap > 0:
            out[P_CAP + _bucket(ncap), p] = 1.0
        # "Self-atari size: how many of own stones would be captured" -- i.e.
        # the size of the chain we would leave on a single liberty.
        if nl == 1:
            out[P_SELFATARI + _bucket(ng), p] = 1.0
        out[P_LIBAFTER + _bucket(nl), p] = 1.0

        if not is_simple_eye(board, nbrs, diags, ndiag, p, to_play):
            out[P_SENSIBLE, p] = 1.0
                                                                                      # +-- THE TWO CONSTANT PLANES ----------------------------------
    # P_ZEROS stays zero, by construction.                                            # | Sensibleness marks moves that are legal and do not fill our
    if with_colour:                                                                   # | own eye. The plane of zeros is in the paper and is kept for
        c = 1.0 if to_play == BLACK else 0.0                                          # | fidelity; it carries no information and exists only as a
        for p in range(nn):                                                           # | fixed reference the network can use or ignore. Colour to
            out[P_COLOUR, p] = c                                                      # | play is added only for the value network, because who is to
    return out                                                                        # | move changes who wins a position but does not change which
                                                                                      # | move is best-shaped.

class FeatureExtractor:                                                               # +-- OWNING THE SCRATCH, HANDING OUT COPIES -------------------
    """Owns the scratch buffers; produces (C, N, N) float32 planes."""                # | The buffers are allocated once and reused across every call,
                                                                                      # | since feature extraction runs millions of times. The output
    def __init__(self, with_colour=False):                                            # | is copied on the way out because the internal buffer is
        self.with_colour = with_colour                                                # | overwritten by the next call, and a caller stacking a batch
        self.n_planes = N_PLANES_VALUE if with_colour else N_PLANES_POLICY            # | would otherwise end up with the same position repeated.
        self.buf = np.empty(NN, dtype=np.int32)                                       # | Planes come out shaped as a square board rather than a flat
        self.seen = np.zeros(NN, dtype=np.int64)                                      # | list because that is what a convolution consumes.
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


def from_dataset(ds, idx, with_colour=False):                                         # +-- FEATURES STRAIGHT FROM STORED RECORDS --------------------
    """Extract planes for rows ``idx`` of a loaded dataset."""                        # | Data sets hold raw game state, not planes, because planes
    from . import data                                                                # | are about twenty times larger and because storing state
    fx = FeatureExtractor(with_colour=with_colour)                                    # | means the feature set can change later without regenerating
    out = np.empty((len(idx), fx.n_planes, N, N), dtype=np.float32)                   # | hours of games. This rebuilds a position from a stored row
    for i, j in enumerate(idx):                                                       # | and extracts from it.
        out[i] = fx(data.decode(ds, j))
    return out


# --------------------------------------------------------------------------          # +-- THE EIGHT WAYS TO LOOK AT A BOARD ------------------------
# dihedral symmetry                                                                   # | A Go position rotated or mirrored is the same position, so
# --------------------------------------------------------------------------          # | one game record is eight training examples. This is used
# "we exploit symmetries at run-time by dynamically transforming each position        # | twice: to multiply the supervised training set, and at
#  s using the dihedral group of eight reflections and rotations".                    # | evaluation time as the ensemble the paper describes, where
# Used two ways here: to augment the supervised training set 8x, and as the           # | either all eight are averaged or one is picked at random per
# implicit symmetry ensemble at evaluation time.                                      # | lookup.

def transform_planes(x, k):
    """Apply dihedral element ``k`` (0..7) to a (..., N, N) array."""
    r, f = k % 4, k // 4
    y = np.rot90(x, r, axes=(-2, -1))
    if f:
        y = np.flip(y, axis=-1)
    return np.ascontiguousarray(y)


def transform_actions(a, k):                                                          # +-- ROTATING THE ANSWER WITH THE QUESTION --------------------
    """Map action indices through the same dihedral element.

    Built by transforming an index grid rather than by deriving the formula:
    the point of the array below is that it is *the same* transform the planes
    get, so the policy target can never fall out of step with the input.
    """
    grid = np.arange(NN, dtype=np.int64).reshape(N, N)                                # | If the board is rotated the target move must rotate with it,
    moved = transform_planes(grid, k).reshape(-1)                                     # | or every augmented example is mislabelled and training
    # moved[i] = index in the ORIGINAL board that lands at position i                 # | quietly gets worse forever. The mapping is not derived by
    inv = np.empty(NN, dtype=np.int64)                                                # | hand: an index grid is put through the very same transform
    inv[moved] = np.arange(NN)                                                        # | the planes get, which makes it impossible for the two to
    a = np.asarray(a)                                                                 # | disagree. Inverting that gives, for each original point,
    out = np.where(a < NN, inv[np.clip(a, 0, NN - 1)], NN)                            # | where it ended up. Pass has no location on the board, so it
    return out                                                                        # | is left alone.
