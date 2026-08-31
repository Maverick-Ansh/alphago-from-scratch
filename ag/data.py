"""Dataset record format.

A training example is a position, the move an expert played there, and the
outcome of the game it came from.  Positions are stored as the raw game state
rather than as expanded feature planes: the planes are 20x larger, and keeping
the state means the feature set can be changed later without regenerating the
data (which costs CPU-hours).

Everything the feature extractor needs is here -- board, side to move, ko point,
per-stone age, and the previous move.
"""

import numpy as np

from . import go
from .go import NN, EMPTY

AGE_CAP = 100          # ages above this are indistinguishable to the features        # +-- STORE THE GAME, NOT THE PICTURE OF IT --------------------
                                                                                      # | A record is the raw game state, not the expanded feature
                                                                                      # | planes. Planes are about twenty times larger, and more
def encode(pos, action, z, game_id=0):                                                # | importantly they freeze the feature set: changing which
    """One record as a tuple of scalars/arrays."""                                    # | planes exist would mean regenerating the games, which costs
    return (pos.board.astype(np.uint8),                                               # | CPU-hours. Stone ages are clipped before storing because the
            np.uint8(pos.to_play),                                                    # | features bucket everything past eight together anyway, so
            np.int16(pos.ko),                                                         # | the exact age of a very old stone is information nobody
            np.clip(pos.move_age, -1, AGE_CAP).astype(np.int8),                       # | reads. Decoding rebuilds a position from a row, which is how
            np.int16(pos.last_move),                                                  # | a stored record becomes something the engine and the
            np.int16(pos.move_no),                                                    # | extractor can both work on.
            np.int16(action),
            np.int8(z),
            np.int32(game_id))


def decode(ds, i):
    """Rebuild a ``go.Position`` from row ``i`` of a loaded dataset."""
    p = go.Position()
    p.board[:] = ds["boards"][i]
    p.to_play = int(ds["to_play"][i])
    p.ko = int(ds["ko"][i])
    p.move_age[:] = ds["move_age"][i]
    p.last_move = int(ds["last_move"][i])
    p.move_no = int(ds["move_no"][i])
    return p


def save(path, records):                                                              # +-- ONE FILE PER SHARD ---------------------------------------
    """Write a list of ``encode`` tuples to a compressed .npz."""                     # | Columns are stacked and written compressed, which matters
    if not records:                                                                   # | because the boards are almost all zeros and the planes are
        raise ValueError("no records")                                                # | binary. Each field is stored at the narrowest type that
    cols = list(zip(*records))                                                        # | holds it, since a data set is hundreds of thousands of rows
    np.savez_compressed(                                                              # | and every wasted byte is multiplied by all of them.
        path,
        boards=np.stack(cols[0]),
        to_play=np.array(cols[1], dtype=np.uint8),
        ko=np.array(cols[2], dtype=np.int16),
        move_age=np.stack(cols[3]),
        last_move=np.array(cols[4], dtype=np.int16),
        move_no=np.array(cols[5], dtype=np.int16),
        action=np.array(cols[6], dtype=np.int16),
        z=np.array(cols[7], dtype=np.int8),
        game_id=np.array(cols[8], dtype=np.int32),
    )


_KEYS = ("boards", "to_play", "ko", "move_age", "last_move", "move_no",               # +-- RENUMBERING GAMES SO SPLITS STAY HONEST ------------------
         "action", "z", "game_id")                                                    # | Shards are written by independent workers that each number
                                                                                      # | their games from zero, so concatenating them naively would
                                                                                      # | merge unrelated games under one identifier. Everything
def load(paths):                                                                      # | downstream splits train from test **by game**, precisely
    """Load and concatenate shards, renumbering game ids so they stay unique."""      # | because positions inside a game differ by one stone and
    if isinstance(paths, (str, bytes)):                                               # | share an outcome; two different games sharing an identifier
        paths = [paths]                                                               # | would put near-duplicate positions on both sides of that
    parts, offset = [], 0                                                             # | split and report a test score that is partly memorised. So
    for pth in paths:                                                                 # | each shard's identifiers are shifted past the highest one
        z = np.load(pth)                                                              # | seen so far.
        d = {k: z[k] for k in _KEYS}
        d["game_id"] = d["game_id"] + offset
        offset = int(d["game_id"].max()) + 1 if len(d["game_id"]) else offset
        parts.append(d)
    return {k: np.concatenate([p[k] for p in parts]) for k in _KEYS}
