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

AGE_CAP = 100          # ages above this are indistinguishable to the features


def encode(pos, action, z, game_id=0):
    """One record as a tuple of scalars/arrays."""
    return (pos.board.astype(np.uint8),
            np.uint8(pos.to_play),
            np.int16(pos.ko),
            np.clip(pos.move_age, -1, AGE_CAP).astype(np.int8),
            np.int16(pos.last_move),
            np.int16(pos.move_no),
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


def save(path, records):
    """Write a list of ``encode`` tuples to a compressed .npz."""
    if not records:
        raise ValueError("no records")
    cols = list(zip(*records))
    np.savez_compressed(
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


_KEYS = ("boards", "to_play", "ko", "move_age", "last_move", "move_no",
         "action", "z", "game_id")


def load(paths):
    """Load and concatenate shards, renumbering game ids so they stay unique."""
    if isinstance(paths, (str, bytes)):
        paths = [paths]
    parts, offset = [], 0
    for pth in paths:
        z = np.load(pth)
        d = {k: z[k] for k in _KEYS}
        d["game_id"] = d["game_id"] + offset
        offset = int(d["game_id"].max()) + 1 if len(d["game_id"]) else offset
        parts.append(d)
    return {k: np.concatenate([p[k] for p in parts]) for k in _KEYS}
