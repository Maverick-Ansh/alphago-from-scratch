"""Feature-plane tests.

The one that matters most is ``test_symmetry_consistency``.  Augmenting the
training set 8x is free accuracy *if* the planes and the policy target are
transformed by the same group element, and silent label noise if they are not
-- and a mismatched rotation looks exactly like "the task is a bit harder than
expected", never like a bug.  So it is pinned against an independent path:
rotate the *board*, extract features from that, and require the result to equal
the rotation of the original features.
"""

import numpy as np

import ag.go as g
import ag.features as F
from ag.go import Position, coord, BLACK, WHITE, EMPTY, PASS, N, NN


def _play(seq):
    p = Position()
    for s in seq:
        p.play_checked(coord(s) if isinstance(s, str) else s)
    return p


def test_plane_count_and_basic_planes():
    fx = F.FeatureExtractor(with_colour=False)
    p = _play(["D4", "E5"])
    x = fx(p)
    assert x.shape == (F.N_PLANES_POLICY, N, N)
    b = x.reshape(F.N_PLANES_POLICY, NN)

    # black played D4, white played E5, black to move again
    assert p.to_play == BLACK
    assert b[F.P_OWN, coord("D4")] == 1        # own = black
    assert b[F.P_OPP, coord("E5")] == 1
    assert b[F.P_EMPTY, coord("A1")] == 1
    assert b[F.P_ONES].all() and not b[F.P_ZEROS].any()
    assert b[F.P_OWN].sum() == 1 and b[F.P_OPP].sum() == 1

    # colour plane only exists for the value network
    fv = F.FeatureExtractor(with_colour=True)
    xv = fv(p).reshape(F.N_PLANES_VALUE, NN)
    assert xv[F.P_COLOUR].all(), "black to play -> plane of ones"
    p2 = _play(["D4"])
    assert not F.FeatureExtractor(with_colour=True)(p2).reshape(
        F.N_PLANES_VALUE, NN)[F.P_COLOUR].any(), "white to play -> zeros"


def test_turns_since():
    fx = F.FeatureExtractor()
    p = _play(["D4", "E5", "F6", "G7"])       # D4 is 3 moves old, G7 is 0
    b = fx(p).reshape(-1, NN)
    assert b[F.P_AGE + 3, coord("D4")] == 1
    assert b[F.P_AGE + 0, coord("G7")] == 1
    assert b[F.P_AGE:F.P_AGE + 8, coord("A1")].sum() == 0, "empty point has no age"


def test_liberties_plane():
    fx = F.FeatureExtractor()
    p = _play(["D4", "A1"])                    # D4 alone: 4 liberties
    b = fx(p).reshape(-1, NN)
    assert b[F.P_LIB + 3, coord("D4")] == 1    # bucket for 4 liberties
    assert b[F.P_LIB + 1, coord("A1")] == 1    # corner stone: 2 liberties
    assert b[F.P_LIB:F.P_LIB + 8, coord("D4")].sum() == 1, "one-hot"


def test_capture_and_selfatari_and_libafter():
    # White D4 has one liberty at E4; black to play.
    p = Position()
    for s in ("D5", "C4", "D3"):
        p.board[coord(s)] = BLACK
    p.board[coord("D4")] = WHITE
    p.to_play = BLACK
    b = F.FeatureExtractor()(p).reshape(-1, NN)
    assert b[F.P_CAP + 0, coord("E4")] == 1, "playing E4 captures exactly 1 stone"
    assert b[F.P_CAP:F.P_CAP + 8, coord("A1")].sum() == 0, "no capture in the corner"

    # A point where black would be left on a single liberty: surround an empty
    # point with white, leaving one escape, then play into the trap.
    q = Position()
    for s in ("C5", "E5", "C3", "E3", "D6", "D2", "B4", "F4"):
        q.board[coord(s)] = WHITE
    for s in ("D5", "D3", "C4"):
        q.board[coord(s)] = WHITE
    q.to_play = BLACK
    bb = F.FeatureExtractor()(q).reshape(-1, NN)
    # D4 is surrounded on 3 sides by white with one liberty at E4
    assert bb[F.P_LIBAFTER + 0, coord("D4")] == 1, "one liberty after playing D4"
    assert bb[F.P_SELFATARI + 0, coord("D4")] == 1, "a 1-stone chain in atari"


def test_sensibleness_excludes_own_eyes():
    p = Position()
    for s in ("D5", "C4", "E4", "D3", "C5", "E5", "C3", "E3"):
        p.board[coord(s)] = BLACK
    p.board[coord("A1")] = WHITE
    p.to_play = BLACK
    b = F.FeatureExtractor()(p).reshape(-1, NN)
    assert b[F.P_SENSIBLE, coord("D4")] == 0, "D4 is black's own eye"
    assert b[F.P_SENSIBLE, coord("G7")] == 1


def _transform_position(p, k):
    """Apply dihedral element k to a Position, independently of features.py."""
    q = Position()
    q.board[:] = F.transform_planes(p.board.reshape(N, N), k).reshape(-1)
    q.move_age[:] = F.transform_planes(
        p.move_age.reshape(N, N), k).reshape(-1)
    q.to_play = p.to_play
    q.move_no = p.move_no
    q.ko = -1 if p.ko < 0 else int(F.transform_actions(np.array([p.ko]), k)[0])
    q.last_move = -1 if p.last_move < 0 else int(
        F.transform_actions(np.array([p.last_move]), k)[0])
    return q


def test_symmetry_consistency():
    """features(rotate(position)) == rotate(features(position)), for all 8.

    This is the check that keeps the 8x data augmentation honest.
    """
    rng = np.random.default_rng(3)
    fx = F.FeatureExtractor(with_colour=True)
    for trial in range(6):
        p = Position()
        for _ in range(int(rng.integers(5, 40))):
            idx = np.flatnonzero(p.legal_actions(exclude_eyes=True))
            idx = idx[idx != PASS]
            if len(idx) == 0:
                break
            p.play(int(rng.choice(idx)))
        base = fx(p)
        for k in range(8):
            direct = fx(_transform_position(p, k))
            rotated = F.transform_planes(base, k)
            assert np.array_equal(direct, rotated), \
                f"symmetry {k} mismatch on trial {trial}"


def test_transform_actions_matches_planes():
    """The action transform must agree with the plane transform, point by point.

    A one-hot plane containing only the move, rotated, must have its 1 exactly
    where ``transform_actions`` says the move went.
    """
    for k in range(8):
        for a in (0, 1, N - 1, N, NN - 1, coord("D4"), coord("B7")):
            onehot = np.zeros((N, N), dtype=np.float32)
            onehot[a // N, a % N] = 1.0
            moved = F.transform_planes(onehot, k).reshape(-1)
            a2 = int(F.transform_actions(np.array([a]), k)[0])
            assert moved[a2] == 1.0, f"k={k} a={a} -> {a2} mismatch"
        assert int(F.transform_actions(np.array([PASS]), k)[0]) == PASS, \
            "pass is invariant under board symmetry"


def test_transform_is_a_group_action():
    """Every element is a bijection on points, and identity is identity."""
    ident = F.transform_actions(np.arange(NN), 0)
    assert np.array_equal(ident, np.arange(NN))
    for k in range(8):
        m = F.transform_actions(np.arange(NN), k)
        assert len(np.unique(m)) == NN, f"symmetry {k} is not a bijection"


if __name__ == "__main__":
    import sys, traceback
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception:
            bad += 1
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
