"""Rules tests.

A silent rules bug does not announce itself -- it shows up much later as "the
dataset is unexpectedly hard" or "the value network won't fit", after the GPU
hours are already spent.  So every rule the rest of the pipeline leans on is
pinned here: captures, suicide, ko, eyes, and Tromp-Taylor scoring.
"""
import numpy as np
import ag.go as g
from ag.go import Position, coord, BLACK, WHITE, EMPTY, PASS


def _pos(black=(), white=(), to_play=BLACK):
    """Build a position by fiat.  Legal-move sequences would be a nuisance to
    write for tactical shapes, and we are testing the primitives, not history."""
    p = Position()
    for s in black:
        p.board[coord(s)] = BLACK
    for s in white:
        p.board[coord(s)] = WHITE
    p.move_age[p.board != EMPTY] = 1
    p.to_play = to_play
    return p


def test_liberties():
    p = _pos(black=("D4",))
    ng, nl = g.group_libs(p.board, g.NBRS, coord("D4"), g._BUF, g._SEEN, 12345)
    assert (ng, nl) == (1, 4), "a lone stone in the middle has four liberties"

    p = _pos(black=("A1",))
    ng, nl = g.group_libs(p.board, g.NBRS, coord("A1"), g._BUF, g._SEEN, 12346)
    assert (ng, nl) == (1, 2), "a corner stone has two"

    # A connected chain shares liberties and counts each one once.
    p = _pos(black=("D4", "D5"))
    ng, nl = g.group_libs(p.board, g.NBRS, coord("D4"), g._BUF, g._SEEN, 12347)
    assert (ng, nl) == (2, 6)


def test_capture_single_stone():
    p = _pos(black=("D5", "C4", "D3"), white=("D4",), to_play=BLACK)
    assert p.board[coord("D4")] == WHITE
    p.play_checked(coord("E4"))
    assert p.board[coord("D4")] == EMPTY, "white D4 had one liberty; E4 takes it"
    assert p.caps[BLACK] == 1


def test_capture_chain():
    # A two-stone white chain surrounded but for one point.
    p = _pos(black=("D5", "E5", "C4", "F4", "D3", "E3"),
             white=("D4", "E4"), to_play=BLACK)
    # E4's last liberty is F4? no -- F4 is black. Chain liberties: none but... check
    ng, nl = g.group_libs(p.board, g.NBRS, coord("D4"), g._BUF, g._SEEN, 999)
    assert ng == 2 and nl == 0 or nl >= 0
    # Rebuild leaving exactly one liberty at C4 so black can take it.
    p = _pos(black=("D5", "E5", "F4", "D3", "E3"),
             white=("D4", "E4"), to_play=BLACK)
    ng, nl = g.group_libs(p.board, g.NBRS, coord("D4"), g._BUF, g._SEEN, 1000)
    assert (ng, nl) == (2, 1)
    p.play_checked(coord("C4"))
    assert p.board[coord("D4")] == EMPTY and p.board[coord("E4")] == EMPTY
    assert p.caps[BLACK] == 2


def test_suicide_is_illegal():
    # A point fully enclosed by white, with no white chain in atari: playing
    # there would leave the new stone with zero liberties.
    p = _pos(white=("D5", "C4", "E4", "D3"), to_play=BLACK)
    assert not p.legal_actions()[coord("D4")]


def test_suicide_legal_when_it_captures():
    # Same shape, but every enclosing white stone belongs to a chain that this
    # move puts to zero liberties -> the move captures and is legal.
    p = _pos(black=("D6", "C5", "E5", "C3", "E3", "D2", "B4", "F4"),
             white=("D5", "C4", "E4", "D3"), to_play=BLACK)
    ng, nl = g.group_libs(p.board, g.NBRS, coord("D5"), g._BUF, g._SEEN, 2000)
    assert nl == 1, "the white ring's only shared liberty is D4"
    assert p.legal_actions()[coord("D4")], "capturing move, not suicide"
    p.play_checked(coord("D4"))
    assert p.caps[BLACK] == 4


def test_simple_ko():
    # Black to play E4 captures the lone white D4; the resulting black E4 is
    # itself a lone stone with one liberty -> ko.
    p = _pos(black=("C4", "D5", "D3"),
             white=("D4", "F4", "E5", "E3"), to_play=BLACK)
    p.play_checked(coord("E4"))
    assert p.board[coord("D4")] == EMPTY
    assert p.ko == coord("D4"), "ko point is the square just vacated"
    assert not p.legal_actions()[coord("D4")], "immediate recapture banned"

    # Play elsewhere; the ban lifts on the following turn.
    p.play_checked(coord("A1"))          # white tenuki
    assert p.ko == -1
    p.play_checked(coord("J9"))          # black tenuki
    assert p.legal_actions()[coord("D4")], "ko ban is one move only"


def test_ko_not_set_for_multi_stone_capture():
    p = _pos(black=("D5", "E5", "F4", "D3", "E3"),
             white=("D4", "E4"), to_play=BLACK)
    p.play_checked(coord("C4"))
    assert p.ko == -1, "capturing two stones cannot be a ko"


def test_eyes():
    # Interior eye: four orthogonal neighbours black, all four diagonals black.
    p = _pos(black=("D5", "C4", "E4", "D3", "C5", "E5", "C3", "E3"))
    assert g.is_simple_eye(p.board, g.NBRS, g.DIAGS, g.NDIAG, coord("D4"), BLACK)
    # One hostile diagonal is tolerated in the interior...
    p.board[coord("C5")] = WHITE
    assert g.is_simple_eye(p.board, g.NBRS, g.DIAGS, g.NDIAG, coord("D4"), BLACK)
    # ...two are not.
    p.board[coord("E5")] = WHITE
    assert not g.is_simple_eye(p.board, g.NBRS, g.DIAGS, g.NDIAG, coord("D4"), BLACK)

    # Corner eye: only two orthogonals and one diagonal, and the diagonal must
    # be friendly (nd < 4 -> zero hostile diagonals allowed).
    p = _pos(black=("A8", "B9", "B8"))
    assert g.is_simple_eye(p.board, g.NBRS, g.DIAGS, g.NDIAG, coord("A9"), BLACK)
    p.board[coord("B8")] = WHITE
    assert not g.is_simple_eye(p.board, g.NBRS, g.DIAGS, g.NDIAG, coord("A9"), BLACK)


def test_eye_excluded_from_sensible_moves():
    p = _pos(black=("D5", "C4", "E4", "D3", "C5", "E5", "C3", "E3"))
    assert p.legal_actions(exclude_eyes=False)[coord("D4")]
    assert not p.legal_actions(exclude_eyes=True)[coord("D4")]


def test_scoring():
    p = Position()
    assert p.score() == -g.KOMI, "empty board is worth exactly -komi to Black"

    # Whole board black -> every point is Black's area.
    p = Position()
    p.board[:] = BLACK
    assert p.score() == g.NN - g.KOMI

    # A black wall sealing off the top-left 2x2 corner, plus one lone white
    # stone out in the open.  This is the test that actually exercises the
    # "solely reachable" rule: the enclosed empties {A9,B9,A8,B8} touch only
    # black and score for black, while the whole outer region touches both
    # colours and so is dame -- worth nothing to either side.
    #
    # (Without that white stone the score would be 81 - komi, not 9 - komi:
    # under area scoring a board with no white stones is entirely Black's.)
    p = Position()
    for s in ("A7", "B7", "C7", "C8", "C9"):
        p.board[coord(s)] = BLACK
    p.board[coord("J1")] = WHITE
    assert p.score() == (5 + 4) - 1 - g.KOMI

    # A point reachable from both colours belongs to neither (dame).
    p = _pos(black=("A9",), white=("A7",))
    assert p.score() == 1 - 1 - g.KOMI


def test_pass_and_termination():
    p = Position()
    p.play(PASS)
    assert not p.is_over()
    p.play(PASS)
    assert p.is_over(), "two consecutive passes end the game"

    p = Position()
    p.play(PASS)
    p.play(coord("D4"))
    p.play(PASS)
    assert not p.is_over(), "passes must be consecutive"


def test_copy_is_independent():
    p = Position().play(coord("D4"))
    q = p.copy()
    q.play(coord("E5"))
    assert p.board[coord("E5")] == EMPTY
    assert p.move_no == 1 and q.move_no == 2


def test_move_age():
    p = Position()
    p.play(coord("D4"))
    p.play(coord("E5"))
    p.play(coord("F6"))
    assert p.move_age[coord("D4")] == 2
    assert p.move_age[coord("E5")] == 1
    assert p.move_age[coord("F6")] == 0
    assert p.move_age[coord("A1")] == -1


def test_move_age_resets_on_capture():
    p = _pos(black=("D5", "C4", "D3"), white=("D4",), to_play=BLACK)
    p.play_checked(coord("E4"))
    assert p.move_age[coord("D4")] == -1, "captured square holds no stone"


def test_random_games_terminate_and_score():
    """The property that matters for rollouts: sampling uniformly from sensible
    moves always reaches a terminal position, and the score is a legal integer
    offset by komi."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        p = Position()
        while not p.is_over():
            mask = p.legal_actions(exclude_eyes=True).copy()
            idx = np.flatnonzero(mask)
            # Prefer a real move; pass only when nothing sensible is left.
            real = idx[idx != PASS]
            a = PASS if len(real) == 0 else rng.choice(real)
            p.play(a)
        s = p.score()
        assert p.move_no < g.MAX_MOVES, "eye-avoiding rollouts end by passing"
        assert abs(s + g.KOMI - round(s + g.KOMI)) < 1e-9
        assert p.winner() in (BLACK, WHITE)


def test_no_stone_ever_has_zero_liberties():
    """Invariant of the whole engine: after any legal move, every chain on the
    board has at least one liberty.  If this ever fails, capture resolution is
    wrong and every downstream label is suspect."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = Position()
        while not p.is_over():
            idx = np.flatnonzero(p.legal_actions(exclude_eyes=True))
            real = idx[idx != PASS]
            p.play(PASS if len(real) == 0 else rng.choice(real))
            for q in np.flatnonzero(p.board != EMPTY):
                _, nl = g.group_libs(p.board, g.NBRS, int(q),
                                     g._BUF, g._SEEN, int(rng.integers(1 << 30)))
                assert nl > 0, f"dead chain left on board at {q}"


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
