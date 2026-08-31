"""Search tests.

A search with an inverted sign somewhere still runs, still returns legal moves,
and still looks like a working program -- it just plays badly, which is
indistinguishable from "the networks are weak".  The tests below pin the sign
chain end to end with synthetic evaluators whose right answer is known, and pin
the property that actually defines a working search: more simulations must play
better than fewer.
"""

import numpy as np

import ag.go as g
from ag.go import Position, coord, BLACK, WHITE, PASS, N_ACTIONS
from ag.mcts import MCTS
from ag.rollout import RolloutPolicy


TARGET = None  # set per test


def owns_target_value(target):
    """A value function that says: 'good for whoever owns `target`'.

    Returned in the frame of the player to move, exactly as ``value_fn`` is
    specified.  If black plays at ``target`` the resulting position has white
    to move and white does not own it, so the value is -1 for white, which the
    search must read as +1 for black.  Any dropped or doubled sign flip on the
    way from the leaf to the root breaks this test.
    """
    def f(pos):
        return 1.0 if pos.board[target] == pos.to_play else -1.0
    return f


def test_value_sign_chain_black():
    t = coord("D4")
    m = MCTS(n_sims=200, lmbda=0.0, value_fn=owns_target_value(t),
             rng=np.random.default_rng(0))
    a, root = m.choose(Position())
    assert a == t, "black should take the square its value function rewards"
    assert m.root_value(root) > 0, "and should evaluate the position as winning"


def test_value_sign_chain_white():
    """The same test with white to move.  This is the half that catches a
    missing flip: a search that ignores `to_play` passes the black test."""
    t = coord("D4")
    p = Position().play(coord("J1"))       # black plays elsewhere; white to move
    assert p.to_play == WHITE
    m = MCTS(n_sims=200, lmbda=0.0, value_fn=owns_target_value(t),
             rng=np.random.default_rng(0))
    a, root = m.choose(p)
    assert a == t, "white should take the same rewarded square"
    assert m.root_value(root) > 0, "white's own value must be positive too"


def test_estimators_run_only_when_lambda_asks():
    """At lambda=1 the value network must never be consulted, and at lambda=0
    no rollout may run.  If they leak into each other, the three arms of C5 are
    not the three arms of C5."""
    rp = RolloutPolicy()
    calls = {"v": 0}

    def counting_value(pos):
        calls["v"] += 1
        return 0.0

    m = MCTS(n_sims=50, lmbda=1.0, rollout=rp, rng=np.random.default_rng(0))
    root = m.search(Position())
    assert calls["v"] == 0
    assert root.Nr.sum() == 50 and root.Nv.sum() == 0

    calls["v"] = 0
    m = MCTS(n_sims=50, lmbda=0.0, value_fn=counting_value,
             rng=np.random.default_rng(0))
    root = m.search(Position())
    assert calls["v"] > 0
    assert root.Nv.sum() == 50 and root.Nr.sum() == 0

    m = MCTS(n_sims=50, lmbda=0.5, rollout=rp, value_fn=counting_value,
             rng=np.random.default_rng(0))
    root = m.search(Position())
    assert root.Nr.sum() == 50 and root.Nv.sum() == 50


def test_visit_count_is_lambda_independent():
    """Nvis must equal the simulation count at every lambda -- this is what
    keeps the PUCT exploration schedule identical across the C5 arms."""
    rp = RolloutPolicy()
    for lmbda in (0.0, 0.5, 1.0):
        m = MCTS(n_sims=64, lmbda=lmbda, rollout=rp,
                 value_fn=(lambda p: 0.0), rng=np.random.default_rng(0))
        root = m.search(Position())
        assert root.Nvis.sum() == 64, f"lambda={lmbda} counted {root.Nvis.sum()}"


def test_prior_is_masked_and_normalised():
    """A policy network knows nothing about legality; the search must."""
    p = Position()
    for s in ("D5", "C4", "E4", "D3"):
        p.board[coord(s)] = WHITE           # D4 is now suicide for black
    p.to_play = BLACK

    def flat_prior(pos):
        return np.ones(N_ACTIONS) / N_ACTIONS

    m = MCTS(n_sims=8, lmbda=1.0, rollout=RolloutPolicy(), prior_fn=flat_prior,
             rng=np.random.default_rng(0))
    root = m.search(p)
    assert root.P[coord("D4")] == 0.0, "illegal move must carry zero prior"
    assert abs(root.P.sum() - 1.0) < 1e-9, "prior must be renormalised"
    assert root.P[coord("A1")] > 0


def test_high_prior_move_gets_the_early_visits():
    """PUCT is prior-driven before evidence arrives."""
    t = coord("E5")

    def peaked(pos):
        q = np.full(N_ACTIONS, 1e-6)
        q[t] = 1.0
        return q

    m = MCTS(n_sims=30, lmbda=1.0, rollout=RolloutPolicy(), prior_fn=peaked,
             rng=np.random.default_rng(0))
    root = m.search(Position())
    assert int(np.argmax(root.Nvis)) == t, \
        "with 30 simulations the prior should still dominate"


def test_terminal_position_uses_the_real_result():
    """Two passes end the game; the search must report the score, not guess."""
    p = Position()
    p.board[:] = g.BLACK                    # black owns everything
    p.play(PASS)
    p.play(PASS)
    assert p.is_over() and p.winner() == BLACK
    m = MCTS(n_sims=4, lmbda=1.0, rollout=RolloutPolicy(),
             rng=np.random.default_rng(0))
    root = m._make_node(p)
    assert root.terminal_z == 1.0


def test_more_simulations_play_better():
    """The property that defines a working search.

    Slow, so it runs few games; the tournament measures this properly.  A
    search with a broken backup typically gets *worse* with more simulations,
    which this catches immediately.
    """
    from ag.players import MCTSPlayer
    from ag.arena import match
    rp = RolloutPolicy()
    strong = MCTSPlayer(name="s", n_sims=60, lmbda=1.0, rollout=rp,
                        rng=np.random.default_rng(1))
    weak = MCTSPlayer(name="w", n_sims=6, lmbda=1.0, rollout=rp,
                      rng=np.random.default_rng(2))
    r = match(strong, weak, n_games=10)
    assert r["wins"] >= 8, f"60 sims only beat 6 sims {r['wins']}/10"


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
