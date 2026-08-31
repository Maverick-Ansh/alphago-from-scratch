"""Network and evaluator tests.

The symmetry round-trip is the one worth writing.  An ensemble that forgets to
undo its own rotation still returns a valid probability distribution, still
trains, and still plays -- just slightly worse, in a way indistinguishable from
"the net is a bit weak".  So it is pinned by its defining property: the
ensembled prediction must *commute* with the symmetry.
"""

import numpy as np
import torch

import ag.go as g
import ag.features as F
import ag.nets as nets
from ag.go import Position, coord, PASS, N, NN


def _random_pos(seed=0, n_moves=25):
    rng = np.random.default_rng(seed)
    p = Position()
    for _ in range(n_moves):
        idx = np.flatnonzero(p.legal_actions(exclude_eyes=True))
        idx = idx[idx != PASS]
        if len(idx) == 0:
            break
        p.play(int(rng.choice(idx)))
    return p


def _transform_position(p, k):
    q = Position()
    q.board[:] = F.transform_planes(p.board.reshape(N, N), k).reshape(-1)
    q.move_age[:] = F.transform_planes(p.move_age.reshape(N, N), k).reshape(-1)
    q.to_play = p.to_play
    q.move_no = p.move_no
    q.ko = -1 if p.ko < 0 else int(F.transform_actions(np.array([p.ko]), k)[0])
    q.last_move = -1 if p.last_move < 0 else int(
        F.transform_actions(np.array([p.last_move]), k)[0])
    return q


def test_shapes_and_ranges():
    torch.manual_seed(0)
    pol = nets.PolicyNet(n_filters=16, n_layers=2)
    val = nets.ValueNet(n_filters=16, n_layers=2)
    x = torch.randn(4, F.N_PLANES_POLICY, N, N)
    xv = torch.randn(4, F.N_PLANES_VALUE, N, N)
    assert pol(x).shape == (4, NN), "policy logits: one per point, no pass"
    v = val(xv)
    assert v.shape == (4,)
    assert (v.abs() <= 1).all(), "tanh output unit"


def test_per_position_bias_is_real():
    """The final layer has 'a different bias for each position'."""
    pol = nets.PolicyNet(n_filters=8, n_layers=1)
    assert pol.pos_bias.shape == (NN,)
    with torch.no_grad():
        pol.pos_bias[coord("D4")] = 10.0
    x = torch.zeros(1, F.N_PLANES_POLICY, N, N)
    out = pol(x)[0]
    assert int(out.argmax()) == coord("D4"), "position bias must reach the logits"


def test_policy_ensemble_commutes_with_symmetry():
    """ensemble(rotate(s)) == rotate(ensemble(s)).

    True for the all-8 ensemble by construction *if* the inverse map is right,
    and false if the rotation is undone in the wrong direction.
    """
    torch.manual_seed(1)
    net = nets.PolicyNet(n_filters=12, n_layers=2)
    ev = nets.PolicyEvaluator(net, symmetry="all", cache=False)
    p = _random_pos(seed=2)
    base = ev(p)[:NN]
    for k in range(8):
        rot = ev(_transform_position(p, k))[:NN]
        fwd = F.transform_actions(np.arange(NN), k)
        # base[a] is the original point a; after symmetry k it sits at fwd[a]
        assert np.allclose(rot[fwd], base, atol=1e-5), \
            f"ensemble does not commute with symmetry {k}"


def test_single_symmetry_roundtrip():
    """Even with one symmetry, evaluating a rotated board and mapping back must
    reproduce evaluating the original board under that same rotation."""
    torch.manual_seed(3)
    net = nets.PolicyNet(n_filters=12, n_layers=2)
    plain = nets.PolicyEvaluator(net, symmetry="none", cache=False)
    p = _random_pos(seed=5)
    for k in range(8):
        q = _transform_position(p, k)
        direct = plain(q)[:NN]                       # in q's frame
        # evaluate q's planes but ask the evaluator to undo symmetry k
        ev_k = nets.PolicyEvaluator(net, symmetry="none", cache=False)
        raw = ev_k._run([F.transform_planes(ev_k._planes(p), k)])
        mapped = ev_k._unify(raw, [k])[:NN]          # back in p's frame
        fwd = F.transform_actions(np.arange(NN), k)
        assert np.allclose(direct[fwd], mapped, atol=1e-5), \
            f"single-symmetry round trip failed for k={k}"


def test_policy_probabilities_are_a_distribution():
    net = nets.PolicyNet(n_filters=8, n_layers=1)
    ev = nets.PolicyEvaluator(net, symmetry="all", cache=False)
    out = ev(_random_pos(seed=7))
    assert out.shape == (NN + 1,)
    assert out[PASS] == 0.0, "no pass output; the player supplies the pass rule"
    assert abs(out[:NN].sum() - 1.0) < 1e-6
    assert (out >= 0).all()


def test_value_ensemble_is_invariant():
    torch.manual_seed(4)
    net = nets.ValueNet(n_filters=12, n_layers=2)
    ev = nets.ValueEvaluator(net, with_colour=True, symmetry="all", cache=False)
    p = _random_pos(seed=9)
    base = ev(p)
    for k in range(8):
        assert abs(ev(_transform_position(p, k)) - base) < 1e-5, \
            "a board value must not depend on how the board is rotated"


def test_cache_returns_same_answer():
    torch.manual_seed(5)
    net = nets.PolicyNet(n_filters=8, n_layers=1)
    ev = nets.PolicyEvaluator(net, symmetry="none", cache=True)
    p = _random_pos(seed=11)
    a, b = ev(p), ev(p.copy())
    assert np.array_equal(a, b) and ev.n_forward == 1, "second call must hit cache"


def test_save_load_roundtrip(tmp_path=None):
    import tempfile, os
    d = tempfile.mkdtemp()
    net = nets.PolicyNet(n_filters=16, n_layers=3)
    path = os.path.join(d, "p.pt")
    nets.save(net, path, note="hello")
    net2, ck = nets.load(path)
    assert ck["note"] == "hello"
    x = torch.randn(2, F.N_PLANES_POLICY, N, N)
    net.eval()
    assert torch.allclose(net(x), net2(x), atol=1e-6)


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
