"""Regression tests for the feature cache, which is shared between processes.

The pipeline launches the three SL widths at the same moment against one cache
file.  ``open_memmap`` creates that file at its full final size, zero filled,
and then fills it row by row.  For the few seconds that takes, a second process
saw a file of exactly the right shape, believed it, and read zeros.

Training tolerated that -- the training set is re-read from the memmap every
step, so it turns real as soon as the build finishes -- but the held-out set is
copied into RAM once at startup and stayed blank for the whole run.  Two of the
three widths reported a test accuracy of 1.4% while their weights were fine.
It was recorded as a step-size collapse, and it was not one.

So the property under test is not "the cache is fast" but **a reader can never
observe a partial cache**: it is built into a temporary file and renamed into
place, and the rows are verifiable because ``P_ONES`` is a constant plane that
no valid row can be missing.
"""

import gc
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

import ag.features as F
from ag import data
from ag.go import N, Position
from train_sl import build_feature_cache, check_cache


def _tiny_dataset(n_games=4, n_moves=6):
    """A handful of short games, encoded the way the pipeline stores them."""
    recs = []
    rng = np.random.default_rng(0)
    for gi in range(n_games):
        pos = Position()
        for _ in range(n_moves):
            legal = np.flatnonzero(pos.legal_actions(exclude_eyes=True))
            legal = legal[legal != N * N]
            if not len(legal):
                break
            a = int(rng.choice(legal))
            recs.append(data.encode(pos.copy(), a, 1, game_id=gi))
            pos.play(a)
    return recs


def _tmpdir():
    """A temp dir that survives a memmap still being open on Windows.

    ``build_feature_cache`` hands back a read-only memmap, which keeps a handle
    on the file.  POSIX unlinks it happily; Windows refuses with WinError 32 and
    the cleanup raises straight through the test.  Releasing the reference is
    the honest fix, but subprocess-held handles in the race test can outlive the
    assertion, so cleanup errors are tolerated too.
    """
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _load_tiny(tmpdir):
    path = os.path.join(tmpdir, "tiny.npz")
    data.save(path, _tiny_dataset())
    return data.load([path])


def test_check_cache_accepts_a_complete_cache():
    with _tmpdir() as d:
        ds = _load_tiny(d)
        arr = build_feature_cache(ds, os.path.join(d, "f.npy"))
        assert check_cache(arr), "a freshly built cache must validate"
        assert np.asarray(arr[:, F.P_ONES]).all()
        del arr, ds
        gc.collect()


def test_check_cache_rejects_unwritten_rows():
    """The exact shape of the bug: right dtype, right shape, blank tail."""
    with _tmpdir() as d:
        ds = _load_tiny(d)
        p = os.path.join(d, "f.npy")
        arr = np.asarray(build_feature_cache(ds, p)).copy()
        arr[-3:] = 0                       # the last rows are the held-out set
        assert not check_cache(arr), \
            "an all-zero row is an unwritten row and must be rejected"
        del ds
        gc.collect()


def test_no_partial_cache_is_ever_visible():
    """Two processes racing on one cache path must both see a complete file.

    The failure this pins is not a crash: before the fix, the loser of the race
    returned an array of the right shape whose tail was zeros, and every number
    computed from it looked like a plausible training result.
    """
    with _tmpdir() as d:
        ds = _load_tiny(d)
        data.save(os.path.join(d, "tiny.npz"), _tiny_dataset())
        prog = (
            "import sys, numpy as np;"
            f"sys.path[:0]=[{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r},"
            f"{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')!r}];"
            "from ag import data;"
            "from train_sl import build_feature_cache, check_cache;"
            f"ds = data.load([{os.path.join(d, 'tiny.npz')!r}]);"
            f"a = build_feature_cache(ds, {os.path.join(d, 'race.npy')!r});"
            "print('OK' if check_cache(a) else 'PARTIAL')")
        procs = [subprocess.Popen([sys.executable, "-c", prog],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
                 for _ in range(3)]
        outs = [p.communicate()[0] for p in procs]
        for o in outs:
            assert "OK" in o, f"a racing reader saw a partial cache:\n{o}"
        assert not any(f.endswith(".lock") for f in os.listdir(d)), \
            "the build lock must not be left behind"


if __name__ == "__main__":
    import traceback
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
