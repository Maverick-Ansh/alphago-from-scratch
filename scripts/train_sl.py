"""Stage 1: supervised learning of the policy network (claim C1).

    "We trained the policy network p_sigma to classify positions according to
     expert moves played in the KGS data set. [...] For each training step, we
     sampled a randomly selected mini-batch of m samples from the augmented KGS
     data set [...] and applied an asynchronous stochastic gradient descent
     update to maximize the log likelihood of the action.  The step size alpha
     was initialized to 0.003 and was halved every 80 million training steps,
     without momentum terms, and a mini-batch size of m = 16."

Kept faithful: plain SGD, **no momentum**, step size halved on a schedule,
maximise log-likelihood of the expert action, 8x dihedral augmentation, and
pass moves excluded from the data set.

Deviations (REPORT.md): mini-batch 256 rather than 16 and a few tens of
thousands of steps rather than 340 million -- 50 GPUs for three weeks is the
one thing that cannot be resized -- with the step size and halving period
rescaled to match.

On the step size: naively rescaling the paper's alpha = 0.003 from batch 16 to
batch 256 gives ~0.048, and the default here is **0.03**, which is close to it.

An earlier version of this file said the opposite -- that 0.03 collapsed two
widths out of three and 0.01 was the safe choice.  That was wrong, and the way
it was wrong is worth keeping.  The two "collapsed" widths were the two that
lost a race on the shared feature cache (see ``build_feature_cache``) and spent
the run measuring themselves against blank boards.  The step size was never
implicated; it was simply the only thing that had been varied deliberately, so
it took the blame.  Re-run against a sound cache, 0.03 is better than 0.01 at
every width -- 24.1/24.1/24.9% against 23.0/23.0/23.5% -- and reaches 17% test
accuracy by step 2,500 where 0.01 is still at chance and stays there until step
7,500.  0.01 is not safer, it is just slower.

The lesson is not about step sizes.  A wrong number that looks like a plausible
training curve will be explained by whatever knob was last turned, and the
explanation will be persuasive.

The split is **by game, not by position**.  Positions within a game differ by
one stone and share an outcome; splitting by position would put near-duplicates
on both sides of the split and report a test accuracy that is partly memorised.
That is the same correlation the paper identifies as fatal for the value
network, and it is just as fatal for measuring C1.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ag import data, features as feat, nets
from ag.go import N, NN, PASS


# --------------------------------------------------------------------------
# feature cache
# --------------------------------------------------------------------------
def check_cache(arr, path=""):
    """Is every row of a feature cache actually written?

    ``P_ONES`` is a constant plane of ones, set at every point of every
    position, so a valid row can never be all zeros.  That makes an unwritten
    row detectable exactly rather than heuristically: reading one plane over
    the whole cache costs a few megabytes and turns a silent wrong answer into
    a refusal.  The rows that go missing are the last ones, which is also the
    held-out set, which is why the symptom was a nonsense test accuracy on an
    otherwise healthy network.
    """
    ok = bool(np.asarray(arr[:, feat.P_ONES]).all())
    if not ok:
        bad = int((~np.asarray(arr[:, feat.P_ONES]).any(axis=(1, 2))).sum())
        print(f"[cache] {path} is INCOMPLETE: {bad}/{len(arr)} rows unwritten",
              flush=True)
    return ok


def build_feature_cache(ds, path, with_colour=False, log_every=20000,
                        wait_timeout=1800):
    """Extract planes once and memoise them as uint8 on disk.

    Every plane is binary, so uint8 costs a quarter of float32 and loses
    nothing.  Extraction is ~250 us/position (it simulates a move at every
    empty point), which would starve the GPU if done per epoch.

    The build is **atomic**, and that is not a nicety.  ``open_memmap`` creates
    the file at its full final size, zero filled, and then fills it row by row
    over several seconds.  A second process starting inside that window found a
    file of exactly the right shape, took the "reusing" branch, and read zeros.
    Training survived it -- the training set is re-read from the memmap every
    step, so it becomes real as soon as the build finishes -- but the held-out
    set is copied into RAM once, at startup, and stayed zeros for the entire
    run.  Two of three widths then reported a test accuracy of 1.4% while
    actually being fine, and it was recorded as a step-size collapse
    (REPORT.md section 4).  A wrong number that looks like a plausible result
    is worse than a crash.

    So: one process builds, under a lock, into a temporary file that is renamed
    into place only when complete, and the others wait for the rename.  A
    reader can now only ever see a finished cache.  If the lock is stale the
    waiter builds its own copy rather than deadlocking -- duplicated work is
    the cheap failure here.
    """
    n = len(ds["boards"])
    n_planes = feat.N_PLANES_VALUE if with_colour else feat.N_PLANES_POLICY
    want = (n, n_planes, N, N)

    def ready():
        if not os.path.exists(path):
            return None
        arr = np.load(path, mmap_mode="r")
        return arr if arr.shape == want and check_cache(arr, path) else False

    arr = ready()
    if arr is not None and arr is not False:
        print(f"[cache] reusing {path} {arr.shape}", flush=True)
        return arr
    if arr is False:
        print(f"[cache] {path} has the wrong shape, expected {want}; "
              f"rebuilding", flush=True)

    lock = path + ".lock"
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        builder = True
    except FileExistsError:
        builder = False

    if not builder:
        t0 = time.time()
        while time.time() - t0 < wait_timeout:
            arr = ready()
            if arr is not None and arr is not False:
                print(f"[cache] another process built {path} "
                      f"({time.time()-t0:.0f}s wait)", flush=True)
                return arr
            time.sleep(2)
        print(f"[cache] waited {wait_timeout}s for {path}; building a private "
              f"copy instead of deadlocking", flush=True)

    tmp = f"{path}.tmp{os.getpid()}"
    try:
        fx = feat.FeatureExtractor(with_colour=with_colour)
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                        shape=want)
        t0 = time.time()
        for i in range(n):
            arr[i] = fx(data.decode(ds, i)).astype(np.uint8)
            if (i + 1) % log_every == 0:
                el = time.time() - t0
                print(f"[cache] {i+1}/{n}  {el:.0f}s  "
                      f"eta {el/(i+1)*(n-i-1):.0f}s", flush=True)
        arr.flush()
        del arr
        os.replace(tmp, path)          # atomic: readers see all or nothing
        print(f"[cache] built {path} in {time.time()-t0:.0f}s", flush=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
        if builder and os.path.exists(lock):
            os.remove(lock)
    return np.load(path, mmap_mode="r")


# --------------------------------------------------------------------------
# GPU-side dihedral augmentation
# --------------------------------------------------------------------------
def sym_action_table():
    return torch.from_numpy(
        np.stack([feat.transform_actions(np.arange(NN), k) for k in range(8)]))


def apply_sym(x, k):
    """Same transform as ``features.transform_planes``, on a torch batch."""
    r, f = k % 4, k // 4
    y = torch.rot90(x, r, dims=(-2, -1))
    if f:
        y = torch.flip(y, dims=(-1,))
    return y


def _check_torch_matches_numpy():
    """The augmentation must agree with the numpy transform the tests pin.

    If torch's rot90 convention ever disagreed with numpy's, the labels would
    rotate one way and the planes the other, and training would simply be
    slightly worse forever.  Cheap to check, so check it.
    """
    a = np.random.default_rng(0).random((2, 3, N, N)).astype(np.float32)
    t = torch.from_numpy(a)
    for k in range(8):
        assert np.allclose(apply_sym(t, k).numpy(), feat.transform_planes(a, k)), \
            f"torch/numpy symmetry mismatch at k={k}"


# --------------------------------------------------------------------------
def split_by_game(ds, test_frac=0.1, seed=0):
    gid = ds["game_id"]
    games = np.unique(gid)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_test = max(1, int(len(games) * test_frac))
    test_games = set(games[:n_test].tolist())
    is_test = np.array([g in test_games for g in gid])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def dead_units(net, Xprobe, device):
    """Fraction of trunk filters that fire for no position in a probe batch.

    The step-size collapse in section 4 of REPORT.md was diagnosed after the
    fact, from a test accuracy frozen to four decimal places and a training
    loss parked at the marginal-move entropy.  Both are indirect.  This is the
    direct measurement: when every rectifier in the trunk is dead the trunk
    gradient is exactly zero and the only parameters still moving are the 81
    per-position output biases, which is a well-defined accuracy that means
    nothing.  One line in the log is cheaper than a second diagnosis.
    """
    net.eval()
    with torch.no_grad():
        xb = torch.from_numpy(np.asarray(Xprobe)).to(device).float()
        h = net.trunk(xb)
        frac = (h <= 0).all(dim=0).float().mean().item()
    net.train()
    return frac


def evaluate(net, X, y, device, batch=1024, sym=True):
    """Test accuracy, and top-5, over the held-out games."""
    net.eval()
    correct = top5 = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(y), batch):
            xb = torch.from_numpy(np.asarray(X[i:i + batch])).to(device).float()
            yb = torch.from_numpy(y[i:i + batch]).to(device).long()
            logits = net(xb)
            loss_sum += Fn.cross_entropy(logits, yb, reduction="sum").item()
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            top5 += (logits.topk(5, dim=1).indices == yb[:, None]).any(1).sum().item()
            total += len(yb)
    net.train()
    return correct / total, top5 / total, loss_sum / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/expert_*.npz")
    ap.add_argument("--cache", default="/content/data/feats_policy.npy")
    ap.add_argument("--out", default="/content/runs")
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--halve-every", type=int, default=12000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    tag = a.tag or f"sl_k{a.filters}"
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    _check_torch_matches_numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(glob.glob(a.data))
    print(f"[{tag}] loading {len(paths)} shards", flush=True)
    ds = data.load(paths)

    # "Pass moves were excluded from the data set."
    keep = ds["action"] != PASS
    ds = {k: v[keep] for k, v in ds.items()}
    print(f"[{tag}] {len(ds['action'])} non-pass positions from "
          f"{len(np.unique(ds['game_id']))} games", flush=True)

    X = build_feature_cache(ds, a.cache, with_colour=False)
    y = ds["action"].astype(np.int64)
    tr, te = split_by_game(ds, test_frac=0.1, seed=a.seed)
    print(f"[{tag}] train {len(tr)} / test {len(te)} (split by game)", flush=True)

    # The held-out set is small enough to hold in RAM as uint8.  It is copied
    # out once, which is exactly why it has to be checked here: a training set
    # re-read from the memmap every step recovers from a half-built cache, and
    # a held-out set copied at startup never does.
    Xte = np.asarray(X[te])
    yte = y[te]
    if not check_cache(Xte, "the held-out slice"):
        raise SystemExit(f"[{tag}] the held-out features are not fully "
                         f"written. Every accuracy this run reported would be "
                         f"measured against blank boards.")

    net = nets.PolicyNet(in_planes=feat.N_PLANES_POLICY,
                         n_filters=a.filters, n_layers=a.layers).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[{tag}] PolicyNet k={a.filters} layers={a.layers}: "
          f"{n_par:,} parameters", flush=True)

    # "without momentum terms"
    opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    sym_act = sym_action_table().to(device)

    rng = np.random.default_rng(a.seed)
    hist = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr * (0.5 ** ((step - 1) // a.halve_every))
        for gp in opt.param_groups:
            gp["lr"] = lr

        idx = np.sort(rng.choice(tr, size=a.batch, replace=False))
        xb = torch.from_numpy(np.asarray(X[idx])).to(device, non_blocking=True).float()
        yb = torch.from_numpy(y[idx]).to(device, non_blocking=True).long()

        # one dihedral element per batch: 8x the effective data, ~0 cost
        k = int(rng.integers(8))
        if k:
            xb = apply_sym(xb, k)
            yb = sym_act[k][yb]

        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            loss = Fn.cross_entropy(net(xb), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % a.eval_every == 0 or step == a.steps:
            acc, t5, tl = evaluate(net, Xte, yte, device)
            dead = dead_units(net, Xte[:256], device)
            el = time.time() - t0
            print(f"[{tag}] step {step:6d}/{a.steps} lr {lr:.5f} "
                  f"train_loss {loss.item():.4f} | test acc {acc:.4f} "
                  f"top5 {t5:.4f} loss {tl:.4f} | dead {dead:.0%} "
                  f"| {el:.0f}s", flush=True)
            if dead > 0.99:
                print(f"[{tag}] COLLAPSED at step {step}: every trunk unit is "
                      f"dead, so only the per-position biases are still "
                      f"training. Lower --lr; this run's accuracy is not a "
                      f"point on the C1 curve.", flush=True)
            hist.append(dict(step=step, lr=lr, train_loss=float(loss.item()),
                             test_acc=acc, test_top5=t5, test_loss=tl,
                             dead_units=dead, seconds=el))
            nets.save(net, os.path.join(a.out, f"{tag}_step{step}.pt"),
                      history=hist, args=vars(a), n_params=n_par)

    nets.save(net, os.path.join(a.out, f"{tag}_final.pt"),
              history=hist, args=vars(a), n_params=n_par)
    with open(os.path.join(a.out, f"{tag}_history.json"), "w") as f:
        json.dump(dict(tag=tag, n_params=n_par, args=vars(a), history=hist), f,
                  indent=2)
    print(f"[{tag}] DONE final test acc {hist[-1]['test_acc']:.4f} "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
