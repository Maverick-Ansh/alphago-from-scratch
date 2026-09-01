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
batch 256 gives ~0.048, and 0.03 **collapses two widths out of three**.  The
failure is not divergence, which would be obvious; it is quiet.  Every ReLU in
the trunk dies, the trunk's gradient becomes exactly zero, and the only
parameters still learning are the 81 free per-position biases on the output.
The network then predicts the same point in every position: training loss
settles at the entropy of the marginal move distribution (~2.9), test accuracy
freezes at whatever fraction of positions happen to share that point, and test
loss climbs past ln(81) as the biases overfit.  It looks like a network that is
training.  The default here is 0.01, which is stable at every width tested.

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
def build_feature_cache(ds, path, with_colour=False, log_every=20000):
    """Extract planes once and memoise them as uint8 on disk.

    Every plane is binary, so uint8 costs a quarter of float32 and loses
    nothing.  Extraction is ~250 us/position (it simulates a move at every
    empty point), which would starve the GPU if done per epoch.
    """
    n = len(ds["boards"])
    n_planes = feat.N_PLANES_VALUE if with_colour else feat.N_PLANES_POLICY
    if os.path.exists(path):
        arr = np.load(path, mmap_mode="r")
        if arr.shape == (n, n_planes, N, N):
            print(f"[cache] reusing {path} {arr.shape}", flush=True)
            return arr
        print(f"[cache] {path} has shape {arr.shape}, expected "
              f"{(n, n_planes, N, N)}; rebuilding", flush=True)
    fx = feat.FeatureExtractor(with_colour=with_colour)
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                    shape=(n, n_planes, N, N))
    t0 = time.time()
    for i in range(n):
        arr[i] = fx(data.decode(ds, i)).astype(np.uint8)
        if (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"[cache] {i+1}/{n}  {el:.0f}s  eta {el/(i+1)*(n-i-1):.0f}s",
                  flush=True)
    arr.flush()
    print(f"[cache] built {path} in {time.time()-t0:.0f}s", flush=True)
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
    ap.add_argument("--lr", type=float, default=0.01)
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

    # The held-out set is small enough to hold in RAM as uint8.
    Xte = np.asarray(X[te])
    yte = y[te]

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
