"""Stage 3b: value-network regression, both arms (claim C3).

    "We train the weights of the value network by regression on state-outcome
     pairs (s, z), using stochastic gradient descent to minimize the mean
     squared error (MSE) between the predicted value v_theta(s), and the
     corresponding outcome z."

    "When trained on the KGS data set in this way, the value network memorized
     the game outcomes rather than generalizing to new positions, achieving a
     minimum MSE of 0.37 on the test set, compared to 0.19 on the training set.
     [...] Training on this data set led to MSEs of 0.226 and 0.234 on the
     training and test set respectively, indicating minimal overfitting."

This is the paper's cleanest ablation: identical network, identical objective,
identical optimiser, one thing changed -- where the positions come from.

  arm ``correlated``   every position of a small number of complete games
  arm ``uncorrelated`` one position each from a large number of games

**Both arms are given the same number of training positions.**  Otherwise the
comparison confounds correlation with data volume, and a smaller train/test gap
would simply be what more data always buys.  Matching positions rather than
games is also what the paper effectively does: 29.4M KGS positions against 30M
self-play positions.

Three things are reported for every arm:

* its own train and test MSE (the paper's numbers);
* MSE on a **common** held-out set of uncorrelated self-play positions, so the
  two arms are scored by the same ruler;
* the **constant-predictor floor**, MSE of always answering E[z].  For
  z in {-1,+1} that is 1 - E[z]^2, near 1.0.  An MSE of 0.9 sounds bad and is
  actually a real signal; an MSE of 0.37 sounds bad and is most of the way to
  perfect.  Reporting one without the other is how value-function results get
  misread.

The step size, and why it is calibrated rather than chosen
----------------------------------------------------------
Run 1 trained both arms at ``--lr 0.02`` and both landed on the floor -- *train*
MSE 1.0000 against a floor of 0.9998.  A train MSE at the floor is not
overfitting, it is a network that never fitted anything: the trunk's rectifiers
die, the trunk gradient goes to exactly zero, and the only parameters still
learning are the two fully connected biases, which converge on E[z].  The arms
then agree perfectly, and reading "no difference between the arms" off that
would have been reporting an optimiser failure as an ablation result.  The same
step size killed two of three widths in the SL stage (REPORT.md section 4).

So the step size is now measured instead of assumed, in two parts.

``calibrate_lr`` runs a short trial of **both** arms at each candidate step size
and keeps the one with the lowest **training** MSE.  Selecting on train MSE
matters: C3 is a claim about the train/test *gap*, and picking the step size by
test MSE, or on one arm only, would let the tuning decide the thing being
measured.  Optimisation quality is what a step size is responsible for, so that
is what it is selected on, and the winner is then used for both arms.

``collapse_report`` makes the failure visible while it happens rather than
inferrable afterwards.  It reports the fraction of dead rectifiers in the trunk
and the spread of the network's own predictions on a fixed probe batch.  A
network sitting at the floor because it collapsed has ``pred_std`` near zero and
a large dead fraction; one sitting near the floor because Go is hard has a
prediction spread and live units.  The two are indistinguishable from the MSE
column alone, which is exactly why run 1 needed a separate diagnosis.
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
from ag.go import BLACK, N, NN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_sl import build_feature_cache, apply_sym, _check_torch_matches_numpy


def z_for_player(ds):
    """Outcome from the perspective of the player to move (the paper's z_t)."""
    zb = ds["z"].astype(np.float32)
    return np.where(ds["to_play"] == BLACK, zb, -zb).astype(np.float32)


def const_floor(z):
    """MSE of the best constant predictor: 1 - E[z]^2 for z in {-1,+1}."""
    return float(np.mean((z - z.mean()) ** 2))


def collapse_report(net, Xprobe, device):
    """Dead-rectifier fraction in the trunk, and the spread of the outputs.

    A value network parked on the constant-predictor floor has two possible
    causes with opposite meanings.  Either it learned that positions are hard
    to call -- in which case it still says different things about different
    positions -- or its trunk is dead and it is emitting one number for every
    board.  ``pred_std`` separates them, and ``dead`` says why.
    """
    net.eval()
    with torch.no_grad():
        xb = torch.from_numpy(Xprobe).to(device).float()
        h = net.trunk(xb)
        dead_unit = (h <= 0).all(dim=0).float().mean().item()   # never fires
        dead_act = (h <= 0).float().mean().item()               # zero right now
        v = net(xb)
        pred_std = v.std().item()
        pred_mean = v.mean().item()
    net.train()
    return dict(dead_units=dead_unit, dead_acts=dead_act,
                pred_std=pred_std, pred_mean=pred_mean)


def mse_on(net, X, z, device, batch=2048):
    net.eval()
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(z), batch):
            xb = torch.from_numpy(np.asarray(X[i:i + batch])).to(device).float()
            zb = torch.from_numpy(z[i:i + batch]).to(device)
            tot += ((net(xb) - zb) ** 2).sum().item()
    net.train()
    return tot / len(z)


def train_arm(name, arm, lr0, steps, a, Xprobe, Xcommon, zcommon,
              floor_common, device, verbose=True, tag=""):
    """Train one arm at one step size.  Returns the net and its history.

    Everything except the data is identical between the arms -- same seed, same
    initialisation, same batch schedule, same step size -- because the one
    thing C3 varies is where the positions came from.
    """
    X, z = arm["X"], arm["z"]
    tr, te = np.sort(arm["train"]), np.sort(arm["test"])
    Xtr, ztr = np.asarray(X[tr]), z[tr]
    Xte, zte = np.asarray(X[te]), z[te]
    floor_tr, floor_te = const_floor(ztr), const_floor(zte)

    torch.manual_seed(a.seed)
    net = nets.ValueNet(in_planes=feat.N_PLANES_VALUE,
                        n_filters=a.filters, n_layers=a.layers).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr0, momentum=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    r = np.random.default_rng(a.seed)
    hist = []
    t0 = time.time()
    if verbose:
        print(f"\n[value/{name}{tag}] lr {lr0} on {len(tr)} positions; "
              f"floors: train {floor_tr:.4f} test {floor_te:.4f}", flush=True)

    eval_every = max(1, min(a.eval_every, steps // 4))
    for step in range(1, steps + 1):
        lr = lr0 * (0.5 ** ((step - 1) // a.halve_every))
        for gp in opt.param_groups:
            gp["lr"] = lr
        idx = np.sort(r.choice(len(tr), size=min(a.batch, len(tr)),
                               replace=False))
        xb = torch.from_numpy(Xtr[idx]).to(device).float()
        zb = torch.from_numpy(ztr[idx]).to(device)
        k = int(r.integers(8))
        if k:
            xb = apply_sym(xb, k)              # value is symmetry-invariant
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=(device == "cuda")):
            loss = Fn.mse_loss(net(xb).float(), zb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % eval_every == 0 or step == steps:
            m_tr = mse_on(net, Xtr, ztr, device)
            m_te = mse_on(net, Xte, zte, device)
            m_co = mse_on(net, Xcommon, zcommon, device)
            d = collapse_report(net, Xprobe, device)
            if verbose:
                print(f"[value/{name}{tag}] step {step:5d} lr {lr:.5f} | "
                      f"train MSE {m_tr:.4f} test MSE {m_te:.4f} "
                      f"gap {m_te - m_tr:+.4f} | common {m_co:.4f} "
                      f"(floor {floor_common:.4f}) | dead {d['dead_units']:.0%} "
                      f"pred_std {d['pred_std']:.3f} | {time.time()-t0:.0f}s",
                      flush=True)
            hist.append(dict(step=step, train_mse=m_tr, test_mse=m_te,
                             common_mse=m_co, lr=lr, **d))
    return net, hist


def calibrate_lr(a, arms, Xprobe, Xcommon, zcommon, floor_common, device):
    """Pick the step size on a short trial of both arms, by TRAINING mse.

    Run 1 assumed 0.02 and got two dead networks that agreed with each other
    perfectly -- an optimiser failure wearing the costume of a null result.
    The candidates are trialled here instead, and the selection criterion is
    deliberately the training error: fitting the training set is what a step
    size is responsible for, and it says nothing about the train/test gap that
    C3 is about.  One winner is used for both arms, so the arms still differ
    by exactly one thing.
    """
    cands = [float(x) for x in a.lr_candidates.split(",")]
    print(f"\n[value] calibrating the step size over {cands}, "
          f"{a.calib_steps} steps per arm", flush=True)
    table = []
    for lr in cands:
        row = dict(lr=lr, arms={})
        for name, arm in arms.items():
            _, h = train_arm(name, arm, lr, a.calib_steps, a, Xprobe, Xcommon,
                             zcommon, floor_common, device, verbose=False)
            row["arms"][name] = h[-1]
        row["mean_train_mse"] = float(np.mean(
            [row["arms"][n]["train_mse"] for n in arms]))
        row["min_pred_std"] = float(min(
            row["arms"][n]["pred_std"] for n in arms))
        row["max_dead_units"] = float(max(
            row["arms"][n]["dead_units"] for n in arms))
        table.append(row)
        print(f"[value/calib] lr {lr:<8} mean train MSE "
              f"{row['mean_train_mse']:.4f} | min pred_std "
              f"{row['min_pred_std']:.4f} | max dead units "
              f"{row['max_dead_units']:.0%}"
              + ("   <- collapsed" if row["min_pred_std"] < 1e-3 else ""),
              flush=True)

    live = [r for r in table if r["min_pred_std"] >= 1e-3]
    if not live:
        raise SystemExit("[value] every candidate step size collapsed at least "
                         "one arm. Widen --lr-candidates downward before "
                         "spending the full budget.")
    best = min(live, key=lambda r: r["mean_train_mse"])
    print(f"[value] chosen step size {best['lr']} "
          f"(mean train MSE {best['mean_train_mse']:.4f}); "
          f"the same value is used for both arms.", flush=True)
    return best["lr"], table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correlated", required=True, help="expert shards glob")
    ap.add_argument("--uncorrelated", required=True, help="self-play npz")
    ap.add_argument("--cache-dir", default="/content/data")
    ap.add_argument("--out", default="/content/runs")
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.0,
                    help="0 = calibrate it (the default); a value pins it")
    ap.add_argument("--lr-candidates", default="0.02,0.01,0.003,0.001")
    ap.add_argument("--calib-steps", type=int, default=1200)
    ap.add_argument("--halve-every", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=0,
                    help="positions per arm; 0 = as many as both arms allow")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    _check_torch_matches_numpy()
    rng = np.random.default_rng(a.seed)

    # ---------------- load both data sets -------------------------------
    cor = data.load(sorted(glob.glob(a.correlated)))
    unc = data.load([a.uncorrelated])
    print(f"[value] correlated: {len(cor['z'])} positions from "
          f"{len(np.unique(cor['game_id']))} games", flush=True)
    print(f"[value] uncorrelated: {len(unc['z'])} positions from "
          f"{len(np.unique(unc['game_id']))} games", flush=True)

    Xc = build_feature_cache(cor, os.path.join(a.cache_dir, "feats_value_cor.npy"),
                             with_colour=True)
    Xu = build_feature_cache(unc, os.path.join(a.cache_dir, "feats_value_unc.npy"),
                             with_colour=True)
    zc, zu = z_for_player(cor), z_for_player(unc)

    # ---------------- splits --------------------------------------------
    # The uncorrelated set is split into train / its own test / a COMMON test
    # that both arms are scored on.
    n_unc = len(zu)
    perm = rng.permutation(n_unc)
    n_common = min(3000, n_unc // 4)
    common = perm[:n_common]
    unc_test = perm[n_common:n_common + n_common]
    unc_train_pool = perm[2 * n_common:]

    # The correlated set is split BY GAME, so its own test set is honest about
    # generalising to unseen games (the fairest possible version of this arm).
    gids = np.unique(cor["game_id"])
    rng.shuffle(gids)
    n_test_g = max(1, len(gids) // 10)
    test_g = set(gids[:n_test_g].tolist())
    is_test = np.array([g in test_g for g in cor["game_id"]])
    cor_train_pool = np.flatnonzero(~is_test)
    cor_test = np.flatnonzero(is_test)

    n_train = a.n_train or min(len(cor_train_pool), len(unc_train_pool))
    if n_train > min(len(cor_train_pool), len(unc_train_pool)):
        raise SystemExit(f"--n-train {n_train} exceeds the smaller pool "
                         f"({min(len(cor_train_pool), len(unc_train_pool))})")
    cor_train = rng.choice(cor_train_pool, n_train, replace=False)
    unc_train = rng.choice(unc_train_pool, n_train, replace=False)
    print(f"[value] each arm trains on {n_train} positions "
          f"(correlated: from {len(np.unique(cor['game_id'][cor_train]))} games; "
          f"uncorrelated: from {len(np.unique(unc['game_id'][unc_train]))} games)",
          flush=True)

    Xcommon, zcommon = np.asarray(Xu[np.sort(common)]), zu[np.sort(common)]
    floor_common = const_floor(zcommon)
    print(f"[value] CONSTANT-PREDICTOR FLOOR on the common test set: "
          f"{floor_common:.4f}  (E[z] = {zcommon.mean():+.3f})", flush=True)

    arms = {
        "correlated": dict(X=Xc, z=zc, train=cor_train, test=cor_test),
        "uncorrelated": dict(X=Xu, z=zu, train=unc_train, test=unc_test),
    }

    Xprobe = np.asarray(Xcommon[:256])

    # ---------------- step-size calibration ------------------------------
    # Both arms, short trials, selected on TRAINING mse -- see the module
    # docstring for why the selection criterion is not the test set.
    if a.lr:
        lr0, calib = a.lr, None
        print(f"\n[value] step size pinned by --lr {lr0}", flush=True)
    else:
        lr0, calib = calibrate_lr(a, arms, Xprobe, Xcommon, zcommon,
                                  floor_common, device)

    results = {}
    for name, arm in arms.items():
        src = cor if name == "correlated" else unc
        net, hist = train_arm(name, arm, lr0, a.steps, a, Xprobe, Xcommon,
                              zcommon, floor_common, device, verbose=True)
        tr = np.sort(arm["train"])
        floor_tr = const_floor(arm["z"][tr])
        floor_te = const_floor(arm["z"][np.sort(arm["test"])])

        nets.save(net, os.path.join(a.out, f"value_{name}.pt"),
                  history=hist, args=vars(a), arm=name, lr_used=lr0)
        best = min(hist, key=lambda h: h["test_mse"])
        results[name] = dict(
            history=hist, floor_train=floor_tr, floor_test=floor_te,
            floor_common=floor_common, lr_used=lr0,
            n_train=int(len(tr)),
            n_train_games=int(len(np.unique(src["game_id"][tr]))),
            best_train_mse=best["train_mse"], best_test_mse=best["test_mse"],
            best_gap=best["test_mse"] - best["train_mse"],
            final_train_mse=hist[-1]["train_mse"],
            final_test_mse=hist[-1]["test_mse"],
            final_common_mse=hist[-1]["common_mse"],
            final_pred_std=hist[-1]["pred_std"],
            final_dead_units=hist[-1]["dead_units"],
            collapsed=bool(hist[-1]["pred_std"] < 1e-3))
        if results[name]["collapsed"]:
            print(f"[value/{name}] COLLAPSED: pred_std "
                  f"{hist[-1]['pred_std']:.2e}, dead units "
                  f"{hist[-1]['dead_units']:.0%}. This arm learned nothing, so "
                  f"C3 cannot be read off it.", flush=True)

    with open(os.path.join(a.out, "value_results.json"), "w") as f:
        json.dump(dict(args=vars(a), lr_used=lr0, lr_calibration=calib,
                       results=results), f, indent=2)

    print("\n=== C3: correlated vs uncorrelated value-net training ===")
    print(f"step size {lr0} for both arms")
    print(f"{'arm':<14}{'games':>8}{'train MSE':>11}{'test MSE':>10}"
          f"{'gap':>9}{'common':>9}{'pred_std':>10}")
    for name, r_ in results.items():
        print(f"{name:<14}{r_['n_train_games']:>8}{r_['best_train_mse']:>11.4f}"
              f"{r_['best_test_mse']:>10.4f}{r_['best_gap']:>+9.4f}"
              f"{r_['final_common_mse']:>9.4f}{r_['final_pred_std']:>10.4f}")
    print(f"{'floor':<14}{'-':>8}{'-':>11}{'-':>10}{'-':>9}"
          f"{results['correlated']['floor_common']:>9.4f}")
    print("paper: correlated 0.19 train / 0.37 test (gap +0.18); "
          "uncorrelated 0.226 / 0.234 (gap +0.008)")


if __name__ == "__main__":
    main()
