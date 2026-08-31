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
    ap.add_argument("--lr", type=float, default=0.02)
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

    results = {}
    for name, arm in arms.items():
        X, z = arm["X"], arm["z"]
        tr, te = np.sort(arm["train"]), np.sort(arm["test"])
        Xtr, ztr = np.asarray(X[tr]), z[tr]
        Xte, zte = np.asarray(X[te]), z[te]
        floor_tr, floor_te = const_floor(ztr), const_floor(zte)

        torch.manual_seed(a.seed)
        net = nets.ValueNet(in_planes=feat.N_PLANES_VALUE,
                            n_filters=a.filters, n_layers=a.layers).to(device)
        opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        r = np.random.default_rng(a.seed)
        hist = []
        t0 = time.time()
        print(f"\n[value/{name}] training on {len(tr)} positions; "
              f"floors: train {floor_tr:.4f} test {floor_te:.4f}", flush=True)

        for step in range(1, a.steps + 1):
            lr = a.lr * (0.5 ** ((step - 1) // a.halve_every))
            for gp in opt.param_groups:
                gp["lr"] = lr
            idx = np.sort(r.choice(len(tr), size=min(a.batch, len(tr)),
                                   replace=False))
            xb = torch.from_numpy(Xtr[idx]).to(device).float()
            zb = torch.from_numpy(ztr[idx]).to(device)
            k = int(r.integers(8))
            if k:
                xb = apply_sym(xb, k)          # value is symmetry-invariant
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=(device == "cuda")):
                loss = Fn.mse_loss(net(xb).float(), zb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % a.eval_every == 0 or step == a.steps:
                m_tr = mse_on(net, Xtr, ztr, device)
                m_te = mse_on(net, Xte, zte, device)
                m_co = mse_on(net, Xcommon, zcommon, device)
                print(f"[value/{name}] step {step:5d} lr {lr:.5f} | "
                      f"train MSE {m_tr:.4f} test MSE {m_te:.4f} "
                      f"gap {m_te - m_tr:+.4f} | common {m_co:.4f} "
                      f"(floor {floor_common:.4f}) | {time.time()-t0:.0f}s",
                      flush=True)
                hist.append(dict(step=step, train_mse=m_tr, test_mse=m_te,
                                 common_mse=m_co))

        nets.save(net, os.path.join(a.out, f"value_{name}.pt"),
                  history=hist, args=vars(a), arm=name)
        best = min(hist, key=lambda h: h["test_mse"])
        results[name] = dict(
            history=hist, floor_train=floor_tr, floor_test=floor_te,
            floor_common=floor_common,
            n_train=int(len(tr)),
            n_train_games=int(len(np.unique(
                (cor if name == "correlated" else unc)["game_id"][tr]))),
            best_train_mse=best["train_mse"], best_test_mse=best["test_mse"],
            best_gap=best["test_mse"] - best["train_mse"],
            final_common_mse=hist[-1]["common_mse"])

    with open(os.path.join(a.out, "value_results.json"), "w") as f:
        json.dump(dict(args=vars(a), results=results), f, indent=2)

    print("\n=== C3: correlated vs uncorrelated value-net training ===")
    print(f"{'arm':<14}{'games':>8}{'train MSE':>11}{'test MSE':>10}"
          f"{'gap':>9}{'common':>9}")
    for name, r_ in results.items():
        print(f"{name:<14}{r_['n_train_games']:>8}{r_['best_train_mse']:>11.4f}"
              f"{r_['best_test_mse']:>10.4f}{r_['best_gap']:>+9.4f}"
              f"{r_['final_common_mse']:>9.4f}")
    print(f"{'floor':<14}{'-':>8}{'-':>11}{'-':>10}{'-':>9}"
          f"{results['correlated']['floor_common']:>9.4f}")
    print("paper: correlated 0.19 train / 0.37 test (gap +0.18); "
          "uncorrelated 0.226 / 0.234 (gap +0.008)")


if __name__ == "__main__":
    main()
