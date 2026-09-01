"""Run the whole reproduction, in dependency order, using both GPUs.

Stages that do not depend on each other run at the same time; stages that do
are made to wait.  Everything logs to a file under ``--runs`` and every stage
is skippable, so a killed run resumes instead of restarting.

The ordering is not arbitrary -- the eval gate sits in the middle on purpose.
It runs as soon as there is a rollout policy and a policy network, and before
any of the expensive stages, because its whole job is to refuse the sweep if
the measurement lacks the range to see the effects being claimed.
"""

import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class Stage:
    def __init__(self, name, argv, gpu=None, produces=None):
        self.name = name
        self.argv = argv
        self.gpu = gpu
        self.produces = produces
        self.proc = None
        self.t0 = None

    def done(self):
        if not self.produces:
            return False
        return all(glob.glob(p) for p in self.produces)


def launch(st, runs, env_base):
    if st.done():
        print(f"[skip] {st.name} (outputs exist)", flush=True)
        return None
    env = dict(env_base)
    if st.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(st.gpu)
    log = open(os.path.join(runs, f"{st.name}.log"), "w")
    st.t0 = time.time()
    st.proc = subprocess.Popen(
        [sys.executable, "-u"] + st.argv, stdout=log,
        stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True)
    print(f"[run ] {st.name}  gpu={st.gpu}  pid={st.proc.pid}", flush=True)
    return st.proc


def wait_all(stages, runs, poll=10):
    live = [s for s in stages if s.proc is not None]
    while any(s.proc.poll() is None for s in live):
        time.sleep(poll)
    ok = True
    for s in live:
        rc = s.proc.returncode
        el = time.time() - s.t0
        tail = ""
        try:
            lines = [l for l in open(os.path.join(runs, f"{s.name}.log")).read()
                     .splitlines() if l.strip()]
            tail = lines[-1][:160] if lines else ""
        except OSError:
            pass
        print(f"[{'ok  ' if rc == 0 else 'FAIL'}] {s.name} rc={rc} "
              f"{el:.0f}s | {tail}", flush=True)
        ok = ok and rc == 0
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data")
    ap.add_argument("--runs", default="/content/runs")
    ap.add_argument("--sl-steps", type=int, default=25000)
    ap.add_argument("--rl-iters", type=int, default=200)
    ap.add_argument("--value-games", type=int, default=24000)
    ap.add_argument("--value-steps", type=int, default=8000)
    ap.add_argument("--tourney-sims", type=int, default=100)
    ap.add_argument("--tourney-games", type=int, default=12)
    ap.add_argument("--c7-ladder", default="50,100,300,1000")
    ap.add_argument("--c7-games", type=int, default=20)
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated phase names")
    a = ap.parse_args()

    os.makedirs(a.runs, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=ROOT,
               NUMBA_CACHE_DIR=os.path.join(a.data, "nbcache"),
               OMP_NUM_THREADS="1")
    D, R = a.data, a.runs
    S = lambda p: os.path.join(HERE, p)
    expert = os.path.join(D, "expert_*.npz")

    phases = {}

    # -- 1. the fast rollout policy, and the SL policy at three widths -----
    phases["distil"] = [
        Stage("rollout", [S("train_rollout.py"), "--data", expert,
                          "--out", f"{R}/rollout.npz", "--epochs", "8"],
              produces=[f"{R}/rollout.npz"]),
    ] + [
        Stage(f"sl_k{k}", [S("train_sl.py"), "--data", expert,
                           "--cache", f"{D}/feats_policy.npy", "--out", R,
                           "--filters", str(k), "--steps", str(a.sl_steps),
                           "--halve-every", str(a.sl_steps // 3),
                           "--eval-every", str(max(a.sl_steps // 10, 500))],
              gpu=i % 2, produces=[f"{R}/sl_k{k}_final.pt"])
        for i, k in enumerate((64, 32, 128))
    ]

    # -- 2. THE GATE ------------------------------------------------------
    phases["gate"] = [
        Stage("gate", [S("check_eval.py"), "--data", expert,
                       "--rollout", f"{R}/rollout.npz", "--sims", "128",
                       "--agree-positions", "120", "--games", "40",
                       "--out", f"{R}/eval_gate.json"],
              produces=[f"{R}/eval_gate.json"]),
    ]

    # -- 3. RL, then the value-network data it is needed for --------------
    phases["rl"] = [
        Stage("rl", [S("train_rl.py"), "--sl", f"{R}/sl_k64_final.pt",
                     "--out", R, "--iters", str(a.rl_iters)],
              gpu=0, produces=[f"{R}/rl_final.pt"]),
    ]
    phases["valuedata"] = [
        Stage("value_data", [S("gen_value_data.py"), "--sl",
                             f"{R}/sl_k64_final.pt", "--rl", f"{R}/rl_final.pt",
                             "--out", f"{D}/selfplay_value.npz",
                             "--games", str(a.value_games)],
              gpu=1, produces=[f"{D}/selfplay_value.npz"]),
    ]
    phases["value"] = [
        Stage("value", [S("train_value.py"), "--correlated", expert,
                        "--uncorrelated", f"{D}/selfplay_value.npz",
                        "--cache-dir", D, "--out", R,
                        "--steps", str(a.value_steps)],
              gpu=0, produces=[f"{R}/value_results.json"]),
    ]

    # -- 4. the measurements ----------------------------------------------
    phases["measure"] = [
        Stage(f"tourney{s}", [S("tournament.py"), "--rollout", f"{R}/rollout.npz",
                              "--sl", f"{R}/sl_k64_final.pt",
                              "--rl", f"{R}/rl_final.pt",
                              "--value", f"{R}/value_uncorrelated.pt",
                              "--sims", str(a.tourney_sims),
                              "--games", str(a.tourney_games),
                              "--out", f"{R}/tourney",
                              "--shard", str(s), "--nshards", "4"],
              gpu=s % 2) for s in range(4)
    ] + [
        Stage("c4", [S("eval_value_vs_rollouts.py"), "--data", expert,
                     "--value", f"{R}/value_uncorrelated.pt",
                     "--rollout", f"{R}/rollout.npz",
                     "--sl", f"{R}/sl_k64_final.pt", "--rl", f"{R}/rl_final.pt",
                     "--positions", "150", "--n-roll", "100",
                     "--out", f"{R}/c4_value_vs_rollouts.json"],
              gpu=1, produces=[f"{R}/c4_value_vs_rollouts.json"]),
    ]
    phases["c7"] = [
        Stage("c7", [S("eval_c7_search_free.py"), "--rollout", f"{R}/rollout.npz",
                     "--sl", f"{R}/sl_k64_final.pt", "--rl", f"{R}/rl_final.pt",
                     "--ladder", a.c7_ladder, "--games", str(a.c7_games),
                     "--out", f"{R}/c7_search_free.json"],
              gpu=1, produces=[f"{R}/c7_search_free.json"]),
    ]
    phases["c1"] = [
        Stage("c1", [S("eval_c1_strength.py"), "--runs", R,
                     "--rollout", f"{R}/rollout.npz", "--games", "30",
                     "--gate", f"{R}/eval_gate.json",
                     "--out", f"{R}/c1_strength.json"],
              gpu=0, produces=[f"{R}/c1_strength.json"]),
    ]
    phases["figures"] = [
        Stage("merge", [S("tournament.py"), "--merge", "--out", f"{R}/tourney"]),
        Stage("figures", [S("make_figures.py"), "--runs", R,
                          "--out", os.path.join(ROOT, "figures")]),
    ]

    order = ["distil", "gate", "rl", "valuedata", "value", "measure", "c7",
             "c1", "figures"]
    if a.only:
        order = [p for p in order if p in a.only.split(",")]

    t_all = time.time()
    for name in order:
        if name == "gate" and a.skip_gate:
            continue
        print(f"\n=== phase: {name} ===", flush=True)
        for st in phases[name]:
            launch(st, R, env)
        ok = wait_all(phases[name], R)
        if not ok:
            if name == "gate":
                print("\nGATE FAILED -- stopping. Read "
                      f"{R}/gate.log before spending more compute.", flush=True)
            else:
                print(f"\nphase {name} had a failure; see {R}/*.log", flush=True)
            sys.exit(1)
    print(f"\nALL DONE in {(time.time()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
