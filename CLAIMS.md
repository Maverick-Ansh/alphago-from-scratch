# Claims under test

Silver et al., *Mastering the game of Go with deep neural networks and tree
search*, Nature 529, 484–489 (2016).

Written **before** the implementation, so the result cannot be chosen after the
fact. Each claim names where it appears in the paper and the number that would
confirm or refute it here.

| # | Claim | Where in the paper | Kind | Confirmed here if |
|---|-------|--------------------|------|-------------------|
| **C1** | A conv policy net trained by supervised learning on expert moves predicts those moves well, and **prediction accuracy translates into playing strength** | Fig. 2a; Ext. Data Table 3 (57.0% test acc; 128→256 filters moves raw-net win rate 36%→67%) | headline | Move-prediction accuracy ≫ chance, and win rate against a fixed reference rises monotonically with accuracy across checkpoints |
| **C2** | Policy-gradient self-play against a **pool of past opponents** improves the SL policy | "the RL policy network won more than 80% of games against the SL policy network" | headline | RL policy beats its SL initialisation head-to-head, with a CI excluding 50% |
| **C3** | Training the value net on **whole games** overfits; one position per self-play game does not | "MSE 0.37 test vs 0.19 train" → "0.234 test vs 0.226 train" | mechanism *(primary ablation)* | The correlated-data arm shows a large train/test MSE gap; the one-position-per-game arm shows a small one. Same net, same budget, one thing changed |
| **C4** | A single value-net forward pass evaluates positions **more accurately than 100 rollouts** of the fast rollout policy, and approaches rollouts using the RL policy | Fig. 2b | mechanism | Value-net MSE < fast-rollout MSE across move-number bins, both bracketed by the constant-predictor floor |
| **C5** | Mixing value net and rollouts (λ=0.5) beats either alone (λ=0, λ=1) | Ext. Data Table 7: α_rvp 2890 vs α_vp 2177 vs α_rp 2416 Elo | headline | In a round-robin, Elo(λ=0.5) > Elo(λ=0) and > Elo(λ=1), with non-overlapping CIs |
| **C6** | The **SL** policy is a better MCTS prior than the stronger **RL** policy | "the SL policy network performed better in AlphaGo than the stronger RL policy network" | mechanism | MCTS[prior=SL] beats MCTS[prior=RL] head-to-head, *while* raw RL beats raw SL (C2). Both halves must hold or the claim is not reproduced |
| **C7** | The raw policy net, **with no search at all**, plays at the level of MCTS programs running thousands of rollouts | Abstract; Ext. Data Table 7 (α_p = 1517 Elo, above Fuego/GnuGo) | headline | Raw policy net (1 forward pass/move) ≥ the win rate of a plain MCTS player given ≥1000 rollouts/move |

## Why these seven

C3 and C5 are the paper's own clean ablations — two arms differing by exactly
one thing — so they are the primary comparisons. C6 is the paper's most
counter-intuitive sentence, stated with no supporting number; a reproduction is
exactly the place to put a number on it. C1/C2/C7 are the headline results.

## The measurement, and how it can lie

Budgeted for before any GPU time (`scripts/check_eval.py`):

* **Elo floor/ceiling.** The tournament includes a uniform-random player (floor)
  and the expert MCTS teacher (ceiling). Any player whose Elo lands outside that
  bracket indicates a broken tournament, not a strong player.
* **Value-net MSE floor.** For outcomes z ∈ {−1,+1}, the constant predictor
  `v ≡ E[z]` has MSE `1 − E[z]²`. Every MSE in C3/C4 is reported against that
  floor; an "MSE of 0.9" means nothing until you know the floor is 0.99.
* **Colour imbalance.** Komi 7.5 on 9×9 is a large handicap. Black's win rate
  under self-play is measured first; all head-to-head matches use **paired
  colours** (each opening played once as black, once as white) regardless.
* **Degenerate shortcut.** A policy that always passes, and one that plays the
  most frequent point, are scored explicitly and put in the report — if either
  is competitive, the benchmark is broken, not the model.
* **Accuracy ceiling for C1.** Expert-move prediction has an intrinsic ceiling
  (the teacher is stochastic). The teacher's *self-agreement* — how often two
  independent runs of the teacher pick the same move — is measured and is the
  ceiling for C1's accuracy number.

## Not being tested

Distributed/asynchronous search, virtual loss (`n_vl`), the tree policy `p_τ`,
the last-good-reply rollout cache, symmetry ensembling at 8 rotations, dynamic
expansion thresholds, handicap play, and anything about 19×19 or human players.
Recorded in full in REPORT.md.
