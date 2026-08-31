# Reproducing AlphaGo at 9×9

Silver et al., *Mastering the game of Go with deep neural networks and tree
search*, **Nature 529, 484–489 (2016)**.

Everything is built from nothing: the Go rules engine, the feature planes, the
policy and value networks, the fast rollout policy, REINFORCE self-play, and
APV-MCTS with PUCT. No Go library, no RL library.

> **Status:** results pending — the run is in progress. Sections 1, 2 and 3 are
> final; 4–6 are filled in as the sweep completes.

---

## 1. Claims under test

Stated in [CLAIMS.md](CLAIMS.md) **before** any code was written, so the result
could not be chosen after the fact. In brief:

| | Claim | Kind |
|---|---|---|
| C1 | SL on expert moves gives high prediction accuracy, and **accuracy predicts strength** | headline |
| C2 | Policy-gradient self-play against a **pool** of past opponents beats the SL policy | headline |
| C3 | Value net on **whole games** overfits; **one position per game** does not | mechanism *(primary)* |
| C4 | One value-net forward pass beats **100 rollouts** of the fast policy | mechanism |
| C5 | Mixing value net and rollouts (λ=0.5) beats either alone | headline |
| C6 | The **SL** policy is a better MCTS prior than the stronger **RL** policy | mechanism |
| C7 | The raw policy net with **no search** matches MCTS running thousands of rollouts | headline |

---

## 2. How it was resized

**The board, not the pipeline.** 19×19 becomes 9×9. Every stage of the paper's
four-stage pipeline is present and runs; nothing was dropped for convenience.

**The expert.** The paper's supervised teacher is 29.4M positions from 160,000
games by KGS 6–9 dan humans. There is no 9×9 equivalent of that archive, so the
teacher here is what was state of the art in computer Go *immediately before*
AlphaGo: Monte-Carlo tree search with rollouts and no neural network — the
paper's own `α_r` row, and the family Pachi and Fuego belong to. This preserves
the structure the claims depend on (a slow strong teacher; a fast network
distilled from it; RL and search on top) while removing the dependence on a
human game archive.

That substitution has a consequence worth stating plainly: **the whole ladder is
anchored to a weaker teacher than the paper's**, so absolute strength here is
not comparable to AlphaGo's. Every claim under test is a *relative* comparison,
which is why they survive the substitution.

**Diversity.** The paper gets it free from 160,000 different human games. Here it
is manufactured: a random opening of 0–4 uniform moves, and the teacher's first
8 moves *played* by sampling from the visit distribution. In both cases the
recorded **label is always the teacher's argmax move**, never the randomised
one, so positions spread out while targets stay clean.

### Deviations table

| | Paper | Here | Why it is still a test of the claim |
|---|---|---|---|
| Board | 19×19 | 9×9 | Same rules, same scoring, same tactics (ladders, ko, life-and-death). Every claim is relative. |
| Komi | 7.5, Chinese rules | 7.5, Chinese rules (Tromp-Taylor) | Unchanged. Measured to be near-fair at 9×9 — see §4. |
| Expert | 29.4M positions, 160k human games | MCTS teacher, 128 sims, 1,400 games | See above. Structure preserved, absolute level lowered. |
| Ko rule | not stated | **simple ko** (no positional superko) | Standard in fast Go engines; superko cycles are additionally bounded by a move cap. |
| Policy trunk | 1×(5×5) + 11×(3×3), k=192 | 1×(5×5) + 5×(3×3), k∈{32,64,128} | Receptive field 15 already spans a 9×9 board; 27 would be wasted. Width is the axis C1 varies. |
| Feature planes | 48 (49 for value) | **46 (47)** — the two *ladder* planes are absent | On 9×9 a ladder resolves against the edge within ~4 moves, and its first move is already visible through *capture size*, *self-atari size* and *liberties after move*, all implemented exactly. |
| Rollout patterns | 3×3 colour **and liberty**, 9⁸ hashed | 3×3 colour only, 4⁸ = 65,536 **dense, no collisions**, dihedral-canonicalised to 8,740 classes | Liberty information moves into the tactical features, where it is exact rather than truncated at "≥3". |
| Rollout response feature | 12-point diamond around the previous move | which of 8 ring slots the candidate occupies | Cheap stand-in; keeps rollouts answering local threats. |
| Rollout tactical features | full board | only within the previous move's 8-neighbourhood | The locality restriction every fast Go engine uses. A capture on the far side of the board gets no bonus. |
| SL optimiser | SGD, no momentum, α=0.003 halved every 80M steps, batch 16, **340M steps** | SGD, no momentum, halved on schedule, batch 256, tens of thousands of steps | 50 GPUs for three weeks is the one thing that cannot be resized. |
| RL learning rate | **not stated in the paper** | a flag; value reported in §4 | A genuine gap in the paper, not an omission here. |
| RL scale | 10,000 minibatches × 128 games | ~200–300 iterations × 32 games | Opponent-pool refresh cadence rescaled to keep the paper's ratio of ~20 snapshots. |
| RL baseline | 0 on the first pass, value net on the second | **0** (first pass only) | Matches the paper's first pass exactly. |
| Value data | 30M positions, one per game | both arms given the **same** number of positions, drawn from very different numbers of games | Matching positions rather than games is what makes C3 measure correlation instead of data volume. |
| MCTS expansion | `n_thr = 40` | `n_thr = 1` | 40 exists to stop the tree outrunning asynchronous GPU evaluation. This search is synchronous; there is nothing to wait for. |
| Search | 40 threads, 48 CPUs, 8 GPUs, virtual loss `n_vl=3`, tree reuse, distributed | single-threaded, synchronous, no virtual loss, no tree reuse | Virtual loss only exists to keep parallel threads apart. |
| Symmetry | explicit 8-ensemble for raw eval, implicit random-1 in search | both implemented, same roles | Unchanged. |
| Prior at expansion | tree policy `p_τ` placeholder, replaced asynchronously by `p_σ^β` | `p_σ` directly, no placeholder, no softmax temperature β | The placeholder exists purely to have *something* before the GPU replies. |

---

## 3. What is not tested

Exhaustively, so the scope is not overstated: distributed and asynchronous
search; virtual loss; the tree policy `p_τ`; the softmax temperature β; the
last-good-reply rollout cache; dynamic expansion thresholds; tree reuse between
moves; resignation; handicap play; time control; 8-way symmetry ensembling at
evaluation *inside* the tournament (the implicit random-1 ensemble is used);
the second RL pass with the value network as baseline; and anything about
19×19, human opponents, or Elo comparable to published programs.

---

## 4. What broke

*(Filled in as the run proceeds — this section records evaluation and
implementation faults found, with the numbers before and after the fix.)*

**Found before the sweep, by writing the tests first:**

1. **PUCT would have confounded the C5 comparison.** The paper writes the
   exploration bonus over `N_r`, its rollout count, which in AlphaGo *is* the
   simulation count because every simulation runs a rollout. Here λ decides
   which estimators run at all: at λ=1 only `N_r` advances, at λ=0 only `N_v`,
   and at λ=0.5 both do. Using either counter — or their sum — inside PUCT would
   have given the three arms of C5 **three different exploration schedules**, so
   the experiment would have measured exploration rather than mixing. Fixed by a
   separate `Nvis` that counts simulations and nothing else; pinned by
   `test_visit_count_is_lambda_independent`.

2. **The policy symmetry ensemble undid its own rotation the wrong way.** The
   inverse permutation was applied in the forward direction, which applies the
   symmetry a second time instead of removing it. The result is still a valid
   probability distribution and still trains and plays — just worse, in a way
   indistinguishable from a weak network. Caught by requiring the ensemble to
   *commute* with the symmetry (`test_policy_ensemble_commutes_with_symmetry`).

3. **Tromp-Taylor scoring is less intuitive than it looks.** An early scoring
   test asserted that a black wall enclosing a 2×2 corner scores 9 − komi. It
   scores 81 − komi: with no white stones anywhere on the board, *every* empty
   point is reachable only from black, so black owns the whole board. The engine
   was right and the test was wrong. The corrected test now places a lone white
   stone specifically to exercise the "solely reachable" rule.

---

## 5. Results

*(pending)*

---

## 6. Verdict per claim

*(pending)*

---

## Reproducing this

```bash
pip install -r requirements.txt
python tests/test_go.py && python tests/test_features.py \
  && python tests/test_nets.py && python tests/test_mcts.py
```

Pipeline order: `gen_expert.py` → `train_rollout.py` → `train_sl.py` →
`check_eval.py` (**gate**) → `train_rl.py` → `gen_value_data.py` →
`train_value.py` → `tournament.py` / `eval_value_vs_rollouts.py` /
`eval_c1_strength.py` → `make_figures.py`.
