# Reproducing AlphaGo at 9×9

Silver et al., *Mastering the game of Go with deep neural networks and tree
search*, **Nature 529, 484–489 (2016)**.

Everything is built from nothing: the Go rules engine, the feature planes, the
policy and value networks, the fast rollout policy, REINFORCE self-play, and
APV-MCTS with PUCT. No Go library, no RL library.

> **Status: complete.** All seven claims measured end to end on 2×T4 — two
> confirmed, four refuted, one split (§6). Run 1's numbers are superseded
> throughout: two of its three policy networks were measured against blank
> boards, and the step-size conclusion it drew from that was wrong (§4).

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

### The expensive one: a 32-bit counter holding a 64-bit tag

Four data-generation workers died **simultaneously**, each after about 2,150
seconds, with `munmap_chunk(): invalid pointer` and `double free or corruption
(out)`. Different seeds, different games, different data — but the *same
elapsed time*. Independent processes failing at the same elapsed time means a
fixed operation count, not a data dependency. That is the whole diagnosis in one
observation.

The flood fill in `group_libs` avoids clearing its 81-entry "visited" array on
every call — which would cost more than the fill itself — by stamping it with a
monotonically increasing tag and comparing against the current one. The tag
counter is `int64`. The array was `int32`.

Past 2³¹ the store silently truncates, so `seen[q] == tag` can never be true
again. The fill then marks *nothing* as visited, re-appends the same stones
forever, and runs off the end of the 81-element `buf` into the heap.

At roughly 324 tags per `legal_mask()` call, 2³¹ ⁄ 324 ≈ 6.6M calls — about 35
minutes of continuous play. Which is exactly when all four died.

**Cost:** 900 completed teacher games, ~35 minutes × 4 cores, lost entirely —
because the workers only wrote their shard at the end.

**Two fixes, for two separate faults:**

* `seen` is now `int64` everywhere, matching the tag width. 648 bytes.
  `group_libs` additionally refuses to write past the end of `buf`, so a future
  width mismatch degrades into a wrong answer instead of heap corruption —
  which is far harder to trace back to its cause.
* `gen_expert.py` writes a partial shard every 40 games. The bug destroyed
  everything only because nothing had been written yet; the worst case is now
  40 games.

Three regression tests, including one that drives the real entry points across
2³¹ and requires legality to be unchanged.

Worth stating plainly: **every test passed before this, and the engine was
correct.** 43 tests covering captures, ko, suicide, scoring, symmetry and search
signs, and not one of them ran long enough to reach 2³¹. The bug was not in the
rules, the model, or the measurement — it was in an optimisation that only
misbehaves after six million calls, and no unit test of a Go rule will ever run
that long. The thing that caught it was a crash, and the thing that made the
crash *diagnosable* in one step was that four independent processes died at the
same elapsed time.

### The quiet one: two networks measured against blank boards

**This section previously blamed the step size. That diagnosis was wrong, and
replacing it is more instructive than the original story was.**

The symptom, seen twice. Three SL widths train at once; two of them report a
test accuracy of about 1.4%, frozen to four decimal places for twenty thousand
steps, with the test loss climbing past ln(81) while the training loss behaves
normally. One width is unaffected. In run 1 the survivor was k=64; in run 2, at
a *different* step size, the survivor was k=128 and k=64 was one of the
casualties.

| step | train loss | test acc | test loss | dead trunk units |
|---|---|---|---|---|
| 2,500 | 4.394 | 1.30% | 4.394 | 56% |
| 7,500 | 3.010 | 1.42% | 5.054 | 35% |
| 15,000 | 2.846 | 1.42% | 8.330 | 34% |
| 25,000 | 2.873 | 1.52% | 10.458 | 34% |

The first explanation on offer was a dead-rectifier collapse driven by too large
a step size, and it fit well enough to be believed: the training loss parks near
2.9, which is the entropy of the marginal move distribution; the test loss runs
away as if only the 81 per-position output biases were still learning; and the
step size was the one thing that had been deliberately changed. Run 1 lowered it
from 0.03 to 0.01, re-ran the two dead widths, got healthy numbers, and wrote it
up.

**The check that broke the story.** Run 2 collapsed at the *lowered* step size,
and took k=64 with it — the width run 1 had reported as the survivor. Loading
the three finished checkpoints and scoring them against a freshly built feature
cache gave 22.65%, 22.85% and 23.20% test accuracy. The weights were fine. All
three networks had trained correctly the whole time. **Only the measurement was
broken**, which is why lowering the step size in run 1 appeared to fix it: the
re-run was sequential, and a sequential run does not race.

The cause is one line in `build_feature_cache`. `np.lib.format.open_memmap`
creates the cache at its **full final size, zero filled**, and then fills it row
by row over several seconds. Any process starting inside that window found a
file of exactly the right shape and dtype, took the `reusing` branch, and read
zeros.

Training tolerated that, which is what hid it. The training set is re-read from
the memmap every step, so it becomes real the moment the build finishes — hence
the loss sitting at exactly ln(81) for the first 5,000 steps and then dropping.
The held-out set does not recover: `Xte = np.asarray(X[te])` copies it into RAM
once, at startup. And `te` is the last 10% of games, which is the *last* region
of the file to be written. The two losers of the race spent the entire run
computing accuracy against empty boards.

Every part of the "collapse" then follows. Accuracy is frozen because the
evaluation input never changes. It is ~1.4% because on an empty board the
network answers the same way every time. The test loss climbs because the
network grows more confident as it learns — on positions it is not being shown.

**Fixes.** The cache is built into a temporary file under a lock and renamed
into place, so a reader can only ever see all of it or none of it; and it is
verified through `P_ONES`, a constant plane of ones that no valid row can be
missing, so an unwritten row is detectable exactly rather than by eye.
`train_sl.py` now refuses to start if its held-out slice is blank. Three
regression tests in `tests/test_cache.py`, one of which races three processes
on one cache path.

**What the step size actually does**, re-measured on a sound cache, all three
widths, 25,000 steps, nothing else changed:

| | k=32 | k=64 | k=128 | at step 2,500 |
|---|---|---|---|---|
| α = 0.01 | 22.95% | 22.95% | 23.50% | still at chance |
| **α = 0.03** | **24.11%** | **24.11%** | **24.87%** | **≈17%** |

No collapse at either. 0.03 is better at every width and reaches in 2,500 steps
what 0.01 has not reached by 7,500 — 0.01 is not safer, it is slower. So the
naive batch-16 → batch-256 rescaling of the paper's α = 0.003 (≈0.048) was
approximately right all along, and the default is now 0.03. The lowering that
"fixed" run 1 fixed nothing and cost a point of accuracy.

Two things are worth taking from this. The first is that **a wrong number which
looks like a plausible training curve will be attributed to whichever knob was
last turned**, and the attribution will be persuasive, self-consistent, and
publishable. The second is what actually broke it open: not a better theory of
step sizes, but re-running the identical configuration and watching a *different*
width survive. A mechanism that cannot predict which of three networks dies is
not the mechanism.

This is also the failure mode the C1 measurement is least able to see. A network
measured against blank boards has a perfectly well-defined accuracy near zero,
and it would have taken its place on the accuracy-versus-strength plot, near the
origin, looking like evidence *for* the claim.

### Found before the sweep, by writing the tests first

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

> **Run 2.** Run 1's numbers are superseded: two of its three policy networks
> were measured against blank boards (§4), so its SL accuracies and its
> step-size conclusion were both wrong. Everything below is from a single
> end-to-end run on a sound cache, with twice the data.

### 5.0 The instrument, measured before the sweep (`check_eval.py` → PASS)

These bracket everything else, and several of them change how the numbers below
should be read.

| quantity | value | why it matters |
|---|---|---|
| **Teacher self-agreement** | **31.7%** (n=120) | The **ceiling on C1**. The teacher is a stochastic MCTS; run twice on the same position it picks the same move only 31.7% of the time. No policy network can exceed this. |
| Teacher vs its stored label | 33.3% | consistency of the recorded targets |
| Constant-predictor value MSE | **1.0000** (E[z] = +0.002) | the **floor for C3/C4**; outcomes are almost perfectly balanced |
| Black win rate, teacher games | 51.0% | komi 7.5 is close to fair at this strength |
| Black win rate, random play | 30.0% | white is favoured under random play; paired colours neutralise it |
| Teacher vs random | 97.5% [86,100] | the Elo bracket is wide enough to hold everything |
| **always-pass vs random** | **loses 100%** | no degenerate shortcut: scoring and komi are sound |
| first-point vs random | 50% [35,65] | a fixed content-free policy is no better than random |
| p_π vs random | 75% [60,86] | the trained rollout policy is meaningfully better than random |

The C1 ceiling moved from 37.5% (run 1) to 31.7% here on the same n=120 — a
reminder that the ceiling is itself an estimate with a wide interval, and that
every "% of ceiling" below inherits that width.

### 5.1 Data

640 teacher games at 128 simulations → **61,817 positions** (59,511 non-pass),
split by game into 53,608 train / 5,903 test. Black wins 51.0%. Generation ran
at 8.8 s/game across 4 workers, ~23 minutes.

### 5.2 The fast rollout policy p_π

The learned tactical weights are interpretable, which is the useful part —
they were fit from data, not set by hand:

| feature | weight |
|---|---|
| captures ≥1 stone | **+0.825** |
| saves a chain from atari | **+0.328** |
| captures ≥2 stones | +0.077 |
| self-atari | +0.042 |

Capture and save-from-atari — the two moves that actually decide Go rollouts —
come out on top on their own, from data, with no hand-set priors.

### 5.3 The SL policy network (C1)

Step size 0.05 (§4), 25,000 steps, mini-batch 256, plain SGD without momentum,
8× dihedral augmentation, split by game.

| width | parameters | test accuracy | top-5 | vs the 31.7% ceiling |
|---|---|---|---|---|
| k=32 | 83,185 | **25.16%** | 48.6% | 79% |
| k=64 | 258,449 | 24.87% | 49.7% | 78% |
| k=128 | 885,457 | 25.12% | 49.9% | 79% |

Read against the ceiling, not against 100%: **the policy network captures about
four-fifths of the teacher's own reproducibility.**

**Width does essentially nothing here, and in the paper it does** (128→256
filters moved accuracy 54.6%→55.9% and the raw-net win rate 36%→67%). The three
widths land within 0.3 points of each other across a 10× parameter range. At
54k training positions capacity is not the bottleneck, data is — so C1's
accuracy axis has to come from training-time checkpoints rather than width, and
the width sweep is reported as the null result it is.

### 5.4 RL policy gradient (C2) — **confirmed**

Sanity check before any update: RL(=SL) vs SL = **0.520**, i.e. ≈ 0.5 as it must
be, which is what says the harness is not scoring one side wrong.

| iteration | 25 | 50 | 75 | 150 | 200 |
|---|---|---|---|---|---|
| RL vs SL win rate (100 games) | 75.0% | 82.0% | 91.0% | 99.0% | **100.0%** |

The paper reports "more than 80% of games against the SL policy network". Here
it passes 80% by iteration 50 and reaches 100/100 by iteration 200 — a stronger
result than the paper's, and a fair reading is that it is *easier* here: the SL
policy is distilled from a 128-simulation MCTS rather than from human dan play,
so there is more headroom above it and less to preserve.

### 5.5 One forward pass against a search ladder (C7)

The round robin gives every search player the same 100 simulations, which cannot
answer a claim about *thousands* of rollouts. So the opponent is swept instead:
the raw network, one forward pass per move and no tree, against `a_r` (MCTS with
p_π rollouts, λ=1, uniform prior, no network anywhere) at a ladder of budgets.
20 paired-colour games per rung.

| opponent `a_r` @ | 50 sims | 100 sims | 300 sims | 1000 sims |
|---|---|---|---|---|
| **p_σ** (SL policy, no search) | 10% [2,31] | 10% [2,31] | 0% [0,19] | 0% [0,19] |
| **p_ρ** (RL policy, no search) | **100%** [81,100] | **100%** [81,100] | **95%** [75,100] | **50%** [30,70] |

Two opposite answers from the same architecture.

**For the SL policy the claim is refuted, and not narrowly** — p_σ loses to
*fifty* rollouts a move. This is the resize showing its teeth, and it is worth
stating precisely. The SL policy is distilled from a 128-simulation MCTS, so it
cannot exceed that teacher; it reaches 79% of the teacher's own self-agreement
(§5.3) and is still beaten 9 games in 10 by a search running a *third* of the
teacher's budget. Being most of the way to a teacher's move distribution is not
the same as being most of the way to its strength — which is C1's premise viewed
from the other end, and the sharpest number in this report against it.

**For the RL policy the claim is confirmed at full strength.** p_ρ is exactly
even with 1,000 rollouts a move (10/20, CI [30,70]) and beats 300 rollouts 95%
of the time. One forward pass is worth on the order of a thousand rollouts.

So policy-gradient self-play moves the same network from "worse than 50
rollouts" to "worth 1,000" with no change whatsoever in inference cost — a ~20×
improvement in search-equivalent strength bought entirely by training, and the
largest single effect measured anywhere in this reproduction.

The paper's α_p row is the **SL** network, so C7 as the paper states it does not
reproduce here; the claim survives only when re-anchored on p_ρ. Recording that
as a pass would be reporting the wrong network.

### 5.6 Correlated vs uncorrelated value data (C3) — **the primary ablation**

Same network, same objective, same optimiser, same step size (0.2, calibrated —
§4), **same number of training positions (17,503)**. One thing differs: where
the positions come from.

| | positions | drawn from | per game | neighbours with opposite labels |
|---|---|---|---|---|
| correlated | 17,503 | **576 games** | 30.4 | **59.3%** |
| uncorrelated | 17,503 | **17,503 games** | 1.0 | 0% |

That last column is the paper's stated mechanism as a number. Because z is from
the mover's point of view, two consecutive positions of one game differ by a
single stone and carry *opposite* labels 59.3% of the time. Nearly identical
inputs with opposite targets, and 17,503 positions carrying only 576 independent
outcomes.

Trained to 20,000 steps, both arms:

| step | correlated train | correlated test | | uncorrelated train | uncorrelated test |
|---|---|---|---|---|---|
| 5,000 | 0.766 | **0.770** | | 0.641 | 0.650 |
| 8,000 | 0.563 | 0.854 | | 0.587 | **0.624** |
| 12,000 | 0.335 | 1.027 | | 0.557 | 0.653 |
| 16,000 | 0.148 | 1.236 | | 0.451 | 0.714 |
| 20,000 | **0.099** | **1.279** | | 0.360 | 0.756 |

**Confirmed, and more starkly than the paper.** The correlated arm memorises —
training MSE falls to 0.099, which for 576 distinct outcomes is exactly what
memorising 576 numbers looks like — while its test MSE climbs to **1.279, worse
than the constant predictor's 1.000**. It ends up less useful than answering
"I don't know" to every position. The uncorrelated arm, on the same budget,
holds test MSE at 0.624 and never crosses the floor.

Read at each arm's best checkpoint, which is the paper's own phrasing:

| | train | test | **common test set** | vs floor 1.000 | vs probe 0.795 |
|---|---|---|---|---|---|
| correlated | 0.766 | 0.770 | 0.733 | better | worse |
| **uncorrelated** | 0.587 | **0.624** | **0.644** | better | **better** |
| paper | 0.19 → 0.226 | 0.37 → 0.234 | | | |

The final train/test gaps are **+1.18 (correlated) against +0.40
(uncorrelated)**; the paper's are +0.18 against +0.008. Same direction, larger
separation, and the reason for the larger separation is that 576 games is a much
easier thing to memorise than 160,000.

One number keeps this honest. The **analytic probe** — `tanh(a·score + b)` on
the Tromp-Taylor score, no network at all — scores 0.795. The uncorrelated arm
beats it (0.644); the correlated arm never does. So the correlated arm is not
merely "worse", it fails to earn its own existence: at no point in training is
it better than counting the stones on the board.

### 5.7 Value net vs rollouts (C4) — **refuted, and the reason is the board**

150 expert positions, 100 rollouts each, MSE against the true outcome:

| estimator | MSE | vs floor 0.998 | seconds/position |
|---|---|---|---|
| value network (1 forward pass) | 0.799 | 0.801 | **0.011** |
| 100 rollouts, uniform random | 0.654 | 0.655 | 0.073 |
| 100 rollouts, p_π | 0.643 | 0.644 | 0.073 |
| 100 rollouts, p_σ | 0.633 | 0.634 | 1.635 |
| 100 rollouts, p_ρ | **0.628** | 0.629 | 1.728 |

The paper's Fig. 2b has the value network *below* every rollout curve. Here it
is **above all four — including uniform random rollouts.** C4 does not
reproduce.

The mechanism is visible in the per-stage breakdown, and it is the board size:

| move number | 0–15 | 15–30 | 30–45 | 45–60 | 60–80 | 80–200 |
|---|---|---|---|---|---|---|
| value net | 1.106 | 1.120 | 1.037 | 0.593 | 0.745 | 0.195 |
| 100 rollouts, p_π | 1.062 | 1.020 | 0.933 | 0.644 | **0.178** | **0.020** |

Past move 60 the rollouts are nearly *exact* (MSE 0.02 by move 80). On a 9×9
board a game is over within about 100 moves, so a rollout from move 80 has
twenty moves of noise left in it and essentially reads out the true result. At
19×19 the same rollout has 250+ moves to go and is genuinely noisy — which is
the entire premise of the paper's claim. **The claim is about rollout variance
over a long horizon, and shortening the board removes the variance it is about.**

This is the clearest case in the reproduction of a claim that fails for the
resize rather than for itself, and no amount of extra training would fix it: the
rollouts are near-perfect late, and nothing can beat near-perfect.

The cost column still holds, for what it is worth — one forward pass is 6.5×
cheaper than 100 p_π rollouts and 150× cheaper than 100 p_ρ rollouts — so the
value net is the better *rate*, just not the better estimator.

### 5.8 The ablation round robin (C5, C6)

Ten players, 45 pairs, 12 paired-colour games each, 100 simulations for every
search player. Random anchored at 0.

| player | components | Elo | 95% CI |
|---|---|---|---|
| `rvp_rl` | p_ρ prior, v_θ, p_π, λ=0.5 | **1524** | [1353, 1701] |
| `p_rl` | RL policy, **no search** | **1358** | [1201, 1552] |
| `a_rp` | p_σ prior, p_π, λ=1 | 894 | [700, 1080] |
| `a_r` | p_π, λ=1, uniform prior | 781 | [624, 957] |
| `a_v` | v_θ, λ=0, uniform prior | 765 | [615, 939] |
| `a_rvp` | p_σ prior, v_θ, p_π, λ=0.5 | 671 | [471, 885] |
| `p_pi` | rollout policy alone | 180 | [−18, 380] |
| `p_sl` | SL policy, no search | 128 | [−55, 342] |
| `a_vp` | p_σ prior, v_θ, λ=0 | 92 | [−94, 281] |
| `random` | — | 0 | — |

**C5 (λ=0.5 beats λ=0 and λ=1): half confirmed, half refuted.**

| | | |
|---|---|---|
| λ=0.5 (`a_rvp`) vs λ=0 (`a_vp`) | **12/12 = 100%** [72,100] | mixing beats value-only ✓ |
| λ=0.5 (`a_rvp`) vs λ=1 (`a_rp`) | **4/12 = 33%** [14,61] | mixing **loses** to rollout-only ✗ |

Elo says the same: 671 for λ=0.5 against 894 for λ=1. The paper has λ=0.5 on top
(α_rvp 2890 > α_rp 2416 > α_vp 2177). Here the ordering is λ=1 > λ=0.5 > λ=0.

This follows directly from §5.7 and is the same fact seen from the search side.
Mixing in an estimator that is *worse* than the one you already have makes the
blend worse, and at 9×9 the value network is worse than the rollouts. C5 is not
independently refuted — it is C4's refutation propagating.

**C6 (the SL policy is a better MCTS prior than the stronger RL policy):
refuted, on both halves.**

| | | |
|---|---|---|
| raw p_ρ vs raw p_σ | **12/12 = 100%** [72,100] | RL is the stronger *player* ✓ (this is C2) |
| MCTS[p_σ] (`a_rvp`) vs MCTS[p_ρ] (`rvp_rl`) | **0/12 = 0%** [0,28] | RL is also the better *prior* ✗ |

The claim requires both halves, and the second fails as completely as a 12-game
match can fail. 1524 Elo against 671: the RL prior is worth ~850 Elo over the SL
prior, in the same program with everything else identical.

The paper's reasoning for C6 is that "humans select a diverse beam of promising
moves, whereas RL optimizes for the single best move" — a claim about the
*diversity* of a prior distilled from **human** play. There are no humans here.
Both priors are distilled from the same MCTS teacher and then one is sharpened
by self-play, so the diversity the paper credits to human variety was never
present to be lost. C6 is the claim in this set that the resize most directly
removes the conditions for, and the number is reported as a refutation of the
*sentence*, not as evidence against the paper's reading of its own system.

**The instrument's own reading.** One row is worth pausing on: `a_vp` (92 Elo)
lands *below* `a_v` (765), so adding the SL prior to a value-only search costs
673 Elo. That is not noise — `a_vp` sits within its CI of `p_sl` (128), which is
what it means for a search to have stopped searching. With λ=0 the only signal
is a value network that §5.7 showed is barely better than counting, so PUCT
follows the prior and reproduces it. Every arm that leans on the value network
(`a_vp`, `a_rvp`) is dragged toward it; every arm that leans on rollouts
(`a_r`, `a_rp`) is not. The whole table is one story: **at 9×9 the value network
is the weak component**, and C4, C5 and this row are three views of it.

### 5.9 Does accuracy buy strength? (C1)

The headline claim, and the one that took three attempts to *measure* rather
than to answer.

| attempt | reference | points | games | accuracy range | win-rate range | mean half-CI | verdict |
|---|---|---|---|---|---|---|---|
| 1 | MCTS-30 | 9 | 30 | 1.2 pts | 27 pts | **15 pts** | unusable |
| 2 | p_π | 15 | 50 | 5.4 pts | 16 pts | **13 pts** | unusable |
| **3** | **p_π** | **30** | **400** | **5.4 pts** | **21 pts** | **5 pts** | **resolvable** |

Attempt 1 sampled checkpoints from the middle of training onward, where accuracy
has already saturated, and played them against a reference that beat all of
them — 1.2 points of x-range and every point pinned near the floor of the
y-axis. Attempt 2 fixed the reference and the checkpoint spread but not the
sample size. Only attempt 3 has a win-rate spread larger than its own error
bars, which is the precondition for the question being answerable at all. The
stage now computes that comparison and prints `NOT RESOLVABLE` when it fails.

**The answer, on the resolvable measurement: no relationship. C1 is refuted.**

Spearman(accuracy, win rate) = **−0.04** over 30 checkpoints × 400 games. Within
every width, accuracy rises over training and strength does not move:

| | accuracy at step 2,500 → 25,000 | win rate at step 2,500 → 25,000 |
|---|---|---|
| k=32 | 20.8% → 25.2% | 50.5% → 45.5% |
| k=64 | 20.2% → 24.9% | 53.8% → 45.5% |
| k=128 | 20.5% → 25.1% | 51.5% → 51.5% |

A checkpoint that agrees with the teacher 20% of the time plays the fixed
reference exactly as well as one that agrees 25% of the time. The paper's Fig.
2a spans 50%→57% accuracy and 25%→70% win rate; the last 5 points of accuracy
available here buy nothing.

Two honest caveats, in opposite directions. The accuracy range is small — 5.4
points against the paper's 7, but at a much lower absolute level (20–25% against
50–57%) and against a ceiling of 31.7%. And this reproduction cannot reach the
part of the curve the paper is describing: to get past 26% here would need more
expert data, not more training, since all three widths saturate together (§5.3).
So the correct statement is narrow and it is what was measured: **over the
accuracy range this setup can produce, accuracy does not predict strength.**

That reading is corroborated from a completely different direction by §5.5. The
SL policy at 79% of its teacher's self-agreement still loses to 50 rollouts a
move. Agreement with this teacher and strength are close to unrelated quantities
here — which is precisely the link C1 asserts.

---

## 6. Verdict per claim

| # | Claim | Verdict | The number |
|---|---|---|---|
| **C1** | Move-prediction accuracy translates into playing strength | **Refuted** | Spearman **−0.04** (30 checkpoints × 400 games; win-rate spread 21 pts vs 5-pt half-CI, so the axes do have range). Accuracy 20.2%→25.6% buys nothing |
| **C2** | Policy-gradient self-play against a pool beats the SL policy | **Confirmed** | RL beats SL **100/100** games (paper: >80%); passes 80% by iteration 50. Pre-update sanity check 0.52 |
| **C3** | Whole-game value data overfits; one position per game does not | **Confirmed, more starkly** | Correlated: train MSE →**0.099**, test →**1.279** (worse than the 1.000 floor). Uncorrelated: **0.624** test, never crosses the floor. Final gaps **+1.18 vs +0.40** (paper: +0.18 vs +0.008) |
| **C4** | One value-net pass beats 100 rollouts | **Refuted** | Value net **0.799**; every rollout estimator beats it, including *uniform random* (0.654). Past move 60 rollouts are near-exact (0.02 by move 80) because a 9×9 game ends in ~100 moves |
| **C5** | λ=0.5 beats λ=0 and λ=1 | **Half confirmed** | λ=0.5 beats λ=0 **12/12**; loses to λ=1 **4/12**. Elo 671 / 92 / 894. Downstream of C4 — mixing in a worse estimator makes the blend worse |
| **C6** | The SL policy is a better MCTS prior than the stronger RL policy | **Refuted** | RL is the stronger player (**12/12**, = C2) *and* the better prior (MCTS[SL] loses **0/12** to MCTS[RL]; 671 vs 1524 Elo). Both halves must hold; the second fails completely |
| **C7** | The raw policy net plays at the level of thousands of rollouts | **Refuted for p_σ, confirmed for p_ρ** | p_σ loses to **50** rollouts (10%). p_ρ is **even with 1,000** (10/20) and beats 300 at 95% |

**Four refuted, two confirmed, one split — and the reproduction is the more
informative for it.** Three of the four failures are traceable to one cause with
a name: at 9×9 a game ends in about a hundred moves, so a rollout is nearly an
exact evaluation and the value network — the component AlphaGo introduced to
*replace* noisy rollouts — has nothing to improve on. C4 is that fact measured
directly, C5 is it propagating into the search, and the `a_vp` row of §5.8 is it
again. The paper's value network is answering a question that a small board does
not ask.

The other two failures are about the teacher, not the board. C1 and the p_σ half
of C7 both say the same thing: a policy distilled to 79% of a 128-simulation
MCTS's self-agreement is not thereby a strong player, and squeezing more
agreement out of it does not make it one. Whether that survives at 19×19 against
160,000 human games is exactly the thing this resize cannot test — and saying so
is the point of having written the claims down first.

What *does* survive the resize is everything about **learning from self-play**.
C2 is confirmed more strongly than the paper states it, C3 is confirmed far more
strongly, and the single largest measured effect in the whole reproduction is
policy-gradient self-play taking one forward pass from "worse than 50 rollouts"
to "worth 1,000" at unchanged inference cost.

---

## Reproducing this

```bash
pip install -r requirements.txt
python tests/test_go.py && python tests/test_features.py \
  && python tests/test_nets.py && python tests/test_mcts.py
```

Everything in one command, dependency-ordered across two GPUs and resumable
(each stage skips if its outputs already exist):

```bash
python scripts/gen_expert.py --games 120 --sims 128 --seed 2000 \
       --out data/expert_0.npz          # x4, one per core, ~20 min
python scripts/run_pipeline.py --data data --runs runs   # ~60 min
```

Stage order inside the driver: `train_rollout` + `train_sl`×3 → **`check_eval`
(the gate)** → `train_rl` → `gen_value_data` → `train_value` → `tournament` /
`eval_value_vs_rollouts` / `eval_c1_strength` → `make_figures`.

Then watch it play:

```bash
python scripts/demo.py --runs runs --black a_rvp --white a_r
```

### Reproducing the numbers above

The two step sizes that matter are **not** the naive rescalings:

* SL: `--lr 0.01`. 0.03 collapses two widths out of three (§4).
* Everything is seeded; `--seed` is threaded through every stage.

Figures are regenerated from the result JSONs alone (`make_figures.py --runs`),
so the JSONs in `runs/` are the durable record — no checkpoint is needed to
redraw a plot.
