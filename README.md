# AlphaGo from scratch

A from-scratch reproduction of **Silver et al. 2016, *Mastering the game of Go
with deep neural networks and tree search*** (Nature **529**, 484–489), resized
from 19×19 to 9×9 so that the entire four-stage pipeline runs on two T4s.

Everything is built here. No Go library, no RL library, no MCTS library:

| | |
|---|---|
| `ag/go.py` | Go rules: chains, liberties, captures, suicide, simple ko, eyes, Tromp-Taylor area scoring — in numba |
| `ag/features.py` | the 48 input planes of Extended Data Table 2 (46 here — see deviations) |
| `ag/rollout.py` | the fast rollout policy `p_π`, with incrementally-updated 3×3 patterns |
| `ag/nets.py` | the policy and value networks of the Methods section, per-position output bias included |
| `ag/mcts.py` | APV-MCTS: PUCT selection, dual leaf evaluation, λ-mixed backup |
| `ag/selfplay.py` | batched self-play, so a network sees one large forward pass per move rather than thousands of tiny ones |
| `ag/arena.py` | paired-colour matches, Agresti–Coull intervals, logistic Elo with bootstrap CIs |

## The point

Not "get a strong 9×9 bot". The point is to state the paper's claims so they
could fail, build the smallest honest apparatus that could falsify them, and
report what happens — including the places the *measurement* turned out to be
the fragile part.

* **[CLAIMS.md](CLAIMS.md)** — seven falsifiable claims, written down before any
  code, each with the number that would confirm it.
* **[REPORT.md](REPORT.md)** — results, the full deviations table, and a "what
  broke" section.

## Reading the code

Four core files carry a **comment rail**: the code on the left, a flowing
plain-English explanation down a `# |` column on the right, explaining *why*
each piece has to be the way it is rather than restating what the line says.

```
@njit(cache=True, inline="always")                        # +-- REPAIRING CODES AFTER A MOVE ---------------
def patch_pat(board, surr, pat, q):                       # | When the colour at one point changes, the only
    d = _digit(board[q])                                  # | codes that become wrong are those of its eight
    for k in range(8):                                    # | ring neighbours, because those are exactly the
        r = surr[q, k]                                    # | points that have it in their own ring. Each of
        if r < 0:                                         # | those codes is wrong in exactly one digit, and
            continue                                      # | the antisymmetry of the ring ordering says
        pw = POW4[7 - k]                                  # | which one ...
```

Annotated: `ag/go.py`, `ag/rollout.py`, `ag/mcts.py`, `ag/features.py`,
`ag/nets.py`.

## Tests

40 tests, and the ones worth knowing about are not the shape checks:

```bash
python tests/test_go.py        # captures, ko and its expiry, suicide-that-captures, area scoring
python tests/test_features.py  # features(rotate(s)) == rotate(features(s)) for all 8 symmetries
python tests/test_nets.py      # the symmetry ensemble must COMMUTE with the symmetry
python tests/test_mcts.py      # the value sign chain, for black AND for white
```

The recurring theme: every one of those failures produces a program that still
runs, still returns legal moves, and merely plays *slightly worse* — which is
indistinguishable from "the network is weak" and would otherwise never be found.

## Running it

```bash
pip install -r requirements.txt
```

```
gen_expert.py        teacher games (MCTS with rollouts, no network)
   ↓
train_rollout.py     fit p_π by softmax regression on the teacher's moves
train_sl.py          the SL policy network p_σ                        [C1]
   ↓
check_eval.py        THE GATE — measures the accuracy ceiling, the MSE floor,
                     the Elo bracket and the degenerate shortcuts, and refuses
                     the sweep if any of them is too narrow to measure in
   ↓
train_rl.py          REINFORCE self-play against a pool of past selves   [C2]
gen_value_data.py    one position per game, three-phase generation
train_value.py       the value network, both data schemes               [C3]
   ↓
tournament.py                 round robin over the ablation rows    [C5 C6 C7]
eval_value_vs_rollouts.py     Fig 2b                                     [C4]
eval_c1_strength.py           Fig 2a                                     [C1]
make_figures.py
```

Board size and komi come from the environment (`AG_BOARD_SIZE`, `AG_KOMI`), so
the whole pipeline re-runs at 7×7 without editing anything.

## What this is not

It is not AlphaGo. The supervised teacher here is a Monte-Carlo tree search
program, not 160,000 games by 6–9 dan humans, so the entire ladder is anchored
lower and absolute strength is not comparable to the paper's. Every claim under
test is a *relative* comparison, which is why they survive the substitution —
and saying so plainly is part of the job.
