# AlphaGo from scratch

A from-scratch reproduction of **Silver et al. 2016, *Mastering the game of Go
with deep neural networks and tree search*** (Nature 529, 484–489), resized from
19×19 to 9×9 so the entire four-stage pipeline runs on two T4s.

Everything is built here: the Go rules engine, the feature planes, the policy and
value networks, the fast rollout policy, REINFORCE self-play, and APV-MCTS with
PUCT — no Go library, no RL library.

* **[CLAIMS.md](CLAIMS.md)** — the seven falsifiable claims, written before the code.
* **REPORT.md** — results, deviations, and what broke. *(in progress)*

## Status

Work in progress. See CLAIMS.md for what is being tested and why.
