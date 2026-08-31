"""Players -- one uniform interface over every agent the tournament compares.

The rows of Extended Data Table 7 are all the same program with different parts
switched off, so they are all ``MCTSPlayer`` with different arguments.  The
degenerate players at the bottom exist only to bracket the measurement: if
"always pass" or "always play the same point" is competitive, the benchmark is
broken and no result above it means anything.
"""

import numpy as np

from . import go
from .go import PASS, BLACK, WHITE
from .mcts import MCTS


class Player:
    name = "player"

    def move(self, pos):
        raise NotImplementedError

    def reset(self):
        pass


def _sensible(pos):
    """Legal moves that are not our own eyes, as an index array."""
    idx = np.flatnonzero(pos.legal_actions(exclude_eyes=True))
    return idx[idx != PASS]


class RandomPlayer(Player):
    """Uniform over sensible moves.  The floor of the Elo scale."""

    name = "random"

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def move(self, pos):
        idx = _sensible(pos)
        return PASS if len(idx) == 0 else int(self.rng.choice(idx))


class PassPlayer(Player):
    """Always passes.  Degenerate-shortcut check: under area scoring with komi
    this should lose essentially every game, and if it does not, the scoring or
    the komi is wrong."""

    name = "always-pass"

    def move(self, pos):
        return PASS


class FirstPointPlayer(Player):
    """Plays the lowest-index sensible point.  The other degenerate control:
    a fixed, content-free policy that still fills the board legally."""

    name = "first-point"

    def move(self, pos):
        idx = _sensible(pos)
        return PASS if len(idx) == 0 else int(idx[0])


class RolloutPolicyPlayer(Player):
    """p_pi playing on its own, with no search at all.

    This is the baseline the value network is measured against in Fig. 2b, and
    it is what a single rollout step actually does.
    """

    name = "p_pi"

    def __init__(self, rollout, greedy=False):
        self.rollout = rollout
        self.greedy = greedy

    def move(self, pos):
        if self.greedy:
            scores = self.rollout.logits(pos)
            legal = pos.legal_actions(exclude_eyes=True)
            scores = np.where(legal[:go.NN], scores, -np.inf)
            if not np.isfinite(scores).any():
                return PASS
            return int(np.argmax(scores))
        return int(self.rollout.sample(pos))


class PolicyNetPlayer(Player):
    """The policy network selecting moves directly -- the paper's alpha_p row,
    "The version solely using the policy network does not perform any search."

    ``temperature=0`` takes the argmax; the paper samples ("sampling each move
    a_t ~ p_rho(.|s_t) from its output probability distribution") when measuring
    the RL policy head-to-head, which is what ``temperature=1`` does.
    """

    def __init__(self, policy_fn, name="p_net", temperature=0.0, rng=None):
        self.policy_fn = policy_fn
        self.name = name
        self.temperature = temperature
        self.rng = rng if rng is not None else np.random.default_rng()

    def move(self, pos):
        p = np.asarray(self.policy_fn(pos), dtype=np.float64).copy()
        legal = pos.legal_actions(exclude_eyes=True)
        p[~legal] = 0.0
        if p.sum() <= 0:
            return PASS
        if self.temperature <= 0:
            return int(np.argmax(p))
        q = p ** (1.0 / self.temperature)
        return int(self.rng.choice(len(q), p=q / q.sum()))


class MCTSPlayer(Player):
    """Any row of Extended Data Table 7, depending on what is passed in."""

    def __init__(self, name="mcts", **kw):
        self.name = name
        self.kw = kw
        self.mcts = MCTS(**kw)

    def move(self, pos):
        a, _ = self.mcts.choose(pos, temperature=0.0)
        return a
