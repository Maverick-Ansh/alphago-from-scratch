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


class Player:                                                                         # +-- ONE INTERFACE FOR EVERY AGENT ----------------------------
    name = "player"                                                                   # | A player is anything that turns a position into a move, so
                                                                                      # | the arena never has to know whether it is driving a network,
    def move(self, pos):                                                              # | a tree search, or a coin flip. Sensible moves are legal
        raise NotImplementedError                                                     # | moves that are not our own eyes; a player that would fill
                                                                                      # | its own eye kills its own group, so every player here draws
    def reset(self):                                                                  # | from that set rather than from bare legality.
        pass


def _sensible(pos):
    """Legal moves that are not our own eyes, as an index array."""
    idx = np.flatnonzero(pos.legal_actions(exclude_eyes=True))
    return idx[idx != PASS]


class RandomPlayer(Player):                                                           # +-- THE PLAYERS THAT EXIST ONLY TO BRACKET THE RESULT --------
    """Uniform over sensible moves.  The floor of the Elo scale."""                   # | Uniform choice is the bottom of the scale, and everything
                                                                                      # | else is measured against it. The other two are shortcut
    name = "random"                                                                   # | checks. If passing every turn is competitive then the
                                                                                      # | scoring or the komi is wrong. If always playing the lowest-
    def __init__(self, rng=None):                                                     # | numbered point is competitive then the game is not
        self.rng = rng if rng is not None else np.random.default_rng()                # | discriminating between players and no number above it means
                                                                                      # | anything. Both are scored explicitly rather than assumed
    def move(self, pos):                                                              # | harmless, because a benchmark with a hole in it produces
        idx = _sensible(pos)                                                          # | results that look perfectly ordinary.
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


class RolloutPolicyPlayer(Player):                                                    # +-- THE ROLLOUT POLICY AS A PLAYER IN ITS OWN RIGHT ----------
    """p_pi playing on its own, with no search at all.

    This is the baseline the value network is measured against in Fig. 2b, and
    it is what a single rollout step actually does.
    """
                                                                                      # | The same weights the search uses inside a rollout, here
    name = "p_pi"                                                                     # | choosing real moves. This is the baseline the value network
                                                                                      # | is measured against, and it answers a question the paper
    def __init__(self, rollout, greedy=False):                                        # | cares about: how much of a rollout's worth comes from the
        self.rollout = rollout                                                        # | policy guiding it rather than from averaging many outcomes.
        self.greedy = greedy                                                          # | Greedy mode takes the best-scoring point instead of
                                                                                      # | sampling, masked to legal moves, since raw scores say
    def move(self, pos):                                                              # | nothing about legality.
        if self.greedy:
            scores = self.rollout.logits(pos)
            legal = pos.legal_actions(exclude_eyes=True)
            scores = np.where(legal[:go.NN], scores, -np.inf)
            if not np.isfinite(scores).any():
                return PASS
            return int(np.argmax(scores))
        return int(self.rollout.sample(pos))


class PolicyNetPlayer(Player):                                                        # +-- A NETWORK PLAYING WITH NO SEARCH AT ALL ------------------
    """The policy network selecting moves directly -- the paper's alpha_p row,
    "The version solely using the policy network does not perform any search."

    ``temperature=0`` takes the argmax; the paper samples ("sampling each move
    a_t ~ p_rho(.|s_t) from its output probability distribution") when measuring
    the RL policy head-to-head, which is what ``temperature=1`` does.
    """
                                                                                      # | One forward pass, one move. This is the row of the paper's
    def __init__(self, policy_fn, name="p_net", temperature=0.0, rng=None):           # | ablation table that carries no tree at all, and claim C7 is
        self.policy_fn = policy_fn                                                    # | about how far it gets. The network puts probability on
        self.name = name                                                              # | occupied and illegal points, so the mask is applied before
        self.temperature = temperature                                                # | anything is chosen; if nothing survives, the only remaining
        self.rng = rng if rng is not None else np.random.default_rng()                # | act is to pass. Zero temperature takes the most likely move,
                                                                                      # | which is what evaluation games use; the paper samples
    def move(self, pos):                                                              # | instead when it measures the reinforcement-learned policy
        p = np.asarray(self.policy_fn(pos), dtype=np.float64).copy()                  # | head to head, and that is what a positive temperature does.
        legal = pos.legal_actions(exclude_eyes=True)
        p[~legal] = 0.0
        if p.sum() <= 0:
            return PASS
        if self.temperature <= 0:
            return int(np.argmax(p))
        q = p ** (1.0 / self.temperature)
        return int(self.rng.choice(len(q), p=q / q.sum()))


class MCTSPlayer(Player):                                                             # +-- EVERY SEARCH-BASED ROW IS THIS ONE CLASS -----------------
    """Any row of Extended Data Table 7, depending on what is passed in."""           # | All the tree-search entries in the ablation table are the
                                                                                      # | same object with different arguments, so nothing about the
    def __init__(self, name="mcts", **kw):                                            # | comparison can differ between arms except what is
        self.name = name                                                              # | deliberately being varied. Moves are always taken at zero
        self.kw = kw                                                                  # | temperature, because randomising a move in an evaluation
        self.mcts = MCTS(**kw)                                                        # | game adds noise to the very quantity being measured.

    def move(self, pos):
        a, _ = self.mcts.choose(pos, temperature=0.0)
        return a
