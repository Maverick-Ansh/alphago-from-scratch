"""Batched self-play.

Both the RL stage and the value-network data set need tens of thousands of
complete games played by neural networks.  Played one at a time, that is tens
of thousands of *sequential* forward passes of batch size 1 -- the worst thing
you can ask a GPU to do.  So games are played in lockstep: every active game
contributes one position to a single batch, one forward pass serves all of
them, and each game takes its own move.

This is the same trick AlphaGo needed and solved differently.  The paper hides
network latency behind 40 asynchronous search threads with a virtual loss to
keep them apart; we have no latency to hide because there is only one policy
evaluation per move, so plain batching across games is both simpler and a
better fit for one GPU.

Feature planes computed here are handed back to the caller.  RL needs exactly
the planes the policy saw in order to differentiate through them, and
recomputing would be pure waste.
"""

import numpy as np
import torch

from . import features as feat, go
from .go import BLACK, WHITE, PASS, N, NN


class BatchPolicy:                                                                    # +-- A NETWORK THAT ANSWERS MANY POSITIONS AT ONCE ------------
    """A torch policy net, evaluated over a list of positions at once."""             # | Feature planes for a whole list of positions, then one
                                                                                      # | forward pass for all of them. This is the only reason self-
    needs_planes = True                                                               # | play is affordable: a graphics processor gets almost none of
                                                                                      # | its throughput on a batch of one, so playing games one at a
    def __init__(self, net, device="cpu", temperature=1.0, rng=None,                  # | time would leave it idle between moves. Half precision is
                 with_colour=False):                                                  # | used because these cards run it several times faster than
        self.net = net.to(device).eval()                                              # | single precision, and the answer only has to be good enough
        self.device = device                                                          # | to pick a move.
        self.temperature = temperature
        self.rng = rng if rng is not None else np.random.default_rng()
        self.fx = feat.FeatureExtractor(with_colour=with_colour)

    def planes(self, positions):
        return np.stack([self.fx(p) for p in positions])

    @torch.no_grad()
    def logits(self, planes):
        x = torch.from_numpy(planes).to(self.device).float()
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=(self.device == "cuda")):
            out = self.net(x)
        return out.float().cpu().numpy()

    def choose(self, positions, planes=None):                                         # +-- FROM SCORES TO A LEGAL MOVE ------------------------------
        """Sample (or argmax) a legal move for each position."""                      # | The network is never told which moves are legal and will put
        if planes is None:                                                            # | weight on occupied points, so illegal and eye-filling moves
            planes = self.planes(positions)                                           # | are removed before anything is sampled. If nothing at all is
        logits = self.logits(planes)                                                  # | left, the only sensible act is to pass, which is also
        actions = np.empty(len(positions), dtype=np.int64)                            # | exactly how a game ends under area scoring once the board is
        for i, pos in enumerate(positions):                                           # | settled. Sampling rather than taking the best move is what
            legal = pos.legal_actions(exclude_eyes=True)[:NN]                         # | the reinforcement learning stage requires, since it can only
            z = np.where(legal, logits[i].astype(np.float64), -np.inf)                # | learn from actions the current policy actually chose; a
            if not np.isfinite(z).any():                                              # | temperature of zero switches back to the best move for
                actions[i] = PASS          # only our own eyes remain: pass           # | evaluation games.
                continue
            if self.temperature <= 0:
                actions[i] = int(np.argmax(z))
            else:
                z = z / self.temperature
                z -= z.max()
                p = np.exp(z)
                p /= p.sum()
                actions[i] = int(self.rng.choice(NN, p=p))
        return actions, planes


class RandomBatchPolicy:                                                              # +-- THE SAME INTERFACE, WITHOUT A NETWORK --------------------
    """Uniform over sensible moves, with the same interface."""                       # | Uniform choice among sensible moves, wearing the same shape
                                                                                      # | so it can be dropped into any slot a network fills. It is
    needs_planes = False                                                              # | the single random move in the middle of the value-network
                                                                                      # | generation recipe, and the floor that every other player is
    def __init__(self, rng=None, with_colour=False):                                  # | measured against.
        self.rng = rng if rng is not None else np.random.default_rng()
        self.fx = feat.FeatureExtractor(with_colour=with_colour)

    def planes(self, positions):
        return np.stack([self.fx(p) for p in positions])

    def choose(self, positions, planes=None):
        actions = np.empty(len(positions), dtype=np.int64)
        for i, pos in enumerate(positions):
            idx = np.flatnonzero(pos.legal_actions(exclude_eyes=True))
            idx = idx[idx != PASS]
            actions[i] = PASS if len(idx) == 0 else int(self.rng.choice(idx))
        return actions, planes


class Game:                                                                           # +-- WHAT A GAME IN FLIGHT CARRIES ----------------------------
    """One game in flight, plus whatever the caller asked to record."""               # | The position, and optionally the record of it: the planes
                                                                                      # | the policy actually saw, the move it chose, and which colour
    __slots__ = ("pos", "planes", "actions", "movers", "done", "z_black",             # | chose it. Keeping the planes rather than recomputing them
                 "meta")                                                              # | later matters because the learning step needs exactly the
                                                                                      # | input the policy was looking at, and rebuilding it would be
    def __init__(self):                                                               # | both slower and a chance for the two to disagree.
        self.pos = go.Position()
        self.planes = []
        self.actions = []
        self.movers = []
        self.done = False
        self.z_black = 0
        self.meta = {}


def run_games(n_games, policy_for, rng=None, max_moves=None, record=False,            # +-- ALL THE GAMES ADVANCE TOGETHER ---------------------------
              on_step=None, init_positions=None):                                     # | Every game contributes one position, they are grouped by
    """Play ``n_games`` in lockstep.

    ``policy_for(colour, game_index, game)`` returns the policy object that
    should move.  Games are grouped by the *identity* of the returned policy so
    that each distinct network sees one batched forward pass per step, however
    the games are split between colours or phases.

    ``init_positions`` starts the games from given positions instead of an
    empty board -- which is how a *rollout from a leaf* is played out in bulk,
    and what claim C4 needs to compare network evaluation against 100 rollouts.

    Returns the finished ``Game`` objects.
    """
    rng = rng or np.random.default_rng()                                              # | which policy is to move, and each distinct network gets
    max_moves = max_moves or go.MAX_MOVES                                             # | exactly one batched call per step. Grouping by the policy
    games = [Game() for _ in range(n_games)]                                          # | object rather than by colour is what lets the same loop
    if init_positions is not None:                                                    # | serve a learner playing its own past self, or a generation
        for g, p0 in zip(games, init_positions):                                      # | recipe that switches between three different move sources
            g.pos = p0.copy()                                                         # | partway through each game, without any of them losing the
                                                                                      # | batching. Games drop out as they finish, so the batch
    while True:                                                                       # | shrinks over time rather than wasting work on finished
        active = [(i, g) for i, g in enumerate(games)                                 # | boards. Starting from given positions is what makes it
                  if not g.pos.is_over() and g.pos.move_no < max_moves]               # | possible to play out thousands of rollouts from a leaf in
        if not active:                                                                # | bulk, which is what claim C4 needs.
            break
        # Group by policy *object* so that each distinct network gets exactly
        # one batched forward pass per step, however the games happen to be
        # split between colours, opponents, or generation phases.
        groups = {}
        for i, g in active:
            pol = policy_for(g.pos.to_play, i, g)
            groups.setdefault(id(pol), (pol, []))[1].append(g)

        for pol, gs in groups.values():
            positions = [g.pos for g in gs]
            need = record or getattr(pol, "needs_planes", True)
            planes = pol.planes(positions) if need else None
            actions, planes = pol.choose(positions, planes)
            for j, (g, a) in enumerate(zip(gs, actions)):
                if record:
                    g.planes.append(planes[j])
                    g.actions.append(int(a))
                    g.movers.append(g.pos.to_play)
                g.pos.play(int(a))
        if on_step is not None:
            on_step(games)
                                                                                      # +-- WHOSE WIN IS IT ------------------------------------------
    for g in games:                                                                   # | Outcomes are stored once, from black's point of view, and
        g.z_black = 1 if g.pos.winner() == BLACK else -1                              # | flipped on reading for whichever colour is asking. One
        g.done = True                                                                 # | stored convention and one conversion point is the whole
    return games                                                                      # | defence against sign errors, which in self-play are
                                                                                      # | invisible: training on negated rewards produces a policy
                                                                                      # | that learns to lose, which looks exactly like a policy that
def outcomes_for(game, colour):                                                       # | fails to learn.
    """z from ``colour``'s perspective: +1 win, -1 loss."""
    return float(game.z_black) if colour == BLACK else -float(game.z_black)
