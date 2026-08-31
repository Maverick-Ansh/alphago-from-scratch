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


class BatchPolicy:
    """A torch policy net, evaluated over a list of positions at once."""

    needs_planes = True

    def __init__(self, net, device="cpu", temperature=1.0, rng=None,
                 with_colour=False):
        self.net = net.to(device).eval()
        self.device = device
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

    def choose(self, positions, planes=None):
        """Sample (or argmax) a legal move for each position."""
        if planes is None:
            planes = self.planes(positions)
        logits = self.logits(planes)
        actions = np.empty(len(positions), dtype=np.int64)
        for i, pos in enumerate(positions):
            legal = pos.legal_actions(exclude_eyes=True)[:NN]
            z = np.where(legal, logits[i].astype(np.float64), -np.inf)
            if not np.isfinite(z).any():
                actions[i] = PASS          # only our own eyes remain: pass
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


class RandomBatchPolicy:
    """Uniform over sensible moves, with the same interface."""

    needs_planes = False

    def __init__(self, rng=None, with_colour=False):
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


class Game:
    """One game in flight, plus whatever the caller asked to record."""

    __slots__ = ("pos", "planes", "actions", "movers", "done", "z_black",
                 "meta")

    def __init__(self):
        self.pos = go.Position()
        self.planes = []
        self.actions = []
        self.movers = []
        self.done = False
        self.z_black = 0
        self.meta = {}


def run_games(n_games, policy_for, rng=None, max_moves=None, record=False,
              on_step=None):
    """Play ``n_games`` in lockstep.

    ``policy_for(colour, game_index, game)`` returns the policy object that
    should move.  Games are grouped by the *identity* of the returned policy so
    that each distinct network sees one batched forward pass per step, however
    the games are split between colours or phases.

    Returns the finished ``Game`` objects.
    """
    rng = rng or np.random.default_rng()
    max_moves = max_moves or go.MAX_MOVES
    games = [Game() for _ in range(n_games)]

    while True:
        active = [(i, g) for i, g in enumerate(games)
                  if not g.pos.is_over() and g.pos.move_no < max_moves]
        if not active:
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

    for g in games:
        g.z_black = 1 if g.pos.winner() == BLACK else -1
        g.done = True
    return games


def outcomes_for(game, colour):
    """z from ``colour``'s perspective: +1 win, -1 loss."""
    return float(game.z_black) if colour == BLACK else -float(game.z_black)
