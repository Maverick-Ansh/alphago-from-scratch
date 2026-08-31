"""APV-MCTS -- the search of Fig. 3 and the "Search algorithm" Methods section.

The paper stores, on every edge (s,a):

    "{P(s,a), N_v(s,a), N_r(s,a), W_v(s,a), W_r(s,a), Q(s,a)}
     where P(s,a) is the prior probability, W_v(s,a) and W_r(s,a) are Monte
     Carlo estimates of total action value, accumulated over N_v(s,a) and
     N_r(s,a) leaf evaluations and rollout rewards, respectively, and Q(s,a) is
     the combined mean action value for that edge."

Two independent statistics per edge, not one.  That separation is the whole
point: the value network and the rollouts are different estimators of the same
quantity, they arrive at different times and at different rates, and mixing
them at *read* time with

    Q(s,a) = (1-lambda) W_v(s,a)/N_v(s,a) + lambda W_r(s,a)/N_r(s,a)

is what claim C5 is about.  Collapsing them into one running mean would make
lambda unimplementable.

Selection uses the paper's PUCT variant:

    u(s,a) = c_puct * P(s,a) * sqrt(sum_b N_r(s,b)) / (1 + N_r(s,a))

    "this search control strategy initially prefers actions with high prior
     probability and low visit count, but asymptotically prefers actions with
     high action value."

Note the numerator is the square root of the *parent's* total rollout count, so
the exploration term for every child grows together as the parent is visited,
while the denominator damps the child that is actually being exploited.

Sign convention
---------------
Every value in this module is stored **from Black's point of view** and flipped
on read for the player to move.  The alternative -- storing values relative to
each node's own player -- needs a sign flip at every backup step and is where
implementations of this algorithm usually go wrong.  Here there is exactly one
flip, at ``_q``, and one at the leaf.

What is not implemented
-----------------------
Virtual loss (n_vl), the asynchronous/distributed machinery, the tree policy
p_tau placeholder priors, and the last-good-reply rollout cache.  All of these
exist to hide GPU latency behind CPU work across 40 threads; this search is
synchronous and single-threaded, so they would be pure overhead.  Recorded in
REPORT.md.
"""

import math
import numpy as np

from . import go
from .go import N_ACTIONS, PASS, BLACK, WHITE

# Extended Data Table 5, "Parameters used by AlphaGo"
C_PUCT = 5.0           # exploration constant
LAMBDA = 0.5           # mixing parameter between value net and rollouts
N_THR = 1              # expansion threshold (paper: 40; see note below)


class Node:
    """One position in the search tree, with per-edge statistics.

    ``Nvis`` deserves a note.  The paper writes the exploration bonus over
    ``N_r``, its rollout count, because in AlphaGo every simulation runs a
    rollout and ``N_r`` *is* the simulation count.  Here lambda decides which
    estimators run at all: at lambda=1 only ``N_r`` advances, at lambda=0 only
    ``N_v``, and at lambda=0.5 both do -- so reusing either one (or their sum)
    inside PUCT would silently give the three arms of claim C5 three different
    exploration schedules, and the comparison would measure that instead of the
    thing it is supposed to measure.  ``Nvis`` counts simulations through the
    edge and nothing else, so selection behaves identically at every lambda.
    """

    __slots__ = ("pos", "legal", "legal_idx", "P", "Nvis", "Nr", "Wr",
                 "Nv", "Wv", "children", "terminal_z")

    def __init__(self, pos, prior, legal):
        self.pos = pos
        self.legal = legal
        self.legal_idx = np.flatnonzero(legal)   # hoisted: _select is hot
        self.P = prior
        self.Nvis = np.zeros(N_ACTIONS, dtype=np.float64)
        self.Nr = np.zeros(N_ACTIONS, dtype=np.float64)
        self.Wr = np.zeros(N_ACTIONS, dtype=np.float64)
        self.Nv = np.zeros(N_ACTIONS, dtype=np.float64)
        self.Wv = np.zeros(N_ACTIONS, dtype=np.float64)
        self.children = {}
        self.terminal_z = None      # +1/-1 (Black's view) once the game is over


class MCTS:
    """Configurable APV-MCTS.

    The knobs are exactly the axes the paper ablates in Extended Data Table 7:

    ``prior_fn``   s -> array over N_ACTIONS.  ``None`` = uniform over legal
                   moves, i.e. no policy network (the paper's ``[p_tau]`` rows).
    ``value_fn``   s -> value in [-1, 1] for the player to move at s.
                   ``None`` = no value network.
    ``lmbda``      0 = value network only (alpha_vp), 1 = rollouts only
                   (alpha_rp), 0.5 = both (alpha_rvp).
    ``rollout``    a RolloutPolicy, or None when lmbda == 0.

    Setting ``lmbda=1, prior_fn=None, value_fn=None`` gives a plain
    rollout-based MCTS -- the pre-AlphaGo state of the art, and the player we
    use as the expert teacher for the supervised stage.
    """

    def __init__(self, n_sims=200, c_puct=C_PUCT, lmbda=LAMBDA, n_thr=N_THR,
                 prior_fn=None, value_fn=None, rollout=None, rng=None,
                 resign_threshold=None):
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.lmbda = lmbda
        self.n_thr = n_thr
        self.prior_fn = prior_fn
        self.value_fn = value_fn
        self.rollout = rollout
        self.rng = rng if rng is not None else np.random.default_rng()
        self.resign_threshold = resign_threshold
        if lmbda > 0 and rollout is None:
            raise ValueError("lmbda > 0 needs a rollout policy")
        if lmbda < 1 and value_fn is None:
            raise ValueError("lmbda < 1 needs a value function")

    # -- tree plumbing ----------------------------------------------------
    def _make_node(self, pos):
        legal = pos.legal_actions(exclude_eyes=True).copy()
        if self.prior_fn is None:
            # No policy network: a uniform prior over legal moves.  This is the
            # paper's "no policy network" ablation, and what the expert teacher
            # uses.
            prior = legal.astype(np.float64)
            s = prior.sum()
            prior /= s if s > 0 else 1.0
        else:
            prior = np.asarray(self.prior_fn(pos), dtype=np.float64).copy()
            prior[~legal] = 0.0
            s = prior.sum()
            if s <= 0:
                prior = legal.astype(np.float64)
                s = prior.sum()
            prior /= s if s > 0 else 1.0
        node = Node(pos, prior, legal)
        if pos.is_over():
            node.terminal_z = 1.0 if pos.winner() == BLACK else -1.0
        return node

    def _q_black(self, node, a):
        """Combined action value of edge (s,a), in **Black's** frame."""
        q = 0.0
        if self.lmbda < 1.0:
            q += (1.0 - self.lmbda) * (node.Wv[a] / node.Nv[a]
                                       if node.Nv[a] > 0 else 0.0)
        if self.lmbda > 0.0:
            q += self.lmbda * (node.Wr[a] / node.Nr[a]
                               if node.Nr[a] > 0 else 0.0)
        return q

    def _select(self, node):
        """argmax_a [ Q(s,a) + u(s,a) ] over legal actions.

        Vectorised over the legal actions only.  ``Q`` is stored in Black's
        frame, so it is flipped once here for the player to move; ``u`` is
        sign-free because exploration is not a preference about who is winning.
        """
        idx = node.legal_idx
        if idx.size == 0:
            return PASS
        sign = 1.0 if node.pos.to_play == BLACK else -1.0
        nvis = node.Nvis[idx]
        sqrt_total = math.sqrt(max(node.Nvis.sum(), 1e-8))
        u = self.c_puct * node.P[idx] * sqrt_total / (1.0 + nvis)

        q = np.zeros(idx.size)
        if self.lmbda < 1.0:
            nv = node.Nv[idx]
            q += (1.0 - self.lmbda) * np.divide(
                node.Wv[idx], nv, out=np.zeros(idx.size), where=nv > 0)
        if self.lmbda > 0.0:
            nr = node.Nr[idx]
            q += self.lmbda * np.divide(
                node.Wr[idx], nr, out=np.zeros(idx.size), where=nr > 0)
        return int(idx[np.argmax(sign * q + u)])

    # -- one simulation ---------------------------------------------------
    def _simulate(self, root):
        path = []
        node = root
        while True:
            if node.terminal_z is not None:
                z_black = v_black = node.terminal_z
                break
            a = self._select(node)
            path.append((node, a))
            child = node.children.get(a)
            if child is not None:
                node = child
                continue
            # No child yet: this simulation ends here.  Whether we *keep* the
            # successor is the paper's expansion rule -- "when the visit count
            # exceeds a threshold, N_r(s,a) > n_thr, the successor state
            # s' = f(s,a) is added to the search tree".  Below the threshold we
            # still evaluate the position, we just do not allocate a node for
            # it, so the tree grows only where the search keeps returning.
            leaf_pos = node.pos.copy().play(a)
            if node.Nvis[a] >= self.n_thr:
                node.children[a] = self._make_node(leaf_pos)
                leaf_pos = node.children[a].pos
            z_black, v_black = self._evaluate(leaf_pos)
            break

        for parent, a in path:
            parent.Nvis[a] += 1.0
            if self.lmbda > 0.0:
                parent.Nr[a] += 1.0
                parent.Wr[a] += z_black
            if self.lmbda < 1.0:
                parent.Nv[a] += 1.0
                parent.Wv[a] += v_black

    def _evaluate(self, pos):
        """Leaf evaluation (Fig. 3c): a rollout outcome and/or a value net read.

        Both are returned in Black's frame.  The rollout is the full
        "play to the end and score it" of the paper; the value network is a
        single forward pass.  Which of them actually reaches the tree is
        decided by lambda in the backup above.
        """
        z_black = 0.0
        v_black = 0.0
        if pos.is_over():
            z = 1.0 if pos.winner() == BLACK else -1.0
            return z, z
        if self.lmbda > 0.0:
            score = self.rollout.rollout(pos)
            z_black = 1.0 if score > 0 else -1.0
        if self.lmbda < 1.0:
            v = float(self.value_fn(pos))          # player-to-move frame
            v_black = v if pos.to_play == BLACK else -v
        return z_black, v_black

    # -- public API -------------------------------------------------------
    def search(self, pos):
        """Run n_sims simulations from ``pos``; return the root node."""
        root = self._make_node(pos.copy())
        for _ in range(self.n_sims):
            self._simulate(root)
        return root

    def visit_counts(self, root):
        return root.Nvis

    def choose(self, pos, temperature=0.0):
        """Pick a move.

        "Once the search is complete, the algorithm chooses the most visited
         move from the root position." -- and, on why visits rather than value:
        "this is less sensitive to outliers than maximizing action value".

        ``temperature > 0`` samples from N^(1/T) instead, which we use only to
        diversify self-play openings, never in evaluation games.
        """
        root = self.search(pos)
        counts = self.visit_counts(root)
        counts = np.where(root.legal, counts, 0.0)
        if counts.sum() <= 0:
            return PASS, root
        if temperature <= 0:
            return int(np.argmax(counts)), root
        p = counts ** (1.0 / temperature)
        p /= p.sum()
        return int(self.rng.choice(len(p), p=p)), root

    def root_value(self, root):
        """Root's own estimate of the winning value for the player to move."""
        counts = self.visit_counts(root)
        tot = counts.sum()
        if tot <= 0:
            return 0.0
        sign = 1.0 if root.pos.to_play == BLACK else -1.0
        q = sum(counts[a] * self._q_black(root, a)
                for a in np.flatnonzero(counts > 0))
        return sign * q / tot
