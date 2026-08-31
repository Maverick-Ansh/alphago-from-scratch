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

import math                                                                         # +-- THE PAPER'S THREE SEARCH CONSTANTS -------------------------
import numpy as np                                                                  # | Straight from the paper's parameter table. c_puct scales how
                                                                                    # | hard a move's prior pulls the search toward it before any
from . import go                                                                    # | games have run through it: at 5 the prior decides early, the
from .go import N_ACTIONS, PASS, BLACK, WHITE                                       # | win rate decides later. lambda weights rollout outcomes
                                                                                    # | against value network readings. n_thr is how many times an
# Extended Data Table 5, "Parameters used by AlphaGo"                               # | edge must be tried before the position behind it gets a
C_PUCT = 5.0           # exploration constant                                       # | permanent node. The paper sets 40 because it waits on GPU
LAMBDA = 0.5           # mixing parameter between value net and rollouts            # | batches and wants the tree growing no faster than evaluations
N_THR = 1              # expansion threshold (paper: 40; see note below)            # | arrive; this search waits for nothing, so it keeps every
                                                                                    # | position it reaches.

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

    __slots__ = ("pos", "legal", "legal_idx", "P", "Nvis", "Nr", "Wr",              # +-- WHAT ONE NODE REMEMBERS ------------------------------------
                 "Nv", "Wv", "children", "terminal_z")                              # | A node is one board position plus five arrays indexed by move
                                                                                    # | number. P holds the prior probability the policy network gave
    def __init__(self, pos, prior, legal):                                          # | each move. Nr and Wr count rollouts through each edge and sum
        self.pos = pos                                                              # | their results; Nv and Wv do the same for value network
        self.legal = legal                                                          # | readings. Keeping the two apart is what makes lambda
        self.legal_idx = np.flatnonzero(legal)   # hoisted: _select is hot          # | adjustable at all, because a single running average could not
        self.P = prior                                                              # | be re-weighted after the fact. Nvis counts simulations and
        self.Nvis = np.zeros(N_ACTIONS, dtype=np.float64)                           # | nothing else. legal_idx is the list of playable move numbers,
        self.Nr = np.zeros(N_ACTIONS, dtype=np.float64)                             # | worked out once here because move selection reads it on every
        self.Wr = np.zeros(N_ACTIONS, dtype=np.float64)                             # | simulation and recomputing it would cost more than the search
        self.Nv = np.zeros(N_ACTIONS, dtype=np.float64)                             # | does. terminal_z is filled only when the game at this node is
        self.Wv = np.zeros(N_ACTIONS, dtype=np.float64)                             # | already decided, so the search stops there and reports the
        self.children = {}                                                          # | real result rather than estimating a position whose answer is
        self.terminal_z = None      # set once the game here is over                # | known.


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

    def __init__(self, n_sims=200, c_puct=C_PUCT, lmbda=LAMBDA, n_thr=N_THR,        # +-- EVERY ABLATION IS AN ARGUMENT HERE -------------------------
                 prior_fn=None, value_fn=None, rollout=None, rng=None,              # | Each row of the paper's ablation table is this same class with
                 resign_threshold=None):                                            # | different arguments. prior_fn supplies move probabilities;
        self.n_sims = n_sims                                                        # | passing None means there is no policy network and every legal
        self.c_puct = c_puct                                                        # | move starts equally likely. value_fn scores a position in one
        self.lmbda = lmbda                                                          # | step. rollout plays it out to the end instead. lambda decides
        self.n_thr = n_thr                                                          # | how much each of those two is believed. The two checks at the
        self.prior_fn = prior_fn                                                    # | bottom refuse a setting that asks for an estimator nobody
        self.value_fn = value_fn                                                    # | supplied: a missing value function at lambda below one would
        self.rollout = rollout                                                      # | otherwise be read as a confident zero on every position, which
        self.rng = rng if rng is not None else np.random.default_rng()              # | is not an error anywhere, just a search that quietly stops
        self.resign_threshold = resign_threshold                                    # | working.
        if lmbda > 0 and rollout is None:
            raise ValueError("lmbda > 0 needs a rollout policy")
        if lmbda < 1 and value_fn is None:
            raise ValueError("lmbda < 1 needs a value function")

    # -- tree plumbing ----------------------------------------------------
    def _make_node(self, pos):                                                      # +-- WHAT THE SEARCH TRIES FIRST --------------------------------
        legal = pos.legal_actions(exclude_eyes=True).copy()                         # | Building a node means deciding the order in which moves get
        if self.prior_fn is None:                                                   # | attention. With no policy network, every legal move gets an
            # No policy network: a uniform prior over legal moves.  This is the     # | equal share. With one, the network's probabilities are read,
            # paper's "no policy network" ablation, and what the expert teacher     # | then zeroed on moves that are illegal or that fill our own
            # uses.                                                                 # | eyes, then renormalised so they still sum to one. That masking
            prior = legal.astype(np.float64)                                        # | is needed because the network was never told which moves are
            s = prior.sum()                                                         # | legal and will put probability on all of them. If masking
            prior /= s if s > 0 else 1.0                                            # | leaves nothing at all, the uniform prior is used rather than
        else:                                                                       # | dividing by zero. A position that is already finished records
            prior = np.asarray(self.prior_fn(pos), dtype=np.float64).copy()         # | its true result immediately, so no rollout is ever spent on a
            prior[~legal] = 0.0                                                     # | decided game.
            s = prior.sum()
            if s <= 0:
                prior = legal.astype(np.float64)
                s = prior.sum()
            prior /= s if s > 0 else 1.0
        node = Node(pos, prior, legal)
        if pos.is_over():
            node.terminal_z = 1.0 if pos.winner() == BLACK else -1.0
        return node

    def _q_black(self, node, a):                                                    # +-- ONE MOVE, TWO ESTIMATES OF ITS VALUE -----------------------
        """Combined action value of edge (s,a), in **Black's** frame."""            # | The value of an edge is a weighted average of two independent
        q = 0.0                                                                     # | estimates: the mean value network reading over the times this
        if self.lmbda < 1.0:                                                        # | edge was evaluated, and the mean rollout result over the times
            q += (1.0 - self.lmbda) * (node.Wv[a] / node.Nv[a]                      # | it was played out. An edge that one estimator has never
                                       if node.Nv[a] > 0 else 0.0)                  # | touched contributes zero from it instead of dividing by zero,
        if self.lmbda > 0.0:                                                        # | which makes an untried move read as exactly even, neither
            q += self.lmbda * (node.Wr[a] / node.Nr[a]                              # | winning nor losing, until evidence arrives.
                               if node.Nr[a] > 0 else 0.0)
        return q

    def _select(self, node):
        """argmax_a [ Q(s,a) + u(s,a) ] over legal actions.

        Vectorised over the legal actions only.  ``Q`` is stored in Black's
        frame, so it is flipped once here for the player to move; ``u`` is
        sign-free because exploration is not a preference about who is winning.
        """
        idx = node.legal_idx                                                        # +-- CHOOSING WHICH MOVE TO TRY NEXT ----------------------------
        if idx.size == 0:                                                           # | Two quantities compete. u is the exploration term: big for
            return PASS                                                             # | moves the policy liked, shrinking as a move gets tried, and
        sign = 1.0 if node.pos.to_play == BLACK else -1.0                           # | growing with the square root of the parent's total simulations
        nvis = node.Nvis[idx]                                                       # | so that every child's appetite for being tried rises together
        sqrt_total = math.sqrt(max(node.Nvis.sum(), 1e-8))                          # | as the node is visited more. q is what the simulations have
        u = self.c_puct * node.P[idx] * sqrt_total / (1.0 + nvis)                   # | actually measured so far. Values are stored from Black's point
                                                                                    # | of view everywhere in this file, so sign flips them once,
        q = np.zeros(idx.size)                                                      # | here, into the frame of whoever is to move; without it White
        if self.lmbda < 1.0:                                                        # | would steer toward positions Black wins. u gets no sign,
            nv = node.Nv[idx]                                                       # | because wanting to try an untested move does not depend on who
            q += (1.0 - self.lmbda) * np.divide(                                    # | is ahead. Everything runs over the legal moves as whole arrays
                node.Wv[idx], nv, out=np.zeros(idx.size), where=nv > 0)             # | rather than a Python loop, since selection happens several
        if self.lmbda > 0.0:                                                        # | times per simulation and there are hundreds of simulations per
            nr = node.Nr[idx]                                                       # | move.
            q += self.lmbda * np.divide(
                node.Wr[idx], nr, out=np.zeros(idx.size), where=nr > 0)
        return int(idx[np.argmax(sign * q + u)])

    # -- one simulation ---------------------------------------------------
    def _simulate(self, root):                                                      # +-- WALKING DOWN, AND WHEN TO KEEP A NODE ----------------------
        path = []                                                                   # | One simulation walks from the root down to a leaf, writing
        node = root                                                                 # | down which edge it took at each step so the result can be
        while True:                                                                 # | pushed back up the same path afterwards. It stops one of two
            if node.terminal_z is not None:                                         # | ways. If it reaches a finished game it takes the real outcome.
                z_black = v_black = node.terminal_z                                 # | Otherwise it reaches an edge with no node behind it, and that
                break                                                               # | position is the leaf to evaluate. Whether a node is actually
            a = self._select(node)                                                  # | allocated there is the expansion rule: an edge tried fewer
            path.append((node, a))                                                  # | than n_thr times is evaluated but not stored, so memory and
            child = node.children.get(a)                                            # | node-building cost go only to lines the search keeps returning
            if child is not None:                                                   # | to. The successor position is built by copying the parent's
                node = child                                                        # | board and playing the move on the copy, never by changing the
                continue                                                            # | parent, because other simulations will descend through that
            # No child yet: this simulation ends here.  Whether we *keep* the       # | same parent again and would find a board with an extra stone
            # successor is the paper's expansion rule -- "when the visit count      # | on it.
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

        for parent, a in path:                                                      # +-- ONE RESULT UPDATES THE WHOLE PATH --------------------------
            parent.Nvis[a] += 1.0                                                   # | The one outcome updates every edge the simulation passed
            if self.lmbda > 0.0:                                                    # | through, not just the last, because each of those moves led to
                parent.Nr[a] += 1.0                                                 # | this result. Nvis advances every time and is the only counter
                parent.Wr[a] += z_black                                             # | selection reads, so exploration behaves the same whether one
            if self.lmbda < 1.0:                                                    # | estimator is running or both. The rollout and value totals
                parent.Nv[a] += 1.0                                                 # | advance only when their own estimator actually produced a
                parent.Wv[a] += v_black                                             # | number this simulation.

    def _evaluate(self, pos):
        """Leaf evaluation (Fig. 3c): a rollout outcome and/or a value net read.

        Both are returned in Black's frame.  The rollout is the full
        "play to the end and score it" of the paper; the value network is a
        single forward pass.  Which of them actually reaches the tree is
        decided by lambda in the backup above.
        """
        z_black = 0.0                                                               # +-- THE TWO WAYS TO JUDGE A LEAF -------------------------------
        v_black = 0.0                                                               # | Leaf evaluation produces up to two numbers about the same
        if pos.is_over():                                                           # | position. The rollout plays the game to the end with the fast
            z = 1.0 if pos.winner() == BLACK else -1.0                              # | policy and reports only who won, throwing away the margin,
            return z, z                                                             # | because the search is picking moves to win games rather than
        if self.lmbda > 0.0:                                                        # | to win by more points. The value network answers in one step,
            score = self.rollout.rollout(pos)                                       # | in the frame of whoever is to move, so it is converted into
            z_black = 1.0 if score > 0 else -1.0                                    # | Black's frame here: this is the only place in the file where
        if self.lmbda < 1.0:                                                        # | that conversion happens, which is why the rest of the file
            v = float(self.value_fn(pos))          # player-to-move frame           # | never has to think about signs. A position that is already
            v_black = v if pos.to_play == BLACK else -v                             # | over returns its true result for both, which costs nothing and
        return z_black, v_black                                                     # | is exact.

    # -- public API -------------------------------------------------------
    def search(self, pos):                                                          # +-- A SEARCH IS JUST REPEATED DESCENTS -------------------------
        """Run n_sims simulations from ``pos``; return the root node."""            # | n_sims independent walks into a tree that grows as they go.
        root = self._make_node(pos.copy())                                          # | The root is built from a copy so the caller's position is
        for _ in range(self.n_sims):                                                # | never disturbed. The paper keeps the subtree under the move it
            self._simulate(root)                                                    # | played and carries its statistics into the next move; this
        return root                                                                 # | does not, so every move starts from an empty tree and pays for
                                                                                    # | the work again.
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
        root = self.search(pos)                                                     # +-- PLAY THE MOST VISITED MOVE, NOT THE BEST-SCORING ONE -------
        counts = self.visit_counts(root)                                            # | These usually agree, and when they disagree the visit count is
        counts = np.where(root.legal, counts, 0.0)                                  # | the safer answer: a high average can come from a handful of
        if counts.sum() <= 0:                                                       # | lucky playouts, while a high visit count can only come from
            return PASS, root                                                       # | the search choosing to return to that move again and again
        if temperature <= 0:                                                        # | after seeing what happened. Counts are masked to legal moves
            return int(np.argmax(counts)), root                                     # | before the maximum is taken. Temperature exists only to make
        p = counts ** (1.0 / temperature)                                           # | self-play games differ from one another so the training data
        p /= p.sum()                                                                # | is not all the same game; evaluation games always take the
        return int(self.rng.choice(len(p), p=p)), root                              # | maximum.

    def root_value(self, root):
        """Root's own estimate of the winning value for the player to move."""
        counts = self.visit_counts(root)                                            # +-- HOW GOOD DOES THE ROOT THINK IT IS DOING -------------------
        tot = counts.sum()                                                          # | An estimate of the current position for the player to move,
        if tot <= 0:                                                                # | formed by averaging the edge values weighted by how much
            return 0.0                                                              # | attention each edge received. Used for logging and for judging
        sign = 1.0 if root.pos.to_play == BLACK else -1.0                           # | a game hopeless, never inside the search itself.
        q = sum(counts[a] * self._q_black(root, a)
                for a in np.flatnonzero(counts > 0))
        return sign * q / tot
