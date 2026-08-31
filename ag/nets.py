"""Policy and value networks -- Methods, "Neural network architecture".

    "The input to the policy network is a 19 x 19 x 48 image stack consisting of
     48 feature planes.  The first hidden layer zero pads the input into a
     23 x 23 image, then convolves k filters of kernel size 5 x 5 with stride 1
     with the input image and applies a rectifier nonlinearity.  Each of the
     subsequent hidden layers 2 to 12 zero pads the respective previous hidden
     layer into a 21 x 21 image, then convolves k filters of kernel size 3 x 3
     with stride 1, again followed by a rectifier nonlinearity.  The final layer
     convolves 1 filter of kernel size 1 x 1 with stride 1, **with a different
     bias for each position**, and applies a softmax function.  The match
     version of AlphaGo used k = 192 filters."

    "The input to the value network is also a 19 x 19 x 48 image stack, with an
     additional binary feature plane describing the current colour to play.
     Hidden layers 2 to 11 are identical to the policy network, hidden layer 12
     is an additional convolution layer, hidden layer 13 convolves 1 filter of
     kernel size 1 x 1 with stride 1, and hidden layer 14 is a fully connected
     linear layer with 256 rectifier units.  The output layer is a fully
     connected linear layer with a single tanh unit."

Note the padding arithmetic: 23 - 5 + 1 = 19 and 21 - 3 + 1 = 19, so "zero pads
to 23 x 23 then convolves 5 x 5" is exactly `padding='same'`.  The paper spells
it out as an explicit pad because that is how the layer was built, but there is
no cropping or shrinking anywhere in the trunk -- every hidden layer is the same
19 x 19 (here 9 x 9) as the board.

The per-position bias in the final layer is a real architectural choice and is
implemented faithfully.  It is a free parameter per intersection added after a
1 x 1 convolution collapses the trunk to one channel, which lets the network
learn a board-wide positional prior (corners and edges behave differently in
Go) without spending trunk capacity on it.

Resizing
--------
On 19 x 19 the paper's 12 convolutional layers give a receptive field of
5 + 11*2 = 27, comfortably larger than the board.  On 9 x 9 a receptive field of
9 already spans everything, so the trunk is shortened to 1 x (5x5) + 5 x (3x3),
for a receptive field of 15.  Width is the axis claim C1 varies (the paper
sweeps k = 128/192/256/384; we sweep k = 32/64/128), because C1 is about
accuracy predicting strength, and width is what moved accuracy in Extended Data
Table 3.

No batch norm, no residual connections: both post-date this paper (AlphaGo Zero
introduced the residual trunk), and adding them would make the result a
reproduction of something else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import features as feat
from .go import N, NN


class Trunk(nn.Module):                                                               # +-- ONE WIDE LAYER, THEN NARROW ONES -------------------------
    """The shared convolutional body: one 5x5 layer then n_layers 3x3 layers."""      # | A single five-by-five layer followed by three-by-three
                                                                                      # | layers. The first is wider because it is the only place raw
    def __init__(self, in_planes, n_filters=64, n_layers=5):                          # | stone positions get combined at all, and a Go shape worth
        super().__init__()                                                            # | naming is about that size; after that, depth is a cheaper
        layers = [nn.Conv2d(in_planes, n_filters, 5, padding=2), nn.ReLU()]           # | way to grow the region a unit can see than width is. Every
        for _ in range(n_layers):                                                     # | layer keeps the board's exact size, so a unit in the last
            layers += [nn.Conv2d(n_filters, n_filters, 3, padding=1), nn.ReLU()]      # | layer still corresponds to one intersection. The paper
        self.body = nn.Sequential(*layers)                                            # | writes this as an explicit zero-pad to twenty-three then a
                                                                                      # | five-by-five, which is the same thing: nothing is ever
    def forward(self, x):                                                             # | cropped.
        return self.body(x)


class PolicyNet(nn.Module):                                                           # +-- A DISTRIBUTION OVER POINTS, WITH A BIAS PER POINT --------
    """p_sigma / p_rho: a distribution over the N*N points.

    There is no "pass" output.  The paper excludes passes from the training set
    ("Pass moves were excluded from the data set"), and under area scoring with
    eye-avoidance the right time to pass is exactly when no sensible move
    remains -- a rule, not a prediction.  The player wrapper applies it.
    """
                                                                                      # | The trunk is collapsed to one number per intersection by a
    def __init__(self, in_planes=feat.N_PLANES_POLICY, n_filters=64,                  # | one-by-one convolution, and then a separate learned bias is
                 n_layers=5):                                                         # | added for every point on the board. That bias is a real
        super().__init__()                                                            # | architectural choice, not a detail: corners, edges and the
        self.in_planes = in_planes                                                    # | centre behave differently in Go for reasons that have
        self.n_filters = n_filters                                                    # | nothing to do with the stones present, and giving the
        self.n_layers = n_layers                                                      # | network a free parameter per intersection lets it learn that
        self.trunk = Trunk(in_planes, n_filters, n_layers)                            # | without spending trunk capacity on it. There is no output
        self.head = nn.Conv2d(n_filters, 1, 1, bias=False)                            # | for passing. The paper drops pass moves from its training
        # "with a different bias for each position"                                   # | set, and under area scoring the moment to pass is when no
        self.pos_bias = nn.Parameter(torch.zeros(N * N))                              # | sensible move is left, which is a rule and not something
                                                                                      # | worth predicting.
    def forward(self, x):
        h = self.trunk(x)
        h = self.head(h).flatten(1)
        return h + self.pos_bias          # logits; softmax lives in the loss

    def config(self):
        return dict(in_planes=self.in_planes, n_filters=self.n_filters,
                    n_layers=self.n_layers, kind="policy")


class ValueNet(nn.Module):                                                            # +-- ONE NUMBER FOR THE WHOLE BOARD ---------------------------
    """v_theta: a scalar in [-1, 1] for the player to move."""                        # | Same trunk, one layer deeper, then the board is collapsed to
                                                                                      # | a single channel and fed through a fully connected layer.
    def __init__(self, in_planes=feat.N_PLANES_VALUE, n_filters=64,                   # | The fully connected layer is what makes this different in
                 n_layers=5, n_hidden=256):                                           # | kind from the policy network: a convolution can only ever
        super().__init__()                                                            # | report about a neighbourhood, and who is winning is a
        self.in_planes = in_planes                                                    # | property of the whole board at once. The final tanh bounds
        self.n_filters = n_filters                                                    # | the output to the range outcomes actually take, so the
        self.n_layers = n_layers                                                      # | network can never predict a win worth more than a win.
        self.n_hidden = n_hidden
        # "hidden layer 12 is an additional convolution layer" -> one deeper
        # than the policy trunk before the 1x1 collapse.
        self.trunk = Trunk(in_planes, n_filters, n_layers + 1)
        self.head = nn.Conv2d(n_filters, 1, 1)
        self.fc1 = nn.Linear(N * N, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        h = self.head(h).flatten(1)
        h = Fn.relu(self.fc1(h))
        return torch.tanh(self.fc2(h)).squeeze(-1)

    def config(self):
        return dict(in_planes=self.in_planes, n_filters=self.n_filters,
                    n_layers=self.n_layers, n_hidden=self.n_hidden,
                    kind="value")


def save(net, path, **extra):                                                         # +-- SAVING THE SHAPE ALONGSIDE THE WEIGHTS -------------------
    torch.save({"config": net.config(), "state": net.state_dict(), **extra},          # | The architecture is written into the checkpoint, so loading
               path)                                                                  # | does not depend on the caller remembering how wide the
                                                                                      # | network was. Anything extra passed in is stored too, which
                                                                                      # | is how training history and accuracy travel with the weights
def load(path, map_location="cpu"):                                                   # | and are still there weeks later when the tournament needs to
    ck = torch.load(path, map_location=map_location, weights_only=False)              # | label a point on a graph.
    cfg = dict(ck["config"])
    kind = cfg.pop("kind")
    net = (PolicyNet if kind == "policy" else ValueNet)(**cfg)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


# --------------------------------------------------------------------------
# inference wrappers used by the players and by MCTS
# --------------------------------------------------------------------------
class NetEvaluator:
    """Turns a torch net into the plain ``pos -> array`` callback MCTS wants.

    Two things happen here that matter for fidelity:

    * **Implicit symmetry ensemble.**  "APV-MCTS makes use of an implicit
      symmetry ensemble that randomly selects a single rotation/reflection
      j in [1,8] for each evaluation.  We compute exactly one evaluation for
      that orientation only; in each simulation we compute the value of leaf
      node s_L by v_theta(d_j(s_L)), and allow the search procedure to average
      over these evaluations."  Set ``symmetry='random'`` for that.  With
      ``symmetry='all'`` it becomes the *explicit* ensemble the paper uses for
      raw network evaluation, averaging all 8.
    * **Caching.**  Positions repeat constantly inside a search tree, and a
      forward pass is three orders of magnitude more expensive than a rollout
      step, so evaluations are memoised on the board bytes.
    """

    def __init__(self, net, device="cpu", with_colour=False,                          # +-- MAKING A NETWORK LOOK LIKE A PLAIN FUNCTION --------------
                 symmetry="none", cache=True, batch_size=64):                         # | The search wants to call something that takes a position and
        self.net = net.to(device).eval()                                              # | returns numbers; this supplies that. The cache is not an
        self.device = device                                                          # | optimisation detail: the same position is reached over and
        self.fx = feat.FeatureExtractor(with_colour=with_colour)                      # | over inside one search tree, and a forward pass costs about
        self.symmetry = symmetry                                                      # | a thousand times what a rollout step does, so answering a
        self.cache = {} if cache else None                                            # | repeat from memory is the difference between a search that
        self.batch_size = batch_size                                                  # | finishes and one that does not. Positions are keyed on the
        self.n_calls = 0                                                              # | stones, the side to move, the ko ban and the last move,
        self.n_forward = 0                                                            # | which is everything the features are built from.

    def _key(self, pos):
        return (pos.board.tobytes(), pos.to_play, pos.ko, pos.last_move)

    def _planes(self, pos):
        return self.fx(pos)

    @torch.no_grad()
    def _run(self, batch):
        x = torch.from_numpy(np.stack(batch)).to(self.device)
        self.n_forward += len(batch)
        return self.net(x).float().cpu().numpy()

    @torch.no_grad()
    def __call__(self, pos):                                                          # +-- EVALUATING UNDER A SYMMETRY ------------------------------
        self.n_calls += 1                                                             # | Three modes. Averaging all eight rotations gives the
        if self.cache is not None:                                                    # | steadiest answer and costs eight forward passes. Picking one
            k = self._key(pos)                                                        # | at random costs one, and is what the paper uses inside the
            hit = self.cache.get(k)                                                   # | search, on the reasoning that the search visits the same
            if hit is not None:                                                       # | position many times and will average over the random choices
                return hit                                                            # | by itself. Doing nothing is the cheapest and is used where a
        x = self._planes(pos)                                                         # | fixed answer is wanted. Whatever is chosen, the network sees
        if self.symmetry == "all":                                                    # | a rotated board and its answer has to be rotated back, which
            batch = [feat.transform_planes(x, j) for j in range(8)]                   # | is what the subclasses below do.
            out = self._run(batch)
            out = self._unify(out, list(range(8)))
        elif self.symmetry == "random":
            j = int(np.random.randint(8))
            out = self._unify(self._run([feat.transform_planes(x, j)]), [j])
        else:
            out = self._unify(self._run([x]), [0])
        if self.cache is not None:
            self.cache[k] = out
        return out

    def _unify(self, raw, syms):
        raise NotImplementedError


class PolicyEvaluator(NetEvaluator):                                                  # +-- ROTATING THE ANSWER BACK ---------------------------------
    """pos -> probability vector of length NN+1 (the pass slot stays at 0)."""        # | Probabilities come out indexed by the rotated board and must
                                                                                      # | be reindexed to the original before they mean anything. The
    def _unify(self, raw, syms):                                                      # | mapping says where each original point ended up, so the
        # Undo each symmetry on the *output plane* before averaging:                  # | network's answer for an original point is read from that
        # "the planes of output probabilities are rotated/reflected back into         # | slot. Reading through the mapping undoes the rotation;
        #  the original orientation, and averaged together".                          # | assigning into it would apply the rotation a second time.
        acc = np.zeros(NN, dtype=np.float64)                                          # | Both produce a valid distribution and only one is right,
        for row, j in zip(raw, syms):                                                 # | which is why this has a test that requires the ensemble to
            p = _softmax(row.astype(np.float64))                                      # | commute with the symmetry rather than merely to run.
            # fwd[a] = the index that original point a occupies after symmetry
            # j, so the network's output for original point a is p[fwd[a]].
            # Reading p through fwd is the *inverse* map; assigning into it
            # would apply the symmetry a second time instead of undoing it.
            fwd = feat.transform_actions(np.arange(NN), j)
            acc += p[fwd]
        acc /= len(syms)
        out = np.zeros(NN + 1)
        out[:NN] = acc
        return out


class ValueEvaluator(NetEvaluator):                                                   # +-- A VALUE DOES NOT CARE WHICH WAY UP THE BOARD IS ----------
    """pos -> scalar value for the player to move.

    The value is invariant under board symmetry, so the ensemble is a plain
    mean: "For the value network, the output values are simply averaged."
    """
                                                                                      # | Rotating a board does not change who is winning, so the
    def _unify(self, raw, syms):                                                      # | ensemble is a plain average with no reindexing at all. The
        return float(np.mean(raw))                                                    # | paper says exactly this. The softmax below subtracts the
                                                                                      # | maximum first, which changes nothing about the result and
                                                                                      # | keeps the exponential from overflowing on a confident
def _softmax(z):                                                                      # | network.
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
