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


class Trunk(nn.Module):
    """The shared convolutional body: one 5x5 layer then n_layers 3x3 layers."""

    def __init__(self, in_planes, n_filters=64, n_layers=5):
        super().__init__()
        layers = [nn.Conv2d(in_planes, n_filters, 5, padding=2), nn.ReLU()]
        for _ in range(n_layers):
            layers += [nn.Conv2d(n_filters, n_filters, 3, padding=1), nn.ReLU()]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class PolicyNet(nn.Module):
    """p_sigma / p_rho: a distribution over the N*N points.

    There is no "pass" output.  The paper excludes passes from the training set
    ("Pass moves were excluded from the data set"), and under area scoring with
    eye-avoidance the right time to pass is exactly when no sensible move
    remains -- a rule, not a prediction.  The player wrapper applies it.
    """

    def __init__(self, in_planes=feat.N_PLANES_POLICY, n_filters=64,
                 n_layers=5):
        super().__init__()
        self.in_planes = in_planes
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.trunk = Trunk(in_planes, n_filters, n_layers)
        self.head = nn.Conv2d(n_filters, 1, 1, bias=False)
        # "with a different bias for each position"
        self.pos_bias = nn.Parameter(torch.zeros(N * N))

    def forward(self, x):
        h = self.trunk(x)
        h = self.head(h).flatten(1)
        return h + self.pos_bias          # logits; softmax lives in the loss

    def config(self):
        return dict(in_planes=self.in_planes, n_filters=self.n_filters,
                    n_layers=self.n_layers, kind="policy")


class ValueNet(nn.Module):
    """v_theta: a scalar in [-1, 1] for the player to move."""

    def __init__(self, in_planes=feat.N_PLANES_VALUE, n_filters=64,
                 n_layers=5, n_hidden=256):
        super().__init__()
        self.in_planes = in_planes
        self.n_filters = n_filters
        self.n_layers = n_layers
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


def save(net, path, **extra):
    torch.save({"config": net.config(), "state": net.state_dict(), **extra},
               path)


def load(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
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

    def __init__(self, net, device="cpu", with_colour=False,
                 symmetry="none", cache=True, batch_size=64):
        self.net = net.to(device).eval()
        self.device = device
        self.fx = feat.FeatureExtractor(with_colour=with_colour)
        self.symmetry = symmetry
        self.cache = {} if cache else None
        self.batch_size = batch_size
        self.n_calls = 0
        self.n_forward = 0

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
    def __call__(self, pos):
        self.n_calls += 1
        if self.cache is not None:
            k = self._key(pos)
            hit = self.cache.get(k)
            if hit is not None:
                return hit
        x = self._planes(pos)
        if self.symmetry == "all":
            batch = [feat.transform_planes(x, j) for j in range(8)]
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


class PolicyEvaluator(NetEvaluator):
    """pos -> probability vector of length NN+1 (the pass slot stays at 0)."""

    def _unify(self, raw, syms):
        # Undo each symmetry on the *output plane* before averaging:
        # "the planes of output probabilities are rotated/reflected back into
        #  the original orientation, and averaged together".
        acc = np.zeros(NN, dtype=np.float64)
        for row, j in zip(raw, syms):
            p = _softmax(row.astype(np.float64))
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


class ValueEvaluator(NetEvaluator):
    """pos -> scalar value for the player to move.

    The value is invariant under board symmetry, so the ensemble is a plain
    mean: "For the value network, the output values are simply averaged."
    """

    def _unify(self, raw, syms):
        return float(np.mean(raw))


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
