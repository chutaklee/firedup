import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box, Discrete


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus


def beta_by_frame(frame_idx, beta_start=0.4, beta_frames=10000):
    return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


class MLP(nn.Module):
    def __init__(
        self, layers,
        activation=torch.tanh,
        output_activation=None,
        output_scale=1,
        output_squeeze=False
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_scale = output_scale
        self.output_squeeze = output_squeeze

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x) * self.output_scale
        else:
            x = self.output_activation(self.layers[-1](x)) * self.output_scale
        return x.squeeze() if self.output_squeeze else x


class CategoricalDQNetwork(nn.Module):
    def __init__(
        self, in_features, action_space,
        num_atoms=50,
        Vmin=-100,
        Vmax=100,
        hidden_sizes=(400, 300),
        activation=torch.relu,
        output_activation=None
    ):
        super(CategoricalDQNetwork, self).__init__()

        self.action_dim = action_space.n
        self.num_atoms = num_atoms
        self.supports = torch.linspace(Vmin, Vmax, num_atoms)

        self.q = MLP(
            layers=[in_features] + list(hidden_sizes) + [self.action_dim * num_atoms],
            activation=activation,
            output_activation=output_activation
        )

    def forward(self, x, log=False):
        q = self.q(x).view(-1, self.action_dim, self.num_atoms)
        if log:
            return F.log_softmax(q, dim=-1)
        else:
            return F.softmax(q, dim=-1)

    def policy(self, x):
        q_dist = self.forward(x)  # (bsz, action_dim, num_atoms)
        q_vals = (q_dist * self.supports.expand_as(q_dist)).sum(-1)  # (bsz, action_dim)
        action = torch.argmax(q_vals, dim=1, keepdim=True)  # (bsz, 1)
        return action


class CategoricalDuelingDQNetwork(nn.Module):
    """
    value head and advantage head need at least two layers to perform well on LunarLander-v2
    """
    def __init__(
        self, in_features, action_space,
        num_atoms=50,
        Vmin=-100,
        Vmax=100,
        hidden_sizes=(400, 300),
        activation=torch.relu,
        output_activation=None
    ):
        super(CategoricalDuelingDQNetwork, self).__init__()

        self.action_dim = action_space.n
        self.num_atoms = num_atoms
        self.supports = torch.linspace(Vmin, Vmax, num_atoms)

        self.enc = MLP(
            layers=[in_features] + list(hidden_sizes)[:-1],
            activation=activation,
            output_activation=None
        )
        self.a = MLP(
            layers=list(hidden_sizes)[-2:] + [num_atoms * self.action_dim],
            activation=activation,
            output_activation=output_activation
        )
        self.v = MLP(
            layers=list(hidden_sizes)[-2:] + [num_atoms],
            activation=activation,
            output_activation=output_activation,
        )

    def forward(self, x, log=False):
        enc = self.enc(x)
        a = self.a(enc).view(-1, self.action_dim, self.num_atoms)
        v = self.v(enc).view(-1, 1, self.num_atoms)
        q_dist = v + a - a.mean(1, keepdim=True)

        if log:
            return F.log_softmax(q_dist, dim=-1)
        else:
            return F.softmax(q_dist, dim=-1)

    def policy(self, x):
        q_dist = self.forward(x)  # (bsz, action_dim, num_atoms)
        q_vals = (q_dist * self.supports.expand_as(q_dist)).sum(-1)  # (bsz, action_dim)
        action = torch.argmax(q_vals, dim=1, keepdim=True)  # (bsz, 1)
        return action
