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
    bonus = np.clip(bonus, 0.0, 1.0 - epsilon)
    return epsilon + bonus


class MLP(nn.Module):
    def __init__(
        self,
        layers,
        activation=torch.tanh,
        output_activation=None,
        output_scale=1,
        output_squeeze=False,
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


class DQNetwork(nn.Module):
    def __init__(
        self,
        in_features,
        action_space,
        num_quantiles=50,
        Vmin=-10,
        Vmax=10,
        hidden_sizes=(400, 300),
        activation=torch.relu,
        output_activation=None,
    ):
        super(DQNetwork, self).__init__()

        self.action_dim = action_space.n
        self.num_quantiles = num_quantiles
        self.supports = torch.linspace(Vmin, Vmax, num_quantiles)

        self.q = MLP(
            layers=[in_features]
            + list(hidden_sizes)
            + [self.action_dim * num_quantiles],
            activation=activation,
            output_activation=output_activation,
        )

    def forward(self, x, log=False):
        q_dist = self.q(x).view(-1, self.action_dim, self.num_quantiles)
        return q_dist

    def policy(self, x):
        q = self.forward(x).mean(-1)  # (bsz, action_dim)
        acts = q.argmax(-1, keepdim=True)
        return acts
