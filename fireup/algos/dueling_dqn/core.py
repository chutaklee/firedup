import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus


class MLP(nn.Module):
    def __init__(self,
                 layers,
                 activation=torch.tanh,
                 output_activation=None,
                 output_scale=1,
                 output_squeeze=False):
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


class DuelingDQNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 action_space,
                 hidden_sizes=(400, 300),
                 activation=torch.relu,
                 output_activation=None):
        super(DuelingDQNetwork, self).__init__()

        self.action_dim = action_space.n

        self.enc = MLP(
            layers=[in_features] + list(hidden_sizes),
            activation=activation,
            output_activation=output_activation)

        self.v = MLP(
            layers=[hidden_sizes[-1], 1],
            activation=activation,
            output_activation=output_activation)

        self.a = MLP(
            layers=[hidden_sizes[-1], self.action_dim],
            activation=activation,
            output_activation=output_activation)

    def forward(self, x):
        enc = self.enc(x)
        a = self.a(enc)
        return self.v(enc) + a - a.mean(1, True)

    def policy(self, x):
        return torch.argmax(self.forward(x), dim=1, keepdim=True)
