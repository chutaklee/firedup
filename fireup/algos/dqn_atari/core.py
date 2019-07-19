
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
    def __init__(
        self, layers,
        activation=torch.relu,
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


class ConvNet(nn.Module):
    """network architecture from deepmind's paper"""
    def __init__(self, num_actions):
        super(ConvNet, self).__init__()

        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channesl=64, out_channels=64, kernel_size=(3, 3), stride=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(3136, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        """
        x has shape of (batch_size, 4, 84, 84) which corresponding to
        (batch_size, channels, height, width)
        """
        assert x.size()[1:] == (4, 84, 84)

        out = self.features(x)  # out has shape of (batch_size, 4, 84, 84)
        batch_size = out.size()[0]
        out = out.view(batch_size, -1)  # out has shape of (batch_size, 2592)
        out = self.fc(out)  # out has shape of (batch_size, num_actions)
        return out


class FastConvNet(nn.Module):
    def __init__(self, num_actions):
        super(FastConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 5, stride=5, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=5, padding=0),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        out = self.features(x).view(x.size()[0], -1)
        out = self.fc(out)
        return out



class DQNetwork(nn.Module):
    def __init__(
        self, in_features, action_space,
        hidden_sizes=(400, 300),
        activation=torch.relu,
        output_activation=None
    ):
        super(DQNetwork, self).__init__()

        action_dim = action_space.n
        # self.q = MLP(layers=[in_features] + list(hidden_sizes) + [action_dim], activation=activation, output_activation=output_activation)
        self.q = ConvNet(action_dim)

    def forward(self, x):
        return self.q(x)

    def policy(self, x):
        return torch.argmax(self.q(x), dim=1, keepdim=True)
