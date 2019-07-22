import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box, Discrete


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class NoisyLinear(nn.Module):
    """Factorised NoisyLinear layer with bias"""

    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon',
                             torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if not self.training:
            return F.linear(x, self.weight_mu, self.bias_mu)
        return F.linear(
            x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, True)


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


class NoisyDQNetwork(nn.Module):
    def __init__(
        self, in_features, action_space,
        hidden_sizes=(400, 300),
        activation=torch.relu,
        output_activation=None
    ):
        super(NoisyDQNetwork, self).__init__()

        action_dim = action_space.n

        self.q = MLP(
            layers=[in_features] + list(hidden_sizes),
            activation=activation,
            output_activation=output_activation)

        self.q.layers.append(NoisyLinear(hidden_sizes[-1], action_dim))

        print(self)

    def forward(self, x):
        return self.q(x)

    def policy(self, x):
        return torch.argmax(self.q(x), dim=1, keepdim=True)

    def reset_noise(self):
        self.q.layers[-1].reset_noise()
