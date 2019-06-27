from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

def make_mlp(hidden_sizes, activation, output_activation):
    layers: List = []
    for i in range(len(hidden_sizes) - 2):
        layers += [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), activation()]
    layers.append(nn.Linear(hidden_sizes[-2], hidden_sizes[-1]))
    if output_activation:
        layers.append(output_activation())
    return nn.Sequential(*layers)

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (torch.exp(log_std) + EPS)) ** 2
                      + 2 * log_std + np.log(2 * np.pi))
    return torch.sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""
class MLPCategoricalPolicy(nn.Module):
    def __init__(self, sizes, activation, output_activation, act_dim):
        self.act_dim = act_dim
        self.model = make_mlp(sizes, activation, output_activation)

    def forward(self, x, a):
        logits = self.model(x)
        logp_all = F.log_softmax(logits)

        a_oh = torch.zeros((a.shape[0], self.act_dim)).scateer_(1, a, 1)
        logp = torch.sum(a_oh * logp_all, dim=1)

        pi = torch.squeeze(torch.multinomial(logits, 1), dim=1)
        pi_oh = torch.zeros((pi.shape[0], self.act_dim))
        logp_pi = torch.sum(pi_oh * logp_all, dim=1)
        return pi, logp, logp_pi


class MLPGaussianPolicy(nn.Module):
    def __init__(self, sizes, activation, output_activation, act_dim):
        self.act_dim = act_dim
        self.model = make_mlp(sizes, activation, output_activation)

    def forward(self, x, a):
        mu = self.model(x)
        log_std = torch.from_numpy(-0.5 * np.ones(self.act_dim, dtype=np.float32))
        std = torch.exp(log_std)
        pi = torch.normal(mu, std)
        logp = gaussian_likelihood(a, mu, log_std)
        logp_pi = gaussian_likelihood(pi, mu, log_std)
        return pi, logp, logp_pi


class MLPActorCritic(nn.Module):
    """ Actor-Critics """
    def __init__(
            self,
            hidden_sizes=[64, 64],
            activation=tf.tanh,
            output_activation=None,
            action_space=None
    ):
        # default policy builder depends on action space
        if isinstance(action_space, Box):
            self.act_dim = action_space.shape[-1]
            self.actor = MLPGaussianPolicy(
                hidden_sizes + [self.act_dim], activation, output_activation
            )
        elif isinstance(action_space, Discrete):
            self.act_dim = action_space.n
            self.actor = MLPCategoricalPolicy(
                hidden_sizes + [self.act_dim], activation, None
            )

        self.critic = make_mlp(hidden_sizes + [1], activation, None)

    def forward(self, x, a):
        pi, logp, logp_pi = self.actor(x, a)
        v = torch.squeeze(self.critic(x), dim=1)
        return pi, logp, logp_pi, v
