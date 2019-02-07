import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import gym
from gym.spaces import Discrete, Box


def mlp(sizes, activation=nn.Tanh, output_activation=None):
    # Build a feedforward neural network.
    # sizes is a list of size from input to output layers
    layers = []
    for i in range(len(sizes)-2):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        layers.append(activation())
    # Last layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2,
          epochs=300, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    network_size = [obs_dim] + hidden_sizes + [n_acts]
    policy_network = mlp(sizes=network_size)

    # make train optimizer
    train_op = optim.Adam(policy_network.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            obs = torch.from_numpy(obs).reshape(1, -1)  # batch_size = 1
            with torch.no_grad():
                logits = policy_network(obs.float())
            act = torch.multinomial(
                    F.softmax(logits, dim=1), num_samples=1).item()
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # make loss function whose gradient, for the right data,
        # is policy gradient
        observations = torch.from_numpy(np.array(batch_obs)).float()

        actions = torch.from_numpy(np.array(batch_acts)).byte()
        action_masks = torch.zeros((len(batch_obs), n_acts))
        action_masks[range(len(batch_obs)), actions.numpy()] = 1

        weights = torch.from_numpy(np.array(batch_weights)).float()

        log_probs = F.log_softmax(policy_network(observations), dim=1)
        masked_log_probs = torch.sum(action_masks * log_probs, dim=1)

        batch_loss = -torch.mean(weights * masked_log_probs)

        # take a single policy gradient update step
        batch_loss.backward()
        train_op.step()

        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
