import math, random

import gym
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from replay import ReplayBuffer
from networks import DQN


# Hyperparameters
NB_FRAMES = 10000
BATCH_SIZE = 32
DISCOUNT   = 0.99
USE_CUDA = torch.cuda.is_available()

# Setup Environment
env = gym.make('CartPole-v0')

# Setup Agent
model = DQN(env.observation_space.shape[0], env.action_space.n)
if USE_CUDA:
    model = model.cuda()
optimizer = optim.Adam(model.parameters())
replay_buffer = ReplayBuffer(1000)

# Setup Epsilon Decay
# TODO Modularize
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# Compute Loss
# TODO Modularize
def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    action     = torch.LongTensor(action)
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(done)

    q_values      = model(state)
    next_q_values = model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + DISCOUNT * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.data).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def train(nb_frames):
    episode_reward = 0
    state = env.reset()
    writer = SummaryWriter()
    for frame_idx in range(1, nb_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            episode_reward = 0

        if len(replay_buffer) > BATCH_SIZE:
            loss = compute_td_loss(BATCH_SIZE)
            writer.add_scalar('data/losses', loss.item(), frame_idx)

        writer.add_scalar('data/rewards', episode_reward, frame_idx)

    writer.close()


if __name__ == '__main__':
    train(NB_FRAMES)
