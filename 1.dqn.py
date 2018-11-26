import gym
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from agents import DQNAgent
from replay import ReplayBuffer
from networks import DQN


# GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NB_FRAMES = 10000
BATCH_SIZE = 32
DISCOUNT   = 0.99

# Setup Environment
env = gym.make('CartPole-v0')

# Setup Agent
model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = optim.Adam(model.parameters())
replay_buffer = ReplayBuffer(1000)
agent = DQNAgent(env, model, replay_buffer, optimizer, discount=0.99)

# Setup Epsilon Decay
# TODO Modularize
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)

def train(nb_frames):
    episode_reward = 0
    state = env.reset()
    writer = SummaryWriter()
    for frame_idx in range(1, nb_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = agent.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            episode_reward = 0

        if len(replay_buffer) > BATCH_SIZE:
            loss = agent.train(BATCH_SIZE)
            writer.add_scalar('data/losses', loss.item(), frame_idx)

        writer.add_scalar('data/rewards', episode_reward, frame_idx)

    writer.close()


if __name__ == '__main__':
    train(NB_FRAMES)
