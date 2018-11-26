import random

import torch


# GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    def __init__(self, env, net, replay, optimizer, discount):
        self.env = env
        self.net = net
        self.replay = replay
        self.optimizer = optimizer
        self.discount = discount

    def act(self, state, epsilon):
        """
        Choose action given state s and epsilon e. The epsilon denotes the
        possibility of random action for epsilon-greedy policy.
        """
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_value = self.net(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.env.action_space.n)

        return action

    def train(self, batch_size):
        """
        Train the agent with one batch and return loss.
        """
        state, action, reward, next_state, done = self.replay.sample(batch_size)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.LongTensor(action).to(device)
        reward     = torch.FloatTensor(reward).to(device)
        done       = torch.FloatTensor(done).to(device)

        q_values      = self.net(state)
        next_q_values = self.net(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.discount * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.data).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
