import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from Masks import CategoricalMasked, CategoricalMap
import matplotlib.pyplot as plt

from typing import Optional

import numpy as np

class TestEnv():
    def __init__(self):
        self.num_actions = 3 # left, stay, right
        self.state = np.array([1,0,0,0,1,0,0,0])
        self.goal_state = np.array([0,0,0,1,0,0,0,1])

    def reset(self):
        self.state = np.array([1,0,0,0,1,0,0,0])
        return self.state

    def step(self,action):
        # print("STATE",self.state)
        new_state = self.state.copy()
        # print("new_state",new_state)
        for cell in range(len(self.state)):
            if self.state[cell]: # has robot
                # print("robot at",cell)
                if action[cell] == 0: # go left
                    # print("goes left")
                    new_pos = max(cell-1,0)
                elif action[cell] == 1: # stay
                    new_pos = cell
                elif action[cell] == 2: # go right
                    # print("goes right")
                    new_pos = min(cell+1,len(self.state)-1)
                else:
                    raise ValueError()
                if not new_state[new_pos]: # collision avoidance
                    # print("moves")
                    new_state[cell] = 0
                    new_state[new_pos] = 1
                # print("new_state",new_state)
        assert sum(self.state) == 2
        self.state = new_state
        reward, done = self.get_reward()
        return self.state, reward, done, None

    def get_reward(self):
        reward, done = 0, False
        if (self.state == self.goal_state).all():
            reward, done = 10, True
        return reward, done

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.num_actions = action_size
        self.action_size = self.state_size*self.num_actions
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        output = output.reshape(self.state_size,self.num_actions)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def trainIters(actor, critic, n_iters, lr):
    scores = []
    optimizerA = optim.Adam(actor.parameters(), lr=lr)
    optimizerC = optim.Adam(critic.parameters(), lr=lr)
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []

        for i in count():
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            # sum up the log_probs of the action taken where there are agents
            log_prob = dist.log_prob(action)[state.bool()].sum(-1).unsqueeze(0)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            if done:
                print('Iteration: {}, Score: {}'.format(iter, i+1))
                scores.append(i+1)
                break

            if i == 500:
                print('attempts exceeded')
                break


        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()

    plt.plot(scores)
    plt.title("Time until goal state reached over training episodes")
    plt.show()

def play(actor):
    state = env.reset()
    print("state:",state)

    for i in range(100):
        state = torch.FloatTensor(state).to(device)
        dist, value = actor(state), critic(state)

        action = dist.sample()
        print("action:",action)
        next_state, reward, done, _ = env.step(action.cpu().numpy())

        state = next_state
        print("state:",state)
        if done:
            print('Done in {} steps'.format(i+1))
            break


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# env = gym.make("CartPole-v0").unwrapped
env = TestEnv()
state_size = env.state.shape[0]
action_size = env.num_actions
print("state_size",state_size)
print("action_size",action_size)
lr = 0.001

actor = Actor(state_size, action_size).to(device)
critic = Critic(state_size, action_size).to(device)
trainIters(actor, critic, n_iters=500, lr=lr)
play(actor)
