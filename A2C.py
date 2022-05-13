import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from Masks import CategoricalMasked, CategoricalMap
import random

from typing import Optional

import numpy as np

class TestEnv():
    def __init__(self):
        self.num_actions = 15 # each edge and self edges
        self.state = np.array([1,0,0,0,1,0,0,0])
        self.goal_state = np.array([0,0,0,1,0,0,0,1])
        self.edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)]

    def reset(self):
        self.state = np.array([1,0,0,0,1,0,0,0])
        return self.state

    def get_indicent_edges(self):
        incident_edges = {}
        for cell in range(len(self.state)):
            if self.state[cell]:
                incident_edges[cell] = torch.zeros(self.num_actions, dtype=torch.bool)
                for edge_idx in range(len(self.edges)):
                    if cell in self.edges[edge_idx]:
                        incident_edges[cell][edge_idx] = 1
        return incident_edges

        incident_edges = {}
        for cell in range(len(self.state)):
            if self.state[cell]: # this is a robot
                incident_edges[cell] = []
                for edge_idx in range(len(self.edges)):
                    if cell in self.edges[edge_idx]:
                        incident_edges[cell].append(edge_idx)
        return incident_edges

    def step(self,action):
        print("STATE",self.state)
        print("ACTION",action)

        assert False

        new_state = self.state.copy()
        actions_taken = torch.zeros(action.shape,dtype=torch.bool)

        for agent_cell, edge_idx_list in  self.get_indicent_edges().items():
            # print("agent at", agent_cell)
            random.shuffle(edge_idx_list)
            for edge_idx in edge_idx_list:
                # print("take edge",edge_idx,"?")
                if action[edge_idx]: # take this action
                    # print("moving")
                    u, v = self.edges[edge_idx]
                    if u == v:
                        actions_taken[edge_idx] = 1
                        break
                    elif agent_cell == u and new_state[v] == 0:
                        new_state[u], new_state[v] = 0, 1
                        actions_taken[edge_idx] = 1
                        # print("new_state",new_state)
                        break
                    elif agent_cell == v and new_state[u] == 0:
                        new_state[u], new_state[v] = 1, 0
                        actions_taken[edge_idx] = 1
                        # print("new_state",new_state)
                        break

        self.state = new_state
        reward, done = self.get_reward()
        return self.state, reward, done, actions_taken

    def get_reward(self):
        reward, done = -1, False
        if (self.state == self.goal_state).all():
            reward, done = 100, True
        return reward, done

    def get_random_action(self):
        r = np.random.randint(2,size=self.num_actions)
        return torch.Tensor(r)

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size*2) # each action can be on or off

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        output = output.reshape(self.action_size,2) # each action can be on or off
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

def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []

        for i in count():
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            print("log prob example", dist.log_prob(torch.Tensor([0])))

            actions = {}
            log_prob_sum = torch.Tensor([0])
            print("log_prob_sum", log_prob_sum)
            for robot, edges in env.get_indicent_edges().items():
                print("robot",robot)
                print("edges",edges)
                print("dist.probs",dist.probs)
                probs = dist.probs[edges]
                # probs = torch.masked_select(dist.probs, edges.unsqueeze(1))
                print("probs",probs)
                robot_dist = Categorical(probs)
                a = robot_dist.sample()
                robot_log_prob = robot_dist.log_prob(a)
                print("log_prob (individual)", robot_log_prob)
                log_prob_sum += robot_log_prob
                actions[robot] = a

            print("actions",actions)
            print("log_prob (joint)", log_prob_sum)

            next_state, reward, done, log_prob_mask = env.step(action.cpu().numpy())

            # sum up the log_probs of the sample where the action was taken
            log_prob = dist.log_prob(action)[log_prob_mask].sum(-1).unsqueeze(0)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            if done:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break

            if i == 1000:
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

def play(actor):
    state = env.reset()
    print("state:",state)

    for i in range(100):
        state = torch.FloatTensor(state).to(device)
        dist, value = actor(state), critic(state)
        print("value:",value)

        action = dist.sample()
        print("action:",action)
        next_state, reward, done, _ = env.step(action.cpu().numpy())

        print("")

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

actor = Actor(state_size, action_size).to(device)
critic = Critic(state_size, action_size).to(device)
trainIters(actor, critic, n_iters=100)
play(actor)
