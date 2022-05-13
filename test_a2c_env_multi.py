import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from typing import Optional

import numpy as np

class CategoricalMasked(Categorical):

    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        # self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

    # def entropy(self):
    #     if self.mask is None:
    #         return super().entropy()
    #     # Elementwise multiplication
    #     p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
    #     # Compute the entropy with possible action only
    #     p_log_p = torch.where(
    #         self.mask,
    #         p_log_p,
    #         torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
    #     )
    #     return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)

class TestEnv():
    def __init__(self):
        self.num_actions = 7
        self.state = np.array([1,0,0,0,1,0,0,0])
        self.goal_state = np.array([0,0,0,1,0,0,0,1])

    def reset(self):
        self.state = np.array([1,0,0,0,1,0,0,0])
        return self.state

    def dist_to_action(self, dist):
        mask = torch.zeros(7, dtype=torch.bool)
        mask[0], mask[1] = True, True
        constrained_dist = CategoricalMasked(dist.logits, mask)
        return constrained_dist.sample_n(2)

    def step1(self,action1):
        new_state = np.zeros(4)
        state = self.state[:4]
        pos = np.argmax(state)
        delta = -1 if action1 == 0 else 1
        new_pose = pos + delta
        new_pose = min(max(new_pose,0),3)
        new_state[new_pose] = 1
        return new_state

    def step2(self,action2):
        new_state = np.zeros(4)
        state = self.state[4:]
        pos = np.argmax(state)
        delta = -1 if action2 == 0 else 1
        new_pose = pos + delta
        new_pose = min(max(new_pose,0),3)
        new_state[new_pose] = 1
        return new_state

    def step(self,action):
        self.state = np.concatenate([self.step1(action[0]),self.step2(action[1])])
        reward, done = 0, False
        if (self.state == self.goal_state).all():
            reward, done = 10, True
        return self.state, reward, done, None

    def get_random_action(self):
        p = random.random()
        if p < 0.25:
            return torch.Tensor([0,0])
        if p < 0.5:
            return torch.Tensor([0,1])
        if p < 0.75:
            return torch.Tensor([1,0])
        return torch.Tensor([1,1])

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
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
            # env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            action = env.dist_to_action(dist)
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action).unsqueeze(0)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            if done:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break


        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs).sum(dim=1) # assumes independence
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        # print("log_probs",log_probs)
        # print("advantage",advantage)
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

        action = env.dist_to_action(dist)
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
lr = 0.0001

actor = Actor(state_size, action_size).to(device)
critic = Critic(state_size, action_size).to(device)
trainIters(actor, critic, n_iters=100)
play(actor)
