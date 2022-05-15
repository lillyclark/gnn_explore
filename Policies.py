import torch
import numpy as np
from Networks import Actor, Critic
from Environments import TestEnv, GraphEnv
import torch.optim as optim
from itertools import count
import matplotlib.pyplot as plt

class Graph_A2C():
    def __init__(self, device, n_iters, lr, gamma):
        self.device = device
        self.gamma = gamma
        self.n_iters = n_iters
        self.lr = lr

    def compute_returns(self,next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self. gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def trainIters(self, env, actor, critic, max_tries=100, plot=False):
        scores = []
        optimizerA = optim.Adam(actor.parameters(), lr=self.lr)
        optimizerC = optim.Adam(critic.parameters(), lr=self.lr)
        for iter in range(self.n_iters):
            state = env.reset()
            log_probs = []
            values = []
            rewards = []
            masks = []
            # print(state.x[:,0].numpy(),state.x[:,1].numpy())

            for i in count():
                dist, value = actor(state), critic(state)

                action = dist.sample()
                next_state, reward, done, _ = env.step(action.cpu().numpy())
                # print(next_state.x[:,0].numpy(),next_state.x[:,1].numpy())

                # sum up the log_probs of the action taken where there are agents
                mask = state.x[:,0]
                log_prob = dist.log_prob(action)[mask.bool()].sum(-1).unsqueeze(0)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor([reward], dtype=torch.float, device=self.device))
                masks.append(torch.tensor([1-done], dtype=torch.float, device=self.device))

                state = next_state

                if done:
                    print('Iteration: {}, Score: {}'.format(iter, i+1))
                    scores.append(i+1)
                    break

                if i == max_tries:
                    print('attempts exceeded')
                    break


            next_value = critic(next_state)
            returns = self.compute_returns(next_value, rewards, masks)

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

        if plot:
            plt.plot(scores)
            plt.title("Time until goal state reached over training episodes")
            plt.show()

    def play(self, env, actor):
        state = env.reset()
        print("state:",state.x[:,0].numpy())

        for i in range(50):
            dist = actor(state)

            action = dist.sample()
            print("action:",action.numpy())
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            state = next_state
            print("state:",state.x[:,0].numpy())
            if done:
                print('Done in {} steps'.format(i+1))
                break


class A2C():
    def __init__(self, device, n_iters, lr, gamma):
        self.device = device
        self.gamma = gamma
        self.n_iters = n_iters
        self.lr = lr

    def compute_returns(self,next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self. gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def trainIters(self, env, actor, critic):
        scores = []
        optimizerA = optim.Adam(actor.parameters(), lr=self.lr)
        optimizerC = optim.Adam(critic.parameters(), lr=self.lr)
        for iter in range(self.n_iters):
            state = env.reset()
            log_probs = []
            values = []
            rewards = []
            masks = []

            for i in count():
                state = torch.FloatTensor(state).to(self.device)
                dist, value = actor(state), critic(state)

                action = dist.sample()
                next_state, reward, done, _ = env.step(action.cpu().numpy())

                # sum up the log_probs of the action taken where there are agents
                log_prob = dist.log_prob(action)[state.bool()].sum(-1).unsqueeze(0)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor([reward], dtype=torch.float, device=self.device))
                masks.append(torch.tensor([1-done], dtype=torch.float, device=self.device))

                state = next_state

                if done:
                    print('Iteration: {}, Score: {}'.format(iter, i+1))
                    scores.append(i+1)
                    break

                if i == 500:
                    print('attempts exceeded')
                    break


            next_state = torch.FloatTensor(next_state).to(self.device)
            next_value = critic(next_state)
            returns = self.compute_returns(next_value, rewards, masks)

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

    def play(self, env, actor):
        state = env.reset()
        print("state:",state)

        for i in range(100):
            state = torch.FloatTensor(state).to(self.device)
            dist = actor(state)

            action = dist.sample()
            print("action:",action)
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            state = next_state
            print("state:",state)
            if done:
                print('Done in {} steps'.format(i+1))
                break
