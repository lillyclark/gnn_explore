import torch
import numpy as np
from Networks import Actor, Critic
from Environments import TestEnv, GraphEnv
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiplicativeLR
from itertools import count
import matplotlib.pyplot as plt

import wandb
wandb.init(project="simple-world-explore", entity="lillyclark", config={})

class Graph_A2C():
    def __init__(self, device, n_iters, a_lr, c_lr, gamma, run_name="tmp"):
        self.device = device
        self.gamma = gamma
        self.n_iters = n_iters
        self.a_lr = a_lr
        self.c_lr = c_lr

        wandb.config.update({
            "gamma": self.gamma,
            "a_lr": self.a_lr,
            "c_lr": self.c_lr,
            "n_iters": self.n_iters})

        wandb.run.name = run_name

    def compute_returns(self,next_value, rewards, dones):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self. gamma * R * dones[step]
            returns.insert(0, R)
        return returns

    def trainIters(self, env, actor, critic, decrease_lr=0.997, max_tries=100, plot=False):
        wandb.config.update({"max_tries":max_tries,
            "actor":actor,
            "actor_k":actor.k,
            "critic":critic,
            "critic_k":critic.k,
            "lr_factor":decrease_lr})

        optimizerA = optim.Adam(actor.parameters(), lr=self.a_lr)
        optimizerC = optim.Adam(critic.parameters(), lr=self.c_lr)
        schedulerA = MultiplicativeLR(optimizerA, lambda iter: decrease_lr)
        schedulerC = MultiplicativeLR(optimizerC, lambda iter: decrease_lr)

        for iter in range(self.n_iters):
            state = env.reset() #env.change_env()
            log_probs = []
            values = []
            rewards = []
            dones = []

            for i in count():
                dist, value = actor(state), critic(state)

                action = dist.sample()
                next_state, reward, done, _ = env.step(action.cpu().numpy())

                # sum up the log_probs/value where there are agents
                mask = state.x[:,env.IS_ROBOT]
                log_prob = dist.log_prob(action)[mask.bool()].sum(-1).unsqueeze(0)
                real_value = value[mask.bool()].sum(-1)

                log_probs.append(log_prob)
                values.append(real_value)
                rewards.append(torch.tensor([reward], dtype=torch.float, device=self.device))
                dones.append(torch.tensor([1-done], dtype=torch.float, device=self.device))

                state = next_state

                if done or (i == max_tries-1):
                    print('Iteration: {}, Steps: {}, Rewards: {}'.format(iter, i+1, torch.sum(torch.cat(rewards)).item()))
                    break

            next_value = critic(next_state)
            next_mask = next_state.x[:,env.IS_ROBOT]
            real_next_value = next_value[next_mask.bool()].sum(-1)
            returns = self.compute_returns(real_next_value, rewards, dones)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)
            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            wandb.log({"actor_loss": actor_loss})
            wandb.log({"critic_loss": critic_loss})
            wandb.log({"explore_time": i+1})
            wandb.log({"sum_reward":torch.sum(torch.cat(rewards))})

            wandb.watch(actor)
            wandb.watch(critic)

            optimizerA.zero_grad()
            optimizerC.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            optimizerA.step()
            optimizerC.step()

            schedulerA.step()
            schedulerC.step()

    def play(self, env, actor, critic, max_tries=50, v=False):
        state = env.reset() #env.change_env()
        rewards = []
        print("state:",state.x[:,env.IS_ROBOT].numpy())
        if v:
            print("known:",state.x[:,env.IS_KNOWN_ROBOT].numpy())
            print("known:",state.x[:,env.IS_KNOWN_BASE].numpy())

        for i in range(max_tries-1):
            dist = actor(state)
            if v:
                print("dist:")
                print(np.round(dist.probs.detach().numpy().T,2))
                value = critic(state)
                print("value:",value.detach().numpy().T)
            action = dist.sample()
            print("action:",action.numpy())
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            print("reward:",reward)
            rewards.append(reward)
            print("")

            state = next_state
            print("state:",state.x[:,env.IS_ROBOT].numpy())
            if v:
                print("known:",state.x[:,env.IS_KNOWN_ROBOT].numpy())
                print("known:",state.x[:,env.IS_KNOWN_BASE].numpy())
            if done:
                print('Done in {} steps'.format(i+1))
                break
        wandb.log({"test_steps":i+1})
        wandb.log({"test_reward":sum(rewards)})

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
