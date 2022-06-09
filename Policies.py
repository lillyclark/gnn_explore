import torch
import numpy as np
from Networks import Actor, Critic
from Environments import TestEnv, GraphEnv
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiplicativeLR
from itertools import count
import matplotlib.pyplot as plt

import wandb

class PolicyGradient():
    def __init__(self, device, n_iters, lr, gamma):
        self.device = device
        self.gamma = gamma
        self.n_iters = n_iters
        self.lr = lr

        wandb.config.update({
            "gamma": self.gamma,
            "lr": self.lr,
            "n_iters": self.n_iters})

    def compute_returns(self,next_value, rewards, not_dones, progresses=None):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * not_dones[step]
            returns.insert(0, R)
        return returns

    def trainIters(self, env, actor, decrease_lr=0.997, max_tries=100, plot=False):

        wandb.config.update({"max_tries":max_tries,
            "actor":actor,
            "actor_k":actor.k,
            "lr_factor":decrease_lr})

        optimizer = optim.Adam(actor.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = MultiplicativeLR(optimizer, lambda iter: decrease_lr)

        for iter in range(self.n_iters):
            state = env.reset() #env.change_env()
            log_probs = []
            entropies = []
            # values = []
            rewards = []
            not_dones = []
            progresses = []

            for i in count():
                dist = actor(state)

                action = dist.sample()
                next_state, reward, done, _ = env.step(action.cpu().numpy())
                progress = _["progress"]

                # sum up the log_probs/value where there are agents
                mask = state.x[:,env.IS_ROBOT]
                log_prob = dist.log_prob(action)[mask.bool()].sum(-1).unsqueeze(0)
                entr = dist.entropy()[mask.bool()].mean(-1).unsqueeze(0)
                # real_value = value[mask.bool()].sum(-1)

                log_probs.append(log_prob)
                # values.append(real_value)
                entropies.append(entr)
                rewards.append(torch.tensor([reward], dtype=torch.float, device=self.device))
                not_dones.append(torch.tensor([1-done], dtype=torch.float, device=self.device))
                progresses.append(torch.tensor([progress], dtype=torch.float, device=self.device))

                state = next_state

                if done or (i == max_tries-1):
                    print('Iteration: {}, Steps: {}, Rewards: {}'.format(iter, i+1, torch.sum(torch.cat(rewards)).item()))
                    break

            returns = self.compute_returns(0, rewards, not_dones, progresses)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            entropy = torch.cat(entropies).sum()

            # DEBUG : JUST REINFORCE (NO BASELINE)
            actor_loss = -(log_probs*returns.detach()).mean()

            wandb.log({"actor_loss": actor_loss})
            wandb.log({"explore_time": i+1})
            wandb.log({"sum_reward":torch.sum(torch.cat(rewards))})
            wandb.log({"entropy": entropy})
            wandb.watch(actor)

            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()
            scheduler.step()

        wandb.finish()

    def play(self, env, actor, max_tries=50, v=False):
        state = env.reset() #env.change_env()
        rewards = []
        print("state:",state.x[:,env.IS_ROBOT].numpy())
        if v:
            print("known:",state.x[:,env.IS_KNOWN_ROBOT].numpy())
            print("known:",state.x[:,env.IS_KNOWN_BASE].numpy())

        for i in range(max_tries):
            dist = actor(state)
            if v:
                print("dist:")
                print(np.round(dist.probs.detach().numpy().T,2))
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
        # wandb.log({"test_steps":i+1})
        # wandb.log({"test_reward":sum(rewards)})


class A2C_Shared():
    def __init__(self, device, n_iters, lr, gamma, run_name="tmp"):
        self.device = device
        self.gamma = gamma
        self.n_iters = n_iters
        self.lr = lr

        # wandb.init(project="simple-world-2", entity="lillyclark", config={})


        wandb.config.update({
            "gamma": self.gamma,
            "lr": self.lr,
            "n_iters": self.n_iters})

        # wandb.run.name = run_name+wandb.run.id

    def compute_returns(self,next_value, rewards, not_dones, progresses=None):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * not_dones[step]
            returns.insert(0, R)
        return returns

    def compute_shaped_returns(self,next_value, rewards, not_dones, progresses=None):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * not_dones[step]
            returns.insert(0, R)
        return returns

    def trainIters(self, env, a2c_net, crit_coeff=1, ent_coeff=0, decrease_lr=0.997, max_tries=100, plot=False):

        wandb.config.update({"max_tries":max_tries,
            "a2c":a2c_net,
            "a2c_k":a2c_net.k,
            "lr_factor":decrease_lr,
            "critic_coeff":crit_coeff,
            "entropy_coeff":ent_coeff})

        optimizer = optim.Adam(a2c_net.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = MultiplicativeLR(optimizer, lambda iter: decrease_lr)

        for iter in range(self.n_iters):
            state = env.reset() #env.change_env()
            log_probs = []
            entropies = []
            values = []
            rewards = []
            not_dones = []
            progresses = []

            for i in count():
                dist, value = a2c_net(state)

                action = dist.sample()
                next_state, reward, done, _ = env.step(action.cpu().numpy())
                progress = _["progress"]

                # sum up the log_probs/value where there are agents
                mask = state.x[:,env.IS_ROBOT]
                log_prob = dist.log_prob(action)[mask.bool()].sum(-1).unsqueeze(0)
                entr = dist.entropy()[mask.bool()].mean(-1).unsqueeze(0)
                real_value = value[mask.bool()].sum(-1)

                log_probs.append(log_prob)
                values.append(real_value)
                entropies.append(entr)
                rewards.append(torch.tensor([reward], dtype=torch.float, device=self.device))
                not_dones.append(torch.tensor([1-done], dtype=torch.float, device=self.device))
                progresses.append(torch.tensor([progress], dtype=torch.float, device=self.device))

                state = next_state

                if done or (i == max_tries-1):
                    print('Iteration: {}, Steps: {}, Rewards: {}'.format(iter, i+1, torch.sum(torch.cat(rewards)).item()))
                    break

            _, next_value = a2c_net(next_state)
            next_mask = next_state.x[:,env.IS_ROBOT]
            real_next_value = next_value[next_mask.bool()].sum(-1)
            returns = self.compute_returns(real_next_value, rewards, not_dones, progresses)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)
            advantage = returns - values
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            entropy = torch.cat(entropies).sum()

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            shared_loss = actor_loss + crit_coeff * critic_loss - ent_coeff * entropy

            # DEBUG : JUST REINFORCE (NO BASELINE)
            shared_loss = -(log_probs*returns.detach()).mean()

            wandb.log({"actor_loss": actor_loss})
            wandb.log({"critic_loss": critic_loss})
            wandb.log({"entropy": entropy})
            wandb.log({"shared_loss": shared_loss})
            wandb.log({"explore_time": i+1})
            wandb.log({"sum_reward":torch.sum(torch.cat(rewards))})
            wandb.watch(a2c_net)

            optimizer.zero_grad()
            shared_loss.backward()
            # torch.nn.utils.clip_grad_norm_(a2c_net.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

    def play(self, env, a2c_net, max_tries=50, v=False):
        state = env.reset() #env.change_env()
        rewards = []
        print("state:",state.x[:,env.IS_ROBOT].numpy())
        if v:
            print("known:",state.x[:,env.IS_KNOWN_ROBOT].numpy())
            print("known:",state.x[:,env.IS_KNOWN_BASE].numpy())

        for i in range(max_tries):
            dist, value = a2c_net(state)
            if v:
                print("dist:")
                print(np.round(dist.probs.detach().numpy().T,2))
                print("value:",np.round(value.detach().numpy().T,2))
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

        optimizerA = optim.Adam(actor.parameters(), lr=self.a_lr, betas=(0.9, 0.999))
        optimizerC = optim.Adam(critic.parameters(), lr=self.c_lr, betas=(0.9, 0.999))
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
