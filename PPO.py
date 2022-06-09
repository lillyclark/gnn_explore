import torch
import numpy as np
from Networks import Actor, Critic
from Environments import TestEnv, GraphEnv
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiplicativeLR
from itertools import count
import matplotlib.pyplot as plt

import wandb

class PPO():
    def __init__(self, device, env, net, lr, gamma, lam, eps, crit_coeff, ent_coeff):
        self.device = device
        self.env = env
        self.model = net
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.epsilon = eps
        self.crit_coeff = crit_coeff
        self.ent_coeff = ent_coeff

        wandb.config.update({
            "gamma": self.gamma,
            "lam": self.lam,
            "lr": self.lr,
            "epsilon": self.epsilon})

        wandb.config.update({"model":self.model,
            "model_k":self.model.k,
            "critic_coeff":self.crit_coeff,
            "entropy_coeff":self.ent_coeff})

    def compute_rewards_to_go(self, rewards, R=0):
        rtg = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R
            rtg.insert(0, R)
        return torch.cat(rtg)

    def apply_normalizer(self, adv):
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def evaluate(self, observations, actions, masks):
        # return V, curr_log_probs, curr_entropy
        V = []
        curr_log_probs = []
        curr_entropy = []
        for s, a, m in zip(observations, actions, masks):
            dist, value = self.model(s)
            log_prob = dist.log_prob(a)[m].sum(-1).unsqueeze(0)
            entr = dist.entropy()[m].mean(-1).unsqueeze(0)
            value = value[m].sum(-1)
            V.append(value)
            curr_log_probs.append(log_prob)
            curr_entropy.append(entr)
        return torch.cat(V), torch.cat(curr_log_probs), torch.cat(curr_entropy)

    def trainIters(self, n_iters=100, max_tries=100, plot=False):
        self.model.train()

        decrease_lr = 1
        print("ignoring dynamic learning rate")

        wandb.config.update({"max_tries":max_tries,
            "n_iters":n_iters})

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = MultiplicativeLR(optimizer, lambda iter: decrease_lr)

        for iter in range(n_iters):
            state = self.env.reset() #env.change_env()

            observations = []
            actions = []
            masks = []
            log_probs = []
            entropies = []
            values = []
            rewards = []
            dones = []

            for i in count():
                dist, value = self.model(state)

                action = dist.sample()
                next_state, reward, done, _ = self.env.step(action.cpu().numpy())

                # sum up the log_probs/value where there are agents
                mask = state.x[:,self.env.IS_ROBOT].bool()
                log_prob = dist.log_prob(action)[mask].sum(-1).unsqueeze(0)
                entr = dist.entropy()[mask].mean(-1).unsqueeze(0)
                value = value[mask].sum(-1)

                observations.append(state)
                actions.append(action)
                masks.append(mask)
                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entr)
                rewards.append(torch.tensor([reward], dtype=torch.float, device=self.device))
                dones.append(torch.tensor([1-done], dtype=torch.float, device=self.device))

                state = next_state

                if done or (i == max_tries-1):
                    print('Iteration: {}, Steps: {}, Rewards: {}'.format(iter, i+1, torch.sum(torch.cat(rewards)).item()))
                    break

            log_probs = torch.cat(log_probs)
            values = torch.cat(values)
            entropies = torch.cat(entropies)

            _, nextV = self.model(next_state)
            next_m = next_state.x[:,self.env.IS_ROBOT].bool()
            nextV = nextV[next_m].sum(-1)
            rewards_to_go = self.compute_rewards_to_go(rewards,R=nextV)

            V, _, _ = self.evaluate(observations, actions, masks)
            adv = rewards_to_go - V.detach()
            adv = self.apply_normalizer(adv)

            for _ in range(1):
                V, curr_log_probs, curr_entropy = self.evaluate(observations, actions, masks)
                ratios = torch.exp(curr_log_probs - log_probs)
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv

                actor_loss = (-torch.min(surr1, surr2)).mean()
                entropy = curr_entropy.sum() # TODO

                value_pred_clipped = values + (V - values).clamp(-self.epsilon, self.epsilon)
                value_losses = (V - rewards_to_go) ** 2
                value_losses_clipped = (value_pred_clipped - rewards_to_go) ** 2
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
                critic_loss = value_loss.mean()
                # critic_loss = (rewards_to_go - V.detach()).pow(2).mean()

                shared_loss = actor_loss + self.crit_coeff * critic_loss - self.ent_coeff * entropy

                wandb.log({"actor_loss": actor_loss})
                wandb.log({"critic_loss": critic_loss})
                wandb.log({"entropy": entropy})
                wandb.log({"shared_loss": shared_loss})
                wandb.log({"explore_time": i+1})
                wandb.log({"sum_reward":torch.sum(torch.cat(rewards))})
                wandb.watch(self.model)

                optimizer.zero_grad()
                shared_loss.backward(retain_graph=True)
                # torch.nn.utils.clip_grad_norm_(a2c_net.parameters(), 0.5)
                optimizer.step()
                scheduler.step()

    def play(self, max_tries=50, v=False):
        self.model.eval()

        state = self.env.reset() #env.change_env()
        rewards = []
        print("state:",state.x[:,self.env.IS_ROBOT].numpy())
        if v:
            print("known:",state.x[:,self.env.IS_KNOWN_ROBOT].numpy())
            print("known:",state.x[:,self.env.IS_KNOWN_BASE].numpy())

        for i in range(max_tries):
            dist, value = self.model(state)
            if v:
                print("dist:")
                print(np.round(dist.probs.detach().numpy().T,2))
                print("value:",np.round(value.detach().numpy().T,2))
            action = dist.sample()
            print("action:",action.numpy())
            next_state, reward, done, _ = self.env.step(action.cpu().numpy())
            print("reward:",reward)
            rewards.append(reward)
            print("")

            state = next_state
            print("state:",state.x[:,self.env.IS_ROBOT].numpy())
            if v:
                print("known:",state.x[:,self.env.IS_KNOWN_ROBOT].numpy())
                print("known:",state.x[:,self.env.IS_KNOWN_BASE].numpy())
            if done:
                print('Done in {} steps'.format(i+1))
                break
        wandb.log({"test_steps":i+1})
        wandb.log({"test_reward":sum(rewards)})
