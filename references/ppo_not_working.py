class PPO():
    def __init__(self, device, n_iters, lr, gamma, lam, eps, run_name="tmp"):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.n_iters = n_iters
        self.lr = lr
        self.epsilon = eps

        wandb.config.update({
            "gamma": self.gamma,
            "lam": self.lam,
            "lr": self.lr,
            "n_iters": self.n_iters})

        wandb.run.name = run_name+wandb.run.id

    def compute_returns(self,next_value, rewards, dones):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * dones[step]
            returns.insert(0, R)
        return returns

    def compute_advantages(self, next_value, dones, rewards, values):
        A = 0
        R = next_value
        advantages = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * R * dones[step] - values[step]
            A = delta + self.gamma * self.lam * A
            advantages.insert(0, A)
            R = values[step]
        return advantages

    def apply_normalizer(self, adv):
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def trainIters(self, env, a2c_net, crit_coeff=1, ent_coeff=0, decrease_lr=0.997, max_tries=100, plot=False):
        a2c_net.train()

        decrease_lr = 1
        print("ignoring dynamic learning rate")

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

            observations = []
            actions = []
            log_probs = []
            entropies = []
            values = []
            rewards = []
            dones = []

            for i in count():
                dist, value = a2c_net(state)

                action = dist.sample()
                next_state, reward, done, _ = env.step(action.cpu().numpy())

                # sum up the log_probs/value where there are agents
                mask = state.x[:,env.IS_ROBOT]
                log_prob = dist.log_prob(action)[mask.bool()].sum(-1).unsqueeze(0)
                entr = dist.entropy()[mask.bool()].mean(-1).unsqueeze(0)
                value = value[mask.bool()].sum(-1)

                observations.append(state)
                actions.append(action)
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
            next_dist, next_value = a2c_net(next_state)
            adv = self.compute_advantages(next_value, dones, rewards, values)
            adv = torch.cat(adv).detach()
            norm_adv = self.apply_normalizer(adv)

            # TODO
            prob_ratio = torch.ones(log_probs.size()) #torch.exp(next_log_probs) / torch.exp(log_probs)
            clipped_ratio = prob_ratio.clamp(min=1.0 - self.epsilon, max=1.0 + self.epsilon)
            actor_loss = -1*torch.min(prob_ratio * norm_adv, clipped_ratio * norm_adv).mean()

            rewards_to_go = norm_adv + values

            entropy = torch.cat(entropies).sum()

            value_pred_clipped = values + (next_value - values).clamp(-self.epsilon, self.epsilon)
            value_losses = (next_value - rewards_to_go) ** 2
            value_losses_clipped = (value_pred_clipped - rewards_to_go) ** 2
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
            critic_loss = value_loss.mean()

            shared_loss = actor_loss + crit_coeff * critic_loss - ent_coeff * entropy

            wandb.log({"actor_loss": actor_loss})
            wandb.log({"critic_loss": critic_loss})
            wandb.log({"entropy": entropy})
            wandb.log({"shared_loss": shared_loss})
            wandb.log({"explore_time": i+1})
            wandb.log({"sum_reward":torch.sum(torch.cat(rewards))})
            wandb.watch(a2c_net)

            optimizer.zero_grad()
            shared_loss.backward()
            torch.nn.utils.clip_grad_norm_(a2c_net.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

    def play(self, env, a2c_net, max_tries=50, v=False):
        a2c_net.eval()

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
