from collections import deque
import numpy as np
import torch
import random
import gc
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class DeepQ(object):
    def __init__(self, environment):
        self.batch = 4 #64
        self.replay_memory = 1000
        self.gamma = 0.99
        self.observe = self.batch
        self.explore = 1000
        self.epoch = 10000
        self.target_update = 10

        self.initial_eps = 0.5 #0.9
        self.final_eps = 0
        self.max_grad_norm = 0.5

        self.buffer = deque()

        self.step_t = 0
        self.epsilon = self.initial_eps
        self.temp_loss = 0
        self.total_reward = np.empty([0,0])

        self.environment_class = environment
        self.environment = self.environment_class()

        self.losses = []

    def running(self, policy_net, target_net, test=False):
        target_net.eval()
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-5)
        loss_data = []

        last_done = 0

        for i in range(self.epoch):
            self.step_t += 1
            # TODO scale down epsilon

            s_t = self.environment.to_state()
            rewards = self.environment.get_rewards()

            # explore
            if random.random() < self.epsilon:
                model_out = np.random.random(self.environment.num_edges)
                a_t = self.environment.value_to_action(s_t, model_out)
            else:
                model_out = self.predict(s_t, policy_net).detach().numpy()
                a_t = self.environment.value_to_action(s_t, model_out)

            r_t = rewards[np.argmax(a_t)]

            is_terminal = self.environment.take_step(a_t)
            s_t1 = self.environment.to_state()

            self.buffer.append((s_t, a_t, r_t, s_t1, is_terminal))

            if len(self.buffer) > self.replay_memory:
                self.buffer.popleft()

            if self.step_t > self.observe:

                if self.step_t % self.target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # if self.step_t % 4 < 3:
                #     continue

                # print("training",self.step_t)
                minibatch = random.sample(self.buffer, self.batch)
                s_batch_loader = DataLoader([exp[0] for exp in minibatch], batch_size=self.batch)
                s1_batch_loader = DataLoader([exp[3] for exp in minibatch], batch_size=self.batch)
                for batch in s_batch_loader:
                    s_batch = batch
                for batch in s1_batch_loader:
                    s1_batch = batch

                target_batch = self.predict(s_batch, target_net).detach().numpy()
                target_batch = np.array(target_batch).reshape(self.batch,-1)

                Q_future_batch = self.predict(s1_batch, target_net).detach().numpy()

                a_batch = [exp[1] for exp in minibatch]
                r_batch = [exp[2] for exp in minibatch]
                was_terminal_batch = [exp[4] for exp in minibatch]

                for j in range(len(minibatch)):
                    Q_future = np.max(Q_future_batch[j])
                    if was_terminal_batch[j]:
                        eff_reward = r_batch[j]
                    else:
                        eff_reward = r_batch[j] + self.gamma * Q_future_batch[j]
                    for possible_edge in range(len(a_batch[j])):
                        if a_batch[j][possible_edge]: # actually took this edge
                            target_batch[j][possible_edge] = eff_reward

                self.train(s_batch, a_batch, target_batch, policy_net, optimizer)
                loss_data.append([self.step_t, self.current_loss])

            status = self.environment.get_status()
            # if i % 10 == 0:
            #     print("TIMESTEP", self.step_t, "/ Q_MAX %e" % np.max(model_out), "/ EXPLORED", status, "/ REWARD", r_t,
            #           "/ Terminal", is_terminal, "\n")

            if is_terminal:
                print("done in",i-last_done,"steps")
                print("TIMESTEP", self.step_t, "/ Q_MAX %e" % np.max(model_out), "/ EXPLORED", status, "/ REWARD", r_t,
                      "/ Terminal", is_terminal, "\n")
                last_done = i
                del self.environment
                gc.collect()
                self.environment = self.environment_class()

            # TODO save progress
            pass

        # TODO save results
        pass

        print(self.losses)
        plt.plot(self.losses)
        plt.show()

    def train(self, state, action, target, model, optimizer):
        model.train()
        optimizer.zero_grad()
        prediction = model(state, 0.5, batch = state.batch)
        target = torch.tensor(target)
        loss = torch.pow(prediction.view(-1) - target.view(-1), 2).sum() / self.batch
        self.current_loss = loss.item()
        self.losses.append(self.current_loss)
        loss.backward()
        # for param in model.parameters():
            # param.grad.data.clamp_(-self.max_grad_norm, self.max_grad_norm)
        optimizer.step()

    def predict(self, state, model):
        model.eval()
        prob = 0.0
        return model(state, prob)
