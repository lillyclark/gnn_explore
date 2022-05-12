import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.autograd as autograd

class TestEnv():
    def __init__(self):
        self.num_actions = 2
        self.state = np.array([1,0,0,0])
        self.goal_state = np.array([0,0,0,1])

    def reset(self):
        self.state = np.array([1,0,0,0])
        return self.state

    def q_to_a(self, state, q):
        return torch.argmax(q,dim=1).unsqueeze(1)

    def step(self,action):
        step = np.argmax(action)
        pos = np.argmax(self.state)
        delta = -1 if step == 0 else 1
        new_pose = pos + delta
        new_pose = min(max(new_pose,0),len(self.state)-1)
        self.state = np.zeros(len(self.state))
        self.state[new_pose] = 1

        reward, done = 0, False
        if (self.state == self.goal_state).all():
            reward, done = 10, True
        return self.state, reward, done, None

    def get_random_action(self):
        random_q = torch.normal(0, 1, size=(1,2))
        return self.q_to_a(self.state, random_q)[0]

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        # print("initial state:", state)
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            # print("output of step:", next_state, reward, done)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward) + " after " + str(step+1) + " steps")
                break

            state = next_state

    return episode_rewards

def play(env, agent):
    state = env.reset()
    print("state:", state)
    for i in range(100):
        action = agent.get_action(state,eps=0.0)
        print("action:", action)
        next_state, reward, done, _ = env.step(action)
        print("state:", state)
        if done:
            print("done :)",i)
            break
        state = next_state

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch), np.array(done_batch))

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals

class DQNAgent:

    def __init__(self, env, gamma=0.99, buffer_size=10000):
        self.env = env
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        print("env.observation_space.shape:",env.state.shape)
        print("env.action_space.n:",env.num_actions)

        self.model = DQN(env.state.shape, env.num_actions).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.50):
        if(np.random.random() < eps):
            random_action = self.env.get_random_action()
            # print("get_action random action:",random_action)
            return random_action

        state_raw = state
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        # print("qvals:",qvals)
        action = env.q_to_a(state_raw, qvals)[0]
        # action = np.argmax(qvals.cpu().detach().numpy())
        # print("get_action action:",action)
        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_out = self.model.forward(states)
        print("curr_out",curr_out)
        print("actions", actions)
        curr_Q = curr_out[actions.bool()]
        print("curr_Q",curr_Q)

        next_out = self.model.forward(next_states)
        # print(next_out)
        next_actions = self.env.q_to_a(next_states,next_out)
        # print(next_actions)
        max_next_Q = next_out.gather(1, next_actions.unsqueeze(1))

        expected_Q = rewards.squeeze(1) + (1 - dones) * self.gamma * max_next_Q.squeeze(1)

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

MAX_EPISODES = 100
MAX_STEPS = 500
BATCH_SIZE = 1 #32

env = TestEnv()
agent = DQNAgent(env,gamma=0.9)
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)

play(env, agent)
