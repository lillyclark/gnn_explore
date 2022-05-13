import torch
import numpy as np
from Networks import Actor, Critic
from Environments import TestEnv
import torch.optim as optim
from itertools import count
import matplotlib.pyplot as plt


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
trainIters(actor, critic, n_iters=200, lr=lr)
play(actor)
