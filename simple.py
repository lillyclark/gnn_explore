import torch
from Networks import Actor, Critic
from Environments import TestEnv
from Policies import A2C

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TestEnv()
state_size = env.state.shape[0]
action_size = env.num_actions
print("state_size",state_size)
print("action_size",action_size)

actor = Actor(state_size, action_size).to(device)
critic = Critic(state_size, action_size).to(device)
A2C = A2C(device=device, n_iters=500, lr=0.001, gamma=0.99)
A2C.trainIters(env, actor, critic)
A2C.play(env, actor)
