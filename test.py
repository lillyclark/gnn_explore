import torch
from Networks import *
from Environments import TestEnv, GraphEnv
from Policies import A2C, Graph_A2C

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env = TestEnv()
# state_size = env.state.shape[0]
# action_size = env.num_actions
# print("state_size",state_size)
# print("action_size",action_size)
#
# actor = Actor(state_size, action_size).to(device)
# critic = Critic(state_size, action_size).to(device)
# A2C = A2C(device=device, n_iters=500, lr=0.001, gamma=0.99)
# A2C.trainIters(env, actor, critic)
# A2C.play(env, actor)

env = GraphEnv(reward_name = "base_reward")
actor = GCNActor(env.num_node_features, env.num_actions).to(device)
critic = GCNCritic(env.num_node_features).to(device)
A2C = Graph_A2C(device=device, n_iters=1000000, lr=0.001, gamma=0.9)
A2C.trainIters(env, actor, critic, max_tries=100, plot=True)
print("GCN with 7 convolutions, lr 0.001, gamma 0.9, a million iters")
A2C.play(env, actor, max_tries=100)
print("GCN with 7 convolutions, lr 0.001, gamma 0.9, a million iters")

# env = GraphEnv(reward_name = "robot_reward")
# actor = GGNNActor(env.num_node_features, env.num_actions).to(device)
# critic = GGNNCritic(env.num_node_features).to(device)
# A2C = Graph_A2C(device=device, n_iters=500, lr=0.001, gamma=0.9)
# A2C.trainIters(env, actor, critic, max_tries=100, plot=True)
# A2C.play(env, actor, max_tries=100)

# env = GraphEnv(reward_name = "robot_reward")
# actor = SimpleActor(env.num_node_features, env.num_nodes, env.num_actions).to(device)
# critic = SimpleCritic(env.num_node_features, env.num_nodes).to(device)
# A2C = Graph_A2C(device=device, n_iters=500, lr=0.001, gamma=0.9)
# A2C.trainIters(env, actor, critic, max_tries=100, plot=True)
# A2C.play(env, actor, max_tries=100)

# env = GraphEnv(reward_name = "robot_reward")
# actor = LinearAggActor(env.num_node_features, env.num_actions).to(device)
# critic = LinearAggCritic(env.num_node_features).to(device)
# A2C = Graph_A2C(device=device, n_iters=1000, lr=0.001, gamma=0.9)
# A2C.trainIters(env, actor, critic, max_tries=100, plot=True)
# A2C.play(env, actor, max_tries=100)
