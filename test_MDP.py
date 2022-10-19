import mdptoolbox
import numpy as np
from Networks import *
from Environments import *
import torch
import time
import wandb
from MDP import *
import testEnv
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### DEFINE ENVIRONMENT
envTest = testEnv.testEnv()
# envTest.createEnv("B")
# print(chr(ord('B')+1))
# exit()
envTest.setEnv("A")
envTest.showGraph()
graph = envTest.getGraph()
plt.clf()
nx.draw_spring(graph, with_labels=True)
plt.show()
# exit()
num_nodes = envTest.num_nodes
right_i = envTest.right_i
right_j = envTest.right_j
env = ConfigureEnv(num_robots=envTest.num_robots, num_actions=envTest.num_actions, num_nodes=num_nodes, right_i=right_i, right_j=right_j)

policy_name = envTest.policy_name
model_name = envTest.model_name
model_name = 'EnvA_test.pt'

mode = ['write_policy']
# mode = ['read_policy', 'train', 'write', 'test']
# mode = ['read_policy', 'read', 'test']
# mode = ['read_policy']

if 'write_policy' in mode:
    transitions, rewards, visited_index, action_index = get_model(env)
    policy = solve(transitions, rewards, discount=0.99)
    test_optimal_policy(env, visited_index, action_index, policy, graph)
    save_optimal_policy(visited_index, action_index, policy, policy_name)

if 'read_policy' in mode:
    visited_index, action_index, policy = load_optimal_policy(policy_name)
    test_optimal_policy(env, visited_index, action_index, policy, graph)

# actor = SimpleActor(env.num_node_features, env.num_nodes, env.num_actions).to(device)
actor = GCNActor(env.num_node_features, env.num_actions)#.to(device)
# actor = LinearAggActor(env.num_node_features, env.num_actions).to(device)
# actor = GGNNActor(env.num_node_features, env.num_actions).to(device)

if 'read' in mode:
    n = torch.load("models/"+model_name)
    actor.load_state_dict(n)

if 'train' in mode:
    wandb.init(project="MDP-medium-world", entity="dmdsouza", config={})
    wandb.run.name = model_name.split(".")[0]+"_"+wandb.run.id
    optimizer = optim.Adam(actor.parameters(), lr=0.001)
    train_agent(env, actor, optimizer, visited_index, action_index, policy, max_tries=500, n_iters=150)
    wandb.finish()

if 'write' in mode:
    torch.save(actor.state_dict(), "models/"+model_name)

if 'test' in mode:
    test_learned_policy(env, actor, visited_index, action_index, policy, graph)
