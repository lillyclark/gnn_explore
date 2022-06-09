import mdptoolbox
import numpy as np
from Networks import *
from Environments import *
import torch
import time
import wandb
from MDP import *

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = GraphEnv(reward_name = "base_reward")

transitions, rewards, visited_index = get_model(env)

policy = solve(transitions, rewards, discount=0.99)

test_optimal_policy(env, visited_index, policy)

wandb.init(project="MDP-learn", entity="lillyclark", config={})
wandb.run.name = "tmp_"+wandb.run.id

# actor = SimpleActor(env.num_node_features, env.num_nodes, env.num_actions).to(device)
actor = GCNActor(env.num_node_features, env.num_actions).to(device)
# actor = LinearAggActor(env.num_node_features, env.num_actions).to(device)
# actor = GGNNActor(env.num_node_features, env.num_actions).to(device)

train_agent(env, actor, visited_index, policy, max_tries=500, n_iters=10)

wandb.finish()

test_learned_policy(env, actor, visited_index, policy)
