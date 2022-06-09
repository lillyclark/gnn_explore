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

env = BranchEnv()
# env = GraphEnv()
policy_name = "branchworld.p"
model_name = "branchworld.pt"

mode = ['read_policy', 'train', 'write', 'test']
# mode = ['read_policy', 'read', 'test']
# mode = ['write_policy']

if 'write_policy' in mode:
    transitions, rewards, visited_index = get_model(env)
    policy = solve(transitions, rewards, discount=0.99)
    test_optimal_policy(env, visited_index, policy)
    save_optimal_policy(visited_index, policy, policy_name)

if 'read_policy' in mode:
    visited_index, policy = load_optimal_policy(policy_name)

# actor = SimpleActor(env.num_node_features, env.num_nodes, env.num_actions).to(device)
actor = GCNActor(env.num_node_features, env.num_actions).to(device)
# actor = LinearAggActor(env.num_node_features, env.num_actions).to(device)
# actor = GGNNActor(env.num_node_features, env.num_actions).to(device)

if 'read' in mode:
    n = torch.load("models/"+model_name)
    actor.load_state_dict(n)

if 'train' in mode:
    wandb.init(project="MDP-learn", entity="lillyclark", config={})
    wandb.run.name = "gcnfast_"+wandb.run.id
    train_agent(env, actor, visited_index, policy, max_tries=500, n_iters=1000)
    wandb.finish()

if 'write' in mode:
    torch.save(actor.state_dict(), "models/"+model_name)

if 'test' in mode:
    test_learned_policy(env, actor, visited_index, policy)
