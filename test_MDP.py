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

#### DEFINE ENVIRONMENT
num_nodes = 11
right_i = torch.Tensor([0,0,1,2,3,4,5,6,6,7,8])
right_j = torch.Tensor([1,2,3,4,5,5,6,7,8,9,10])
env = ConfigureEnv(num_robots=2, num_actions=4, num_nodes=num_nodes, right_i=right_i, right_j=right_j)

policy_name = "Env_daniel.p"
model_name = "Env_daniel.pt"
# model_name = "EnvA_simple.pt"


mode = ['write_policy']
# mode = ['read_policy']
# mode = ['read_policy', 'train', 'write', 'test']
# mode = ['read_policy', 'read', 'train', 'write', 'test']
# mode = ['read_policy', 'train', 'test']

if 'write_policy' in mode:
    transitions, rewards, visited_index, action_index = get_model(env)
    index_action = {v:k for k,v in action_index.items()}
    policy = solve(transitions, rewards, discount=0.99)
    test_optimal_policy(env, visited_index, index_action, policy)
    save_optimal_policy(visited_index, action_index, policy, policy_name)

if 'read_policy' in mode:
    visited_index, index_action, policy = load_optimal_policy(policy_name)
    test_optimal_policy(env, visited_index, index_action, policy)

# actor = SimpleActor(env.num_node_features, env.num_nodes, env.num_actions).to(device)
actor = GCNActor(env.num_node_features, env.num_actions).to(device)
actor.k = 2
# actor = LinearAggActor(env.num_node_features, env.num_actions).to(device)
# actor = GGNNActor(env.num_node_features, env.num_actions).to(device)

if 'read' in mode:
    n = torch.load("models/"+model_name)
    actor.load_state_dict(n)

if 'train' in mode:
    wandb_log = False
    if wandb_log:
        wandb.init(project="MDP-medium-world", entity="lillyclark", config={})
        wandb.run.name = model_name.split(".")[0]+"_"+wandb.run.id
    optimizer = optim.Adam(actor.parameters(), lr=0.001)
    train_agent(env, actor, optimizer, visited_index, index_action, policy, max_tries=500, n_iters=1000, wandb_log=wandb_log)
    if wandb_log:
        wandb.finish()

if 'write' in mode:
    torch.save(actor.state_dict(), "models/"+model_name)

if 'test' in mode:
    test_learned_policy(env, actor, visited_index, index_action, policy)
