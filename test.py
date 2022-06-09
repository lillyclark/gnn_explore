import torch
from Networks import *
from Environments import TestEnv, GraphEnv
from Policies import *
import wandb

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mode = ['read','train','test','write']
# mode = ['test']
# mode = ['read','test']
mode = ['train','test']
# mode = ['train','test','write']

RUN_NAME = "GCN_A2C_"

wandb.init(project="smaller-world", entity="lillyclark", config={})
wandb.run.name = RUN_NAME+wandb.run.id

model_name = 'models/'+RUN_NAME+'_policy_gradient.pt'

env = GraphEnv(reward_name = "base_reward", has_master = False)

# actor = SimpleActor(env.num_node_features, env.num_nodes, env.num_actions).to(device)
# actor = GCNActor(env.num_node_features, env.num_actions).to(device)
actor = GCNA2C(env.num_node_features, env.num_actions).to(device)
# actor = LinearAggActor(env.num_node_features, env.num_actions).to(device)
# actor = GGNNActor(env.num_node_features, env.num_actions).to(device)

# agent = PolicyGradient(device=device, n_iters=2000, lr=0.005, gamma=0.99)
agent = A2C_Shared(device=device, n_iters=1000, lr=0.005, gamma=0.99)


if 'read' in mode:
    n = torch.load(model_name)
    actor.load_state_dict(n)

if 'train' in mode:
    agent.trainIters(env, actor, max_tries=500, plot=False)

if 'write' in mode:
    torch.save(actor.state_dict(), model_name)

if 'test' in mode:
    agent.play(env, actor, max_tries=100, v=True)

wandb.finish()
