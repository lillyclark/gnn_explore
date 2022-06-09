import torch
from Networks import SimpleA2C
from Environments import TestEnv, GraphEnv
from PPO import PPO
import wandb
import numpy as np

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

mode = ['train','test']

RUN_NAME = "smaller-world"
run = wandb.init(project="ppo-gnn-explore", entity="lillyclark", config={})
run.name = RUN_NAME+run.id

model_name = 'models/'+RUN_NAME+'_ppo.pt'

env = GraphEnv(reward_name = "base_reward", has_master = False)

net = SimpleA2C(env.num_node_features, env.num_nodes, env.num_actions).to(device)
# net = GCNA2C(env.num_node_features, env.num_actions).to(device)
ppo = PPO(device=device,
            env=env,
            net=net,
            lr=0.001,
            gamma=0.99,
            lam=0.95,
            eps=0.2,
            crit_coeff=1.0,
            ent_coeff=0.0)

if 'read' in mode:
    n = torch.load(model_name)
    net.load_state_dict(n)

if 'train' in mode:
    ppo.trainIters(n_iters=1000, max_tries=500, plot=False)

if 'write' in mode:
    torch.save(net.state_dict(), model_name)

if 'test' in mode:
    ppo.play(max_tries=100, v=True)

wandb.finish()
