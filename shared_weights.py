import torch
from Networks import *
from Environments import TestEnv, GraphEnv
from Policies import A2C, Graph_A2C, A2C_Shared

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mode = ['read','train','test','write']
# mode = ['read','test']
# mode = ['read','train','test']
# mode = ['train','test','write']
mode = ['train','test']

RUN_NAME = "no_entropy"

a2c_name = 'models/'+RUN_NAME+'_a2c.pt'

env = GraphEnv(reward_name = "base_reward", has_master = False)

# a2c_net = SimpleA2C(env.num_node_features, env.num_nodes, env.num_actions).to(device)
a2c_net = GCNA2C(env.num_node_features, env.num_actions).to(device)
A2C = A2C_Shared(device=device, n_iters=1000, lr=0.001, gamma=0.99, run_name=RUN_NAME)

if 'read' in mode:
    n = torch.load(a2c_name)
    a2c_net.load_state_dict(n)

if 'train' in mode:
    A2C.trainIters(env, a2c_net, crit_coeff=0.25, ent_coeff=0.0, max_tries=500, plot=False)

if 'write' in mode:
    torch.save(a2c_net.state_dict(), a2c_name)

if 'test' in mode:
    A2C.play(env, a2c_net, max_tries=100, v=True)
