import torch
from Networks import *
from Environments import TestEnv, GraphEnv
from Policies import A2C, Graph_A2C

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = ['train','test','write']
# mode = ['read']
# mode = ['read','train','test','write']
# mode = ['train','test']

a_name = 'models/sf6_gcn_a.pt'
c_name = 'models/sf6_gcn_c.pt'
a_name1 = 'models/sf6_gcn_a.pt'
c_name1 = 'models/sf6_gcn_c.pt'

# a_name = 'models/s_a.pt'
# c_name = 'models/s_c.pt'
# a_name1 = 'models/s_a.pt'
# c_name1 = 'models/s_c.pt'

env = GraphEnv(reward_name = "base_reward", has_master = False)

actor = GCNActor(env.num_node_features, env.num_actions).to(device)
critic = GCNCritic(env.num_node_features).to(device)
# actor = SimpleActor(env.num_node_features, env.num_nodes, env.num_actions).to(device)
# critic = SimpleCritic(env.num_node_features, env.num_nodes).to(device)

A2C = Graph_A2C(device=device, n_iters=5000, a_lr=0.0001, c_lr=0.001, gamma=0.99)

if 'read' in mode:
    a = torch.load(a_name)
    c = torch.load(c_name)
    actor.load_state_dict(a)
    critic.load_state_dict(c)

if 'train' in mode:
    A2C.trainIters(env, actor, critic, max_tries=500, plot=True)

if 'write' in mode:
    torch.save(actor.state_dict(), a_name1)
    torch.save(critic.state_dict(), c_name1)

if 'test' in mode:
    A2C.play(env, actor, critic, max_tries=100, v=True)
