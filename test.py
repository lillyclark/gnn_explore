import torch
from Networks import *
from Environments import TestEnv, GraphEnv
from Policies import A2C, Graph_A2C

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mode = ['train','write']
# mode = ['read']
# mode = ['read','train','write']
mode = ['train','test']

a_name = 'models/sa_.pt'
c_name = 'models/sc_.pt'
a_name1 = 'models/sa_.pt'
c_name1 = 'models/sc_.pt'

env = GraphEnv(reward_name = "base_reward", has_master = False)

# actor = GCNActor(env.num_node_features, env.num_actions).to(device)
# critic = GCNCritic(env.num_node_features).to(device)
actor = SimpleActor(env.num_node_features, env.num_nodes, env.num_actions).to(device)
critic = SimpleCritic(env.num_node_features, env.num_nodes).to(device)

A2C = Graph_A2C(device=device, n_iters=100, lr=0.001, gamma=0.9)

if 'read' in mode:
    a = torch.load(a_name)
    c = torch.load(c_name)
    actor.load_state_dict(a)
    critic.load_state_dict(c)

if 'train' in mode:
    A2C.trainIters(env, actor, critic, max_tries=500, plot=False)

if 'write' in mode:
    torch.save(actor.state_dict(), a_name1)
    torch.save(critic.state_dict(), c_name1)

if 'test' in mode:
    A2C.play(env, actor, critic, max_tries=100)
