import torch
from Networks import *
from Environments import TestEnv, GraphEnv
from Policies import A2C, Graph_A2C, A2C_Shared
from PPO import PPO

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# mode = ['read','train','test','write']
# mode = ['read','test']
# mode = ['read','train','test']
# mode = ['train','test','write']
mode = ['train']#,'test']

RUN_NAME = "debug_critic_loss"

model_name = 'models/'+RUN_NAME+'_ppo.pt'

env = GraphEnv(reward_name = "base_reward", has_master = False)
# env = FeatureEnv()

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
            ent_coeff=0.0,
            run_name=RUN_NAME)

if 'read' in mode:
    n = torch.load(model_name)
    net.load_state_dict(n)

if 'train' in mode:
    ppo.trainIters(n_iters=1000, max_tries=500, plot=False)

if 'write' in mode:
    torch.save(net.state_dict(), model_name)

if 'test' in mode:
    ppo.play(max_tries=100, v=True)
