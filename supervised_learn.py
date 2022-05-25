import torch
from Networks import *
from Environments import TestEnv, GraphEnv
from Policies import A2C, Graph_A2C

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = GraphEnv(reward_name = "base_reward", has_master = False)
# actor = SimpleActor(env.num_node_features, env.num_nodes, env.num_actions).to(device)
# actor = GCNActor(env.num_node_features, env.num_actions).to(device)
actor = GGNNActor(env.num_node_features, env.num_actions).to(device)
# actor = LinearAggActor(env.num_node_features, env.num_actions).to(device)
optimizerA = optim.Adam(actor.parameters(), lr=0.001)

def compute_target(state,env):
    right_dist = torch.Tensor([[0,0,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[1,0,0]])
    left_dist = torch.Tensor([[0,0,1],[0,0,1],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]])
    if state.x[-1][env.IS_KNOWN_ROBOT]:
        return left_dist
    return right_dist

max_tries = 500
losses = []
for iter in range(200):
    state = env.reset()
    for i in count():
        dist = actor(state)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy())

        target = compute_target(state,env)
        ce = torch.nn.CrossEntropyLoss()
        actor_loss = ce(dist.probs,target)
        losses.append(actor_loss.item())

        optimizerA.zero_grad()
        actor_loss.backward()
        optimizerA.step()

        state = next_state

        if done or i == max_tries:
            print(f'Iter: {iter}, Steps: {i+1}, Loss: {actor_loss.item()}')
            break

A2C = Graph_A2C(device=device, n_iters=100, lr=0.001, gamma=0.9)
A2C.play(env, actor, critic=None, max_tries=100)
