import torch
from Networks import *
from Environments import TestEnv, GraphEnv
from Policies import A2C, Graph_A2C

# seed for consistent results
np.random.seed(0)
torch.manual_seed(0)

# use cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# world from Environments.py
env = GraphEnv(reward_name = "base_reward", has_master = False)

# NN model from Networks.py (choose one)
# actor = SimpleActor(env.num_node_features, env.num_nodes, env.num_actions).to(device)
# actor = GCNActor(env.num_node_features, env.num_actions).to(device)
# actor = GGNNActor(env.num_node_features, env.num_actions).to(device)
actor = LinearAggActor(env.num_node_features, env.num_actions).to(device)

# Adam for gradient descent
optimizerA = optim.Adam(actor.parameters(), lr=0.001)

# acts as the optimal policies
def compute_target(state,env):
    # desired probability of going [left, right, stay] from each node
    right_dist = torch.Tensor([[0,0,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[1,0,0]])
    left_dist = torch.Tensor([[0,0,1],[0,0,1],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]])
    if state.x[-1][env.IS_KNOWN_ROBOT]:
        # if the robot has visited the last node
        return left_dist
    return right_dist

# train
max_tries = 500
losses = []
for iter in range(100):
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

# test
state = env.reset()
print("state:",state.x[:,env.IS_ROBOT].numpy())

for i in range(max_tries):
    dist = actor(state)

    action = dist.sample()
    print("action:",action)
    next_state, reward, done, _ = env.step(action.cpu().numpy())
    if reward:
        print("reward:",reward)

    state = next_state
    print("state:",state.x[:,env.IS_ROBOT].numpy())
    if done:
        print('Done in {} steps'.format(i+1))
        break
