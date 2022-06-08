import mdptoolbox
import numpy as np
from Networks import *
from Environments import *
import torch
import time

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = GraphEnv(reward_name = "base_reward")

state = env.reset()
s = state.x

#### CREATE TRANSITION AND REWARD MODELS
Q = [s]
visited_index = {}
t_dict = dict([(x,{}) for x in range(env.num_actions)])
r_dict = dict([(x,{}) for x in range(env.num_actions)])

count = 0
start = time.time()
done = False
while Q: # and not done and count < 10000:
    count += 1
    u = Q.pop(0)

    if u not in visited_index:
        tu = tuple(u.flatten().numpy())
        visited_index[tu] = len(visited_index)
        a = torch.zeros(env.num_nodes).long()

        for n in range(env.num_nodes):
            if u[n,env.IS_ROBOT]: # TODO
                for a_index in range(env.num_actions):
                    a_ = a.detach().clone()
                    a_[n] = a_index

                    env.set_features(u)
                    state, reward, done, _ = env.step(a_.numpy())
                    v = state.x
                    tv = tuple(v.flatten().numpy())

                    t_dict[a_index].setdefault(tu, {})[tv] = 1
                    r_dict[a_index].setdefault(tu, {})[tv] = reward

                    if tv not in visited_index:
                        Q.append(v)

                    # if done:
                    #     print("found a solution")
                    #     # print(a_index, visited_index[tu])
                    #     # break

    else:
        print("u in visited_index")

# if stopped early
if Q:
    print("stopped early")
for s in Q:
    ts = tuple(s.flatten().numpy())
    visited_index[ts] = len(visited_index)

print(f"Done searching in {time.time()-start} secs")
n_states = len(visited_index)
print(f"{n_states} states were considered exhaustively")

transitions = np.zeros((env.num_actions, n_states, n_states))
rewards = np.zeros((env.num_actions, n_states, n_states))

for a in t_dict:
    for u in t_dict[a]:
        for v in t_dict[a][u]:
            transitions[a, visited_index[u], visited_index[v]] = t_dict[a][u][v]

for a in r_dict:
    for u in r_dict[a]:
        for v in r_dict[a][u]:
            rewards[a, visited_index[u], visited_index[v]] = r_dict[a][u][v]

print("transitions:", transitions.shape)
print("rewards:", rewards.shape)

#### SOLVE FOR MDP POLICY
discount = 0.99
start = time.time()
mdp = mdptoolbox.mdp.PolicyIteration(transitions,
                                    rewards,
                                    discount,
                                    policy0=None,
                                    max_iter=100,
                                    eval_type=0)
mdp.run()
print(f"Done solving MDP in {time.time()-start} secs")
policy = mdp.policy

def compute_target(state, env, visited_index, policy):
    s = state.x
    st = tuple(s.flatten().numpy())
    s_idx = visited_index[st]
    action_idx = policy[s_idx]
    action = torch.ones(env.num_nodes).long()
    for n in range(env.num_nodes): # TODO
        if s[n,env.IS_ROBOT]:
            action[n] = action_idx
    dist = torch.nn.functional.one_hot(action, num_classes=env.num_actions).float()
    return dist

def TEST_compute_target(state, env, visited_index, policy):
    # desired probability of going [left, right, stay] from each node
    right_dist = torch.Tensor([[0,0,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[1,0,0]])
    left_dist = torch.Tensor([[0,0,1],[0,0,1],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]])
    if state.x[-1][env.IS_KNOWN_ROBOT]:
        # if the robot has visited the last node
        return left_dist
    return right_dist

##### TEST MDP BEFORE TRAINING
print("test MDP policy")
state = env.reset()
print("pose:",state.x[:,env.IS_ROBOT].numpy())

for i in range(50):
    dist = compute_target(state, env, visited_index, policy)
    action = torch.argmax(dist,1)
    print("action:",action)
    next_state, reward, done, _ = env.step(action.cpu().numpy())
    if reward:
        print("reward:",reward)
        print("***")

    state = next_state
    print("pose:",state.x[:,env.IS_ROBOT].numpy())
    if done:
        print('Done in {} steps'.format(i+1))
        break

##### TRAIN
print("Train NN agent")
actor = SimpleActor(env.num_node_features, env.num_nodes, env.num_actions).to(device)
optimizerA = optim.Adam(actor.parameters(), lr=0.0001)

max_tries = 500
losses = []
for iter in range(500):
    state = env.reset()

    for i in range(max_tries):
        dist = actor(state)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy())

        target = compute_target(state, env, visited_index, policy)
        ce = torch.nn.CrossEntropyLoss()

        mask = state.x[:,env.IS_ROBOT].bool()
        actor_loss = ce(dist.probs[mask],target[mask])
        losses.append(actor_loss.item())

        optimizerA.zero_grad()
        actor_loss.backward()
        optimizerA.step()

        state = next_state

        if done:
            print(f'Iter: {iter}, Steps: {i+1}, Loss: {actor_loss.item()}')
            break

    if not done:
        print(f'Iter: {iter}, Steps: {i+1}, Loss: {actor_loss.item()}')


#### PLAY
print("")
print("PLAYING WITH LEARNED POLICY")
state = env.reset()
print("pose:", state.x[:,env.IS_ROBOT].numpy())

for i in range(50):
    dist = actor(state)
    target = compute_target(state, env, visited_index, policy)
    print(np.round(dist.probs.detach().numpy().T,2))
    print(np.round(target.numpy().T,2))
    action = dist.sample()
    for n in range(env.num_nodes):
        if state.x[n,env.IS_ROBOT]:
            print("action:",action.numpy()[n])
    next_state, reward, done, _ = env.step(action.cpu().numpy())
    if reward:
        print("reward:",reward)
        print("***")
    print(" ")
    state = next_state
    print("pose:",state.x[:,env.IS_ROBOT].numpy())
    if done:
        print('Done in {} steps'.format(i+1))
        break
