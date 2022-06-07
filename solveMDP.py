import mdptoolbox
import numpy as np
from Networks import *
from Environments import *
import torch
import time

env = GraphEnv(reward_name = "base_reward")

state = env.reset()
s = state.x

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

# for a_idx in range(transitions.shape[0]):
#     transitions[a_idx] = np.identity(n_states)
#     for u_idx in range(transitions.shape[1]):
#         tmp_row = np.zeros(transitions.shape[2])
#         for v_idx in range(transitions.shape[2]):
#             try:
#                 tmp_row[v_idx] = t_dict[a_idx][u_idx][v_idx]
#             except KeyError:
#                 pass
#         if np.sum(tmp_row) == 1:
#             transitions[a_idx][u_idx] = tmp_row

# print(t_dict)

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

discount = 0.99
start = time.time()
mdp = mdptoolbox.mdp.PolicyIteration(transitions,
                                    rewards,
                                    discount,
                                    policy0=None,
                                    max_iter=1000,
                                    eval_type=0)
mdp.run()
print(f"Done solving MDP in {time.time()-start} secs")

print(mdp.policy)


##### PLAY
state = env.reset()
print("pose:",state.x[:,env.IS_ROBOT].numpy())

for i in range(50):
    s = state.x
    st = tuple(s.flatten().numpy())
    s_idx = visited_index[st]
    action_idx = mdp.policy[s_idx]
    action = torch.zeros(env.num_nodes).long()
    for n in range(env.num_nodes):
        if s[n,env.IS_ROBOT]:
            action[n] = action_idx

    print("action:",action)
    next_state, reward, done, _ = env.step(action.cpu().numpy())
    if reward:
        print("reward:",reward)
        print("***")

    state = next_state
    print("state:",state.x[:,env.IS_ROBOT].numpy())
    if done:
        print('Done in {} steps'.format(i+1))
        break
