import mdptoolbox
import numpy as np
from Networks import *
from Environments import *
import torch
import time
import wandb
import pickle
from itertools import product
from scipy.sparse import csr_matrix

#### CREATE TRANSITION AND REWARD MODELS
def get_model(env):
    # return transitions, rewards, visited_index
    state = env.reset()
    s = state.x

    action_index = {}
    action_generator = product(list(range(env.num_actions)), repeat=len(env.robots))
    for action_combo in list(action_generator):
        action_index[action_combo] = len(action_index)

    Q = [s]
    visited_index = {}

    t_dict = {action_combo: {} for action_combo in action_index}
    r_dict = {action_combo: {} for action_combo in action_index}

    start = time.time()
    done = False
    print("searching state space exhaustively...")
    e_break = 0
    while Q: # and e_break < 100:
        u = Q.pop(0)
        e_break += 1

        tu = tuple(u.flatten().numpy())
        if tu not in visited_index:
            visited_index[tu] = len(visited_index)
            a = torch.zeros(env.num_nodes).long()

            is_robot = env.is_robot(u)
            robot_nodes = torch.where(is_robot)[0]

            for action_combo in action_index:
                a_ = a.detach().clone()
                a_[robot_nodes] = torch.Tensor(action_combo).long()

                env.set_features(u)
                state, reward, done, _ = env.step(a_.numpy())
                v = state.x
                tv = tuple(v.flatten().numpy())

                t_dict[action_combo].setdefault(tu, {})[tv] = 1
                if reward:
                    r_dict[action_combo].setdefault(tu, {})[tv] = reward

                if tv not in visited_index:
                    Q.append(v)

        if e_break % 100 == 0:
            print('.')

    # if stopped early
    if Q:
        print("stopped early")
    for s in Q:
        ts = tuple(s.flatten().numpy())
        visited_index[ts] = len(visited_index)

    print(f"Done searching in {time.time()-start} secs")
    n_states = len(visited_index)
    print(f"{n_states} states were considered exhaustively")

    # n_states = 11776
    # transitions = np.zeros((len(action_index), n_states, n_states), dtype=np.uint8)
    # rewards = np.zeros((len(action_index), n_states, n_states), dtype=np.uint8)

    transitions = [csr_matrix((n_states, n_states), dtype=np.int8) for _ in range(len(action_index))]
    rewards = [csr_matrix((n_states, n_states), dtype=np.int8) for _ in range(len(action_index))]

    print("transitions shape:", transitions.shape)
    print("rewards shape:", rewards.shape)

    for a in t_dict:
        a_idx = action_index[a]
        for u in t_dict[a]:
            for v in t_dict[a][u]:
                transitions[a_idx][visited_index[u],visited_index[v]] = t_dict[a][u][v]

    for a in r_dict:
        a_idx = action_index[a]
        for u in r_dict[a]:
            for v in r_dict[a][u]:
                rewards[a_idx][visited_index[u],visited_index[v]] = r_dict[a][u][v]

    if transitions.sum(axis=1) != np.ones(transitions.shape[0]):
        print("stochastic error?")
        print(transitions.sum(axis=1))
        print("at index", torch.argmax(transitions.sum(axis=1)))

    return transitions, rewards, visited_index, action_index

#### SOLVE FOR MDP POLICY
def solve(transitions, rewards, discount=0.99):
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
    return policy

def save_optimal_policy(visited_index, action_index, policy, filename="policy.p"):
    print("saving optimal policy")
    policy_dict = {"visited_index":visited_index, "policy":policy, "action_index":action_index}
    pickle.dump(policy_dict, open("policies/"+filename, "wb"))

def load_optimal_policy(filename="policy.p"):
    print("loading optimal policy", filename)
    policy_dict = pickle.load(open("policies/"+filename, "rb"))
    index_action = {v:k for k,v in policy_dict["action_index"].items()}
    return policy_dict["visited_index"], index_action, policy_dict["policy"]

def compute_target(state, env, visited_index, index_action, policy):
    s = state.x
    st = tuple(s.flatten().numpy())
    s_idx = visited_index[st]
    action_idx = policy[s_idx]
    action_combo = index_action[action_idx]
    action = torch.zeros(env.num_nodes).long()
    is_robot = env.is_robot(s)
    robot_nodes = torch.where(is_robot)[0]
    action[robot_nodes] = torch.Tensor(action_combo).long()
    dist = torch.nn.functional.one_hot(action, num_classes=env.num_actions).float()
    return dist

# def TEST_compute_target(state, env, visited_index, policy):
#     # desired probability of going [left, right, stay] from each node
#     right_dist = torch.Tensor([[0,0,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[1,0,0]])
#     left_dist = torch.Tensor([[0,0,1],[0,0,1],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]])
#     if state.x[-1][env.IS_KNOWN_ROBOT]:
#         # if the robot has visited the last node
#         return left_dist
#     return right_dist

def test_optimal_policy(env, visited_index, index_action, policy):
    ##### TEST MDP BEFORE TRAINING
    print("test MDP policy")
    state = env.reset()

    for i in range(50):
        print("pose:",torch.argmax(state.x[:,env.IS_ROBOT_1]),torch.argmax(state.x[:,env.IS_ROBOT_2]))
        dist = compute_target(state, env, visited_index, index_action, policy)
        action = torch.argmax(dist,1)
        # print("action:",action)
        next_state, reward, done, _ = env.step(action.cpu().numpy())
        if reward:
            print("reward:",reward)
            print("***")

        state = next_state
        if done:
            print('Done in {} steps'.format(i+1))
            break

def train_agent(env, actor, optimizer, visited_index, index_action, policy, max_tries=500, n_iters=1000, wandb_log=False):
    ##### TRAIN
    print("Train NN agent")
    ce = torch.nn.CrossEntropyLoss()

    for iter in range(n_iters):
        try:
            state = env.reset()

            outputs = []
            targets = []

            for i in range(max_tries):
                dist = actor(state)
                action = dist.sample()
                next_state, reward, done, _ = env.step(action.cpu().numpy())

                target = compute_target(state, env, visited_index, index_action, policy)

                # TODO check mask
                mask = env.is_robot(state.x).bool()
                outputs.append(dist.probs[mask])
                targets.append(target[mask])

                # outputs.append(dist.probs)
                # targets.append(target)

                state = next_state

                if done:
                    break

                if dist.entropy().sum().item() < 0.01:
                    print('Reached a close-to-zero entropy solution...')
                    return

            outputs = torch.cat(outputs)
            targets = torch.cat(targets)
            actor_loss = ce(outputs, targets)

            if wandb_log:
                wandb.log({"actor_loss": actor_loss})

            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()
            print(f'Iter: {iter}, Steps: {i+1}, Loss: {actor_loss.item()}')
        except KeyboardInterrupt:
            print('Keyboard interrupt')
            return

#### PLAY
def test_learned_policy(env, actor, visited_index=None, index_action=None, policy=None):
    print("PLAYING WITH LEARNED POLICY")
    state = env.reset()

    for i in range(50):
        print("pose:",env.is_robot(state.x).numpy())
        dist = actor(state)
        if visited_index and index_action and policy:
            target = compute_target(state, env, visited_index, index_action, policy)
        action = dist.sample()
        print("action:", action.numpy()[env.is_robot(state.x).bool()])
        next_state, reward, done, _ = env.step(action.cpu().numpy())
        if reward:
            print("reward:",reward)
            print("***")
        state = next_state
        if done:
            print('Done in {} steps'.format(i+1))
            break
