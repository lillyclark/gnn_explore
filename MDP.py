from timeit import repeat
import matplotlib
from matplotlib.animation import FuncAnimation
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
    print("loading optimal policy")
    policy_dict = pickle.load(open("policies/"+filename, "rb"))
    return policy_dict["visited_index"], policy_dict["action_index"], policy_dict["policy"]

def compute_target(state, env, visited_index, action_index, policy):
    index_action = {v:k for k,v in action_index.items()}
    s = state.x
    st = tuple(s.flatten().numpy())
    s_idx = visited_index[st]
    action_idx = policy[s_idx]
    action_combo = index_action[action_idx]
    action = torch.ones(env.num_nodes).long()
    for i, robot in enumerate(env.robots):
        robot_pose = torch.argmax(s[:,robot])
        action[robot_pose] = action_combo[i]
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

def test_optimal_policy(env, visited_index, action_index, policy, graph):
    ##### TEST MDP BEFORE TRAINING
    print("test MDP policy")
    state = env.reset()
    # print("pose:",state.x[:,env.IS_ROBOT].numpy())
    print("pose:",env.is_robot(state.x).numpy())
    print("known:",state.x[:,env.IS_KNOWN_BASE].numpy())

    fig = plt.gcf()
    plt.clf()
    def update(i):
        nonlocal state, visited_index, action_index, policy, graph, ani
        dist = compute_target(state, env, visited_index, action_index, policy)
        action = torch.argmax(dist,1)
        # print("action:",action)
        next_state, reward, done, _ = env.step(action.cpu().numpy())
        print(np.round(dist.detach().numpy().T,2))
        # print(np.round(target.numpy().T,2))
        if reward:
            print("reward:",reward)
            print("***")

        state = next_state
        # print("pose:",state.x[:,env.IS_ROBOT].numpy())
        print("pose:",env.is_robot(state.x).numpy())
        print("known:",state.x[:,env.IS_KNOWN_BASE].numpy())
        graph = env.update_graph(graph)
        if done:
            print('Done in {} steps'.format(i+1))
            ani.event_source.stop()
            time.sleep(5)
            plt.close()
    ani = FuncAnimation(fig, update, frames=50, interval=1000, repeat=False)
    plt.show()

def train_agent(env, actor, optimizer, visited_index, action_index, policy, max_tries=500, n_iters=1000):
    ##### TRAIN
    print("Train NN agent")

    for iter in range(n_iters):
        state = env.reset()

        outputs = []
        targets = []

        for i in range(max_tries):
            dist = actor(state)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            # TODO: check if compute target is correct
            target = compute_target(state, env, visited_index, action_index, policy)
            ce = torch.nn.CrossEntropyLoss()
            # mask = state.x[:,env.IS_ROBOT].bool()
            mask = env.is_robot(state.x).bool()
            outputs.append(dist.probs[mask])
            targets.append(target[mask])

            state = next_state

            if done:
                break

        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        actor_loss = ce(outputs, targets)

        wandb.log({"actor_loss": actor_loss})

        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()
        print(f'Iter: {iter}, Steps: {i+1}, Loss: {actor_loss.item()}')

#### PLAY
def test_learned_policy(env, actor, visited_index=None, action_index=None, policy=None, graph=None):
    print("PLAYING WITH LEARNED POLICY")
    state = env.reset()
    # print("pose:", state.x[:,env.IS_ROBOT].numpy())


    print("pose:",env.is_robot(state.x).numpy())
    print("known:",state.x[:,env.IS_KNOWN_BASE].numpy())
    
    # for idx, node in enumerate(graph):
    # nx.set_edge_attributes(graph, values = 1, name = 'weight')
    # env.render(graph)
    # graph = env.update_graph(graph)
    fig = plt.gcf()
    plt.clf()
    print("*******STARTING SIMULATION*******")
    count = 0
    def update(i):
        nonlocal state, visited_index, action_index, policy, graph, ani, count
        # nx.set_edge_attributes(graph, values = 1, name = 'weight')
        plt.clf()
        print(f'*********count: {count}**********')       
        
        dist = actor(state)
        # print(f'distribution: {dist}')
        if visited_index and action_index and policy:
            target = compute_target(state, env, visited_index, action_index, policy)
            # print(np.round(dist.probs.detach().numpy().T,2))
            # print(np.round(target.numpy().T,2))
        action = dist.sample()
        for n in range(env.num_nodes):
            is_robot = env.is_robot(state.x)
            if is_robot[n]:
                print("action:",action.numpy()[n])
        next_state, reward, done, _ = env.step(action.cpu().numpy())
        if reward:
            print("reward:",reward)
            print("***")
        print(" ")
        state = next_state
        # print("pose:",state.x[:,env.IS_ROBOT].numpy())
        # Updating the graph        
        # nx.set_edge_attributes(graph, values = 1, name = 'weight')
        print(f'{count} pose:',env.is_robot(state.x).numpy())
        print(f'{count} known:',state.x[:,env.IS_KNOWN_BASE].numpy())
        count += 1 
        graph = env.update_graph(graph)
        if done:
            print('Done in {} steps'.format(i+1))
            ani.event_source.stop()
            time.sleep(5)
            plt.close()
    ani = FuncAnimation(fig, update, frames=50, interval=1000, repeat=False)
    plt.show()


    # for i in range(50):
        
            # break
