import mdptoolbox
import numpy as np
from Networks import *
from Environments import *
import torch
import time
import wandb
import pickle

#### CREATE TRANSITION AND REWARD MODELS
def get_model(env):
    # return transitions, rewards, visited_index
    state = env.reset()
    s = state.x
    Q = [s]
    visited_index = {}
    t_dict = dict([(x,{}) for x in range(env.num_actions)])
    r_dict = dict([(x,{}) for x in range(env.num_actions)])

    start = time.time()
    done = False
    while Q:
        u = Q.pop(0)

        tu = tuple(u.flatten().numpy())
        if tu not in visited_index:
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

    return transitions, rewards, visited_index

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

def save_optimal_policy(visited_index, policy, filename="policy.p"):
    print("saving optimal policy")
    policy_dict = {"visited_index":visited_index, "policy":policy}
    pickle.dump(policy_dict, open("policies/"+filename, "wb"))

def load_optimal_policy(filename="policy.p"):
    print("loading optimal policy")
    policy_dict = pickle.load(open("policies/"+filename, "rb"))
    return policy_dict["visited_index"], policy_dict["policy"]

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

def test_optimal_policy(env, visited_index, policy):
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

def train_agent(env, actor, visited_index, policy, max_tries=500, n_iters=1000):
    ##### TRAIN
    print("Train NN agent")

    optimizerA = optim.Adam(actor.parameters(), lr=0.005)

    for iter in range(n_iters):
        state = env.reset()

        outputs = []
        targets = []

        for i in range(max_tries):
            dist = actor(state)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            target = compute_target(state, env, visited_index, policy)
            ce = torch.nn.CrossEntropyLoss()

            mask = state.x[:,env.IS_ROBOT].bool()
            outputs.append(dist.probs[mask])
            targets.append(target[mask])

            state = next_state

            if done:
                break

        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        actor_loss = ce(outputs, targets)

        wandb.log({"actor_loss": actor_loss})

        optimizerA.zero_grad()
        actor_loss.backward()
        optimizerA.step()
        print(f'Iter: {iter}, Steps: {i+1}, Loss: {actor_loss.item()}')

#### PLAY
def test_learned_policy(env, actor, visited_index=None, policy=None):
    print("PLAYING WITH LEARNED POLICY")
    state = env.reset()
    print("pose:", state.x[:,env.IS_ROBOT].numpy())

    for i in range(50):
        dist = actor(state)
        if visited_index and policy:
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
