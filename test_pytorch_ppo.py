from spinup import ppo_pytorch as ppo
from Environments import GymGraphEnv
from Networks import GCNActor, GCNCritic
import torch.nn as nn
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class MyActor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.gcn = GCNActor(env.num_node_features, env.num_actions)

    def forward(self, state, act=None):
        if isinstance(state, torch.Tensor):
            try:
                state = self.env.to_state(state)
            except:
                state = self.env.batch_obs_to_batch_state(state)

        if isinstance(state, Data):
            dist = self.gcn(state)
            log_prob = None
            if act is not None:
                mask = self.env.get_mask(state)
                log_prob = dist.log_prob(act)[mask].sum(-1).unsqueeze(0)
            return dist, log_prob

        elif isinstance(state, DataLoader):
            for b in state:
                batch = b
            dist = self.gcn(batch)
            log_prob = None
            if act is not None:
                act = torch.flatten(act)
                mask = self.env.get_mask(state)
                log_prob = dist.log_prob(act)[mask]
                log_prob = log_prob.reshape(batch.num_graphs,-1)#.sum(-1).unsqueeze(0)
            return dist, log_prob


class MyCritic(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.gcn = GCNCritic(env.num_node_features)

    def forward(self, state):
        if isinstance(state, torch.Tensor):
            try:
                state = self.env.to_state(state)
            except:
                state = self.env.batch_obs_to_batch_state(state)
        if isinstance(state, DataLoader):
            for b in state:
                batch = b
            value = self.gcn(batch)
            value = value.reshape(batch.num_graphs,self.env.num_nodes,1)
        else:
            value = self.gcn(state)
        return value

class MyActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, env):
        super().__init__()
        self.pi = MyActor(env)
        self.v = MyCritic(env)
        self.env = env
        self.obs_shape = observation_space.shape

    def step(self, batch_obs):
        with torch.no_grad():
            if batch_obs.shape == self.obs_shape:
                batch_state = self.env.to_state(batch_obs)
            else:
                batch_state = self.env.batch_obs_to_batch_state(batch_obs)
            dist, _ = self.pi(batch_state)
            value = self.v(batch_state)
            action = dist.sample()
            mask = self.env.get_mask(batch_state)
            log_prob = dist.log_prob(action)[mask].sum(-1).unsqueeze(0)
            value = value[mask].sum(-1)
        return action.numpy(), value.numpy(), log_prob.numpy()

    def act(self, obs):
        return self.step(obs)[0]

# mpi_fork(8)

env = GymGraphEnv()
env_fn = lambda: GymGraphEnv()
# ppo(env_fn, actor_critic=<MagicMock spec='str' id='140554322637768'>, ac_kwargs={},
    # seed=0, steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=0.0003,
    # vf_lr=0.001, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
    # target_kl=0.01, logger_kwargs={}, save_freq=10)
actor, critic = ppo(env_fn=env_fn, run_name="multi-time-step",
                    actor_critic=MyActorCritic, ac_kwargs=dict(env=env),
                    epochs=5, steps_per_epoch=100, max_ep_len=100,
                    pi_lr=0.001, vf_lr=0.01)

print("******DONE TRAINING**********")

# test
obs = env.reset()
state = env.to_state(obs)
print("state:",state.x[:,env.IS_ROBOT].numpy())

for i in range(500):
    dist, _ = actor(state)
    action = dist.sample()
    print("action:",action)

    next_obs, reward, done, _ = env.step(action.cpu().numpy())
    if reward:
        print("reward:",reward)

    state = env.to_state(next_obs)
    print("state:",state.x[:,env.IS_ROBOT].numpy())
    if done:
        print('Done in {} steps'.format(i+1))
        break
