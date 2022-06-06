from Networks import SimpleActor
from Environments import GraphEnv
import torch

def compute_target(state,env):
    # desired probability of going [left, right, stay] from each node
    right_dist = torch.Tensor([[0,0,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[1,0,0]])
    left_dist = torch.Tensor([[0,0,1],[0,0,1],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]])
    if state.x[-1][env.IS_KNOWN_ROBOT]:
        # if the robot has visited the last node
        return left_dist
    return right_dist

a = SimpleActor(4,8,3)

env = GraphEnv()

state = env.reset()

dist = a(state)
ce = torch.nn.CrossEntropyLoss()

loss = ce(dist.probs, compute_target(state, env))

for p in a.parameters():
    print(p.grad)

loss.backward()

for p in a.parameters():
    print(p.grad)
