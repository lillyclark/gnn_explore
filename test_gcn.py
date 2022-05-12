import Networks
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt

model = Networks.GCN()

# robot at 0
x0 = torch.tensor([[1,0],[0,1],[0,2],[0,3],[0,4]],dtype=torch.float)
y0 = torch.tensor([0.5,1.0,1.5,2])

# robot at 1
x1 = torch.tensor([[0,0],[1,1],[0,2],[0,3],[0,4]],dtype=torch.float)
y1 = torch.tensor([0,1.0,1.5,2])

# robot at 2
x2 = torch.tensor([[0,0],[0,1],[1,2],[0,3],[0,4]],dtype=torch.float)
y2 = torch.tensor([0,0,1.5,2])

# robot at 3
x3 = torch.tensor([[0,0],[0,1],[0,2],[1,3],[0,4]],dtype=torch.float)
y3 = torch.tensor([0,0,0,2])

# robot at 4
x4 = torch.tensor([[0,0],[0,1],[0,2],[0,3],[1,4]],dtype=torch.float)
y4 = torch.tensor([0,0,0,0])

edge_index = torch.tensor([[0,1,2,3],[1,2,3,4]])

d0 = Data(x=x0, edge_index=edge_index, y=y0)
d1 = Data(x=x1, edge_index=edge_index, y=y1)
d2 = Data(x=x2, edge_index=edge_index, y=y2)
d3 = Data(x=x3, edge_index=edge_index, y=y3)
d4 = Data(x=x4, edge_index=edge_index, y=y4)
datasets = [d0,d1,d2,d3,d4]

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
losses = []
for epoch in range(10000):
    i = epoch % 5
    model.train()
    optimizer.zero_grad()
    pred = model(datasets[i],0.0)
    loss = torch.pow(pred - datasets[i].y,2).sum() / len(pred)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

print("TEST")
model.eval()
for i in range(5):
    pred = model(datasets[i],0.0)
    print("predicted:",pred)
    print("actual:",datasets[i].y)
    print("MSE:",torch.pow(pred - datasets[i].y,2).sum().item() / len(pred))
    print("")

plt.plot(losses)
plt.show()
