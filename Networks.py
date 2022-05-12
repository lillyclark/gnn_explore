import os
import torch
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_scatter import scatter_max, scatter_add
from torch_geometric.nn import GCNConv, TopKPooling, GatedGraphConv, global_max_pool, global_mean_pool
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops, softmax)
from torch_geometric.utils.repeat import repeat


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, 16, improved=True)
        self.conv2 = GCNConv(16, 16, improved=True)
        self.fully_con1 = torch.nn.Linear(16, 1)

    def forward(self, data, prob, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index) #, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index) #, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=prob)
        x = self.fully_con1(x)
        return (x[edge_index[0]] * x[edge_index[1]]).sum(dim=-1)
        # return x
