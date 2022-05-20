from torch_geometric.nn import GCNConv, GatedGraphConv, MessagePassing
# , TopKPooling, GatedGraphConv, global_max_pool, global_mean_pool
# from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   # remove_self_loops, softmax)
# from torch_geometric.utils.repeat import repeat

import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from Masks import CategoricalMasked, CategoricalMap

from typing import Optional

import numpy as np

class LinearAgg(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean',node_dim=0)

    def forward(self, x, edge_index):
        # message -> aggregate -> update
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out

class LinearAggActor(torch.nn.Module):
    def __init__(self, num_node_features, num_actions):
        super(LinearAggActor, self).__init__()
        self.receptive_field = 7
        self.num_node_features = num_node_features
        self.num_actions = num_actions
        self.hidden_units = 16

        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.num_node_features,self.hidden_units),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.hidden_units,self.hidden_units),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.hidden_units,self.num_node_features))
        self.agg_layer = LinearAgg()
        self.fully_con1 = torch.nn.Linear((1+self.receptive_field)*self.num_node_features,self.num_node_features)
        self.fully_con2 = torch.nn.Linear(self.num_node_features,self.num_actions)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        fields = [self.mlp(self.mlp(x))]
        for i in range(self.receptive_field):
            x_ = self.mlp(x)
            for j in range(i+1):
                x_ = self.agg_layer(x_, edge_index)
            x_ = self.mlp(x_)
            fields.append(x_)
        fields = torch.cat(fields,dim=1)
        x = self.fully_con1(fields)
        x = self.fully_con2(x)
        return Categorical(F.softmax(x, dim=-1))

class LinearAggCritic(torch.nn.Module):
    def __init__(self, num_node_features):
        super(LinearAggCritic, self).__init__()
        self.receptive_field = 7
        self.num_node_features = num_node_features
        self.hidden_units = 16

        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.num_node_features,self.hidden_units),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.hidden_units,self.hidden_units),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.hidden_units,self.num_node_features))
        self.agg_layer = LinearAgg()
        self.fully_con1 = torch.nn.Linear((1+self.receptive_field)*self.num_node_features,self.num_node_features)
        self.fully_con2 = torch.nn.Linear(self.num_node_features,1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        fields = [self.mlp(self.mlp(x))]
        for i in range(self.receptive_field):
            x_ = self.mlp(x)
            for j in range(i+1):
                x_ = self.agg_layer(x_, edge_index)
            x_ = self.mlp(x_)
            fields.append(x_)
        fields = torch.cat(fields,dim=1)
        x = self.fully_con1(fields)
        x = self.fully_con2(x)
        return x

class SimpleActor(torch.nn.Module):
    def __init__(self, num_node_features, num_nodes, num_actions):
        super(SimpleActor, self).__init__()
        self.num_node_features = num_node_features
        self.num_nodes = num_nodes
        self.num_actions = num_actions

        self.lin1 = torch.nn.Linear(self.num_node_features*self.num_nodes, 128)
        self.lin2 = torch.nn.Linear(128, 256)
        self.lin3 = torch.nn.Linear(256, self.num_actions*self.num_nodes)

    def forward(self, data, prob=0.0, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = x.reshape(-1,self.num_node_features*self.num_nodes)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = x.reshape(self.num_nodes,self.num_actions)
        return Categorical(F.softmax(x, dim=-1))

class SimpleCritic(torch.nn.Module):
    def __init__(self, num_node_features, num_nodes):
        super(SimpleCritic, self).__init__()
        self.num_node_features = num_node_features
        self.num_nodes = num_nodes

        self.lin1 = torch.nn.Linear(self.num_node_features*self.num_nodes, 128)
        self.lin2 = torch.nn.Linear(128, 256)
        self.lin3 = torch.nn.Linear(256, self.num_nodes)

    def forward(self, data, prob=0.0, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = x.reshape(-1,self.num_node_features*self.num_nodes)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = x.reshape(self.num_nodes,-1)
        return x

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

class GCNActor(torch.nn.Module):
    def __init__(self, num_node_features, num_actions):
        super(GCNActor, self).__init__()
        self.input_size = num_node_features
        self.output_size = num_actions
        self.conv1 = GCNConv(self.input_size, 128, improved=True)
        self.conv2 = GCNConv(128, 128, improved=True)
        self.fully_con1 = torch.nn.Linear(128, self.output_size)

    def forward(self, data, prob=0.0, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        for i in range(6):
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
        x = F.dropout(x, p=prob)
        x = self.fully_con1(x)
        return Categorical(F.softmax(x, dim=-1))

class GCNCritic(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCNCritic, self).__init__()
        self.input_size = num_node_features
        self.conv1 = GCNConv(self.input_size, 128, improved=True)
        self.conv2 = GCNConv(128, 128, improved=True)
        self.fully_con1 = torch.nn.Linear(128, 1)

    def forward(self, data, prob=0.0, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        for i in range(6):
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
        x = F.dropout(x, p=prob)
        x = self.fully_con1(x)
        return x

class GGNN(torch.nn.Module):
    def __init__(self):
        super(GGNN, self).__init__()
        self.gconv1 = GatedGraphConv(1000, 3)
        self.fully_con1 = torch.nn.Linear(1000, 1)

    def forward(self, data, prob, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.gconv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=prob)
        x = self.fully_con1(x)
        return x


class GGNNActor(torch.nn.Module):
    def __init__(self, num_node_features, num_actions):
        super(GGNNActor, self).__init__()
        self.input_size = num_node_features
        self.output_size = num_actions
        self.gconv1 = GatedGraphConv(256, 3)
        self.fully_con1 = torch.nn.Linear(256, self.output_size)

    def forward(self, data, prob=0.0, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for i in range(1):
            x = self.gconv1(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=prob)
        x = self.fully_con1(x)
        return Categorical(F.softmax(x, dim=-1))


class GGNNCritic(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GGNNCritic, self).__init__()
        self.input_size = num_node_features
        self.gconv1 = GatedGraphConv(256, 3)
        self.fully_con1 = torch.nn.Linear(256, 1) #100)

    def forward(self, data, prob=0.0, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        for i in range(1):
            x = self.gconv1(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=prob)
        x = self.fully_con1(x)
        # x = global_mean_pool(x, batch).mean(dim=1)
        return x

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.num_actions = action_size
        self.action_size = self.state_size*self.num_actions
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        output = output.reshape(self.state_size,self.num_actions)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value
