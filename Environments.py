# import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

class SimpleEnvironment():
    def __init__(self):
        # num_nodes, num_node_features
        self.num_nodes = 5
        self.num_edges = (self.num_nodes - 1)
        self.num_node_features = 2         # features: is robot, is goal
        self.feature_matrix = np.zeros((self.num_nodes, self.num_node_features))
        self.feature_matrix[0][0] = 1
        self.feature_matrix[-1][1] = 1

        self.edge_index = torch.tensor([[0,1,2,3],[1,2,3,4]])
        self.edge_attr = np.zeros(self.num_edges) # TODO

    def get_feature_matrix(self):
        return self.feature_matrix

    def get_edge_index(self):
        return self.edge_index

    def get_edge_attr(self):
        return self.edge_attr

    def to_state(self):
        feature_matrix = torch.tensor(self.get_feature_matrix(), dtype=torch.float)
        edge_index = torch.tensor(self.get_edge_index())
        # edge_index = self.get_edge_index()
        # edge_attr = torch.tensor(self.get_edge_attr())
        edge_attr = self.get_edge_attr()
        state = Data(feature_matrix, edge_index, edge_attr)
        return state

    def get_rewards(self):
        rewards = np.zeros(self.num_edges)
        for i in range(self.num_edges):
            edge_v = self.edge_index[1][i]
            if self.feature_matrix[edge_v][1] == 1:
                rewards[i] = 1000
        return rewards

    def take_step_given_value(self, value):
        action = value
        # action is a vector of length num_edges
        # for each node that is a robot, it moves along the edge with the highest value
        all_moves = {}
        for node_idx in range(self.num_nodes):
            if self.feature_matrix[node_idx][0]:
                # this is a robot
                best_a_value = 0
                next_node_idx = None
                for edge_idx in range(self.num_edges):
                    if self.edge_index[0][edge_idx] == node_idx:
                        # this is an outgoing edge
                        if action[edge_idx] > best_a_value:
                            best_a_value = action[edge_idx]
                            next_node_idx = self.edge_index[1][edge_idx]
                if best_a_value:
                    all_moves[node_idx] = next_node_idx
        for robot_start, robot_end in all_moves.items():
            self.feature_matrix[robot_start][0] = 0
            self.feature_matrix[robot_end][0] = 1
        is_terminal = self.feature_matrix[-1][0] == 1
        return is_terminal

    def take_step(self, action):
        all_moves = {}
        for node_idx in range(self.num_nodes):
            if self.feature_matrix[node_idx][0]:
                for edge_idx in range(self.num_edges):
                    if self.edge_index[0][edge_idx] == node_idx:
                        if action[edge_idx]:
                            all_moves[node_idx] = self.edge_index[1][edge_idx]
        for robot_start, robot_end in all_moves.items():
            self.feature_matrix[robot_start][0] = 0
            self.feature_matrix[robot_end][0] = 1
            # print("MOVE:",robot_start,robot_end)
        is_terminal = self.feature_matrix[-1][0] == 1
        return is_terminal

    def get_status(self):
        return self.feature_matrix[-1][0] == 1

    def value_to_action(self, state, value):
        action = np.zeros(self.num_edges)
        feature_matrix = state.x
        edge_index = state.edge_index
        for node_idx in range(self.num_nodes):
            if feature_matrix[node_idx][0]:
                # this is a robot
                best_value = 0
                best_edge_idx = None
                for edge_idx in range(self.num_edges):
                    if edge_index[0][edge_idx] == node_idx:
                        # this is an outgoing edge
                        if value[edge_idx] > best_value:
                            best_value = value[edge_idx]
                            best_edge_idx = edge_idx
                if best_edge_idx is not None:
                    action[best_edge_idx] = 1
        return action
