import numpy as np
import torch
from torch_geometric.data import Data

class GraphEnv():
    def __init__(self):
        self.IS_ROBOT = 0
        self.IS_ROBOT_GOAL = 1
        self.num_node_features = 2 # is_robot, is_goal

        self.num_actions = 3 # each node has max 2 edges
        self.num_nodes = 8

        self.feature_matrix = torch.zeros((self.num_nodes, self.num_node_features))
        self.feature_matrix[0][self.IS_ROBOT], self.feature_matrix[4][self.IS_ROBOT] = True, True
        self.feature_matrix[3][self.IS_ROBOT_GOAL], self.feature_matrix[7][self.IS_ROBOT_GOAL] = True, True

        self.goal_feature_matrix = torch.zeros(self.feature_matrix.shape)
        self.goal_feature_matrix[3][self.IS_ROBOT], self.goal_feature_matrix[7][self.IS_ROBOT] = True, True
        self.goal_feature_matrix[3][self.IS_ROBOT_GOAL], self.goal_feature_matrix[7][self.IS_ROBOT_GOAL] = True, True

        self.edge_index = torch.Tensor([list(range(0,self.num_nodes-1)),list(range(1,self.num_nodes))]).long()
        self.edge_attr = torch.ones(self.num_nodes-1)
        self.state = Data(x=self.feature_matrix, edge_index=self.edge_index, edge_attr=self.edge_attr)

    def reset(self):
        self.__init__()
        return self.state

    def incident_edges(self, node_idx): # list of tuples
        incident_edges = []
        for u, v in zip(self.edge_index[0],self.edge_index[1]):
            if node_idx == u or node_idx == v:
                incident_edges.append((u,v))
        for extra_edge in range(self.num_actions - len(incident_edges)):
            incident_edges.append((node_idx,node_idx))
        return incident_edges

    def step(self,action):
        new_feature_matrix = torch.zeros(self.feature_matrix.shape)
        new_feature_matrix[:,self.IS_ROBOT_GOAL] = self.feature_matrix[:,self.IS_ROBOT_GOAL]
        for n in range(self.feature_matrix.size()[self.IS_ROBOT]):
            if self.feature_matrix[n][self.IS_ROBOT]: # is robot
                u, v = self.incident_edges(n)[action[n]]
                if u == n and v == n: # stay
                    new_feature_matrix[v][self.IS_ROBOT] = True
                elif u == n and not self.feature_matrix[v][self.IS_ROBOT] and not new_feature_matrix[v][self.IS_ROBOT]:
                    new_feature_matrix[v][self.IS_ROBOT] = True
                elif v == n and not self.feature_matrix[u][self.IS_ROBOT] and not new_feature_matrix[u][self.IS_ROBOT]:
                    new_feature_matrix[u][self.IS_ROBOT] = True
                else:
                    new_feature_matrix[n][self.IS_ROBOT] = True # wanted to move but couldn't
        self.feature_matrix = new_feature_matrix
        self.state = Data(x=self.feature_matrix, edge_index=self.edge_index, edge_attr=self.edge_attr)
        reward, done = self.get_reward()
        return self.state, reward, done, None

    def get_reward(self):
        reward, done = 0, False
        if (self.feature_matrix == self.goal_feature_matrix).all():
            reward, done = 10, True
        return reward, done

class TestEnv():
    def __init__(self):
        self.num_actions = 3 # left, stay, right
        self.state = np.array([1,0,0,0,1,0,0,0])
        self.goal_state = np.array([0,0,0,1,0,0,0,1])

    def reset(self):
        self.state = np.array([1,0,0,0,1,0,0,0])
        return self.state

    def step(self,action):
        # print("STATE",self.state)
        new_state = self.state.copy()
        # print("new_state",new_state)
        for cell in range(len(self.state)):
            if self.state[cell]: # has robot
                # print("robot at",cell)
                if action[cell] == 0: # go left
                    # print("goes left")
                    new_pos = max(cell-1,0)
                elif action[cell] == 1: # stay
                    new_pos = cell
                elif action[cell] == 2: # go right
                    # print("goes right")
                    new_pos = min(cell+1,len(self.state)-1)
                else:
                    raise ValueError()
                if not new_state[new_pos]: # collision avoidance
                    # print("moves")
                    new_state[cell] = 0
                    new_state[new_pos] = 1
                # print("new_state",new_state)
        assert sum(self.state) == 2
        self.state = new_state
        reward, done = self.get_reward()
        return self.state, reward, done, None

    def get_reward(self):
        reward, done = 0, False
        if (self.state == self.goal_state).all():
            reward, done = 10, True
        return reward, done
