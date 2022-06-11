import numpy as np
import torch
from torch_geometric.data import Data

class MABranchEnv():
    def __init__(self):

        self.num_actions = 4 # each node has max 3 edges

        self.IS_BASE = 0
        self.IS_KNOWN_BASE = 1

        self.IS_ROBOT_1 = 2
        self.IS_ROBOT_2 = 3
        self.IS_KNOWN_ROBOT_1 = 4
        self.IS_KNOWN_ROBOT_2 = 5
        self.robots = [self.IS_ROBOT_1, self.IS_ROBOT_2]
        self.agents = self.robots + [self.IS_BASE]
        self.agent_knows = {self.IS_BASE: self.IS_KNOWN_BASE,
                            self.IS_ROBOT_1: self.IS_KNOWN_ROBOT_1,
                            self.IS_ROBOT_2:self.IS_KNOWN_ROBOT_2}

        self.num_node_features = 6

        self.num_nodes = 8

        self.feature_matrix = torch.zeros((self.num_nodes, self.num_node_features))
        self.feature_matrix[0][self.IS_BASE], self.feature_matrix[0][self.IS_KNOWN_BASE] = True, True
        self.feature_matrix[1][self.IS_ROBOT_1], self.feature_matrix[1][self.IS_KNOWN_ROBOT_1] = True, True
        self.feature_matrix[2][self.IS_ROBOT_2], self.feature_matrix[2][self.IS_KNOWN_ROBOT_2] = True, True


        right_i = torch.Tensor([0,1,2,2,3,5,4,6])
        right_j = torch.Tensor([1,2,3,5,4,6,7,7])
        left_i = right_j
        left_j = right_i
        self.edge_index = torch.stack([torch.cat([right_i,left_i]),torch.cat([right_j,left_j])]).long()
        self.edge_attr = torch.ones(self.edge_index.size()[1])

        # start with everyone knowing, otherwise there is a reward for no action
        # print("init: before share")
        # print(self.feature_matrix[:,self.IS_KNOWN_BASE])
        self.feature_matrix = self.share_k(self.feature_matrix, k=len(self.robots))
        # print("init: after share")
        # print(self.feature_matrix[:,self.IS_KNOWN_BASE])

        self.state = Data(x=self.feature_matrix, edge_index=self.edge_index, edge_attr=self.edge_attr)

    def is_robot(self, feature_matrix):
        is_robot = torch.zeros(self.num_nodes)
        for r in self.robots:
            is_robot = torch.max(is_robot, feature_matrix[:,r])
        return is_robot

    def reset(self):
        self.__init__()
        return self.state

    def set_features(self, state_x):
        self.__init__()
        self.feature_matrix = state_x
        self.state = Data(x=self.feature_matrix, edge_index=self.edge_index, edge_attr=self.edge_attr)

    def is_incident(self,n,u,v):
        return (n == v)

    def incident_edges(self, node_idx): # list of tuples
        incident_edges = []
        for u, v in zip(self.edge_index[0],self.edge_index[1]):
            if self.is_incident(node_idx, u, v):
                incident_edges.append((u,v))
        for extra_edge in range(self.num_actions - len(incident_edges)):
            incident_edges.append((node_idx,node_idx))
        return incident_edges

    def share(self, feature_matrix):
        for n in range(self.num_nodes):
            for x in self.agents:
                if feature_matrix[n][x]: # is robot
                    for y in self.agents:
                        if (torch.argmax(feature_matrix[:,y]), n) in self.incident_edges(n):
                            feature_matrix[:,y] = torch.max(feature_matrix[:,x], feature_matrix[:,y])
                            feature_matrix[:,x] = torch.max(feature_matrix[:,x], feature_matrix[:,y])
        return feature_matrix

    def share_k(self, feature_matrix, k=1):
        new_feature_matrix = torch.zeros(feature_matrix.shape)
        for static in self.agents:
            new_feature_matrix[:,static] = feature_matrix[:,static]
        for hop in range(k):
            for agent1 in self.agents:
                known = feature_matrix[:,self.agent_knows[agent1]]
                for agent2 in self.agents:
                    if agent1 == agent2:
                        continue
                    agent1_pose = torch.argmax(feature_matrix[:,agent1])
                    agent2_pose = torch.argmax(feature_matrix[:,agent2])
                    if (agent2_pose, agent1_pose) in self.incident_edges(agent1_pose):
                        known = torch.max(known, feature_matrix[:,self.agent_knows[agent2]])
                new_feature_matrix[:,self.agent_knows[agent1]] = known
            feature_matrix = new_feature_matrix
        return new_feature_matrix


    def old_share_k(self, feature_matrix, k=1):
        print("")
        new_feature_matrix = torch.zeros(feature_matrix.shape)
        for static in self.agents:
            new_feature_matrix[:,static] = feature_matrix[:,static]
        for hop in range(k):
            for n in range(self.num_nodes):
                for u,v in self.incident_edges(n):
                    for agent1 in self.agents:
                        for agent2 in self.agents:
                            if agent1 == agent2:
                                continue
                            if feature_matrix[v][agent1] and feature_matrix[u][agent2]:
                                # if agent1 == 0:
                                #     print("agent1",agent1,feature_matrix[:,self.agent_knows[agent1]])
                                #     print("agent2",agent2,feature_matrix[:,self.agent_knows[agent2]])
                                shared = torch.max(feature_matrix[:,self.agent_knows[agent1]], feature_matrix[:,self.agent_knows[agent2]])
                                # if agent1 == 0:
                                #     print("max",shared)
                                #     print("old",new_feature_matrix[:,self.agent_knows[agent1]])
                                new_feature_matrix[:,self.agent_knows[agent1]] = torch.max(shared, new_feature_matrix[:,self.agent_knows[agent1]])
                                # if agent1 == 0:
                                #     print("updated",new_feature_matrix[:,self.agent_knows[agent1]])
                                #     print("column",self.agent_knows[agent1])
                                #     print(new_feature_matrix)
                                if new_feature_matrix[:,self.IS_KNOWN_BASE].sum() == 0:
                                    print("n",n,"u",u,"v",v,"agent1",agent1,"agent2",agent2)
                                    print(feature_matrix)
                                    print(shared)
                                    print("updated column",self.agent_knows[agent1],self.IS_KNOWN_BASE)
                                    print(new_feature_matrix[:,self.agent_knows[agent1]])
                                    print(new_feature_matrix)
                                    print(k)
                                    assert False
            feature_matrix = new_feature_matrix.detach().clone()
        return new_feature_matrix

    def robot_at(self, feature_matrix, n):
        for r in self.robots:
            if feature_matrix[n][r]:
                return True, r
        return False, None

    def step(self,action):
        new_feature_matrix = torch.zeros(self.feature_matrix.shape)

        # BASE STAYS
        new_feature_matrix[:,self.IS_BASE] = self.feature_matrix[:,self.IS_BASE]

        # WHAT WAS KNOWN IS STILL KNOWN
        for static in self.agents:
            new_feature_matrix[:,self.agent_knows[static]] = self.feature_matrix[:,self.agent_knows[static]]

        # MOVE ROBOT & EXPLORE
        for n in range(self.num_nodes):
            is_robot, r = self.robot_at(self.feature_matrix, n)
            if is_robot:
                u, v = self.incident_edges(n)[action[n]] # v = n
                also_robot, _ = self.robot_at(self.feature_matrix, u)
                already_robot, _ = self.robot_at(new_feature_matrix, u)
                is_base = new_feature_matrix[u][self.IS_BASE]
                if not also_robot and not already_robot and not is_base:
                    new_feature_matrix[u][r] = True
                    new_feature_matrix[u][self.agent_knows[r]] = True
                else:
                    new_feature_matrix[n][r] = True # wanted to move but couldn't

        # SHARE
        new_feature_matrix = self.share_k(new_feature_matrix)

        # COLLECT IMMEDIATE REWARD
        reward, done, progress = self.get_reward(self.feature_matrix, new_feature_matrix)
        self.feature_matrix = new_feature_matrix
        self.state = Data(x=self.feature_matrix, edge_index=self.edge_index, edge_attr=self.edge_attr)
        return self.state, reward, done, {"progress":progress}

    def get_reward(self, old_feature_matrix, new_feature_matrix):
        done = new_feature_matrix[:,self.IS_KNOWN_BASE].all().long()
        reward = 1*(new_feature_matrix[:,self.IS_KNOWN_BASE] - old_feature_matrix[:,self.IS_KNOWN_BASE]).sum().item()
        return reward, done, 0

class BranchEnv():
    def __init__(self):

        self.num_actions = 4 # each node has max 3 edges

        self.IS_BASE = 0
        self.IS_ROBOT = 1
        self.IS_KNOWN_BASE = 2
        self.IS_KNOWN_ROBOT = 3
        self.num_node_features = 4
        self.num_nodes = 8

        self.feature_matrix = torch.zeros((self.num_nodes, self.num_node_features))
        self.feature_matrix[0][self.IS_BASE], self.feature_matrix[0][self.IS_KNOWN_BASE], self.feature_matrix[0][self.IS_KNOWN_ROBOT] = True, True, True
        self.feature_matrix[1][self.IS_ROBOT], self.feature_matrix[1][self.IS_KNOWN_ROBOT], self.feature_matrix[1][self.IS_KNOWN_BASE] = True, True, True

        right_i = torch.Tensor([0,1,2,2,3,5,4,6])
        right_j = torch.Tensor([1,2,3,5,4,6,7,7])
        left_i = right_j
        left_j = right_i
        self.edge_index = torch.stack([torch.cat([right_i,left_i]),torch.cat([right_j,left_j])]).long()
        self.edge_attr = torch.ones(self.edge_index.size()[1])
        self.state = Data(x=self.feature_matrix, edge_index=self.edge_index, edge_attr=self.edge_attr)

    def reset(self):
        self.__init__()
        return self.state

    def set_features(self, state_x):
        self.__init__()
        self.feature_matrix = state_x
        self.state = Data(x=self.feature_matrix, edge_index=self.edge_index, edge_attr=self.edge_attr)

    def is_incident(self,n,u,v):
        return (n == v)

    def incident_edges(self, node_idx): # list of tuples
        incident_edges = []
        for u, v in zip(self.edge_index[0],self.edge_index[1]):
            if self.is_incident(node_idx, u, v):
                incident_edges.append((u,v))
        for extra_edge in range(self.num_actions - len(incident_edges)):
            incident_edges.append((node_idx,node_idx))
        return incident_edges

    def step(self,action):
        new_feature_matrix = torch.zeros(self.feature_matrix.shape)

        # BASE STAYS
        new_feature_matrix[:,self.IS_BASE] = self.feature_matrix[:,self.IS_BASE]

        # WHAT WAS KNOWN IS STILL KNOWN
        new_feature_matrix[:,self.IS_KNOWN_ROBOT] = self.feature_matrix[:,self.IS_KNOWN_ROBOT]
        new_feature_matrix[:,self.IS_KNOWN_BASE] = self.feature_matrix[:,self.IS_KNOWN_BASE]

        # MOVE ROBOT & EXPLORE
        for n in range(self.num_nodes):
            if self.feature_matrix[n][self.IS_ROBOT]: # is robot
                u, v = self.incident_edges(n)[action[n]] # v = n
                if not self.feature_matrix[u][self.IS_ROBOT] and not new_feature_matrix[u][self.IS_ROBOT] and not new_feature_matrix[u][self.IS_BASE]: # collision avoidance
                    new_feature_matrix[u][self.IS_ROBOT] = True
                    new_feature_matrix[u][self.IS_KNOWN_ROBOT] = True
                else:
                    new_feature_matrix[n][self.IS_ROBOT] = True # wanted to move but couldn't

        # SHARE
        for n in range(self.num_nodes):
            if new_feature_matrix[n][self.IS_ROBOT]: # is robot
                if (torch.argmax(new_feature_matrix[:,self.IS_BASE]), n) in self.incident_edges(n):
                    # todo (right now a fully two way sync)
                    new_feature_matrix[:,self.IS_KNOWN_BASE] = torch.max(new_feature_matrix[:,self.IS_KNOWN_ROBOT], new_feature_matrix[:,self.IS_KNOWN_BASE])
                    new_feature_matrix[:,self.IS_KNOWN_ROBOT] = torch.max(new_feature_matrix[:,self.IS_KNOWN_ROBOT], new_feature_matrix[:,self.IS_KNOWN_BASE])
                    # TODO share with other robots

        # COLLECT IMMEDIATE REWARD
        reward, done, progress = self.get_reward(self.feature_matrix, new_feature_matrix)
        self.feature_matrix = new_feature_matrix
        self.state = Data(x=self.feature_matrix, edge_index=self.edge_index, edge_attr=self.edge_attr)
        return self.state, reward, done, {"progress":progress}

    def get_reward(self, old_feature_matrix, new_feature_matrix):
        done = new_feature_matrix[:,self.IS_KNOWN_BASE].all().long()
        reward = 1*(new_feature_matrix[:,self.IS_KNOWN_BASE] - old_feature_matrix[:,self.IS_KNOWN_BASE]).sum().item()
        return reward, done, 0

class GraphEnv():
    def __init__(self, reward_name="base_reward", has_master=False):
        self.reward_name = reward_name
        self.has_master = has_master

        self.num_actions = 3 # each node has max 2 edges

        self.IS_BASE = 0
        self.IS_ROBOT = 1
        self.IS_KNOWN_BASE = 2
        self.IS_KNOWN_ROBOT = 3
        self.num_node_features = 4
        self.num_nodes = 8

        if self.has_master:
            self.IS_MASTER_NODE = 4 # not really part of the env
            self.num_node_features = 5
            self.num_nodes = 9

        self.feature_matrix = torch.zeros((self.num_nodes, self.num_node_features))
        self.feature_matrix[0][self.IS_BASE], self.feature_matrix[0][self.IS_KNOWN_BASE], self.feature_matrix[0][self.IS_KNOWN_ROBOT] = True, True, True
        self.feature_matrix[1][self.IS_ROBOT], self.feature_matrix[1][self.IS_KNOWN_ROBOT], self.feature_matrix[1][self.IS_KNOWN_BASE] = True, True, True

        if self.has_master:
            self.feature_matrix[-1][self.IS_MASTER_NODE], self.feature_matrix[-1][self.IS_KNOWN_ROBOT], self.feature_matrix[-1][self.IS_KNOWN_BASE] = True, True, True

        right_i = torch.Tensor(list(range(0,self.num_nodes-1)))
        right_j = torch.Tensor(list(range(1,self.num_nodes)))
        left_i = right_j
        left_j = right_i
        self.edge_index = torch.stack([torch.cat([right_i,left_i]),torch.cat([right_j,left_j])]).long()

        if self.has_master:
            right_i = torch.Tensor(list(range(0,self.num_nodes-2)))
            right_j = torch.Tensor(list(range(1,self.num_nodes-1)))
            left_i = right_j
            left_j = right_i
            to_master_i = torch.Tensor(list(range(0,self.num_nodes-1)))
            to_master_j = torch.Tensor([self.num_nodes-1]*(self.num_nodes-1))
            from_master_i = to_master_j
            from_master_j = to_master_i
            self.edge_index = torch.stack([torch.cat([right_i,left_i,to_master_i,from_master_i]),torch.cat([right_j,left_j,to_master_j,from_master_j])]).long()

        # self.edge_index = torch.Tensor([list(range(0,self.num_nodes-2))+list(range(1,self.num_nodes-1)),list(range(1,self.num_nodes-1))+list(range(0,self.num_nodes-2))).long()
        self.edge_attr = torch.ones(self.edge_index.size()[1])
        self.state = Data(x=self.feature_matrix, edge_index=self.edge_index, edge_attr=self.edge_attr)

    def reset(self):
        self.__init__(reward_name=self.reward_name, has_master=self.has_master)
        return self.state

    def set_features(self, state_x):
        self.__init__(reward_name=self.reward_name, has_master=self.has_master)
        self.feature_matrix = state_x
        self.state = Data(x=self.feature_matrix, edge_index=self.edge_index, edge_attr=self.edge_attr)

    def change_env(self):
        self.feature_matrix = torch.zeros((self.num_nodes, self.num_node_features))
        self.feature_matrix[0][self.IS_BASE], self.feature_matrix[0][self.IS_KNOWN_BASE], self.feature_matrix[0][self.IS_KNOWN_ROBOT] = True, True, False
        self.feature_matrix[4][self.IS_ROBOT], self.feature_matrix[4][self.IS_KNOWN_ROBOT], self.feature_matrix[4][self.IS_KNOWN_BASE] = True, True, False
        if self.has_master:
            self.feature_matrix[-1][self.IS_MASTER_NODE], self.feature_matrix[-1][self.IS_KNOWN_ROBOT], self.feature_matrix[-1][self.IS_KNOWN_BASE] = True, True, True
        self.state = Data(x=self.feature_matrix, edge_index=self.edge_index, edge_attr=self.edge_attr)
        return self.state

    def is_incident(self,n,u,v):
        if self.has_master:
            return (n == v and not self.feature_matrix[u][self.IS_MASTER_NODE])
        return (n == v)

    def incident_edges(self, node_idx): # list of tuples
        incident_edges = []
        for u, v in zip(self.edge_index[0],self.edge_index[1]):
            if self.is_incident(node_idx, u, v):
                incident_edges.append((u,v))
        for extra_edge in range(self.num_actions - len(incident_edges)):
            incident_edges.append((node_idx,node_idx))
        return incident_edges

    def step(self,action):
        new_feature_matrix = torch.zeros(self.feature_matrix.shape)

        # MASTER_NODE STAYS
        if self.has_master:
            new_feature_matrix[:,self.IS_MASTER_NODE] = self.feature_matrix[:,self.IS_MASTER_NODE]

        # BASE STAYS
        new_feature_matrix[:,self.IS_BASE] = self.feature_matrix[:,self.IS_BASE]

        # WHAT WAS KNOWN IS STILL KNOWN
        new_feature_matrix[:,self.IS_KNOWN_ROBOT] = self.feature_matrix[:,self.IS_KNOWN_ROBOT]
        new_feature_matrix[:,self.IS_KNOWN_BASE] = self.feature_matrix[:,self.IS_KNOWN_BASE]

        # MOVE ROBOT & EXPLORE
        for n in range(self.num_nodes):
            if self.feature_matrix[n][self.IS_ROBOT]: # is robot
                u, v = self.incident_edges(n)[action[n]] # v = n
                if not self.feature_matrix[u][self.IS_ROBOT] and not new_feature_matrix[u][self.IS_ROBOT] and not new_feature_matrix[u][self.IS_BASE]: # collision avoidance
                    new_feature_matrix[u][self.IS_ROBOT] = True
                    new_feature_matrix[u][self.IS_KNOWN_ROBOT] = True
                else:
                    new_feature_matrix[n][self.IS_ROBOT] = True # wanted to move but couldn't

        # SHARE
        for n in range(self.num_nodes):
            if new_feature_matrix[n][self.IS_ROBOT]: # is robot
                if (torch.argmax(new_feature_matrix[:,self.IS_BASE]), n) in self.incident_edges(n):
                    # todo (right now a fully two way sync)
                    new_feature_matrix[:,self.IS_KNOWN_BASE] = torch.max(new_feature_matrix[:,self.IS_KNOWN_ROBOT], new_feature_matrix[:,self.IS_KNOWN_BASE])
                    new_feature_matrix[:,self.IS_KNOWN_ROBOT] = torch.max(new_feature_matrix[:,self.IS_KNOWN_ROBOT], new_feature_matrix[:,self.IS_KNOWN_BASE])

        # COLLECT IMMEDIATE REWARD
        if self.reward_name == "base_reward":
            reward, done, progress = self.get_reward(self.feature_matrix, new_feature_matrix)
        elif self.reward_name == "robot_reward":
            reward, done, progress = self.get_robot_reward(self.feature_matrix, new_feature_matrix)
        elif self.reward_name == "shaped_reward":
            reward, done, progress = self.get_shaped_reward(self.feature_matrix, new_feature_matrix)
        elif self.reward_name == "both_reward":
            reward, done, progress = self.get_both_reward(self.feature_matrix, new_feature_matrix)
        else:
            raise NotImplementedError
        self.feature_matrix = new_feature_matrix
        self.state = Data(x=self.feature_matrix, edge_index=self.edge_index, edge_attr=self.edge_attr)
        return self.state, reward, done, {"progress":progress}

    def get_both_reward(self, old_feature_matrix, new_feature_matrix):
        done = new_feature_matrix[:,self.IS_KNOWN_BASE].all().long()
        base_reward = 0.5*(new_feature_matrix[:,self.IS_KNOWN_BASE] - old_feature_matrix[:,self.IS_KNOWN_BASE]).sum().item()
        robot_reward = 0.5*(new_feature_matrix[:,self.IS_KNOWN_ROBOT] - old_feature_matrix[:,self.IS_KNOWN_ROBOT]).sum().item()
        progress = robot_reward > 0
        return base_reward+robot_reward, done, progress

    def get_shaped_reward(self, old_feature_matrix, new_feature_matrix):
        done = new_feature_matrix[:,self.IS_KNOWN_BASE].all().long()
        base_reward = 1*(new_feature_matrix[:,self.IS_KNOWN_BASE] - old_feature_matrix[:,self.IS_KNOWN_BASE]).sum().item()
        robot_reward = 1*(new_feature_matrix[:,self.IS_KNOWN_ROBOT] - old_feature_matrix[:,self.IS_KNOWN_ROBOT]).sum().item()
        progress = robot_reward > 0
        return base_reward, done, progress

    def get_reward(self, old_feature_matrix, new_feature_matrix):
        done = new_feature_matrix[:,self.IS_KNOWN_BASE].all().long()
        reward = 1*(new_feature_matrix[:,self.IS_KNOWN_BASE] - old_feature_matrix[:,self.IS_KNOWN_BASE]).sum().item()
        return reward, done, 0

    def get_robot_reward(self, old_feature_matrix, new_feature_matrix):
        done = new_feature_matrix[:,self.IS_KNOWN_ROBOT].all().long()
        reward = 1*(new_feature_matrix[:,self.IS_KNOWN_ROBOT] - old_feature_matrix[:,self.IS_KNOWN_ROBOT]).sum().item()
        return reward, done, 0

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
