from itertools import combinations, groupby
import random
import torch
import networkx as nx
import matplotlib.pyplot as plt


class testEnv():
    def __init__(self) -> None:
        self.policy_name = ".p"
        self.right_i = torch.Tensor([])
        self.right_j = torch.Tensor([])
        self.model_name = ".pt"
        self.num_nodes = 0
        self.num_actions = 0
        self.envName = ""
        self.num_robots = 2
        self.G = None

    def gnp_random_connected_graph(self, n, p):
        edges = combinations(range(n), 2)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        if p <= 0:
            return G
        if p >= 1:
            return nx.complete_graph(n, create_using=G)
        for _, node_edges in groupby(edges, key=lambda x: x[0]):
            node_edges = list(node_edges)
            random_edge = random.choice(node_edges)
            G.add_edge(*random_edge)
            for edge in node_edges:
                if random.random() < p:
                    G.add_edge(*edge)
        return G
       
    def setEnv(self, env_option):
        envName = "Env"
        if env_option == "A":
            envName += env_option
            self.envName = envName
            self.policy_name = envName + self.policy_name
            self.model_name = envName + self.model_name
            self.right_i = torch.Tensor([0,0,1,2,3,4,5,6,6])
            self.right_j = torch.Tensor([1,2,3,4,5,5,6,7,8])
            self.num_nodes = 9
            self.num_actions = 4
        if env_option == "B":
            envName += env_option
            self.envName = envName
            self.policy_name = envName + self.policy_name
            self.model_name = envName + self.model_name
            self.right_i = torch.Tensor([0,0,1,2,3,4,5,6])
            self.right_j = torch.Tensor([1,2,3,4,5,6,7,8])
            self.num_nodes = 9
            self.num_actions = 3
        if env_option == "C":
            envName += env_option
            self.envName = envName
            self.policy_name = envName + self.policy_name
            self.model_name = envName + self.model_name
            self.right_i = torch.Tensor([0,0,1,2,3,4,4,5,6])
            self.right_j = torch.Tensor([1,2,3,3,4,5,6,7,8])
            self.num_nodes = 9
            self.num_actions = 4
        if env_option == "D":
            envName += env_option
            self.envName = envName
            self.policy_name = envName + self.policy_name
            self.model_name = envName + self.model_name
            self.right_i = torch.Tensor([0,0,1,2,3,4,4,5,6,7,8])
            self.right_j = torch.Tensor([1,2,3,3,4,5,6,7,8,9,10])
            self.num_nodes = 11
            self.num_actions = 4
        if env_option == "E":
            envName += env_option
            self.envName = envName
            self.policy_name = envName + self.policy_name
            self.model_name = envName + self.model_name
            self.right_i = torch.Tensor([0,0,1,2,3,4,5,6,6,7,8])
            self.right_j = torch.Tensor([1,2,3,4,5,5,6,7,8,9,10])
            self.num_nodes = 11
            self.num_actions = 4
    
    def showGraph(self, G=None):
        if self.envName == "":
            print("test environment not initialized")
            return
        if G:
            nx.draw_spring(G, with_labels=True)
            return
        G = nx.Graph()
        
        for idx in range (len(self.right_i)):
            G.add_edge(int(self.right_i[idx]), int(self.right_j[idx]), weight=1.0)
        nx.draw_spectral(G, with_labels=True)
        self.G = G
        # plt.savefig('filename.png')
        # TODO add animation to the graph
        # plt.animation()
    
    def getGraph(self):
        return self.G
    
    def createEnv(self, envName, num_nodes = 8, seed=246, p=0.01):
        random.seed(246)
        if envName == "B":
            G = self.gnp_random_connected_graph(num_nodes, p)
            # G = nx.erdos_renyi_graph(num_nodes, 0.5, seed=246)
            array_i = []
            array_j = []
            for line in nx.generate_edgelist(G, data=False):
                values = line.split(' ')
                array_i.append(int(values[0]))
                array_j.append(int(values[1]))
            print(array_i)
            print(array_j)
            self.right_i = torch.tensor(array_i)
            self.right_j = torch.tensor(array_j)
            nx.draw(G, with_labels=True)
            plt.show()
    
# class displayGraph():
#     def __init__(self) -> None:
#         pass
#     def show(self, right_i, right_j):


        
        
        
            

