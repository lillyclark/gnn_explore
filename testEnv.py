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
    
    def showGraph(self):
        if self.envName == "":
            print("test environment not initialized")
            return
        G = nx.Graph()
        for idx in range (len(self.right_i)):
            G.add_edge(int(self.right_i[idx]), int(self.right_j[idx]))
        nx.draw_spectral(G, with_labels=True)
        plt.savefig('filename.png')
        plt.show()
    
# class displayGraph():
#     def __init__(self) -> None:
#         pass
#     def show(self, right_i, right_j):


        
        
        
            

