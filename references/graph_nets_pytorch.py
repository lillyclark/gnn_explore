import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import randint
import collections

NODES = "nodes"
EDGES = "edges"
RECEIVERS = "receivers"
SENDERS = "senders"
GLOBALS = "globals"
N_NODE = "n_node"
N_EDGE = "n_edge"

GRAPH_FEATURE_FIELDS = (NODES, EDGES, GLOBALS)
GRAPH_INDEX_FIELDS = (RECEIVERS, SENDERS)
GRAPH_DATA_FIELDS = (NODES, EDGES, RECEIVERS, SENDERS, GLOBALS)

class GraphsTuple(collections.namedtuple("GraphsTuple",GRAPH_DATA_FIELDS)):

    def __init__(self, *args, **kwargs):
        del args, kwargs
        # The fields of a `namedtuple` are filled in the `__new__` method.
        # `__init__` does not accept parameters.
        super(GraphsTuple, self).__init__()

    def replace(self, **kwargs):
        output = self._replace(**kwargs)
        return output

    def map(self, field_fn, fields=GRAPH_FEATURE_FIELDS):
        """Applies `field_fn` to the fields `fields` of the instance.

        `field_fn` is applied exactly once per field in `fields`. The result must
        satisfy the `GraphsTuple` requirement w.r.t. `None` fields, i.e. the
        `SENDERS` cannot be `None` if the `EDGES` or `RECEIVERS` are not `None`,
        etc.

        Args:
          field_fn: A callable that take a single argument.
          fields: (iterable of `str`). An iterable of the fields to apply
            `field_fn` to.

        Returns:
          A copy of the instance, with the fields in `fields` replaced by the result
          of applying `field_fn` to them.
        """
        return self.replace(**{k: field_fn(getattr(self, k)) for k in fields})

class GraphNetwork(nn.Module):
    def __init__(self,graph):
        super(GraphNetwork,self).__init__()
        self._edge_block = EdgeBlock(graph)
        self._node_block = NodeBlock(graph)
        self._global_block = GlobalBlock(graph)
    def forward(self, graph):
        return self._node_block(self._edge_block(graph))

class Aggregator(nn.Module):
    def __init__(self, mode):
        super(Aggregator, self).__init__()
        self.mode = mode

    def forward(self, graph):
        edges = graph.edges
        nodes = graph.nodes
        if self.mode == 'receivers':
            indeces = graph.receivers
        elif self.mode == 'senders':
            indeces = graph.senders
        else:
            raise AttributeError("invalid parameter `mode`")
        N_edges, N_features = edges.shape
        N_nodes=nodes.shape[0]
        aggrated_list = []
        for i in range(N_nodes):
            aggrated = edges[indeces == i]
            if aggrated.shape[0] == 0:
                aggrated = torch.zeros(1, N_features)
            aggrated_list.append(torch.sum(aggrated, dim=0))
        return torch.stack(aggrated_list,dim=0)

class EdgeBlock(nn.Module):
    def __init__(self,
                 graph: GraphsTuple,
                 use_edges=True,
                 use_receiver_nodes=True,
                 use_sender_nodes=True,
                 use_globals=True):
        super(EdgeBlock, self).__init__()
        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes
        self._use_globals = use_globals
        N_features = 0
        pre_features=graph.edges.shape[-1]
        if self._use_edges:
            N_features += graph.edges.shape[-1]
        if self._use_receiver_nodes:
            N_features += graph.nodes.shape[-1]
        if self._use_sender_nodes:
            N_features += graph.nodes.shape[-1]
        if self._use_globals:
            N_features += graph.globals.shape[-1]
        self.linear = nn.Linear(N_features, pre_features)

    def forward(self, graph: GraphsTuple):
        edges_to_collect = []

        if self._use_edges:
            edges_to_collect.append(graph.edges)  # edge feature  (50,6) 50边，6特征

        if self._use_receiver_nodes:
            edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph))  # (50,5)
            # 顶点有5个特征 receiver=(50,) 表示 每个边的汇点index
            #            得到的是每个边发射终点的顶点的feature

        if self._use_sender_nodes:
            edges_to_collect.append(broadcast_sender_nodes_to_edges(graph))  # (50,5)
            #            同上，只不过换成了起点

        if self._use_globals:
            edges_to_collect.append(broadcast_globals_to_edges(graph))  # (50,)

        collected_edges = torch.cat(edges_to_collect, dim=1)
        updated_edges = self.linear(collected_edges)
        return graph.replace(edges=updated_edges)

class NodeBlock(nn.Module):

    def __init__(self,
                 graph,
                 use_received_edges=True,
                 use_sent_edges=False,
                 use_nodes=True,
                 use_globals=True):
        super(NodeBlock, self).__init__()
        self._use_received_edges = use_received_edges
        self._use_sent_edges = use_sent_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals
        N_features = 0
        pre_features=graph.nodes.shape[-1]
        if self._use_nodes:
            N_features += graph.nodes.shape[-1]
        if self._use_received_edges:
            N_features += graph.edges.shape[-1]
        if self._use_sent_edges:
            N_features += graph.edges.shape[-1]
        if self._use_globals:
            N_features += graph.globals.shape[-1]
        self.linear = nn.Linear(N_features, pre_features)
        self._received_edges_aggregator = Aggregator('receivers')
        self._sent_edges_aggregator = Aggregator('senders')

    def forward(self, graph):

        nodes_to_collect = []
        # nodes: (24,5)
        # edges: (50,10)  # 上一轮更新了
        # global: (4,4)

        if self._use_received_edges:
            nodes_to_collect.append(self._received_edges_aggregator(graph))  # (24,10)
            # 在上一轮对边的处理中， 使用的是 _received_nodes_aggregator 将边相连的顶点信息考虑进来
            # 现在是将与顶点相连的边考虑进来

        if self._use_sent_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(graph))

        if self._use_nodes:
            nodes_to_collect.append(graph.nodes)

        if self._use_globals:
            nodes_to_collect.append(broadcast_globals_to_nodes(graph))  # (24,4)

        collected_nodes = torch.cat(nodes_to_collect, dim=1)  # 24,19
        updated_nodes = self.linear(collected_nodes)  # 24,11
        return graph.replace(nodes=updated_nodes)

class GlobalBlock(nn.Module):
    def __init__(self,
                 use_edges=True,
                 use_nodes=True,
                 use_globals=True):

        super(GlobalBlock, self).__init__()

        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals


    def forward(self, graph):
        globals_to_collect = []

        if self._use_edges:
            globals_to_collect.append(self._edges_aggregator(graph))

        if self._use_nodes:
            globals_to_collect.append(self._nodes_aggregator(graph))

        if self._use_globals:
            globals_to_collect.append(graph.globals)

        collected_globals = torch.cat(globals_to_collect, dim=1)
        updated_globals = self._global_model(collected_globals)
        return graph.replace(globals=updated_globals)

def broadcast_receiver_nodes_to_edges(graph: GraphsTuple):
    return graph.nodes.index_select(index=graph.receivers.long(), dim=0)

def broadcast_sender_nodes_to_edges(graph: GraphsTuple):
    return graph.nodes.index_select(index=graph.senders.long(), dim=0)

def broadcast_globals_to_edges(graph: GraphsTuple):
    N_edges = graph.edges.shape[0]
    return graph.globals.repeat(N_edges, 1)

def broadcast_globals_to_nodes(graph: GraphsTuple):
    N_nodes = graph.nodes.shape[0]
    return graph.globals.repeat(N_nodes, 1)

def data_dicts_to_graphs_tuple(graph_dicts:dict):
    for k,v in graph_dicts.items():
        graph_dicts[k]=torch.tensor(v)
    return GraphsTuple(**graph_dicts)
