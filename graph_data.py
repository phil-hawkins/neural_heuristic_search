import torch
import numpy as np


class GraphData():
    node_attr_dim = 4
    edge_attr_dim = 1
    global_attr_dim = 1

    def __init__(self, edge_index, node_attr, zero_edge_attr=True, edge_attr=None, action_edge_index=None, global_attr=None, edge_strut_slots=None, device=None):
        if device is None:
            device = edge_index.device
        self.device = device
        if zero_edge_attr:
            edge_attr = torch.zeros((edge_index.size(1), GraphData.edge_attr_dim), dtype=torch.float, device=device)
        if edge_attr is None:
            # TODO remove this
            # initial edge attribute is the number of simplices it participates in
            adj = torch.sparse_coo_tensor(
                indices=edge_index, 
                values=torch.ones(edge_index.size(1), device=device), 
                size=(node_attr.size(0), node_attr.size(0)),
                device=device
            ).to_dense()
            simplex_num = adj.matmul(adj) * adj
            edge_attr = simplex_num[edge_index[0], edge_index[1]].unsqueeze(dim=1) + 1.
        if global_attr is None:
            global_attr = torch.zeros((1, GraphData.global_attr_dim), dtype=torch.float, device=device)

        self.edge_index = edge_index    
        self.node_attr = node_attr
        self.edge_attr = edge_attr
        self.global_attr = global_attr
        self.action_edge_index = action_edge_index
        self.node_batch = torch.zeros((self.node_attr.size(0)), dtype=torch.long, device=device)
        self.edge_strut_slots = edge_strut_slots
        if self.action_edge_index is not None:
            self.action_batch = torch.stack([
                torch.zeros((self.action_edge_index.size(0)), dtype=torch.long, device=device),
                torch.arange(self.action_edge_index.size(0), device=device)
            ])

        assert self.node_attr.size(1) == self.node_attr_dim
        assert self.edge_attr.size(1) == self.edge_attr_dim
        assert self.global_attr.size(1) == self.global_attr_dim

    def zero_edge_attr(self):
        self.edge_attr = torch.zeros((self.edge_index.size(1), self.__class__.edge_attr_dim), dtype=torch.float, device=self.device)

    def get_cannon_str(self):
        """
        Creates a cannonical string representation that is identical for all
        topologicaly equivalent graphs

        Returns:
            a string representation for the truss state graph
        """
        node_polar = self.node_attr[:, -2:].numpy()
        sort_ndx = np.lexsort((node_polar[:,1], node_polar[:,0]))
        node_polar = np.rint(node_polar[sort_ndx] * 10).astype(int)
        cmap = {ni:i for i, ni in enumerate(sort_ndx)}
        edge_index = np.vectorize(cmap.__getitem__)(self.edge_index.T)
        edge_index = edge_index[np.lexsort((edge_index[:,1], edge_index[:,0]))]
        srep = (str(node_polar) + str(edge_index)).replace("\n", "")

        return srep


class GraphDataBatch():
    """
    Groups multiple graphs into a single disconnected graph as a batch for training

    Args:
        graphs: a list of Graph objects
        device: torch device to place the batch
    """
    def __init__(self, graphs, device=None):
        if device is None:
            device = graphs[0].device
        GClass = graphs[0].__class__
        self.device = device
        node_counts = [g.node_attr.size(0) for g in graphs]
        node_start = np.cumsum([0] + node_counts)
        self.edge_index = torch.cat([g.edge_index + node_start[i] for i, g in enumerate(graphs)], dim=1)
        self.edge_strut_slots = torch.cat([g.edge_strut_slots for g in graphs], dim=0)
        self.node_attr = torch.cat([g.node_attr for g in graphs], dim=0)
        self.edge_attr = torch.cat([g.edge_attr for g in graphs], dim=0)
        self.global_attr = torch.cat([g.global_attr for g in graphs], dim=0)
        self.node_batch = torch.cat([g.node_batch + i for i, g in enumerate(graphs)], dim=0)

        self.edge_index = self.edge_index.to(device=device)
        self.edge_strut_slots = self.edge_strut_slots.to(device=device)
        self.node_attr = self.node_attr.to(device=device)
        self.edge_attr = self.edge_attr.to(device=device)
        self.global_attr = self.global_attr.to(device=device)
        self.node_batch = self.node_batch.to(device=device)

        assert self.node_attr.size(1) == GClass.node_attr_dim
        assert self.edge_attr.size(1) == GClass.edge_attr_dim
        assert self.global_attr.size(1) == GClass.global_attr_dim