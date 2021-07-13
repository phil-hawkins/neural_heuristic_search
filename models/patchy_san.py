import os,sys
sys.path.insert(0, os.path.abspath('.'))
import torch
from torch.nn import Module, Linear
from torch_geometric.utils import k_hop_subgraph
from models.wl_conv import WLConv

from models.utils import MLP

class PatchySan(Module):
    def __init__(self, args):
        super().__init__()
        self._args = args
        self._node_lin = Linear(args['node_attr_in'], args['node_dims'])
        self._wl = WLConv()
        self._conv = Linear(
            in_features=args['node_dims'] * self._args['k'], 
            out_features=args['node_dims']
        )
        self._heuristic_readout = MLP(
            in_features=args['node_dims'] * self._args['w'], 
            hidden_dims=args['heuristic_readout_hdims'],
            hidden_layers=args['heuristic_readout_hlayers'],
            out_features=1
        )

    def get_neighbourhood(self, node_idx, edge_index, k):
        node_list = [node_idx]
        nodes = set(node_list)
        while len(nodes) < k:
            node_idx = node_list.pop(0)
            node_list.extend(k_hop_subgraph(node_idx=node_idx, num_hops=1, edge_index=edge_index)[0].tolist())
            nodes.update(node_list)

        return torch.tensor(list(nodes)[:k], device=edge_index.device)

    def get_cnn_frame(self, x, edge_index, node_batch):
        """
        Rearrange the node attributes as a PATCH-SAN structured tensor using the Weisfeiler Lehman operator
        as the canonical node labeling scheme
        """
        batch_sz = node_batch[-1]+1
        out = torch.zeros((batch_sz, self._args['w'], self._args['k'], self._args['node_dims']), device=x.device)
        # label nodes canonically according to WL1
        wl_labels = self._wl(torch.ones(x.size(0), 1), edge_index)
        
        for graph_idx in range(batch_sz):
            # get the next graph from the the batch super-graph
            g_node_mask = (node_batch == graph_idx)
            _, g_sort_ndx = torch.sort(wl_labels[g_node_mask])
            # convert the subgraph index to a full batch index
            g_sort_ndx = g_node_mask.nonzero(as_tuple=True)[0][g_sort_ndx].tolist()

            # find the neighbourhoods of the first w nodes
            for i, i_node_idx in enumerate(g_sort_ndx[:self._args['w']]):
                k = min(self._args['k'], len(g_sort_ndx))
                n_idx = self.get_neighbourhood(i_node_idx, edge_index, k)
                # order the neighbourhood nodes canonically
                n_order = (wl_labels[n_idx] - wl_labels[i_node_idx]).abs()
                _, n_sort_ndx = torch.sort(n_order)
                n_idx = n_idx[n_sort_ndx]
                # add the node attributes to the neighbourhood vector
                for j, j_node_idx in enumerate(n_idx.tolist()):
                    out[graph_idx, i, j] = x[j_node_idx]

        out = out.view(batch_sz * self._args['w'], self._args['k'] * self._args['node_dims'])

        return out

    def forward(self, data):
        """
        """
        x = self._node_lin(data.node_attr)
        x = self.get_cnn_frame(x, data.edge_index, data.node_batch)
        x = self._conv(x).relu()
        x = x.view(-1, self._args['w'] * self._args['node_dims'])
        h = self._heuristic_readout(x)

        return h