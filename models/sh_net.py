import os,sys
sys.path.insert(0, os.path.abspath('.'))
import torch
from torch.nn import Module, ModuleList, Linear, ReLU
from torch_geometric.nn import GATConv, GINConv, GlobalAttention
from models.utils import MLP
from math import pi


class ConvWrapper(Module):
    def __init__(self, args):
        super().__init__()
        self._args = args

    def forward(self, x, edge_index, edge_slot, edge_attr, u, batch):
        return self.conv(x, edge_index)


class GATConvW(ConvWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.conv = GATConv(
            in_channels=args['node_dims'], 
            out_channels=args['node_dims'] // args['heads'], 
            heads=args['heads'])


class GINConvW(ConvWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.node_mlp = MLP(
            in_features=self._args['node_dims'], 
            hidden_dims=self._args['node_mlp1_hdims'], 
            hidden_layers=self._args['node_mlp1_hlayers'], 
            out_features=self._args['node_dims']
        )
        self.conv = GINConv(nn=self.node_mlp)




class ResBlock(Module):
    def __init__(self, args):
        super().__init__()
        GCM = args['graph_conv']
        self._gc1 = GCM(args)
        self._gc2 = GCM(args)
        self._nl = ReLU()

    def forward(self, x, edge_index, edge_slot, edge_attr, u, batch):
        out = self._gc1(x, edge_index, edge_slot, edge_attr, u, batch)
        out = self._nl(out)
        out = self._gc2(out, edge_index, edge_slot, edge_attr, u, batch)
        out += x
        out = self._nl(out)

        return out


class SHNet(Module):
    def __init__(self, args):
        super().__init__()
        self._args = args
        self._trunk = ModuleList()
        for _ in range(args['layers']):
            self._trunk.append(ResBlock(args))

        self._node_lin = Linear(args['node_attr_in'], args['node_dims'])
        gate_nn = MLP(
            in_features=args['node_dims'],
            hidden_dims=args['gate_hdims'],
            hidden_layers=args['gate_hlayers'],
            out_features=1
        )
        nn = MLP(
            in_features=args['node_dims'],
            hidden_dims=args['heuristic_readout_hdims'],
            hidden_layers=args['heuristic_readout_hlayers'],
            out_features=1
        )
        self._attn = GlobalAttention(gate_nn=gate_nn, nn=nn)
        
    def forward(self, g):
        """
        Returns:
            h: the predicted search heuristic value for each structure graph
        """
        # project the input attributes to the correct dimensions
        x = self._node_lin(g.node_attr)

        for layer in self._trunk:
            x = layer(
                x=x, 
                edge_index=g.edge_index, 
                edge_slot=g.edge_strut_slots,
                edge_attr=g.edge_attr,
                u=g.global_attr, 
                batch=g.node_batch
            )

        h = self._attn(x=x, batch=g.node_batch)

        return h