
from graph_data import GraphData
from models.sh_net import SHNet, LGC, GATConvW, GCNConvW, GINConvW, GMMConvW
from models.patchy_san import PatchySan

# trained with 2000 epochs, lr=1e-2

shared_args = {
        'nnet': SHNet,
        'layers': 2,
        'edge_attr_in': GraphData.edge_attr_dim, 
        'node_attr_in': GraphData.node_attr_dim,
        'node_dims': 50,
        'heuristic_readout_hdims': 100,
        'heuristic_readout_hlayers': 1,
        'gate_hdims': 100,
        'gate_hlayers': 1,
}

args = {
    'LGC': { 
        'graph_conv': LGC,
        'node_in_degree': 6,
        'node_mlp1_hlayers': 2,
        'node_mlp1_hdims': 100,
        'node_mlp2_hlayers': 1,
        'node_mlp2_hdims': 100,
    },
    'GAT': {
        'graph_conv': GATConvW,
    },
    'GCN': {
        'graph_conv': GCNConvW,
    },
    'GIN': {
        'graph_conv': GINConvW,
        'node_mlp1_hlayers': 2,
        'node_mlp1_hdims': 100,
    },
    'MoNet': {
        'graph_conv': GMMConvW,
        'kernel_size': 50,
    },
    'PatchySan': {
        'nnet': PatchySan,
        'edge_attr_in': GraphData.edge_attr_dim, 
        'node_attr_in': GraphData.node_attr_dim,
        'node_dims': 50,
        'heuristic_readout_hdims': 100,
        'heuristic_readout_hlayers': 1,
        'w': 5,
        'k': 7
    },
}

for net_name in args.keys():
    if not net_name == 'PatchySan':
        args[net_name] = {**args[net_name], **shared_args}