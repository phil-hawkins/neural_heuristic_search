
from graph_data import GraphData
from models.sh_net import SHNet, GATConvW, GINConvW

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
    'GAT': {
        'graph_conv': GATConvW,
        'heads': 5
    },
    'GIN': {
        'graph_conv': GINConvW,
        'node_mlp1_hlayers': 2,
        'node_mlp1_hdims': 100,
    },
}

for net_name in args.keys():
    args[net_name] = {**args[net_name], **shared_args}