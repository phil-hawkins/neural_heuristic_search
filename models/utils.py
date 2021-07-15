import torch
import torch.nn as nn
import time


class MLP(nn.Module):
    def __init__(self, in_features=3, hidden_dims=150, hidden_layers=3, out_features=256):
        super().__init__()
        self._mlist = nn.ModuleList([
            nn.Linear(in_features, hidden_dims),
            nn.ReLU()
        ])
        for _ in range(hidden_layers):
            self._mlist.append(nn.Linear(hidden_dims, hidden_dims))
            self._mlist.append(nn.ReLU())
        self._mlist.append(nn.Linear(hidden_dims, out_features))

    def forward(self, x):
        for layer in self._mlist:
            x = layer(x)

        return x

    
class Timer():
    """
    handles measuring elapsed time and checking for timeouts
    """
    def __init__(self, timeout=0):
        """
        Args:
            timeout: timeout is seconds or unlimited if 0
        """
        self._tic = time.time()
        self._timeout = self._tic + timeout
        self._no_timeout = (timeout == 0)

    @property
    def is_timed_out(self):
        return False if self._no_timeout else (time.time() > self._timeout)

    @property
    def timing(self):
        return time.time() - self._tic


def scatter_slots(out_size, node_index, slot_index, src):
    """
    scatters the 2D src tensor rows vectors to slots (dim 1) in the 3D output tensor 

    Args:
        out_size: size of output tensor
        node_index: the node to assign each src vector to [E]
        slot_index: the slot to assign each src vector to [E]
        src: the src vectors by edge [E, attr]

    Returns:
        a tensor of size=out_size with src rows scattered according to the indicies 
        with zeros elsewhere
    """
    node_index = node_index.unsqueeze(1).expand_as(src)
    out = torch.zeros(out_size, dtype=src.dtype, device=src.device)
    slot_count = out_size[1]
    for s in range(slot_count):
        s_msk = slot_index == s
        out[:, s, :].scatter_(0, node_index[s_msk], src[s_msk])

    return out

def get_sequence_index(batch):
    # get the length of the longest sequence
    max_seq = batch.unique(return_counts=True)[1].max().item()
    l_batch = batch[-1].item()
    i_a = max_seq
    seq_index = []
    for i_b in batch.flip([0]):
        if i_b == l_batch:
            i_a -= 1
        else:
            i_a = max_seq - 1
            l_batch = i_b
        seq_index.append(i_a)
    seq_index.reverse()
    
    index = torch.stack([torch.tensor(seq_index, device=batch.device), batch], dim=0)
    return index

def pack_node_sequence(x, batch):
    """
    packs the nodes into sequences by graph in batch 

    Args:
        x: node input [nodes, channels]
        batch: graph index of each node in batch [nodes]
        
    Returns:
        a packed sequence object with a sequence of nodes attributes for each graph in the batch
    """
    # get the length of the longest sequence
    lengths = batch.unique(return_counts=True)[1].tolist()
    max_seq_length = max(lengths)
    batch_count = batch.max().item() + 1
    channels = x.size(1)

    l_batch = -1
    i_a = 0
    seq_index = []
    for i_b in batch:
        if i_b == l_batch:
            i_a += 1
        else:
            i_a = 0
            l_batch = i_b
        seq_index.append(i_a)
    
    seq_idx = torch.stack([batch, torch.tensor(seq_index, device=batch.device)], dim=0)
    x = torch.sparse_coo_tensor(
        size=(batch_count, max_seq_length, channels), 
        indices=seq_idx, 
        values=x, 
        device=x.device).to_dense()
    x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

    return x

def unpack_rnn_out(x):   
    """
    unpacks the final RNN output vector from the packed sequences output from a RNN

    Args:
        x: PackedSequence of RNN output
        
    Returns:
        tensor with RNN output as a row vector for each graph in the batch
    """
    x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
    x = x[torch.arange(lengths.numel(), device=x.device), lengths-1]

    return x

class AverageMeter(object):
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
