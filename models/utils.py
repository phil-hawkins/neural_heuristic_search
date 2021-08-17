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
