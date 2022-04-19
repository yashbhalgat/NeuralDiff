import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GRUCell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        
        return hy


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bias=False):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
         
        # Number of hidden layers
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, bias=bias, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x: B x L x d
        '''
        gru_output = self.gru(x)[0] # B x L x d
        out = self.fc(gru_output) # B x L x d
        return out