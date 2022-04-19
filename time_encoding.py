import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from hash_encoding import HashEmbedder
from gru import GRUModel


def triangle_wave(t, tau):
    # if torch.abs(t-tau) > 1:
    #     return torch.tensor(0.0)
    # else:
    #     return 1.0-torch.abs(t-tau)
    return (1.0-torch.abs(t-tau)) * (torch.abs(t-tau) < 1)


class XYZ_TimeOnOff_Encoding(nn.Module):
    def __init__(self, xyz_encoder, t_bounds, t_embed_dim=32, t_hidden_dim=64):
        super(XYZ_TimeOnOff_Encoding, self).__init__()
        self.xyz_encoder = xyz_encoder
        self.t_bounds = t_bounds
        self.time_idxs = time_idxs = [t for t in range(int(t_bounds[0]), int(t_bounds[1]))]
        self.t_embed_dim = t_embed_dim
        
        self.time_coeff_MLP = nn.Sequential(
            nn.Linear(xyz_encoder.out_dim, t_hidden_dim),
            nn.ReLU(),
            nn.Linear(t_hidden_dim, t_hidden_dim),
            nn.ReLU(),
            nn.Linear(t_hidden_dim, len(time_idxs)),
            nn.Sigmoid()
        )
        
        # self.time_embeddings = nn.Linear(len(time_idxs), t_embed_dim, bias=False)
        self.time_embeddings = nn.Linear(1, t_embed_dim, bias=False)

        self.triangle_fns = []
        for tau in time_idxs:
            self.triangle_fns.append(lambda t, fn=triangle_wave, tau=tau : fn(t, tau))
        
        self.out_dim = xyz_encoder.out_dim + t_embed_dim

    def forward(self, xyzt):
        xyz, t = xyzt[..., :3], xyzt[..., 3]
        xyz_embed, _ = self.xyz_encoder(xyz)
        triangle_out = torch.stack([fn(t) for fn in self.triangle_fns], dim=-1)
        pdb.set_trace()
        time_coeffs = self.time_coeff_MLP(xyz_embed) * triangle_out
        # time_embed = self.time_embeddings(time_coeffs)
        time_embed = self.time_embeddings(time_coeffs.sum(dim=-1).unsqueeze(-1))
        return torch.cat([xyz_embed, time_embed], dim=-1)


class XYZ_TimePiecewiseConstant(nn.Module):
    def __init__(self, xyzt_bounds, n_levels=16, n_features_per_level=2,\
                base_resolution=128, finest_resolution=4096, n_pieces=10,
                init_temperature=100.0):
        super(XYZ_TimePiecewiseConstant, self).__init__()
        self.xyzt_bounds = xyzt_bounds
        self.xyz_bounds = xyzt_bounds[0][:3], xyzt_bounds[1][:3]
        self.t_bounds = xyzt_bounds[0][3], xyzt_bounds[1][3]
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.n_pieces = n_pieces
        self.temperature = init_temperature
        self.b = torch.exp((torch.log(self.finest_resolution.float())-torch.log(self.base_resolution.float()))/(n_levels-1))

        # Define embeddings
        # self.embeddings = nn.ModuleList([nn.Embedding(torch.floor(self.base_resolution*self.b**i).int().item()**3 * n_pieces, \
                                        # self.n_features_per_level) for i in range(n_levels)])
        self.embeddings = nn.ModuleList([
            HashEmbedder(bounding_box=self.xyz_bounds,
                         n_levels=self.n_levels,
                         n_features_per_level=self.n_features_per_level,
                         base_resolution=self.base_resolution,
                         finest_resolution=self.finest_resolution
            ) for _ in range(n_pieces)
        ])
        
        self.out_dim = self.n_features_per_level * n_levels

        # MLP to get time anchors
        # self.time_anchor_MLP = nn.Sequential(
        #                             nn.Linear(self.out_dim, 64, bias=False),
        #                             nn.Softplus(),
        #                             nn.Linear(64, 1, bias=False),
        #                             nn.Sigmoid()
        #                         )
        self.time_anchor_MLP = GRUModel(input_dim=self.out_dim, hidden_dim=64, num_layers=1, output_dim=1, bias=False)

    def forward(self, xyzt):
        xyz, t = xyzt[..., :3], xyzt[..., 3]
        
        xyz_embeddings = [self.embeddings[i](xyz)[0] for i in range(self.n_pieces)] # list: n_pieces x (B, n_features_per_level*n_levels)
        xyz_embeddings = torch.stack(xyz_embeddings, dim=-2) # (B, n_pieces, n_features_per_level*n_levels)

        # get time anchors
        # To ensure time_anchors are monotonically increasing, we predict delta_t and then cumulatively sum them
        delta_time_anchors = self.time_anchor_MLP(xyz_embeddings).squeeze(-1) # (B, n_pieces)
        delta_time_anchors = delta_time_anchors * 2*(self.t_bounds[1]-self.t_bounds[0])/self.n_pieces
        time_anchors = self.t_bounds[0] + torch.cumsum(delta_time_anchors, dim=-1)
        # pdb.set_trace()

        # Get piecewise constant embedding
        weights = F.softmax(-torch.abs(t[...,None]-time_anchors)/self.temperature, dim=-1) # (B, n_pieces)
        # soft clustering using inverse distance 
        # weights = 1./(torch.abs(t[...,None]-time_anchors)/self.temperature + 1e-6) # (B, n_pieces)
        weights = weights / weights.sum(dim=-1, keepdim=True) # (B, n_pieces)
        xyzt_embedding = (weights[...,None] * xyz_embeddings).sum(dim=-2) # (B, n_features_per_level*n_levels)

        return xyzt_embedding