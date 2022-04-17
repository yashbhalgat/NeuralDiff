import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

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
                base_resolution=16, finest_resolution=128, n_pieces=10):
        super(XYZ_TimePiecewiseConstant, self).__init__()
        self.xyzt_bounds = xyzt_bounds
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.n_pieces = n_pieces
        self.b = torch.exp((torch.log(self.finest_resolution.float())-torch.log(self.base_resolution.float()))/(n_levels-1))

        # Define embeddings
        self.embeddings = nn.ModuleList([nn.Embedding(torch.floor(self.base_resolution*self.b**i).int().item()**3 * n_pieces, \
                                        self.n_features_per_level) for i in range(n_levels)])
        
        self.out_dim = self.n_features_per_level * n_levels

        # MLP to get time anchors
        self.time_anchor_MLP = nn.Sequential(
                                    nn.Linear(self.out_dim, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1),
                                    nn.Sigmoid()
                                )

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x n_features_per_level
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def get_voxel_embeddings_at_time(self, embeddings, voxel_indices, xyz_res, t):
        '''
        voxel_indices: (B, 8, 3)
        Returns:
            embedding: (B, 8, n_features_per_level)
        '''
        # flat_idxs: (B, 8)
        flat_idxs = np.ravel_multi_index([voxel_indices[...,0], voxel_indices[...,1], voxel_indices[...,2], t*torch.ones_like(voxel_indices[...,0])], \
                                                (xyz_res, xyz_res, xyz_res, self.n_pieces))
        
        return embeddings(torch.tensor(flat_idxs))

    def get_xyz_embeddings(self, xyz):
        xyz_embeddings = []
        for t in range(self.n_pieces):
            xyz_embedded_at_t = []
            for i in range(self.n_levels):
                xyz_res = torch.floor(self.base_resolution*self.b**i)
                box_min, box_max = self.xyzt_bounds[0][:3], self.xyzt_bounds[1][:3]
                xyz_grid_sz = (box_max-box_min)/xyz_res
                
                bottom_left_idx = torch.floor((xyz-box_min)/xyz_grid_sz).int()
                voxel_min_vertex = bottom_left_idx*xyz_grid_sz + box_min
                voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0 for _ in range(voxel_min_vertex.shape[-1])]).to(voxel_min_vertex.device)*xyz_grid_sz
                BOX_OFFSETS_3D = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], device='cuda')
                # voxel_indices: (B, 8, 3)
                voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS_3D
    
                voxel_embedds = self.get_voxel_embeddings_at_time(self.embeddings[i], voxel_indices.cpu(), xyz_res.int().item(), t)
                xyz_embedded_at_t.append(self.trilinear_interp(xyz, voxel_min_vertex, voxel_max_vertex, voxel_embedds))
            
            xyz_embeddings.append(torch.cat(xyz_embedded_at_t, dim=-1))
            
        return xyz_embeddings # list: n_pieces x (B, n_features_per_level*n_levels)

    def forward(self, xyzt):
        xyz, t = xyzt[..., :3], xyzt[..., 3]
        
        xyz_embeddings = self.get_xyz_embeddings(xyz) # list: n_pieces x (B, n_features_per_level*n_levels)
        xyz_embeddings = torch.stack(xyz_embeddings, dim=-2) # (B, n_pieces, n_features_per_level*n_levels)

        # get time anchors
        time_anchors = [] # --> list: n_pieces x (B, 1)
        for i in range(self.n_pieces):
            time_anchors.append(self.time_anchor_MLP(xyz_embeddings[:,i,:]))
        time_anchors = torch.cat(time_anchors, dim=-1) # (B, n_pieces)
        time_anchors = self.xyzt_bounds[0][3] + (self.xyzt_bounds[1][3]-self.xyzt_bounds[0][3])*time_anchors

        T = 100.0 # temperature
        weights = F.softmax(-torch.abs(t[...,None]-time_anchors)/T, dim=-1) # (B, n_pieces)
        xyzt_embedding = (weights[...,None] * xyz_embeddings).sum(dim=-2) # (B, n_features_per_level*n_levels)

        return xyzt_embedding