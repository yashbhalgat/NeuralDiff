import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from hash_utils import get_voxel_vertices, get_interval_vertices

class HashEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size

        if hasattr(base_resolution, "__len__") and len(base_resolution)>1:
            # if base_resolution is a list, ...
            self.base_resolution = torch.tensor(base_resolution)
        else:
            # if just one value is given, use it for all 4 (or 3) dimensions
            self.base_resolution = torch.tensor([base_resolution for _ in range(4)])
        
        if hasattr(finest_resolution, "__len__") and len(finest_resolution)>1:
            self.finest_resolution = torch.tensor(finest_resolution)
        else:
            self.finest_resolution = torch.tensor([finest_resolution for _ in range(4)])

        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution.float())-torch.log(self.base_resolution.float()))/(n_levels-1)) # 4 values

        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                        self.n_features_per_level) for _ in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()
        
    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
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

    def quadrilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 4
        voxel_min_vertex: B x 4
        voxel_max_vertex: B x 4
        voxel_embedds: B x 16 x 2
        '''
        # extension of: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 4

        # step 1
        # 0->0000, 1->0001, 2->0010, 3->0011, 4->0100, 5->0101, 6->0110, 7->0111
        # 8->1000, 9->1001, 10->1010, 11->1011, 12->1100, 13->1101, 14->1110, 15->1111
        c000 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,8]*weights[:,0][:,None]
        c001 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,9]*weights[:,0][:,None]
        c010 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,10]*weights[:,0][:,None]
        c011 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,11]*weights[:,0][:,None]
        c100 = voxel_embedds[:,4]*(1-weights[:,0][:,None]) + voxel_embedds[:,12]*weights[:,0][:,None]
        c101 = voxel_embedds[:,5]*(1-weights[:,0][:,None]) + voxel_embedds[:,13]*weights[:,0][:,None]
        c110 = voxel_embedds[:,6]*(1-weights[:,0][:,None]) + voxel_embedds[:,14]*weights[:,0][:,None]
        c111 = voxel_embedds[:,7]*(1-weights[:,0][:,None]) + voxel_embedds[:,15]*weights[:,0][:,None]

        # step 2
        c00 = c000*(1-weights[:,1][:,None]) + c100*weights[:,1][:,None]
        c01 = c001*(1-weights[:,1][:,None]) + c101*weights[:,1][:,None]
        c10 = c010*(1-weights[:,1][:,None]) + c110*weights[:,1][:,None]
        c11 = c011*(1-weights[:,1][:,None]) + c111*weights[:,1][:,None]

        # step 3
        c0 = c00*(1-weights[:,2][:,None]) + c10*weights[:,2][:,None]
        c1 = c01*(1-weights[:,2][:,None]) + c11*weights[:,2][:,None]

        # step 3
        c = c0*(1-weights[:,3][:,None]) + c1*weights[:,3][:,None]

        return c

    def forward(self, x):
        # x is 3D (xyz) or 4D (xyzt) point position: B x 3 or B x 4
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(\
                                                x, self.bounding_box, \
                                                resolution[:x.shape[-1]], self.log2_hashmap_size)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            if x.shape[-1]==3:
                x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            elif x.shape[-1]==4:
                x_embedded = self.quadrilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        keep_mask = keep_mask.sum(dim=-1)==keep_mask.shape[-1]
        return torch.cat(x_embedded_all, dim=-1), keep_mask


class Linear_HashEmbedder(nn.Module):
    def __init__(self, bounding_range, n_levels=4, n_features_per_level=2,\
                log2_hashmap_size=11, base_resolution=16, finest_resolution=2048):
        super(Linear_HashEmbedder, self).__init__()
        self.bounding_range = bounding_range
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution.float())-torch.log(self.base_resolution.float()))/(n_levels-1))

        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                        self.n_features_per_level) for _ in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()
        

    def linear_interp(self, x, interval_min, interval_max, interval_embedds):
        '''
        x: B x 1
        voxel_min_vertex: B x 1
        voxel_max_vertex: B x 1
        voxel_embedds: B x 2 x 2
        '''
        weight = (x - interval_min)/(interval_max-interval_min) # B x 1
        c = interval_embedds[:,0]*(1-weight) + interval_embedds[:,1]*weight
        return c

    def forward(self, x):
        # x is 1D point position: B x 1
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            interval_min_vertex, interval_max_vertex, hashed_interval_indices = get_interval_vertices(\
                                                x, self.bounding_range, \
                                                resolution, self.log2_hashmap_size)
 
            interval_embedds = self.embeddings[i](hashed_interval_indices)

            x_embedded = self.linear_interp(x, interval_min_vertex, interval_max_vertex, interval_embedds)
            x_embedded_all.append(x_embedded)

        return torch.cat(x_embedded_all, dim=-1)


class XYZplusT_HashEmbedder(nn.Module):
    def __init__(self, bounding_box, xyz_embedder=None, t_embedder=None, \
                n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(XYZplusT_HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        if xyz_embedder is None:
            self.n_levels = n_levels
            self.base_resolution = torch.tensor(base_resolution)
            self.finest_resolution = torch.tensor(finest_resolution)
            self.b = torch.exp((torch.log(self.finest_resolution.float())-torch.log(self.base_resolution.float()))/(n_levels-1))
            self.log2_hashmap_size = log2_hashmap_size
            self.xyz_bounding_box = bounding_box[0][:3], bounding_box[1][:3]
            self.xyz_embedder = HashEmbedder(self.xyz_bounding_box, finest_resolution=finest_resolution, 
                                            log2_hashmap_size=log2_hashmap_size)
        else:
            self.n_levels = xyz_embedder.n_levels
            self.base_resolution = torch.tensor(xyz_embedder.base_resolution)
            self.finest_resolution = torch.tensor(xyz_embedder.finest_resolution)
            self.b = torch.exp((torch.log(self.finest_resolution.float())-torch.log(self.base_resolution.float()))/(self.n_levels-1))
            self.log2_hashmap_size = xyz_embedder.log2_hashmap_size
            self.xyz_bounding_box = xyz_embedder.bounding_box
            self.xyz_embedder = xyz_embedder

        if t_embedder is None:
            self.t_bounding_range = bounding_box[0][3], bounding_box[1][3]
            self.t_embedder = Linear_HashEmbedder(self.t_bounding_range)
        else:
            self.t_bounding_range = t_embedder.bounding_range
            self.t_embedder = t_embedder
        
        self.out_dim = self.xyz_embedder.out_dim + self.t_embedder.out_dim
        self.time_dim = self.t_embedder.out_dim

    def forward(self, xyzt):
        xyz, t = xyzt[..., :3], xyzt[..., 3].unsqueeze(-1)
        return torch.cat([self.xyz_embedder(xyz), self.t_embedder(t)], dim=-1)


class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
    
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                #result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result
