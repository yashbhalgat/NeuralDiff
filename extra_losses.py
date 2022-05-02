from itertools import combinations, chain, product
from random import sample
from math import exp, log, floor
from this import d
import torch
from torch import nn
import torch.nn.functional as F
from skimage.color import rgb2hsv
import pdb

from hash_utils import hash
from hash_encoding import HashEmbedder

from run_nerf_helpers import img2mse_with_uncertainty_perray, img2mse, img2mse_perray


def total_variation_loss_3D(embeddings, min_resolution, max_resolution, level, log2_hashmap_size, n_levels=16, scaled=False):
    # Get resolution
    b = exp((log(max_resolution)-log(min_resolution))/(n_levels-1))
    resolution = torch.tensor(floor(min_resolution * b**level))

    # Cube size to apply TV loss
    min_cube_size = min_resolution - 1
    max_cube_size = min_resolution+50 # can be tuned
    if min_cube_size > max_cube_size:
        print("ALERT! min cuboid size greater than max!")
        pdb.set_trace()
    cube_size = torch.floor(torch.clip(resolution/10.0, min_cube_size, max_cube_size)).int()

    # Sample cuboid
    min_vertex = torch.randint(0, resolution-cube_size, (3,))
    idx = min_vertex + torch.stack([torch.arange(cube_size+1) for _ in range(3)], dim=-1)
    cube_indices = torch.stack(torch.meshgrid(idx[:,0], idx[:,1], idx[:,2]), dim=-1)

    hashed_indices = hash(cube_indices, log2_hashmap_size)
    cube_embeddings = embeddings(hashed_indices)

    tv_x = torch.pow(cube_embeddings[1:,:,:]-cube_embeddings[:-1,:,:], 2).sum()
    tv_y = torch.pow(cube_embeddings[:,1:,:]-cube_embeddings[:,:-1,:], 2).sum()
    tv_z = torch.pow(cube_embeddings[:,:,1:]-cube_embeddings[:,:,:-1], 2).sum()

    tv_loss = (tv_x + tv_y + tv_z)/cube_size
    scaled_tv_loss = tv_loss * log(resolution)

    if scaled:
        return scaled_tv_loss
    else:
        return tv_loss
    

def total_variation_loss_4D(embeddings, min_resolution, max_resolution, level, log2_hashmap_size, n_levels=16, scaled=False):
    # Get resolution
    b = exp((log(max_resolution)-log(min_resolution))/(n_levels-1))
    resolution = torch.tensor(floor(min_resolution * b**level))

    # Cube size to apply TV loss
    min_cube_size = 20
    max_cube_size = 50 # can be tuned
    if min_cube_size > max_cube_size:
        print("ALERT! min cuboid size greater than max!")
        pdb.set_trace()
    cube_size = torch.floor(torch.clip(resolution/100.0, min_cube_size, max_cube_size)).int()

    # Sample cuboid
    min_vertex = torch.randint(0, resolution-cube_size, (4,))
    idx = min_vertex + torch.stack([torch.arange(cube_size+1) for _ in range(4)], dim=-1)
    cube_indices = torch.stack(torch.meshgrid(idx[:,0], idx[:,1], idx[:,2], idx[:,3]), dim=-1)

    hashed_indices = hash(cube_indices, log2_hashmap_size)
    cube_embeddings = embeddings(hashed_indices)

    tv_x = torch.pow(cube_embeddings[1:,:,:,:]-cube_embeddings[:-1,:,:,:], 2).sum()
    tv_y = torch.pow(cube_embeddings[:,1:,:,:]-cube_embeddings[:,:-1,:,:], 2).sum()
    tv_z = torch.pow(cube_embeddings[:,:,1:,:]-cube_embeddings[:,:,:-1,:], 2).sum()
    tv_t = torch.pow(cube_embeddings[:,:,:,1:]-cube_embeddings[:,:,:,:-1], 2).sum()

    tv_loss = (tv_x + tv_y + tv_z + tv_t)/cube_size
    scaled_tv_loss = tv_loss * log(resolution)

    if scaled:
        return scaled_tv_loss
    else:
        return tv_loss


def total_variation_loss_1D(embeddings, min_resolution, max_resolution, level, log2_hashmap_size, n_levels=4, scaled=False):
    # Get resolution
    b = exp((log(max_resolution)-log(min_resolution))/(n_levels-1))
    resolution = torch.tensor(floor(min_resolution * b**level))

    bin_size = resolution - 1

    # Sample bin
    min_vertex = torch.randint(0, resolution-bin_size, (1,))
    idx = min_vertex + torch.stack([torch.arange(bin_size+1) for _ in range(1)], dim=-1)
    bin_indices = torch.stack(torch.meshgrid(idx[:,0]), dim=-1)

    hashed_indices = hash(bin_indices, log2_hashmap_size)
    bin_embeddings = embeddings(hashed_indices)

    tv_x = torch.pow(bin_embeddings[1:]-bin_embeddings[:-1], 2).sum()

    tv_loss = tv_x/bin_size
    scaled_tv_loss = tv_loss * log(resolution)

    if scaled:
        return scaled_tv_loss
    else:
        return tv_loss 


def push_pull_loss_xyzt(embeddings, min_resolution, max_resolution, level, log2_hashmap_size, n_levels=16):
    # Get resolution
    b = exp((log(max_resolution)-log(min_resolution))/(n_levels-1))
    
    resolution = torch.tensor(floor(min_resolution * b**level))
    grid_size = 1.0/resolution    # upto a scaling factor

    N = 100
    # sample N unique pairs --> time (t_i, t_j)
    pairs = sample(list(combinations(range(resolution),2)), N)
    t_0 = torch.tensor([p[0] for p in pairs])
    t_1 = torch.tensor([p[1] for p in pairs])

    # generate random xyz cuboid of size 30
    min_vertex = torch.randint(0, resolution-30, (3,))
    idx = min_vertex + torch.stack([torch.arange(30+1) for _ in range(3)], dim=-1)

    # cartesian product of xyz and t
    xyzt_0 = torch.stack(torch.meshgrid(idx[:,0], idx[:,1], idx[:,2], t_0), dim=-1)
    xyzt_1 = torch.stack(torch.meshgrid(idx[:,0], idx[:,1], idx[:,2], t_1), dim=-1)

    # get embeddings
    hashed_indices_0 = hash(xyzt_0, log2_hashmap_size)
    hashed_indices_1 = hash(xyzt_1, log2_hashmap_size)
    embeddings_0 = embeddings(hashed_indices_0)
    embeddings_1 = embeddings(hashed_indices_1)

    # Get Pull and Push loss
    std = 10.0/min_resolution
    std_dist = 0.0001
    dist_sq = torch.pow((t_0-t_1)*grid_size, 2) / std**2
    embed_dist_sq = torch.sum((embeddings_0-embeddings_1)**2, dim=-1) / std_dist**2
    pull = torch.exp(-dist_sq) * embed_dist_sq.mean(dim=(0,1,2))
    push = dist_sq * torch.exp(-embed_dist_sq).mean(dim=(0,1,2))

    return pull.mean() + push.mean()


class PushPullLoss(nn.Module):
    def __init__(self, xyz_bounding_box, base_res, finest_res, log2_hashmap_size, n_levels=16, n_features_per_level=4, n_centroids=10):
        super(PushPullLoss, self).__init__()
        self.xyz_bounding_box = xyz_bounding_box
        self.base_res = base_res
        self.finest_res = finest_res
        self.log2_hashmap_size = log2_hashmap_size - 3 # less 3 for 1/8th size because we have ~10 embedders
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.b = exp((log(finest_res)-log(base_res))/(n_levels-1))
        self.n_centroids = n_centroids
        self.centroid_embedders = nn.ModuleList([
            HashEmbedder(bounding_box=self.xyz_bounding_box,
                         base_resolution=base_res,
                         finest_resolution=finest_res,
                         n_levels=n_levels,
                         n_features_per_level=n_features_per_level,
                         log2_hashmap_size=self.log2_hashmap_size) 
            for _ in range(n_centroids)
        ])

    def _cdist_squared(self, x, y):
        # x: L x L x L x N x D
        # y: L x L x L x M x D
        # return: L x L x L x N x M
        x_norm = (x**2).sum(dim=-1, keepdim=True)
        y_norm = (y**2).sum(dim=-1, keepdim=True).transpose(-1,-2)
        return x_norm + y_norm - 2*torch.matmul(x, y.transpose(-1,-2))

    def forward(self, xyzt_embedder):
        total_push, total_pull = 0.0, 0.0
        for level in range(self.n_levels):
            resolution = torch.tensor(floor(self.base_res * self.b**level))
            # generate random xyz cuboid
            cube_size = floor(torch.clip(resolution/100, 20, 30))
            min_vertex = torch.randint(0, resolution-cube_size, (3,))
            idx = min_vertex + torch.stack([torch.arange(cube_size+1) for _ in range(3)], dim=-1)
            
            # sample cube_size number of time points
            t = torch.tensor(sample(list(range(resolution)), int(cube_size*1.5)))
            xyzt = torch.stack(torch.meshgrid(idx[:,0], idx[:,1], idx[:,2], t), dim=-1) # N x N x N x 100 x 4coords
            hashed_xyzt = hash(xyzt, xyzt_embedder.log2_hashmap_size) # N x N x N x 100
            xyzt_embeddings = xyzt_embedder.embeddings[level](hashed_xyzt) # N x N x N x 100 x 2
        
            # get centroid embeddings
            xyz = torch.stack(torch.meshgrid(idx[:,0], idx[:,1], idx[:,2]), dim=-1) # N x N x N x 3coords
            hashed_xyz = hash(xyz, self.log2_hashmap_size) # N x N x N
            centroids = torch.stack([self.centroid_embedders[i].embeddings[level](hashed_xyz) for i in range(self.n_centroids)], dim=-2) # N x N x N x 10 x 2

            # compute soft assignment
            dists_from_centroids = self._cdist_squared(xyzt_embeddings, centroids) # N x N x N x 100 x 10
            soft_assignment = torch.exp(-dists_from_centroids/(2*0.0001**2)) # N x N x N x 100 x 10
            soft_assignment = soft_assignment / (soft_assignment.sum(dim=-1, keepdim=True)+1e-10) # N x N x N x 100 x 10
            soft_centroids = soft_assignment @ centroids # N x N x N x 100 x 2

            # compute push and pull loss
            total_pull += ((xyzt_embeddings-soft_centroids)/1.0).pow(2).sum(dim=-1).mean()
            # torch.cdist(soft_centroids, soft_centroids) --> N x N x N x 10 x 10
            # total_push += torch.exp(-torch.cdist(soft_centroids, soft_centroids, p=2).pow(2)/(2*1.0**2)).mean()
            total_push += (1/(1e-15+torch.cdist(soft_centroids, soft_centroids, p=2).pow(2)/(2*1.0**2))).mean()

        print("Push: {}, Pull: {}".format(total_push, total_pull))
        return total_push + total_pull


class BGguidedLoss(nn.Module):
    def __init__(self):
        super(BGguidedLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.threshold_param = nn.Parameter(torch.tensor(-2.0))
        self.steepness = nn.Parameter(torch.tensor(0.1))

    def get_diff(self, gt, BG_map):
        # return torch.abs(gt - BG_map.detach().clone()).max(dim=-1)[0]
        gt_hsv, BG_hsv = torch.tensor(rgb2hsv(gt.cpu())).to(BG_map.device), torch.tensor(rgb2hsv(BG_map.detach().clone().cpu())).to(BG_map.device)
        return torch.norm(gt_hsv[...,(0,2)] - BG_hsv[...,(0,2)], dim=-1)

    def get_mask(self, gt, BG_map):
        # threshold = 1.0 - torch.sigmoid(self.threshold_param)
        threshold = 1.414*(1.0 - torch.sigmoid(self.threshold_param))
        diff = self.get_diff(gt, BG_map)
        mask = torch.sigmoid((diff-threshold)/0.1)
        return mask

    def forward(self, gt, BG_map, FG_map, FG_acc, FG_uncertainties, iter):
        threshold = 1.414*(1.0 - torch.sigmoid(self.threshold_param))

        diff = self.get_diff(gt, BG_map)
        # diff = torch.abs(gt - BG_map.detach().clone()).max(dim=-1)[0]
        # mask = torch.sigmoid((diff-threshold)/F.softplus(self.steepness))
        mask = torch.sigmoid((diff-threshold)/0.1)

        if iter > 300:
            BG_loss = img2mse_perray(gt, BG_map) * (1-mask)
            # mask_loss = torch.mean(-mask*torch.log(FG_acc) - (1.0-mask)*torch.log(1.0-FG_acc))
            mask_loss = 0.0

            # FG_loss = img2mse_with_uncertainty_perray(gt, FG_map, FG_uncertainties) * F.relu(torch.exp((diff-threshold))-1.0)
            FG_loss = img2mse_with_uncertainty_perray(gt, FG_map, FG_uncertainties) * mask

            return BG_loss.mean(dim=-1) + FG_loss.mean(dim=-1) + 0.001*mask_loss
        else:
            BG_loss = img2mse(gt, BG_map)
            return BG_loss.mean(dim=-1)


class BGguidedSparsityWeights(nn.Module):
    def __init__(self):
        super(BGguidedSparsityWeights, self).__init__()
        self.threshold_param = nn.Parameter(torch.tensor(-2.0))

    def get_diff(self, gt, BG_map):
        # return torch.abs(gt - BG_map.detach().clone()).max(dim=-1)[0]
        gt_hsv, BG_hsv = torch.tensor(rgb2hsv(gt.cpu())).to(BG_map.device), torch.tensor(rgb2hsv(BG_map.detach().clone().cpu())).to(BG_map.device)
        return torch.norm(gt_hsv - BG_hsv, dim=-1)

    def forward(self, gt, BG_map):
        threshold = 1.732*(1.0 - torch.sigmoid(self.threshold_param))
        diff = self.get_diff(gt, BG_map)
        
        weights = torch.exp(-3.*(diff-threshold)) # lower the diff, higher the sparsity constraint

        return weights