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