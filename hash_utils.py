#################################################################
### Hash Encoding Utils
#################################################################
import pdb
import torch
from tqdm import tqdm

from ray_utils import get_rays, get_ray_directions, get_ndc_rays

BOX_OFFSETS_3D = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                               device='cuda')
BOX_OFFSETS_4D = torch.tensor([[[i,j,k,l] for i in [0, 1] for j in [0, 1] for k in [0, 1] for l in [0, 1]]],
                               device='cuda')

def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
    
    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result


def get_voxel_vertices(xyzt, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3 or B x 4
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    keep_mask = xyzt==torch.max(torch.min(xyzt, box_max), box_min)
    if not torch.all(xyzt <= box_max) or not torch.all(xyzt >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        # pdb.set_trace()
        # xyzt = torch.clamp(xyzt, min=box_min, max=box_max)
        xyzt = torch.max(torch.min(xyzt, box_max), box_min)

    grid_size = (box_max-box_min)/resolution

    bottom_left_idx = torch.floor((xyzt-box_min)/grid_size).int()
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0 for _ in range(voxel_min_vertex.shape[-1])]).to(voxel_min_vertex.device)*grid_size

    if xyzt.shape[-1]==3:
        voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS_3D
    elif xyzt.shape[-1]==4:
        voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS_4D

    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask


def get_interval_vertices(t, bounding_range, resolution, log2_hashmap_size):
    '''
    t: 1D time coordinates of samples. B x 1
    bounding_box: min and max t coordinates
    resolution: number of intervals per axis
    '''
    range_min, range_max = bounding_range

    if not torch.all(t <= range_max) or not torch.all(t >= range_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        pdb.set_trace()
        t = torch.clamp(t, min=range_min, max=range_max)

    grid_size = (range_max-range_min)/resolution

    left_idx = torch.floor((t-range_min)/grid_size).int() # B x 1
    right_idx = left_idx + 1 # B x 1
    interval_indices = torch.stack([left_idx, right_idx], dim=-2)
    interval_min_vertex = left_idx*grid_size + range_min
    interval_max_vertex = interval_min_vertex + torch.tensor([1.0])*grid_size

    hashed_interval_indices = hash(interval_indices, log2_hashmap_size)

    return interval_min_vertex, interval_max_vertex, hashed_interval_indices


# def get_bbox3d_for_epickitchens(metas, H, W, near, far):
#     focal = metas["intrinsics"][0][0]
#     poses = [torch.FloatTensor(metas["poses"][str(idx)]) for idx in metas["ids_train"]]

#     # ray directions in camera coordinates
#     directions = get_ray_directions(H, W, focal)

#     min_bound = [100, 100, 100]
#     max_bound = [-100, -100, -100]
#     points = []
#     for pose in poses:
#         rays_o, rays_d = get_rays(directions, pose)

#         def find_min_max(pt):
#             for i in range(3):
#                 if(min_bound[i] > pt[i]):
#                     min_bound[i] = pt[i]
#                 if(max_bound[i] < pt[i]):
#                     max_bound[i] = pt[i]
#             return

#         for i in [0, W-1, H*W-W, H*W-1]:
#             min_point = rays_o[i] + near*rays_d[i]
#             max_point = rays_o[i] + far*rays_d[i]
#             points += [min_point, max_point]
#             find_min_max(min_point)
#             find_min_max(max_point)

#     return (torch.tensor(min_bound)-torch.tensor([4.0,4.0,4.0]), torch.tensor(max_bound)+torch.tensor([4.0,4.0,4.0]))

def get_bbox3d_for_epickitchens(metas, H, W, near, far):
    focal = metas["intrinsics"][0][0]
    poses = [torch.FloatTensor(metas["poses"][str(idx)]) for idx in metas["ids_train"]]

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)
    directions = directions.view(-1,3)
    directions = directions[[0, W-1, H*W-W, H*W-1]]

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]
    print("Getting bounding box")
    for pose in tqdm(poses):
        rays_o, rays_d = get_rays(directions, pose)

        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in range(4):
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([4.0,4.0,4.0]), torch.tensor(max_bound)+torch.tensor([4.0,4.0,4.0]))


def get_bbox3d_for_epickitchens_incameraframe(metas, H, W, near, far):
    focal = metas["intrinsics"][0][0]

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    def find_min_max(pt):
        for i in range(3):
            if(min_bound[i] > pt[i]):
                min_bound[i] = pt[i]
            if(max_bound[i] < pt[i]):
                max_bound[i] = pt[i]
        return

    for i in [0, -1]:
        for j in [0, -1]:
            min_point = near*directions[i,j]
            max_point = far*directions[i,j]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([0.1, 0.1, 0.1]), torch.tensor(max_bound)+torch.tensor([0.1, 0.1, 0.1]))