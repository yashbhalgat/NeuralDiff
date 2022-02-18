import warnings

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import torch
import torchvision
from PIL import Image

import model
import opt
import train
from dataset import EPICDiff
from evaluation.utils import tqdm


def set_deterministic():

    import random

    import numpy
    import torch

    torch.manual_seed(0)
    random.seed(0)
    numpy.random.seed(0)
    torch.backends.cudnn.benchmark = False


def adjust_jupyter_argv():
    import sys

    sys.argv = sys.argv[:1]


def write_mp4(name, frames, fps=10):
    imageio.mimwrite(name + ".mp4", frames, "mp4", fps=fps)


def overlay_image(im, im_overlay, coord=(100, 70)):
    # assumes that im is 3 channel and im_overlay 4 (with alpha)
    alpha = im_overlay[:, :, 3]
    offset_rows = im_overlay.shape[0]
    offset_cols = im_overlay.shape[1]
    row = coord[0]
    col = coord[1]
    im[row : row + offset_rows, col : col + offset_cols, :] = (
        1 - alpha[:, :, None]
    ) * im[row : row + offset_rows, col : col + offset_cols, :] + alpha[
        :, :, None
    ] * im_overlay[
        :, :, :3
    ]
    return im


def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:
        # single pytorch model
        parameters += list(models.parameters())
    return parameters


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = torchvision.transforms.ToTensor()(x_)  # (3, H, W)
    return x_


def assign_appearance(ids_train, ids_unassigned):
    # described in experiments, (3) NeRF-W: reassign each test embedding to closest train embedding
    ids = sorted(ids_train + ids_unassigned)
    g = {}
    for id in ids_unassigned:
        pos = ids.index(id)
        if pos == 0:
            # then only possible to assign to next embedding
            id_reassign = ids[1]
        elif pos == len(ids) - 1:
            # then only possible to assign to previous embedding
            id_reassign = ids[pos - 1]
        else:
            # otherwise the one that is closes according to frame index
            id_prev = ids[pos - 1]
            id_next = ids[pos + 1]
            id_reassign = min(
                (abs(ids[pos] - id_prev), id_prev), (abs(ids[pos] - id_next), id_next)
            )[1]
        g[ids[pos]] = id_reassign
    return g


def init_model(ckpt_path, dataset):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    opt_hp = opt.get_opts(dataset.vid)
    for j in ckpt["hyper_parameters"]:
        setattr(opt_hp, j, ckpt["hyper_parameters"][j])
    model = train.NeuralDiffSystem(
        opt_hp, train_dataset=dataset, val_dataset=dataset
    ).cuda()
    model.load_state_dict(ckpt["state_dict"])

    g_test = assign_appearance(dataset.img_ids_train, dataset.img_ids_test)
    g_val = assign_appearance(dataset.img_ids_train, dataset.img_ids_val)

    for g in [g_test, g_val]:
        for i, i_train in g.items():
            model.embedding_a.weight.data[i] = model.embedding_a.weight.data[
                i_train
            ]

    return model



#################################################################
### Hash Encoding Utils
#################################################################
from ray_utils import get_rays, get_ray_directions, get_ndc_rays

BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                               device='cuda')

def hash(coords, log2_hashmap_size):
    '''
    coords: 3D coordinates. B x 3
    log2T:  logarithm of T w.r.t 2
    '''
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    return ((1<<log2_hashmap_size)-1) & (x*73856093 ^ y*19349663 ^ z*83492791)


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        pdb.set_trace()
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution

    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0])*grid_size

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices
