import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import pdb

from utils import get_bbox3d_for_epickitchens

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_epic_kitchens_data(basedir, half_res=False):
    splits = ["train", "val", "test"]
    with open(os.path.join(basedir, 'meta.json'), 'r') as fp:
        metas = json.load(fp)

    # all_ids, train_ids, val_ids, test_ids = metas['ids_all'], metas['ids_train'], metas['ids_val'], metas['ids_test']

    all_imgs = []
    all_poses = []
    all_frame_idxs = []
    counts = [0]
    for s in splits:
        imgs = []
        poses = []
        frame_indices = []
            
        for idx in metas['ids_'+s]:
            fname = os.path.join(basedir, "frames", metas['images'][str(idx)])
            frame_indices.append(idx)
            imgs.append(imageio.imread(fname))
            poses.append(np.array(metas['poses'][str(idx)]))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_frame_idxs += frame_indices

    all_frame_idxs = np.array(all_frame_idxs)    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    focal = metas["intrinsics"][0][0]
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    near, far = min(metas["nears"].values()), max(metas["fars"].values())

    xyz_bounding_box = get_bbox3d_for_epickitchens(metas, H, W, near=near, far=far)
    min_time = torch.FloatTensor([min(all_frame_idxs)]).to(xyz_bounding_box[0].device)
    max_time = torch.FloatTensor([max(all_frame_idxs)]).to(xyz_bounding_box[1].device)
    xyzt_bounding_box = (torch.cat([xyz_bounding_box[0], min_time]), torch.cat([xyz_bounding_box[1], max_time]))

    return imgs, poses, render_poses, [H, W, focal], i_split, xyzt_bounding_box, [near, far], all_frame_idxs
