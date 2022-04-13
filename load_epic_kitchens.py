import os
import random
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import pdb
from tqdm import tqdm
from joblib import Parallel, delayed

from hash_utils import get_bbox3d_for_epickitchens, get_bbox3d_for_epickitchens_incameraframe

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


def load_epic_kitchens_data(basedir, half_res=False, get_masks=True, parallel=False):
    splits = ["train", "val", "test"]
    with open(os.path.join(basedir, 'meta.json'), 'r') as fp:
        metas = json.load(fp)

    ### Define function for parallel loading
    def load_data_single(idx):
        fname = os.path.join(basedir, "frames", metas['images'][str(idx)])
        return idx, imageio.imread(fname), np.array(metas['poses'][str(idx)])

    all_imgs = []
    all_poses = []
    all_frame_idxs = []
    mask_gts = []
    counts = [0]
    for s in splits:
        print("Loading {} data".format(s))
        imgs = []
        poses = []
        frame_indices = []
        
        if parallel:
            idx_img_pose = Parallel(n_jobs=16)(delayed(load_data_single)(idx) for idx in tqdm(metas['ids_'+s]))
            pdb.set_trace()
            idx_img_pose.sort(key=lambda x: x[0])
            frame_indices = [x[0] for x in idx_img_pose]
            imgs = [x[1] for x in idx_img_pose]
            poses = [x[2] for x in idx_img_pose]
        else:
            for idx in tqdm(metas['ids_'+s]):
                fname = os.path.join(basedir, "frames", metas['images'][str(idx)])
                frame_indices.append(idx)
                imgs.append(imageio.imread(fname))
                poses.append(np.array(metas['poses'][str(idx)]))

        if get_masks and s == "test":
            for idx in metas['ids_'+s]:
                mask_fname = os.path.join(basedir, "annotations", metas['images'][str(idx)])
                mask_gt = np.array(imageio.imread(mask_fname))
                mask_gts.append((mask_gt / 255.).astype(np.float32))

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
    if get_masks:
        mask_gts = np.stack(mask_gts, 0)
    else:
        mask_gts = None
    
    H, W = imgs[0].shape[:2]
    focal = metas["intrinsics"][0][0]
    
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
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
    xyz_bounding_box_incameraframe = get_bbox3d_for_epickitchens_incameraframe(metas, H, W, near=near, far=far)
    min_time = torch.FloatTensor([min(all_frame_idxs)]).to(xyz_bounding_box[0].device)
    max_time = torch.FloatTensor([max(all_frame_idxs)]).to(xyz_bounding_box[1].device)
    xyzt_bounding_box = (torch.cat([xyz_bounding_box[0], min_time]), torch.cat([xyz_bounding_box[1], max_time]))
    xyzt_bounding_box_incameraframe = (torch.cat([xyz_bounding_box_incameraframe[0], min_time]), torch.cat([xyz_bounding_box_incameraframe[1], max_time]))
    return imgs, mask_gts, poses, [H, W, focal], i_split, xyzt_bounding_box, xyzt_bounding_box_incameraframe, [near, far], all_frame_idxs

#############################################################################################

from run_nerf_helpers import get_rays_np, get_rays_incameraframe_np

class EpicKitchensDataset:
    def __init__(self, basedir, device, buffer_size=1000):
        self.images, self.mask_gts, self.poses, hwf, i_split, self.bounding_box, self.bounding_box_incameraframe, near_far, self.all_frame_idxs = load_epic_kitchens_data(basedir, get_masks=False, parallel=False)
        H, W, focal = hwf
        self.H, self.W = int(H), int(W)
        self.hwf = [self.H, self.W, focal]
        self.K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        self.i_train, self.i_val, self.i_test = i_split
        self.near, self.far = near_far
        self.buffer_size = buffer_size
        self.device = device
        self.reload_iter = 1000 # reload buffer after these many iterations
        
        self.buffer = None
        self.i_batch = 0  # reload_buffer function will anyway set this to 0
        self.reload_buffer()

    def reload_buffer(self, loss_sampling=False):
        # sample non-repeating indices
        print("Reloading Buffer")
        image_idxs = random.sample(range(len(self.i_train)), self.buffer_size)
        images = self.images[self.i_train][image_idxs]
        poses = self.poses[self.i_train][image_idxs]
        rays = np.stack([get_rays_np(self.H, self.W, self.K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        rays_c = np.stack([get_rays_incameraframe_np(self.H, self.W, self.K) for _ in poses[:,:3,:4]], 0)
        rays_rgb = np.concatenate([rays, rays_c, images[:,None]], 1)
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+roc+rdc+rgb, 3]
        N = rays_rgb.shape[0]

        times = self.all_frame_idxs[self.i_train][image_idxs]
        broadcasted_times = np.broadcast_to(times[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis], [N, self.H, self.W, 1, 3])
        rays_rgb = np.concatenate([rays_rgb, broadcasted_times], 3) # [N, H, W, ro+rd+roc+rdc+rgb+t, 3]

        rays_rgb = np.reshape(rays_rgb, [-1,rays_rgb.shape[-2],rays_rgb.shape[-1]]) # [(N-1)*H*W, ro+rd+roc+rdc+rgb+t, 3]
        rays_rgb = rays_rgb.astype(np.float32)

        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        
        self.buffer = torch.Tensor(rays_rgb).to(self.device)
        self.i_batch = 0
        print("Loaded buffer ", self.buffer.shape)

    def get_ray_batch(self, batch_size, iter):
        batch = self.buffer[self.i_batch:self.i_batch+batch_size] # [B, 2+2+1+1, 3]
        batch = torch.transpose(batch, 0, 1)
        ## including time coordinates in the ray tensor too
        batch_rays, target_s = batch[[0, 1, 2, 3, 5]], batch[4]

        self.i_batch += batch_size
        if self.i_batch >= self.buffer.shape[0] or iter % self.reload_iter == 0:
            self.reload_buffer()
        
        return batch_rays, target_s
