import math
import os
import numpy as np
import imageio
import pdb
import time
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm, trange
import pickle
from sklearn.metrics import average_precision_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage

from run_nerf_helpers import *
from radam import RAdam
from extra_losses import *
from hash_encoding import HashEmbedder, XYZplusT_HashEmbedder, Linear_HashEmbedder
from robust_loss_pytorch.adaptive import AdaptiveLossFunction
from robust_loss_pytorch.general import lossfun

from load_epic_kitchens import load_epic_kitchens_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, inputs_cam, viewdirs, viewdirs_cam, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    inputs_cam_flat = torch.reshape(inputs_cam, [-1, inputs_cam.shape[-1]])
    embedded_cam = embed_fn(inputs_cam_flat)
    embedded = torch.cat([embedded, embedded_cam], -1)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(list(inputs.shape[:-1]) + [viewdirs.shape[-1]])
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        
        input_dirs_cam = viewdirs_cam[:,None].expand(list(inputs.shape[:-1]) + [viewdirs_cam.shape[-1]])
        input_dirs_cam_flat = torch.reshape(input_dirs_cam, [-1, input_dirs_cam.shape[-1]])
        embedded_dirs_cam = embeddirs_fn(input_dirs_cam_flat)
        embedded = torch.cat([embedded, embedded_dirs, embedded_dirs_cam], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ego_c2w=None, time_coord=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
        # rays_o_cam, rays_d_cam = get_rays_incameraframe(H, W, K)
        ## The above line would get coordinates in the fixed camera frame!!
        ## We need the coordinates in the ego-camera frame (which is different)!!
        assert ego_c2w is not None, "I need the EGO pose for actor model!"
        
        c2c = torch.zeros_like(ego_c2w)
        # c2c[:3,:3] = ego_c2w[:3,:3].t() @ c2w[:3,:3]
        # c2c[:3,-1] = ego_c2w[:3,:3].t() @ (c2w[:3,-1]-ego_c2w[:3,-1])
        c2c[:3,:3] = torch.eye(3,3).to(ego_c2w.device)
        rays_o_cam, rays_d_cam = get_rays(H, W, K, c2c)

        if time_coord is None:
            pdb.set_trace()
            print("Alert! for single image case, time_coord should be provided")
        time_coords = time_coord*torch.ones_like(rays_d[...,:1]).view(-1,1)
        time_coords = time_coords[:,0]
    else:
        # use provided ray batch
        rays_o, rays_d, rays_o_cam, rays_d_cam, time_coords = rays
        time_coords = time_coords[:,0]

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

        viewdirs_cam = rays_d_cam
        viewdirs_cam = viewdirs_cam / torch.norm(viewdirs_cam, dim=-1, keepdim=True)
        viewdirs_cam = torch.reshape(viewdirs_cam, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    ### Note: ignoring NDC for now...
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    rays_o_cam = torch.reshape(rays_o_cam, [-1,3]).float()
    rays_d_cam = torch.reshape(rays_d_cam, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, rays_o_cam, rays_d_cam, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs, viewdirs_cam], -1)

    # Attach time coordinates
    rays = torch.cat([rays, time_coords[:,None]], -1)
    
    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'depth_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_video(render_poses, fixed_pose_idx, render_frame_idxs, hwf, K, chunk, render_kwargs, gt_imgs, savedir, render_factor=0):
    H, W, focal = hwf
    fixed_pose = render_poses[fixed_pose_idx]

    numpy_figs = []
    psnrs = []
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        if render_frame_idxs[i] <= 920:
            continue
        print(i, time.time() - t)
        t = time.time()

        ### Render video
        rgb, _, _, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], ego_c2w=c2w[:3,:4], time_coord=render_frame_idxs[i], **render_kwargs)
        
        ### Render fixed point video
        fixed_rgb, _, _, fixed_extras = render(H, W, K, chunk=chunk, c2w=fixed_pose[:3,:4], ego_c2w=c2w[:3,:4], time_coord=render_frame_idxs[i], **render_kwargs)
        
        ### Get PSNR
        try:
            gt_img = gt_imgs[i].cpu().numpy()
        except:
            gt_img = gt_imgs[i]
        p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_img)))
        print(p)
        psnrs.append(p)

        ### Create figure
        f, p = plt.subplots(3, 5, figsize=(20,6))
        p[0,0].imshow(to8b(gt_img))
        p[0,0].set_ylabel("Ego Viewpoint", fontdict={"fontsize":15}, rotation='horizontal', ha='right')
        p[0,0].set_xlabel("Ground Truth", fontdict={"fontsize":15})
        p[0,0].xaxis.set_label_position('top') 
        
        p[0,1].imshow(to8b(rgb.cpu().numpy()))
        p[0,1].set_xlabel("Composite", fontdict={"fontsize":15})
        p[0,1].xaxis.set_label_position('top')
        
        p[0,2].imshow(to8b(extras["BG_rgb_map"].cpu().numpy()))
        p[0,2].set_xlabel("Static", fontdict={"fontsize":15})
        p[0,2].xaxis.set_label_position('top')
        
        p[0,3].imshow(to8b(extras["FG_rgb_map"].cpu().numpy()))
        p[0,3].set_xlabel("Transient", fontdict={"fontsize":15})
        p[0,3].xaxis.set_label_position('top')
        
        p[0,4].imshow(to8b(extras["ACTOR_rgb_map"].cpu().numpy()))
        p[0,4].set_xlabel("Actor", fontdict={"fontsize":15})
        p[0,4].xaxis.set_label_position('top')
        
        p[1,0].axis("off")

        p[1,1].imshow(to8b(fixed_rgb.cpu().numpy()))
        p[1,1].set_ylabel("Fixed Viewpoint", fontdict={"fontsize":15}, rotation='horizontal', ha='right')
        p[1,1].set_xlabel("Composite", fontdict={"fontsize":15})
        p[1,1].xaxis.set_label_position('top')
        
        p[1,2].imshow(to8b(fixed_extras["BG_rgb_map"].cpu().numpy()))
        p[1,2].set_xlabel("Static", fontdict={"fontsize":15})
        p[1,2].xaxis.set_label_position('top')
        
        p[1,3].imshow(to8b(fixed_extras["FG_rgb_map"].cpu().numpy()))
        p[1,3].set_xlabel("Transient", fontdict={"fontsize":15})
        p[1,3].xaxis.set_label_position('top')
        
        p[1,4].imshow(to8b(fixed_extras["ACTOR_rgb_map"].cpu().numpy()))
        p[1,4].set_xlabel("Actor", fontdict={"fontsize":15})
        p[1,4].xaxis.set_label_position('top')
        
        p[2,0].axis("off")
        p[2,1].axis("off")
        p[2,2].axis("off")
                
        p[2,3].imshow(to8b(extras["FG_acc_map"].cpu().numpy()))
        p[2,3].set_xlabel("Transient", fontdict={"fontsize":15})
        p[2,3].xaxis.set_label_position('top')
        
        p[2,4].imshow(to8b(extras["ACTOR_acc_map"].cpu().numpy()))
        p[2,4].set_xlabel("Actor", fontdict={"fontsize":15})
        p[2,4].xaxis.set_label_position('top')

        for r in range(2):
            for c in range(5):
                p[r,c].set_xticklabels([])
                p[r,c].set_yticklabels([])
                p[r,c].set_xticks([])
                p[r,c].set_yticks([])
        f.savefig(os.path.join(savedir, '{:03d}_combined_withmasks.png'.format(int(render_frame_idxs[i]))), bbox_inches='tight')

        numpy_fig = mplfig_to_npimage(f)
        numpy_figs.append(numpy_fig)
        plt.close()

        with open(os.path.join(savedir, '{:03d}_acc_embed.pkl'.format(int(render_frame_idxs[i]))), 'wb') as fp:
            pickle.dump({
                "acc_map": extras["FG_acc_map"].cpu().numpy(),
                "embed_map": extras["FG_embedding_map"].cpu().numpy()
                }, fp)


    numpy_figs = np.stack(numpy_figs, 0)
    avg_psnr = sum(psnrs)/len(psnrs)
    print("Avg PSNR over Test set: ", avg_psnr)
    with open(os.path.join(savedir, "test_psnrs_avg{:0.2f}.pkl".format(avg_psnr)), "wb") as fp:
        pickle.dump(psnrs, fp)

    return numpy_figs


def render_path(render_poses, render_frame_idxs, hwf, K, chunk, render_kwargs, gt_imgs=None, mask_gts=None, BGguided_loss=None, BGguided_sparsity=None, savedir=None, render_factor=0):

    H, W, focal = hwf
    near, far = 0.1, render_kwargs["far"]

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    depths = []
    uncertainty_maps = []
    psnrs = []
    mAPs = []
    ssim_list = []
    lpips_list = []
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, depth, _, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], ego_c2w=c2w[:3,:4], time_coord=render_frame_idxs[i], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        depths.append(depth.cpu().numpy())
        if 'uncertainty_map' in extras:
            uncertainty_map = extras['uncertainty_map'].cpu().numpy()
            uncertainty_map = (uncertainty_map - np.min(uncertainty_map)) / (np.max(uncertainty_map)-np.min(uncertainty_map))
            uncertainty_maps.append(uncertainty_map)

        if i==0:
            print(rgb.shape, depth.shape)

        if gt_imgs is not None and render_factor==0:
            try:
                gt_img = gt_imgs[i].cpu().numpy()
            except:
                gt_img = gt_imgs[i]
            psnr, ssim, lpips = get_perceptual_metrics(rgb.cpu().numpy(), gt_img, lpips_batch_size=8, device='cuda')
            print("PSNR: ", psnr, "SSIM: ", ssim, "LPIPS: ", lpips)
            psnrs.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(int(render_frame_idxs[i])))
            imageio.imwrite(filename, rgb8)

            if 'uncertainty_map' in extras:
                uncertainty_filename = os.path.join(savedir, '{:03d}_uncertainty.png'.format(int(render_frame_idxs[i])))
                uncertainty_map = to8b(uncertainty_maps[-1])
                imageio.imwrite(uncertainty_filename, uncertainty_map)

            if extras["BG_rgb_map"] is not None and gt_imgs is not None:
                ### compute mAP over mask predictions
                mask_pred = extras["FG_acc_map"].cpu().numpy() + extras["ACTOR_acc_map"].cpu().numpy()
                mAP = average_precision_score(mask_gts[i].reshape(-1), mask_pred.reshape(-1))
                print("mAP: ", mAP)
                mAPs.append(mAP)

                f, p = plt.subplots(3+1*(BGguided_loss is not None or BGguided_sparsity is not None), 3, figsize=(8,5))
                # f.suptitle(f"Sample: {int(render_frame_idxs[i])} \n PSNR: {psnr:.2f}, SSIM: {ssim:.2f}, LPIPS: {lpips:.2f}, mAP: {mAP:.3f} \n")
                f.suptitle("Sample: {0}".format(int(render_frame_idxs[i])) + "\n" + r"PSNR: $\bf{{{0:.2f}}}$, SSIM: $\bf{{{1:.2f}}}$, LPIPS: $\bf{{{2:.2f}}}$, mAP: $\bf{{{3:.3f}}}$".format(psnr, ssim, lpips, mAP) + "\n")
                f.tight_layout()
                p[0,0].imshow(to8b(gt_img))
                p[0,0].title.set_text("GT")
                p[0,0].axis("off")
                p[0,1].imshow(to8b(rgb.cpu().numpy()))
                p[0,1].title.set_text("Composite")
                p[0,1].axis("off")
                # depth_normalized = (depth.cpu().numpy() - near) / (far - near)
                # pdb.set_trace()
                disp = 1./torch.max(1e-10 * torch.ones_like(depth), depth)
                disp_normalized = (disp - 1./far) / (1./near - 1./far)
                p[0,2].imshow(to8b(disp_normalized.clamp(0,1).cpu().numpy()))
                p[0,2].title.set_text("Depth")
                p[0,2].axis("off")
                psnr_static = -10. * np.log10(np.mean(np.square(extras["BG_rgb_map"].cpu().numpy() - gt_img)))
                p[1,0].imshow(to8b(extras["BG_rgb_map"].cpu().numpy()))
                p[1,0].axis("off")
                p[1,0].title.set_text(f"Static. PSNR: {psnr_static:.2f}")
                p[1,1].imshow(to8b(extras["FG_rgb_map"].cpu().numpy()))
                p[1,1].axis("off")
                p[1,1].title.set_text("Transient+Person")
                p[1,2].imshow(to8b(extras["ACTOR_rgb_map"].cpu().numpy()))
                p[1,2].axis("off")
                p[1,2].title.set_text("Person")

                p[2,0].imshow(to8b(mask_gts[i]))
                p[2,0].axis("off")
                p[2,0].title.set_text("GT Mask")
                p[2,1].imshow(to8b(mask_pred))
                p[2,1].axis("off")
                p[2,1].title.set_text("Mask: Transient+Person")
                # p[2,2].imshow(to8b(mask_pred))
                p[2,2].axis("off")
                # p[2,2].title.set_text("Mask: Transient")

                if BGguided_loss is not None:
                    p[3,0].imshow(to8b(BGguided_loss.get_diff(torch.tensor(gt_img), extras["BG_rgb_map"]).cpu().numpy()))
                    p[3,0].axis("off")
                    p[3,0].title.set_text("BGguided Diff")
                    p[3,1].imshow(to8b(BGguided_loss.get_mask(torch.tensor(gt_img), extras["BG_rgb_map"]).cpu().numpy()))
                    p[3,1].axis("off")
                    p[3,1].title.set_text("BGguided Mask")
                    p[3,2].axis("off")

                if BGguided_sparsity is not None:
                    p[3,0].imshow(to8b(BGguided_sparsity.get_diff(torch.tensor(gt_img), extras["BG_rgb_map"]).cpu().numpy()))
                    p[3,0].axis("off")
                    p[3,0].title.set_text("BGguided Diff")
                    p[3,1].axis("off")
                    p[3,2].axis("off")

                f.savefig(os.path.join(savedir, '{:03d}_combined.png'.format(int(render_frame_idxs[i]))), bbox_inches='tight')
                plt.close(f)

                # with open(os.path.join(savedir, '{:03d}_acc_embed.pkl'.format(int(render_frame_idxs[i]))), 'wb') as fp:
                #     pickle.dump({
                #         "acc_map": extras["FG_acc_map"].cpu().numpy(),
                #         "embed_map": extras["FG_embedding_map"].cpu().numpy()
                #         }, fp)


    rgbs = np.stack(rgbs, 0)
    depths = np.stack(depths, 0)
    if uncertainty_maps:
        uncertainty_maps = np.stack(uncertainty_maps, 0)
    if gt_imgs is not None and render_factor==0:
        avg_psnr = sum(psnrs)/len(psnrs)
        print("Avg PSNR over Test set: ", avg_psnr)
        with open(os.path.join(savedir, "test_psnrs_avg{:0.2f}.pkl".format(avg_psnr)), "wb") as fp:
            pickle.dump(psnrs, fp)
        
        print("Avg mAP over Test set: ", sum(mAPs)/len(mAPs))
        with open(os.path.join(savedir, "test_mAPs_avg{:0.3f}.pkl".format(sum(mAPs)/len(mAPs))), "wb") as fp:
            pickle.dump(mAPs, fp)

        print("Avg LPIPS over Test set: ", sum(lpips_list)/len(lpips_list))
        with open(os.path.join(savedir, "test_LPIPSs_avg{:0.2f}.pkl".format(sum(lpips_list)/len(lpips_list))), "wb") as fp:
            pickle.dump(lpips_list, fp)
        
        print("Avg SSIM over Test set: ", sum(ssim_list)/len(ssim_list))
        with open(os.path.join(savedir, "test_SSIMs_avg{:0.2f}.pkl".format(sum(ssim_list)/len(ssim_list))), "wb") as fp:
            pickle.dump(ssim_list, fp)

    return rgbs, depths, uncertainty_maps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch, time_dim = get_embedder(args.multires, args, i=args.i_embed)
    # if args.i_embed==1:
        # commenting for now
        # embedding_params = list(embed_fn.parameters())

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embeddirs_fn, input_ch_views, _ = get_embedder(args.multires_views, args, i=args.i_embed_views)
    
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    
    xyz_bounding_box = args.bounding_box[0][:3], args.bounding_box[1][:3]
    xyz_bounding_box_cam = args.bounding_box_incameraframe[0][:3], args.bounding_box_incameraframe[1][:3]
    t_bounding_range = args.bounding_box[0][3], args.bounding_box[1][3]
    
    ### DEFINING THE FOUNDATIONAL BLOCKS
    world_grid_embed = HashEmbedder(bounding_box=xyz_bounding_box, \
                        n_levels=args.xyzt_embed_levels, \
                        n_features_per_level=4 if args.big_world_embed else 1, \
                        log2_hashmap_size=args.log2_hashmap_size, \
                        finest_resolution=1024) # always 1024 for "coarse" static grid. use fine_res for "fine" static grid
    if args.coarse_fine_separate:
        world_grid_embed_fine = HashEmbedder(bounding_box=xyz_bounding_box, \
                        n_levels=args.xyzt_embed_levels,
                        n_features_per_level=4 if args.big_world_embed else 1, \
                        log2_hashmap_size=args.log2_hashmap_size, \
                        finest_resolution=args.finest_res)
    else:
        world_grid_embed_fine = world_grid_embed

    world_grid_embed_FG = None
    if args.bg_fg_separate or args.on_off_encoding:
        # high freq embedding for FG
        world_grid_embed_FG = HashEmbedder(bounding_box=xyz_bounding_box, \
                                n_features_per_level=4 if args.big_world_embed else 2, \
                                base_resolution=128,
                                log2_hashmap_size=args.log2_hashmap_size, \
                                finest_resolution=args.finest_res*4)
    elif args.xyzt_model:
        world_grid_embed_FG = HashEmbedder(bounding_box=args.bounding_box, \
                                n_levels=args.xyzt_embed_levels, \
                                n_features_per_level=4 if args.big_world_embed else 1, \
                                base_resolution=[128,128,128,16],
                                log2_hashmap_size=args.log2_hashmap_size, \
                                finest_resolution=[args.finest_res*4,args.finest_res*4,args.finest_res*2,1024])

    camera_grid_embed, time_grid_embed = None, None
    if args.xyzt_model:
        if args.use_actor_xyzt:
            camera_grid_embed = HashEmbedder(bounding_box=args.bounding_box_incameraframe, \
                            n_levels=args.xyzt_embed_levels, \
                            n_features_per_level=4 if args.big_world_embed else 1, \
                            base_resolution=128,
                            log2_hashmap_size=args.log2_hashmap_size, \
                            finest_resolution=args.finest_res*4)
    else:
        camera_grid_embed = HashEmbedder(bounding_box=xyz_bounding_box_cam, \
                            base_resolution=128 if args.actor_high_freq else 16,
                            log2_hashmap_size=args.log2_hashmap_size, \
                            finest_resolution=args.finest_res)

        ### separate time embedding only when not using XYZT model
        if args.big_time:
            time_grid_embed = Linear_HashEmbedder(t_bounding_range, n_levels=6, n_features_per_level=3)
        else:
            time_grid_embed = Linear_HashEmbedder(t_bounding_range)

    # model_class = BackgroundForegroundActorNeRF_separateEmbeddings
    if args.bg_fg_separate:
        model_class = NeuralDiff_BGFGSeparate
    else:
        model_class = NeuralDiff

    if args.xyzt_model:
        if args.use_xyzt_bottleneck:
            model = BGFG_XYZT_Bottleneck(num_layers=2, # doesn't matter
                            hidden_dim=64,
                            geo_feat_dim=64,
                            num_layers_color=3, # doesn't matter
                            input_ch=input_ch, input_cam_ch=input_ch,
                            input_ch_views=input_ch_views, input_ch_views_cam=input_ch_views,
                            use_uncertainties=args.use_uncertainties,
                            static_grid=world_grid_embed, xyzt_grid=world_grid_embed_FG,
                            coarse=True).to(device)
        else:
            model = BGFG_XYZT(num_layers=2,
                            hidden_dim=64,
                            geo_feat_dim=64,
                            num_layers_color=3,
                            input_ch=input_ch, input_cam_ch=input_ch,
                            input_ch_views=input_ch_views, input_ch_views_cam=input_ch_views,
                            use_uncertainties=args.use_uncertainties,
                            static_grid=world_grid_embed, xyzt_grid=world_grid_embed_FG, xyzt_grid_cam=camera_grid_embed,
                            coarse=True,
                            use_viewdirs_FG=not args.no_views_FG).to(device)
    elif args.on_off_encoding:
        model = BGFG_OnOffEncoding(t_bounds=t_bounding_range,
                                   num_layers=2,
                                   hidden_dim=64,
                                   geo_feat_dim=15,
                                   num_layers_color=3,
                                   input_ch=input_ch, input_cam_ch=input_ch,
                                   input_ch_views=input_ch_views, input_ch_views_cam=input_ch_views,
                                   use_uncertainties=args.use_uncertainties,
                                   static_grid=world_grid_embed, FG_xyz_grid=None,
                                   coarse=True).to(device)
    elif args.piecewise_constant:
        model = BGFG_PiecewiseConst(xyzt_bounds=args.bounding_box,
                                   num_layers=2,
                                   hidden_dim=64,
                                   geo_feat_dim=15,
                                   num_layers_color=3,
                                   input_ch=input_ch, input_cam_ch=input_ch,
                                   input_ch_views=input_ch_views, input_ch_views_cam=input_ch_views,
                                   use_uncertainties=args.use_uncertainties,
                                   static_grid=world_grid_embed,
                                   coarse=True).to(device)
    else:
        model = model_class(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        input_ch=input_ch, input_cam_ch=input_ch,
                        input_ch_views=input_ch_views, input_ch_views_cam=input_ch_views,
                        use_uncertainties=args.use_uncertainties,
                        world_grid_embed=world_grid_embed, world_grid_embed_FG=world_grid_embed_FG, camera_grid_embed=camera_grid_embed, time_grid_embed=time_grid_embed,
                        big=args.big,
                        coarse=True).to(device)
    
    embedding_params = []
    grad_vars = []
    for _, child in model.named_children():
        if isinstance(child, HashEmbedder) or isinstance(child, Linear_HashEmbedder) or isinstance(child, XYZplusT_HashEmbedder):
            embedding_params += list(child.parameters())
        else:
            grad_vars += list(child.parameters())

    model_fine = None
    if args.N_importance > 0:
        # model_class = BackgroundForegroundActorNeRF_separateEmbeddings
        if args.bg_fg_separate:
            model_class = NeuralDiff_BGFGSeparate
        else:
            model_class = NeuralDiff
        
        if args.xyzt_model:
            if args.use_xyzt_bottleneck:
                model_fine = BGFG_XYZT_Bottleneck(num_layers=2, # doesn't matter
                                hidden_dim=64,
                                geo_feat_dim=64,
                                num_layers_color=3, # doesn't matter
                                input_ch=input_ch, input_cam_ch=input_ch,
                                input_ch_views=input_ch_views, input_ch_views_cam=input_ch_views,
                                use_uncertainties=args.use_uncertainties,
                                static_grid=world_grid_embed_fine, xyzt_grid=world_grid_embed_FG,
                                coarse=False).to(device)
            else:
                model_fine = BGFG_XYZT(num_layers=2,
                                hidden_dim=64,
                                geo_feat_dim=64,
                                num_layers_color=3,
                                input_ch=input_ch, input_cam_ch=input_ch,
                                input_ch_views=input_ch_views, input_ch_views_cam=input_ch_views,
                                use_uncertainties=args.use_uncertainties,
                                static_grid=world_grid_embed_fine, xyzt_grid=world_grid_embed_FG, xyzt_grid_cam=camera_grid_embed,
                                coarse=False,
                                use_viewdirs_FG=not args.no_views_FG,
                                use_actor=args.use_actor_xyzt,
                                small_MLPs_dyn=args.small_MLPs_dynamic).to(device)
        elif args.on_off_encoding:
            model_fine = BGFG_OnOffEncoding(t_bounds=t_bounding_range,
                                            num_layers=2,
                                            hidden_dim=64,
                                            geo_feat_dim=15,
                                            num_layers_color=3,
                                            input_ch=input_ch, input_cam_ch=input_ch,
                                            input_ch_views=input_ch_views, input_ch_views_cam=input_ch_views,
                                            use_uncertainties=args.use_uncertainties,
                                            static_grid=world_grid_embed_fine, FG_xyz_grid=world_grid_embed_FG,
                                            coarse=False).to(device)
        elif args.piecewise_constant:
            model_fine = BGFG_PiecewiseConst(xyzt_bounds=args.bounding_box,
                                            num_layers=2,
                                            hidden_dim=64,
                                            geo_feat_dim=15,
                                            num_layers_color=3,
                                            input_ch=input_ch, input_cam_ch=input_ch,
                                            input_ch_views=input_ch_views, input_ch_views_cam=input_ch_views,
                                            use_uncertainties=args.use_uncertainties,
                                            static_grid=world_grid_embed_fine,
                                            coarse=False,
                                            init_temperature=args.init_temp,
                                            n_pieces=args.n_pieces).to(device)
        else:
            model_fine = model_class(num_layers=2,
                            hidden_dim=64,
                            geo_feat_dim=15,
                            num_layers_color=3,
                            input_ch=input_ch, input_cam_ch=input_ch,
                            input_ch_views=input_ch_views, input_ch_views_cam=input_ch_views,
                            use_uncertainties=args.use_uncertainties,
                            world_grid_embed=world_grid_embed_fine, world_grid_embed_FG=world_grid_embed_FG, camera_grid_embed=camera_grid_embed, time_grid_embed=time_grid_embed,
                            big=args.big,
                            coarse=False).to(device)

        for _, child in model_fine.named_children():
            if isinstance(child, HashEmbedder) or isinstance(child, Linear_HashEmbedder) or isinstance(child, XYZplusT_HashEmbedder):
                embedding_params += list(child.parameters())
            else:
                grad_vars += list(child.parameters())
            # grad_vars += list(model_fine.parameters())
    
    network_query_fn = lambda inputs, inputs_cam, viewdirs, viewdirs_cam, network_fn : run_network(inputs, inputs_cam, viewdirs, viewdirs_cam, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    ### Define adaptive robust loss
    if args.fixed_adaptive_loss:
        adaptive_loss = AdaptiveLossFunction(num_dims=3, float_dtype=torch.float32, device=torch.device("cuda:0"))
        grad_vars += list(adaptive_loss.parameters())
    else:
        adaptive_loss = None

    if args.bg_guided_loss:
        BGguided_loss = BGguidedLoss()
        grad_vars += list(BGguided_loss.parameters())
    else:
        BGguided_loss = None

    if args.bg_guided_sparsity:
        BGguided_sparsity = BGguidedSparsityWeights()
        grad_vars += list(BGguided_sparsity.parameters())
    else:
        BGguided_sparsity = None

    if args.push_pull_weight>0 and args.xyzt_model:
        Push_Pull_loss = PushPullLoss(xyz_bounding_box=xyz_bounding_box, 
                                base_res=world_grid_embed_FG.base_resolution, 
                                finest_res=world_grid_embed_FG.finest_resolution, 
                                log2_hashmap_size=world_grid_embed_FG.log2_hashmap_size, 
                                n_levels=world_grid_embed_FG.n_levels,
                                n_features_per_level=world_grid_embed_FG.n_features_per_level, 
                                n_centroids=20)
        grad_vars += list(Push_Pull_loss.parameters())
    else:
        Push_Pull_loss = None


    # Create optimizer
    if args.i_embed==1:
        # sparse_opt = torch.optim.SparseAdam(embedding_params, lr=args.lrate, betas=(0.9, 0.99), eps=1e-15)
        # dense_opt = torch.optim.Adam(grad_vars, lr=args.lrate, betas=(0.9, 0.99), weight_decay=1e-6)
        # optimizer = MultiOptimizer(optimizers={"sparse_opt": sparse_opt, "dense_opt": dense_opt})
        embedding_params = list(set(embedding_params))
        grad_vars = list(set(grad_vars))
        optimizer = torch.optim.Adam([
                            {'params': grad_vars, 'weight_decay': 1e-6},
                            {'params': embedding_params}
                        ], lr=args.lrate, betas=(0.9, 0.99), eps=1e-15)
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(args.ckptdir, expname, f) for f in sorted(os.listdir(os.path.join(args.ckptdir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if args.i_embed==1:
            embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])
        if world_grid_embed:
            world_grid_embed.load_state_dict(ckpt['world_grid_state_dict'])
        if world_grid_embed_FG:
            world_grid_embed_FG.load_state_dict(ckpt['world_grid_FG_state_dict'])
        if camera_grid_embed:
            camera_grid_embed.load_state_dict(ckpt['camera_grid_state_dict'])
        if time_grid_embed:
            time_grid_embed.load_state_dict(ckpt['time_grid_state_dict'])


    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'embed_fn': embed_fn,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'embedders': {'world_grid_embed': world_grid_embed, \
                      'world_grid_embed_FG': world_grid_embed_FG, \
                      'camera_grid_embed': camera_grid_embed, \
                      'time_grid_embed': time_grid_embed},
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['test_time'] = True # if "test_time" in render_kwargs_test, then you are in testing mode

    return render_kwargs_train, render_kwargs_test, start, adaptive_loss, BGguided_loss, BGguided_sparsity, Push_Pull_loss, optimizer


def get_weights_from_sigmas(sigmas, dists, noise=0.0):
    alpha = 1.-torch.exp(-F.relu(sigmas + noise)*dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    return weights


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, test_time=False, coarse=False, sigma_cauchy=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    if not raw.shape[-1] > 4:
        rgb = raw[...,:3]  # [N_rays, N_samples, 3]
        sigma = raw[...,3]
        BG_rgb_map, FG_rgb_map, ACTOR_rgb_map = None, None, None
        FG_acc_map, ACTOR_acc_map = None, None
        FG_embedding_map = None
    else:
        BG_rgb = raw[...,:3]
        FG_rgb = raw[...,3:6]
        ACTOR_rgb = raw[...,6:9]
        BG_sigma, FG_sigma, ACTOR_sigma = raw[...,9], raw[...,10], raw[...,11]
        
        if not test_time and not coarse: ## same as "if training time and if using model_fine"
            BG_sigma[:, -1:] = 1000
            BG_rgb[:, -1, :3] = torch.rand(BG_rgb.shape[0], 3).to(BG_rgb.device)

        sigma = BG_sigma + FG_sigma + ACTOR_sigma + 1e-9
        rgb = (BG_sigma/sigma)[...,None] * BG_rgb + (FG_sigma/sigma)[...,None] * FG_rgb + (ACTOR_sigma/sigma)[...,None] * ACTOR_rgb

    # overall rgb map
    alphas = 1.-torch.exp(-F.relu(sigma + noise)*dists)
    transmittance = torch.cumprod(torch.cat([torch.ones((alphas.shape[0], 1)), 1.-alphas + 1e-10], -1), -1)[:, :-1]
    weights = alphas * transmittance
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    # background, foreground and actor weights
    if raw.shape[-1] > 4:
        BG_alphas = (BG_sigma/sigma)*alphas
        BG_alphas[:,-1:] = 1.-torch.exp(-F.relu(BG_sigma + noise)*dists)[:,-1:]
        FG_alphas = (FG_sigma/sigma)*alphas
        FG_alphas[:,-1:] = 1.-torch.exp(-F.relu(FG_sigma + noise)*dists)[:,-1:]
        ACTOR_alphas = (ACTOR_sigma/sigma)*alphas
        ACTOR_alphas[:,-1:] = 1.-torch.exp(-F.relu(ACTOR_sigma + noise)*dists)[:,-1:]

        if test_time:
            BG_weights = BG_alphas * torch.cumprod(torch.cat([torch.ones((BG_alphas.shape[0], 1)), 1.-BG_alphas + 1e-10], -1), -1)[:, :-1]
            # FG_weights = FG_alphas * torch.cumprod(torch.cat([torch.ones((FG_alphas.shape[0], 1)), 1.-FG_alphas + 1e-10], -1), -1)[:, :-1]
            # ACTOR_weights = ACTOR_alphas * torch.cumprod(torch.cat([torch.ones((ACTOR_alphas.shape[0], 1)), 1.-ACTOR_alphas + 1e-10], -1), -1)[:, :-1]
            ### Hacky. Must remove in future!
            # FG_weights = FG_alphas * transmittance 
            # ACTOR_weights = ACTOR_alphas * transmittance
            ### Another hack
            FG_weights = (FG_sigma/(FG_sigma+BG_sigma+1e-9))*alphas * transmittance 
            ACTOR_weights = (ACTOR_sigma/(ACTOR_sigma+BG_sigma+1e-9))*alphas * transmittance
        else:
            BG_weights = BG_alphas * transmittance
            FG_weights = FG_alphas * transmittance
            ACTOR_weights = ACTOR_alphas * transmittance

    uncertainty_map = None
    if raw.shape[-1] > 4 and not test_time:
        foreground_uncertainty = F.relu(raw[..., 12])
        if raw.shape[-1] > 5:
            uncertainty_map = torch.sum(FG_weights * foreground_uncertainty, -1)
            if raw.shape[-1] > 6:
                actor_uncertainty = F.relu(raw[..., 13])
                uncertainty_map += torch.sum(ACTOR_weights * actor_uncertainty, -1)
        else:
            uncertainty_map = torch.sum(weights * foreground_uncertainty, -1)
        # adding beta_min as in NeRF-W
        uncertainty_map += 0.03

    if raw.shape[-1] > 4:
        ### Defining *temp* weights for BG and FG to compute BG/FG rgb maps
        BG_weights_temp = BG_alphas * torch.cumprod(torch.cat([torch.ones((BG_alphas.shape[0], 1)), 1.-BG_alphas + 1e-10], -1), -1)[:, :-1]
        if test_time:
            FG_weights_temp = (FG_sigma/(FG_sigma+BG_sigma+1e-9))*alphas * transmittance
        else:
            FG_weights_temp = FG_alphas * torch.cumprod(torch.cat([torch.ones((FG_alphas.shape[0], 1)), 1.-FG_alphas + 1e-10], -1), -1)[:, :-1]
        BG_rgb_map = torch.sum(BG_weights_temp[...,None] * BG_rgb, -2)
        FG_rgb_map = torch.sum(FG_weights_temp[...,None] * FG_rgb, -2)
        ACTOR_rgb_map = torch.sum(ACTOR_weights[...,None] * ACTOR_rgb, -2)

        FG_acc_map = torch.sum(FG_weights, -1)
        ACTOR_acc_map = torch.sum(ACTOR_weights, -1)

        FG_embedding_map = torch.sum(FG_weights[...,None] * raw[...,14:], -2)

    sigma_L1 = torch.zeros(raw.shape[0])
    ordering_loss = None
    if raw.shape[-1] > 4 and not test_time:
        if not sigma_cauchy:
            sigma_L1 += FG_sigma.sum(dim=-1) + ACTOR_sigma.sum(dim=-1) # Just use L1 loss
        else:
            sigma_L1 = torch.log(1+20*FG_sigma**2).sum(dim=-1) + torch.log(1+20*ACTOR_sigma**2).sum(dim=-1)

        ordering_loss = BGFG_ordering_loss(BG_weights, FG_weights)
        # entropy = Categorical(probs = torch.stack([BG_sigma/sigma, FG_sigma/sigma, ACTOR_sigma/sigma], dim=-1) + 1e-8).entropy()
        # sigma_L1 += entropy.sum(dim=-1)

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    ## Calculate weights sparsity loss
    # mask = weights.sum(-1) > 0.5
    # entropy = Categorical(probs = weights+1e-5).entropy()
    # sparsity_loss = entropy * mask

    return rgb_map, disp_map, acc_map, weights, depth_map, sigma_L1, ordering_loss, uncertainty_map, BG_rgb_map, FG_rgb_map, ACTOR_rgb_map, FG_acc_map, ACTOR_acc_map, FG_embedding_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                embedders=None,
                embed_fn=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                test_time=False,
                sigma_cauchy=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d, rays_o_cam, rays_d_cam = ray_batch[:,0:3], ray_batch[:,3:6], ray_batch[:,6:9], ray_batch[:,9:12] # [N_rays, 3] each
    time_coords = ray_batch[:,-1]
    viewdirs = ray_batch[:,-7:-4] if ray_batch.shape[-1] > 15 else None
    viewdirs_cam = ray_batch[:,-4:-1] if ray_batch.shape[-1] > 15 else None
    bounds = torch.reshape(ray_batch[...,12:14], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    pts_cam = rays_o_cam[...,None,:] + rays_d_cam[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    pts = torch.cat([pts, time_coords[:,None].expand(pts.shape[:-1])[:,:,None]], dim=-1)
    pts_cam = torch.cat([pts_cam, time_coords[:,None].expand(pts_cam.shape[:-1])[:,:,None]], dim=-1)

#     raw = run_network(pts)
    raw = network_query_fn(pts, pts_cam, viewdirs, viewdirs_cam, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss, ordering_loss, uncertainty_map, BG_rgb_map, FG_rgb_map, ACTOR_rgb_map, FG_acc_map, ACTOR_acc_map, FG_embedding_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, test_time=test_time, coarse=network_fn.coarse, sigma_cauchy=sigma_cauchy)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0, sparsity_loss_0, ordering_loss_0, uncertainty_map_0, BG_rgb_map_0, FG_rgb_map_0, ACTOR_rgb_map_0 = rgb_map, disp_map, acc_map, depth_map, sparsity_loss, ordering_loss, uncertainty_map, BG_rgb_map, FG_rgb_map, ACTOR_rgb_map
        FG_acc_map_0, ACTOR_acc_map_0 = FG_acc_map, ACTOR_acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        pts_cam = rays_o_cam[...,None,:] + rays_d_cam[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        pts = torch.cat([pts, time_coords[:,None].expand(pts.shape[:-1])[:,:,None]], dim=-1)
        pts_cam = torch.cat([pts_cam, time_coords[:,None].expand(pts_cam.shape[:-1])[:,:,None]], dim=-1)

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, pts_cam, viewdirs, viewdirs_cam, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss, ordering_loss, uncertainty_map, BG_rgb_map, FG_rgb_map, ACTOR_rgb_map, FG_acc_map, ACTOR_acc_map, FG_embedding_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, test_time=test_time, coarse=run_fn.coarse, sigma_cauchy=sigma_cauchy)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'sparsity_loss': sparsity_loss, 'depth_map': depth_map}
    ret["BG_rgb_map"] = BG_rgb_map
    ret["FG_rgb_map"] = FG_rgb_map
    ret["ACTOR_rgb_map"] = ACTOR_rgb_map
    ret["FG_acc_map"] = FG_acc_map
    ret["ACTOR_acc_map"] = ACTOR_acc_map
    ret["FG_embedding_map"] = FG_embedding_map
    if ordering_loss is not None:
        ret["ordering_loss"] = ordering_loss
    if uncertainty_map is not None:
        ret['uncertainty_map'] = uncertainty_map
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['depth0'] = depth_map_0
        ret['acc0'] = acc_map_0
        ret['sparsity_loss0'] = sparsity_loss_0
        if uncertainty_map_0 is not None:
            ret['uncertainty_map0'] = uncertainty_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--ckptdir", type=str, default='/work/yashsb/NeuralDiff/logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=1, 
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--i_embed_views", type=int, default=2, 
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_fixed_viewpoint_video", action='store_true', 
                        help='do not optimize, reload weights and render video from a fixed viewpoint')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=3000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=1000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=500000, 
                        help='frequency of render_poses video saving')

    parser.add_argument("--finest_res",   type=int, default=1024, 
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size",   type=int, default=19, 
                        help='log2 of hashmap size')
    parser.add_argument("--sparse-loss-weight", type=float, default=1e-10,
                        help='learning rate')
    parser.add_argument("--tv-loss-weight", type=float, default=0.0,
                        help='weight for Total Variation loss')
    parser.add_argument("--use-uncertainties", action='store_true', 
                        help='use gaussian observation loss with uncertainty')
    parser.add_argument("--big", action='store_true', 
                        help='use bigger actor and foreground model')
    parser.add_argument("--big-time", action='store_true', 
                        help='use bigger embedding for time')
    parser.add_argument("--big-world-embed", action='store_true', 
                        help='use bigger size for world embeddings')
    parser.add_argument("--actor-high-freq", action='store_true', 
                        help='high freq for actor embedding')
    parser.add_argument("--fixed-adaptive-loss", action='store_true', 
                        help='fixed version of adaptive loss')
    parser.add_argument("--bg-fg-separate", action='store_true', 
                        help='separate embedding grids for BG/FG')
    parser.add_argument("--entropy-weight", type=float, default=0.0,
                        help='weight for Entropy loss on sigmas')
    parser.add_argument("--xyzt-model", action='store_true', 
                        help='use XYZT model')
    parser.add_argument("--no-views-FG", action='store_true', 
                        help='dont use views for FG in XYZT model')
    parser.add_argument("--use-actor-xyzt", action='store_true', 
                        help='use Actor model in XYZT model')
    parser.add_argument("--scaled-tvloss", action='store_true', 
                        help='Use scaled TV loss')
    parser.add_argument("--small-MLPs-dynamic", action='store_true',
                        help='Use small MLPs for foreground and actor models')
    parser.add_argument("--xyzt-embed-levels", type=int, default=16, 
                        help='number of levels in XYZT embeddings')
    parser.add_argument("--bg-guided-loss", action='store_true', 
                        help='Joao idea: weight loss with diff between GT and BG')
    parser.add_argument("--bg-guided-sparsity", action='store_true', 
                        help='BG guided sparsity constraint on FG sigmas')
    parser.add_argument("--on-off-encoding", action='store_true',
                        help='Use OnOffEncoding for time')
    parser.add_argument("--piecewise-constant", action='store_true',
                        help="Use piecewise constant for time")
    parser.add_argument("--init-temp", type=float, default=100.0,
                        help='Temperature for piecewise constant embedding')
    parser.add_argument("--temperature-decay", type=float, default=0.001,
                        help='Temperature decay for piecewise constant embedding')
    parser.add_argument("--n-pieces", type=int, default=10, 
                        help='number of pieces in PiecewiseConstant Time encoding')
    parser.add_argument("--push-pull-weight", type=float, default=0.0,
                        help='weight for Push-Pull loss with centroids')
    parser.add_argument("--fake-loss-weight", type=float, default=0.0,
                        help='weight for Fake time loss')
    parser.add_argument("--ordering-loss-weight", type=float, default=0.0,
                        help='weight for BGFG ordering loss')
    parser.add_argument("--use-xyzt-bottleneck", action='store_true',
                        help="Use bottleneck with XYZT model")
    parser.add_argument("--coarse-fine-separate", action='store_true',
                        help="different embedding grid for coarse and fine models")

    ### The following didn't work:
    # parser.add_argument("--entropy-color", action='store_true', 
    #                     help='Use entropy loss for colors')
    # parser.add_argument("--actor-small-embed", action='store_true', 
    #                     help='Less levels for actor embedding')
    # parser.add_argument("--small-actor-MLP", action='store_true', 
    #                     help='Less levels for actor embedding')
    # parser.add_argument("--only-background", action='store_true', 
    #                     help='train only the static/background model')
    # parser.add_argument("--only-foreground", action='store_true', 
    #                     help='train only the dynamic/foreground model')
    # parser.add_argument("--sigma-cauchy", action='store_true', 
    #                     help='Use cauchy loss for sigma values')
    
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None

    if args.dataset_type=="epic_kitchens":
        images, mask_gts, poses, hwf, i_split, bounding_box, bounding_box_incameraframe, near_far, all_frame_idxs = load_epic_kitchens_data(args.datadir)
        near, far = near_far
        
        args.bounding_box = bounding_box
        args.bounding_box_incameraframe = bounding_box_incameraframe
        print('Loaded llff', images.shape, hwf, args.datadir)

        i_train, i_val, i_test = i_split
        render_poses = poses[i_test]
        render_frame_idxs = all_frame_idxs[i_test]
    
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])
        render_frame_idxs = all_frame_idxs[i_test]

    # Create log dir and copy the config file
    basedir = args.basedir
    args.expname += "_fine"+str(args.finest_res) + "_log2T"+str(args.log2_hashmap_size)
    args.expname += "_lr"+str(args.lrate) + "_decay"+str(args.lrate_decay)
    if args.xyzt_model:
        args.expname += "_BGFG_XYZT"
        if args.use_xyzt_bottleneck:
            args.expname += "BNeck"
        if args.use_actor_xyzt:
            args.expname += "withAct"
        if args.fake_loss_weight > 0:
            args.expname += "_fake"+str(args.fake_loss_weight)
        if args.ordering_loss_weight > 0:
            args.expname += "_Order"+str(args.ordering_loss_weight)
        args.expname += "_lev"+str(args.xyzt_embed_levels)
        if args.coarse_fine_separate:
            args.expname += "_CFdiff"
    elif args.on_off_encoding:
        args.expname += "_BGFG_OnOff"
    elif args.piecewise_constant:
        args.expname += "_BGFG_PConGRU"
        args.expname += "_T"+str(args.init_temp) + "_Dec"+str(args.temperature_decay) + "_nP"+str(args.n_pieces)
    else:
        args.expname += "_NeuralDiffsig_CoarseAdapt"
    if args.fixed_adaptive_loss:
        args.expname += "_Fixed"
    else:
        args.expname += "_L2"
    if args.use_uncertainties:
        args.expname += "_withUncert"
    if args.big:
        args.expname += "_BIGGER"
    args.expname += "_smallBG"
    if args.big_time:
        args.expname += "_BigTime"
    if args.big_world_embed:
        args.expname += "_BWor"
    if args.sparse_loss_weight > 0:
        args.expname += "_sparse" + str(args.sparse_loss_weight)
    if args.entropy_weight > 0:
        args.expname += "_ent" + str(args.entropy_weight)
    args.expname += "_scTV" if args.scaled_tvloss else "_TV"
    args.expname += "4D" if args.xyzt_model else ""
    args.expname += str(args.tv_loss_weight)
    if args.bg_fg_separate:
        args.expname += "_BGFGDiff"
    else:
        args.expname += "_BGFGsame"
    args.expname += "_BGsignoise"
    if args.actor_high_freq:
        args.expname += "_ActHighFreq"
    if args.no_views_FG:
        args.expname += "_NoViewsFG"
    if args.small_MLPs_dynamic:
        args.expname += "_smMLPs"
    if args.bg_guided_loss:
        args.expname += "_BGguided"
    if args.bg_guided_sparsity:
        args.expname += "_BGSparsityHSVallch"
    if args.push_pull_weight>0:
        args.expname += "_PP" + str(args.push_pull_weight) + "cent20"
    expname = args.expname   

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(args.ckptdir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, adaptive_loss, BGguided_loss, BGguided_sparsity, Push_Pull_loss, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    render_frame_idxs = torch.Tensor(render_frame_idxs).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None
                mask_gts = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _, uncertainty_maps = render_path(render_poses, render_frame_idxs, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, mask_gts=mask_gts, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            if len(uncertainty_maps)>0:
                imageio.mimwrite(os.path.join(testsavedir, 'uncertainty_video.mp4'), to8b(uncertainty_maps), fps=30, quality=8)

            return

    if args.render_fixed_viewpoint_video:
        print("RENDER from a Fixed View-point")
        with torch.no_grad():
            fixed_pose_idx = 608
            # fixed_pose = render_poses[fixed_pose_idx]
            # fixed_poses = fixed_pose.repeat(all_frame_idxs.shape[0], 1, 1) # repeat for number of frames
            
            videobase = os.path.join(basedir, expname, 'fixedviewpoint_comb_withmasks_{:04d}_{:06d}'.format(fixed_pose_idx, start))
            os.makedirs(videobase, exist_ok=True)
            
            sorted_idx = np.argsort(all_frame_idxs)
            # rgbs, _, uncertainty_maps = render_path(fixed_poses, np.sort(all_frame_idxs), hwf, K, args.chunk, render_kwargs_test, savedir=videobase)
            numpy_figs = render_video(torch.tensor(poses[sorted_idx]), fixed_pose_idx, all_frame_idxs[sorted_idx], hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[sorted_idx], savedir=videobase)
            print('Done, saving', numpy_figs.shape)
            imageio.mimwrite(videobase + '_figs.mp4', numpy_figs, fps=30, quality=8)
            
            return
            

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        rays_incameraframe = np.stack([get_rays_incameraframe_np(H, W, K) for p in poses[:,:3,:4]], 0)
        print('done, concats')
        rays_rgb = np.concatenate([rays, rays_incameraframe, images[:,None]], 1) # [N, ro+rd+roc+rdc+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+roc+rdc+rgb, 3]
        N, H, W = rays_rgb.shape[:3]

        ##### Appending time coordinates!!!
        broadcasted_frame_idxs = np.broadcast_to(all_frame_idxs[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis], [N, H, W, 1, 3])
        rays_rgb = np.concatenate([rays_rgb, broadcasted_frame_idxs], 3) # [N, H, W, ro+rd+roc+rdc+rgb+t, 3]

        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,rays_rgb.shape[-2],rays_rgb.shape[-1]]) # [(N-1)*H*W, ro+rd+roc+rdc+rgb+t, 3]

        if args.fake_loss_weight > 0:
            # rays_rgb is [(N-1)*H*W, ro+rd+roc+rdc+rgb+t, 3]
            fake_times = np.random.choice(all_frame_idxs[i_train], size=rays_rgb.shape[0]) # [N-1*H*W]
            rays_rgb_fake = np.concatenate([rays_rgb[:,:-1,:], np.broadcast_to(fake_times[:,None,None], [rays_rgb.shape[0], 1, rays_rgb.shape[2]])], 1) # [N-1*H*W, ro+rd+roc+rdc+rgb+t, 3]
            rays_rgb_fake = rays_rgb_fake.astype(np.float32)   

        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        rand_idx = np.random.permutation(rays_rgb.shape[0])
        rays_rgb = rays_rgb[rand_idx]
        if args.fake_loss_weight > 0:
            rays_rgb_fake = rays_rgb_fake[rand_idx]

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)
        if args.fake_loss_weight > 0:
            rays_rgb_fake = torch.Tensor(rays_rgb_fake).to(device)

    N_iters = int(50000 * 1024/args.N_rand) + 1
    args.lrate = args.lrate * math.sqrt(args.N_rand/1024.0)

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    
    loss_list = []
    psnr_list = []
    time_list = []
    start = start + 1
    time0 = time.time()
    for i in trange(start, N_iters):
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            # batch_rays, target_s = batch[:2], batch[2]
            ## including time coordinates in the ray tensor too
            batch_rays, target_s = batch[[0, 1, 2, 3, 5]], batch[4]

            if args.fake_loss_weight > 0:
                batch_fake = rays_rgb_fake[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
                batch_fake = torch.transpose(batch_fake, 0, 1)
                batch_fake_rays, _ = batch_fake[[0, 1, 2, 3, 5]], batch_fake[4] # fake batch uses same target values

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                if args.fake_loss_weight > 0:
                    rays_rgb_fake = rays_rgb_fake[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        ### Compute FAKE outputs --> Assume pose p_i and image I_i are from a time t_j
        if args.fake_loss_weight > 0:
            rgb_fake, _, _, extras_fake = render(H, W, K, chunk=args.chunk, rays=batch_fake_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        if 'uncertainty_map' not in extras:
            img_loss = img2mse(rgb, target_s)
        else:
            if args.bg_guided_loss:
                img_loss = BGguided_loss(target_s, extras['BG_rgb_map'], extras['FG_rgb_map'], extras['FG_acc_map'], extras['uncertainty_map'], iter=i)
            else:
                img_loss = img2mse_with_uncertainty(rgb, target_s, extras['uncertainty_map'].unsqueeze(-1))

        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img2mse(rgb, target_s))

        if 'rgb0' in extras:
            # if 'uncertainty_map0' not in extras:
            # img_loss0 = img2mse(extras['rgb0'], target_s)
            # img_loss0 = img2L1(extras['rgb0'], target_s)
            if args.fixed_adaptive_loss:
                img_loss0 = lossfun(extras['rgb0']-target_s, alpha=torch.Tensor([-2.]), scale=torch.Tensor([0.01])).sum(dim=-1).mean()
            else:
                # img_loss0 = adaptive_loss.lossfun(extras['rgb0']-target_s).sum(dim=-1).mean()
                img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        ### Adding FAKE losses
        if args.fake_loss_weight > 0:
            img_loss_fake = img2mse_with_uncertainty(rgb_fake, target_s, extras_fake['uncertainty_map'].unsqueeze(-1))
            loss = loss + args.fake_loss_weight * img_loss_fake

            if 'rgb0' in extras_fake:
                img_loss_fake0 = lossfun(extras_fake['rgb0']-target_s, alpha=torch.Tensor([-2.]), scale=torch.Tensor([0.01])).sum(dim=-1).mean()
                loss = loss + args.fake_loss_weight * img_loss_fake0

        if args.bg_guided_sparsity:
            extras["sparsity_loss"] = extras["sparsity_loss"] * BGguided_sparsity(target_s, extras['BG_rgb_map'])
        sparsity_loss = args.sparse_loss_weight*(extras["sparsity_loss"].mean() + extras["sparsity_loss0"].mean())
        loss = loss + sparsity_loss

        if args.ordering_loss_weight > 0:
            ordering_loss = extras["ordering_loss"].mean()
            loss = loss + args.ordering_loss_weight * ordering_loss

        ### Add entropy loss
        if args.entropy_weight>0:
            BG_sigma, FG_sigma, ACTOR_sigma = extras["raw"][...,9], extras["raw"][...,10], extras["raw"][...,11]
            sigma = BG_sigma + FG_sigma + ACTOR_sigma + 1e-9
            entropy = Categorical(probs = torch.stack([BG_sigma/sigma, FG_sigma/sigma, ACTOR_sigma/sigma], dim=-1) + 1e-8).entropy()
            entropy_loss = entropy.sum(dim=-1)
            loss = loss + args.entropy_weight*entropy_loss.mean()

        # add Total Variation loss
        if args.i_embed==1:
            if args.tv_loss_weight > 0:
                TV_loss = 0.0
                for embed in ['camera_grid_embed', 'world_grid_embed_FG']:
                    if render_kwargs_train['embedders'][embed] is None:
                        continue
                    n_levels, min_res, max_res, log2_hashmap_size = render_kwargs_train['embedders'][embed].n_levels, render_kwargs_train['embedders'][embed].base_resolution, render_kwargs_train['embedders'][embed].finest_resolution, render_kwargs_train['embedders'][embed].log2_hashmap_size

                    if args.xyzt_model and embed in ['camera_grid_embed', 'world_grid_embed_FG']:
                        tv_loss_fn = total_variation_loss_4D
                    else:
                        tv_loss_fn = total_variation_loss_3D

                    TV_loss += 0.001*sum(tv_loss_fn(render_kwargs_train['embedders'][embed].embeddings[i], \
                                                    min_res, max_res, \
                                                    i, log2_hashmap_size, \
                                                    n_levels=n_levels,
                                                    scaled=args.scaled_tvloss) for i in range(n_levels))
                
                ### TV loss on time
                if render_kwargs_train['embedders']['time_grid_embed'] is not None:
                    embed = 'time_grid_embed'
                    n_levels, min_res, max_res, log2_hashmap_size = render_kwargs_train['embedders'][embed].n_levels, render_kwargs_train['embedders'][embed].base_resolution, render_kwargs_train['embedders'][embed].finest_resolution, render_kwargs_train['embedders'][embed].log2_hashmap_size
                    TV_loss += sum(total_variation_loss_1D(render_kwargs_train['embedders'][embed].embeddings[i], \
                                                    min_res, max_res, \
                                                    i, log2_hashmap_size, \
                                                    n_levels=n_levels,
                                                    scaled=args.scaled_tvloss) for i in range(n_levels))

                loss = loss + args.tv_loss_weight * TV_loss

        # add Push Pull clustering loss on time embeddings
        if args.push_pull_weight>0:
            # push_pull = 0.0
            # if args.xyzt_model:
            #     for embed in ['camera_grid_embed', 'world_grid_embed_FG']:
            #         if render_kwargs_train['embedders'][embed] is None:
            #             continue
            #         n_levels, min_res, max_res, log2_hashmap_size = render_kwargs_train['embedders'][embed].n_levels, render_kwargs_train['embedders'][embed].base_resolution, render_kwargs_train['embedders'][embed].finest_resolution, render_kwargs_train['embedders'][embed].log2_hashmap_size
            #         push_pull += sum(push_pull_loss_xyzt(render_kwargs_train['embedders'][embed].embeddings[i], \
            #                                             min_res, max_res, \
            #                                             i, log2_hashmap_size, \
            #                                             n_levels=n_levels) for i in range(n_levels))
            push_pull = Push_Pull_loss(render_kwargs_train['embedders']['world_grid_embed_FG'])
            loss = loss + args.push_pull_weight * push_pull
 
        loss.backward()
        # pdb.set_trace()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # update temperature of PiecewiseConstant embeddings
        if args.piecewise_constant:
            render_kwargs_train['network_fine'].FG_xyzt_encoder.temperature *= args.temperature_decay**(1.0 / decay_steps)

        t = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(args.ckptdir, expname, '{:06d}.tar'.format(i))
            if args.i_embed==1:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'embed_fn_state_dict': render_kwargs_train['embed_fn'].state_dict(),
                    'world_grid_state_dict': render_kwargs_train['embedders']['world_grid_embed'].state_dict(),
                    'world_grid_FG_state_dict': render_kwargs_train['embedders']['world_grid_embed_FG'].state_dict() if render_kwargs_train['embedders']['world_grid_embed_FG'] else [],
                    'camera_grid_state_dict': render_kwargs_train['embedders']['camera_grid_embed'].state_dict() if render_kwargs_train['embedders']['camera_grid_embed'] else [],
                    'time_grid_state_dict': render_kwargs_train['embedders']['time_grid_embed'].state_dict() if render_kwargs_train['embedders']['time_grid_embed'] else [],
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps, uncertainty_maps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            if len(uncertainty_maps)>0:
                imageio.mimwrite(moviebase + 'uncertainties.mp4', to8b(uncertainty_maps), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), render_frame_idxs, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], mask_gts=mask_gts, BGguided_loss=BGguided_loss, BGguided_sparsity=BGguided_sparsity, savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            if args.bg_guided_loss:
                threshold = 1.0 - torch.sigmoid(BGguided_loss.threshold_param)
                tqdm.write(f"threshold: {threshold.item()} steepness: {BGguided_loss.steepness.item()}")
            if args.bg_guided_sparsity:
                threshold = 1.0 - torch.sigmoid(BGguided_sparsity.threshold_param)
                tqdm.write(f"threshold: {threshold.item()}")
            loss_list.append(loss.item())
            psnr_list.append(psnr.item())
            time_list.append(t)
            loss_psnr_time = {
                "losses": loss_list,
                "psnr": psnr_list,
                "time": time_list
            }
            with open(os.path.join(basedir, expname, "loss_vs_time.pkl"), "wb") as fp:
                pickle.dump(loss_psnr_time, fp)
        
        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
