import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from lpips import LPIPS

from hash_encoding import HashEmbedder, SHEncoder, XYZplusT_HashEmbedder
from time_encoding import XYZ_TimeOnOff_Encoding, XYZ_TimePiecewiseConstant

# Misc

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2L1 = lambda x, y : torch.mean(0.01 * (torch.sqrt(1 + ((x-y)**2).sum(dim=-1)/0.0001) - 1))
img2mse_with_uncertainty = lambda x, y, u : torch.mean(((x - y) ** 2)/(2*(u+1e-9)**2) + torch.log((u+1e-9)**2))
img2mse_perray = lambda x, y : ((x - y) ** 2).sum(dim=-1)
img2mse_with_uncertainty_perray = lambda x, y, u : ((x - y) ** 2).sum(dim=-1)/(2*(u+1e-9)**2) + 0.5*torch.log((u+1e-9)**2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
relu_act = nn.Softplus()
bce_loss = nn.BCELoss()

lpips_vgg = None

@torch.no_grad()
def get_perceptual_metrics(rgbs, gts, lpips_batch_size=8, device='cuda'):
    # rgbs and gts should be numpy arrays of the same shape. Can be just 1 x H x W x 3
    # From pixelNeRF https://github.com/sxyu/pixel-nerf/blob/2929708e90b246dbd0329ce2a128ef381bd8c25d/eval/calc_metrics.py#L188
    global lpips_vgg
    ssim = [structural_similarity(rgb, gt, multichannel=True, data_range=1) for rgb, gt in zip(rgbs, gts)]
    ssim = np.mean(ssim)
    psnr = [peak_signal_noise_ratio(rgb, gt, data_range=1) for rgb, gt in zip(rgbs, gts)]
    psnr = np.mean(psnr)

    # From pixelNeRF https://github.com/sxyu/pixel-nerf/blob/2929708e90b246dbd0329ce2a128ef381bd8c25d/eval/calc_metrics.py#L238
    if lpips_vgg is None:
        lpips_vgg = LPIPS(net="vgg").to(device=device)
    lpips_all = []
    preds_spl = torch.split(torch.from_numpy(rgbs).unsqueeze(0).permute(0,3,1,2).float(), lpips_batch_size, dim=0)
    gts_spl = torch.split(torch.from_numpy(gts).unsqueeze(0).permute(0,3,1,2).float(), lpips_batch_size, dim=0)
    for predi, gti in zip(preds_spl, gts_spl):
        lpips_i = lpips_vgg(predi.to(device=device), gti.to(device=device))
        lpips_all.append(lpips_i)
    lpips = torch.cat(lpips_all)
    lpips = lpips.mean().item()

    return psnr, ssim, lpips



# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, args, i=0):
    time_dim = None
    if i == -1:
        return nn.Identity(), 4, None
    elif i==0:
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }
        
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        out_dim = embedder_obj.out_dim
    elif i==1:
        # # embed = HashEmbedder(bounding_box=args.bounding_box, \
        # embed = XYZplusT_HashEmbedder(bounding_box=args.bounding_box, \
        #                     log2_hashmap_size=args.log2_hashmap_size, \
        #                     finest_resolution=args.finest_res)
        # out_dim = embed.out_dim
        # time_dim = embed.time_dim
        return nn.Identity(), 4, None ### I am initializing embeddings in NeRF models itself
    elif i==2:
        embed = SHEncoder()
        out_dim = embed.out_dim
    
    # out_dim means dimension of the TOTAL embedding, which could be XYZ or XYZT or XYZ+T
    # time_dim means dimension of time embedding
    return embed, out_dim, time_dim


def create_sigma_and_color_MLP(
    num_layers, num_layers_color,
    hidden_dim, hidden_dim_color,
    input_ch,
    input_ch_views,
    geo_feat_dim,
    use_viewdirs=True
):
    # sigma network
    sigma_net = []
    for l in range(num_layers):
        if l == 0:
            in_dim = input_ch
        else:
            in_dim = hidden_dim
        
        if l == num_layers - 1:
            out_dim = 1 + 1 + geo_feat_dim # 1 sigma + 1 uncertainty + 15 SH features for color
        else:
            out_dim = hidden_dim
        
        sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

    # color network
    color_net =  []
    for l in range(num_layers_color):
        if l == 0:
            if use_viewdirs:
                in_dim = input_ch_views + geo_feat_dim
            else:
                in_dim = geo_feat_dim
        else:
            in_dim = hidden_dim
        
        if l == num_layers_color - 1:
            out_dim = 3 # 3 rgb
        else:
            out_dim = hidden_dim
        
        color_net.append(nn.Linear(in_dim, out_dim, bias=False))

    return nn.ModuleList(sigma_net), nn.ModuleList(color_net)


def forward_through_MLP(sigma_net, color_net, embedded_x, embedded_views, \
                        num_layers, num_layers_color):
        # sigma
        h = embedded_x
        for l in range(num_layers):
            h = sigma_net[l](h)
            if l != num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma, uncertainties, geo_feat = h[..., 0], h[..., 1], h[..., 2:]
        # always predict uncertainties, just don't use them sometimes!
        uncertainties = relu_act(uncertainties)
        sigma = relu_act(sigma)

        # color
        if embedded_views is not None:
            h = torch.cat([embedded_views, geo_feat], dim=-1)
        else:
            h = geo_feat

        for l in range(num_layers_color):
            h = color_net[l](h)
            if l != num_layers_color - 1:
                h = F.relu(h, inplace=True)
            
        color = h
        return sigma, color, uncertainties


# Small NeRF for Hash embeddings
class NeRFSmall(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,
                 use_uncertainties=False
                 ):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_uncertainties = use_uncertainties

        # sigma network
        self.num_layers = num_layers
        self.num_layers_color = num_layers_color
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        self.sigma_net, self.color_net = create_sigma_and_color_MLP(
                                            num_layers, num_layers_color,
                                            hidden_dim, hidden_dim_color,
                                            input_ch,
                                            input_ch_views,
                                            geo_feat_dim
                                         )
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # sigma
        sigma, color, uncertainties = forward_through_MLP(self.sigma_net, self.color_net, \
                                                          input_pts, input_views, \
                                                          self.num_layers, self.num_layers_color)

        if self.use_uncertainties:
            outputs = torch.cat([color, sigma.unsqueeze(dim=-1), uncertainties.unsqueeze(dim=-1)], -1)
        else:
            outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

        return outputs


class BackgroundForegroundNeRF(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,
                 time_dim=8,
                 use_uncertainties=False,
                 only_background=False # we will rarely use this option
                 ):
        super(BackgroundForegroundNeRF, self).__init__()

        self.input_ch = input_ch # size of XYZ and time embeddings combined!
        self.input_ch_views = input_ch_views
        self.time_dim = time_dim

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.use_uncertainties = use_uncertainties
        self.only_background = only_background

        ### Background Network
        self.background_sigma_net, self.background_color_net = create_sigma_and_color_MLP(
                                                                    num_layers, num_layers_color,
                                                                    hidden_dim, hidden_dim_color,
                                                                    input_ch - time_dim, # only XYZ
                                                                    input_ch_views,
                                                                    geo_feat_dim
                                                               )

        ### Foreground Network
        self.foreground_sigma_net, self.foreground_color_net = create_sigma_and_color_MLP(
                                                                    num_layers, num_layers_color,
                                                                    hidden_dim, hidden_dim_color,
                                                                    input_ch, # XYZ + time embedding
                                                                    input_ch_views,
                                                                    geo_feat_dim
                                                               )
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        input_xyz, _ = input_pts[..., :-self.time_dim], input_pts[..., -self.time_dim:]

        background_sigma, background_color, background_uncertainties = forward_through_MLP( \
                                                                        self.background_sigma_net, self.background_color_net, \
                                                                        input_xyz, input_views, \
                                                                        self.num_layers, self.num_layers_color)
        if not self.only_background:
            foreground_sigma, foreground_color, foreground_uncertainties = forward_through_MLP( \
                                                                            self.foreground_sigma_net, self.foreground_color_net, \
                                                                            input_pts, input_views, \
                                                                            self.num_layers, self.num_layers_color)
        else:
            # dummies for foreground
            foreground_sigma, foreground_color, foreground_uncertainties = torch.zeros_like(background_sigma), torch.zeros_like(background_color), None

        # Principled color mixing
        sigma = background_sigma + foreground_sigma + 1e-9
        color = (background_sigma/sigma)[:,None] * background_color + (foreground_sigma/sigma)[:,None] * foreground_color

        if self.only_background:
            if self.use_uncertainties:
                return torch.cat([color, \
                                sigma.unsqueeze(dim=-1), \
                                background_uncertainties.unsqueeze(dim=-1)], -1)
            else:
                return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

        if not self.only_background:
            if self.use_uncertainties:
                return torch.cat([color, \
                                    sigma.unsqueeze(dim=-1), \
                                    foreground_uncertainties.unsqueeze(dim=-1), \
                                    foreground_sigma.unsqueeze(dim=-1)], -1)
            else:
                return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)


class BackgroundForegroundNeRF_separateEmbeddings(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,
                 use_uncertainties=False,
                 only_background=False, # we will rarely use this option
                 args=None
                 ):
        super(BackgroundForegroundNeRF_separateEmbeddings, self).__init__()

        self.input_ch = input_ch # it's raw xyzt, so input_ch=4
        self.input_ch_views = input_ch_views # has embedded views

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.use_uncertainties = use_uncertainties
        self.only_background = only_background

        xyz_bounding_box = args.bounding_box[0][:3], args.bounding_box[1][:3]
        self.BG_embedder = HashEmbedder(bounding_box=xyz_bounding_box, \
                            log2_hashmap_size=args.log2_hashmap_size, \
                            finest_resolution=args.finest_res)
        self.FG_embedder = XYZplusT_HashEmbedder(bounding_box=args.bounding_box, \
                            log2_hashmap_size=args.log2_hashmap_size, \
                            finest_resolution=args.finest_res)

        ### Background Network
        self.BG_sigma_net, self.BG_color_net = create_sigma_and_color_MLP(num_layers, num_layers_color,
                                                                    hidden_dim, hidden_dim_color,
                                                                    self.BG_embedder.out_dim, # only XYZ
                                                                    input_ch_views, geo_feat_dim
                                                               )

        ### Foreground Network
        self.FG_sigma_net, self.FG_color_net = create_sigma_and_color_MLP(num_layers, num_layers_color,
                                                                    hidden_dim, hidden_dim_color,
                                                                    self.FG_embedder.out_dim, # XYZ + time embedding
                                                                    input_ch_views, geo_feat_dim
                                                               )
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        BG_embedded_xyz = self.BG_embedder(input_pts[...,:3])
        FG_embedded_xyzt = self.FG_embedder(input_pts)

        BG_sigma, BG_color, BG_uncertainties = forward_through_MLP(self.BG_sigma_net, self.BG_color_net, \
                                                    BG_embedded_xyz, input_views, \
                                                    self.num_layers, self.num_layers_color)
        if not self.only_background:
            FG_sigma, FG_color, FG_uncertainties = forward_through_MLP(self.FG_sigma_net, self.FG_color_net, \
                                                    FG_embedded_xyzt, input_views, \
                                                    self.num_layers, self.num_layers_color)
        else:
            # dummies for foreground
            FG_sigma, FG_color, FG_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), None

        # Principled color mixing
        sigma = BG_sigma + FG_sigma + 1e-9
        color = (BG_sigma/sigma)[:,None] * BG_color + (FG_sigma/sigma)[:,None] * FG_color

        if self.only_background:
            if self.use_uncertainties:
                return torch.cat([color, sigma.unsqueeze(dim=-1), BG_uncertainties.unsqueeze(dim=-1)], -1)
            else:
                return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

        if not self.only_background:
            if self.use_uncertainties:
                return torch.cat([color, sigma.unsqueeze(dim=-1), \
                                    FG_uncertainties.unsqueeze(dim=-1), \
                                    FG_sigma.unsqueeze(dim=-1)], -1)
            else:
                return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)


class BackgroundForegroundActorNeRF_separateEmbeddings(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=4, input_cam_ch=4,
                 input_ch_views=3, input_ch_views_cam=3,
                 use_uncertainties=False,
                 BG_embedder=None, FG_embedder=None, ACTOR_embedder=None,
                 big=False
                 ):
        super(BackgroundForegroundActorNeRF_separateEmbeddings, self).__init__()

        self.input_ch = input_ch # it's raw xyzt, so input_ch=4
        self.input_ch_views = input_ch_views # has embedded views
        self.input_cam_ch = input_cam_ch # should be 4
        self.input_ch_views_cam = input_ch_views_cam # has embedded views

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.use_uncertainties = use_uncertainties
        self.big = big

        self.BG_embedder, self.FG_embedder, self.ACTOR_embedder = BG_embedder, FG_embedder, ACTOR_embedder

        ### Background Network
        self.BG_sigma_net, self.BG_color_net = create_sigma_and_color_MLP(num_layers, num_layers_color,
                                                                    hidden_dim, hidden_dim_color,
                                                                    self.BG_embedder.out_dim, # only XYZ
                                                                    input_ch_views, geo_feat_dim
                                                               )

        ### Foreground Network
        self.FG_sigma_net, self.FG_color_net = create_sigma_and_color_MLP(num_layers+2*big, num_layers_color+2*big,
                                                                    hidden_dim+64*big, hidden_dim_color+64*big,
                                                                    self.FG_embedder.out_dim, # XYZ + time embedding
                                                                    input_ch_views, geo_feat_dim
                                                               )

        ### Actor Network
        self.ACTOR_sigma_net, self.ACTOR_color_net = create_sigma_and_color_MLP(num_layers+2*big, num_layers_color+2*big,
                                                                    hidden_dim+64*big, hidden_dim_color+64*big,
                                                                    self.ACTOR_embedder.out_dim, # XYZ + time embedding
                                                                    input_ch_views_cam, geo_feat_dim
                                                               )
    
    def forward(self, x):
        input_pts, input_pts_cam, input_views, input_views_cam = torch.split(x, [self.input_ch, self.input_cam_ch, self.input_ch_views, self.input_ch_views_cam], dim=-1)
        BG_embedded_xyz = self.BG_embedder(input_pts[...,:3])
        FG_embedded_xyzt = self.FG_embedder(input_pts)
        ACTOR_embedded_xyzt = self.ACTOR_embedder(input_pts_cam)

        BG_sigma, BG_color, _ = forward_through_MLP(self.BG_sigma_net, self.BG_color_net, \
                                                    BG_embedded_xyz, input_views, \
                                                    self.num_layers, self.num_layers_color)
        FG_sigma, FG_color, FG_uncertainties = forward_through_MLP(self.FG_sigma_net, self.FG_color_net, \
                                                    FG_embedded_xyzt, input_views, \
                                                    self.num_layers+2*self.big, self.num_layers_color+2*self.big)
        ACTOR_sigma, ACTOR_color, ACTOR_uncertainties = forward_through_MLP(self.ACTOR_sigma_net, self.ACTOR_color_net, \
                                                    ACTOR_embedded_xyzt, input_views_cam, \
                                                    self.num_layers+2*self.big, self.num_layers_color+2*self.big)

        # Principled color mixing
        sigma = BG_sigma + FG_sigma + ACTOR_sigma + 1e-9
        color = (BG_sigma/sigma)[:,None] * BG_color + (FG_sigma/sigma)[:,None] * FG_color + (ACTOR_sigma/sigma)[:,None] * ACTOR_color

        if self.use_uncertainties:
            return torch.cat([color, sigma.unsqueeze(dim=-1), \
                                FG_uncertainties.unsqueeze(dim=-1), \
                                FG_sigma.unsqueeze(dim=-1), \
                                ACTOR_uncertainties.unsqueeze(dim=-1), \
                                ACTOR_sigma.unsqueeze(dim=-1)], -1)
        else:
            return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)


class NeuralDiff(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 num_layers_FG=4, num_layers_ACTOR=4,
                 input_ch=4, input_cam_ch=4,
                 input_ch_views=3, input_ch_views_cam=3,
                 use_uncertainties=False,
                 world_grid_embed=None, world_grid_embed_FG=None, camera_grid_embed=None, time_grid_embed=None,
                 big=False,
                 coarse=True):
        super(NeuralDiff, self).__init__()

        self.input_ch, self.input_cam_ch = input_ch, input_cam_ch # it's raw xyzt, so input_ch=4
        self.input_ch_views, self.input_ch_views_cam = input_ch_views, input_ch_views_cam # has embedded views

        self.num_layers, self.num_layers_color, self.hidden_dim = num_layers, num_layers_color, hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_FG, self.num_layers_ACTOR = num_layers_FG, num_layers_ACTOR
        if coarse:
            self.use_uncertainties = False
        else:
            self.use_uncertainties = use_uncertainties
        self.big = big
        self.coarse = coarse # Is this a coarse model?
        
        self.world_grid_embed, self.camera_grid_embed, self.time_grid_embed = world_grid_embed, camera_grid_embed, time_grid_embed

        ### Background Network
        # sigma network
        BG_sigma_net = []
        for l in range(num_layers):
            in_dim = world_grid_embed.out_dim if l == 0 else (hidden_dim + 64)*(self.big+1)
            out_dim = 1 + geo_feat_dim if l==num_layers-1 else (hidden_dim + 64)*(self.big+1)   # 1 sigma + 15 SH features for color
            BG_sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l!=num_layers-1:
                BG_sigma_net.append(nn.ReLU())

        # color network
        BG_color_net =  []
        for l in range(num_layers_color):
            in_dim = input_ch_views + geo_feat_dim if l == 0 else hidden_dim*(self.big+1)
            out_dim = 3 if l==num_layers_color-1 else hidden_dim*(self.big+1)  # 3 rgb
            BG_color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l!=num_layers_color-1:
                BG_color_net.append(nn.ReLU())

        self.BG_sigma_net, self.BG_color_net = nn.Sequential(*BG_sigma_net), nn.Sequential(*BG_color_net)

        if not coarse: # if this is a "fine" model, use dynamic components
            ### Foreground Network
            FG_net = []
            for l in range(num_layers_FG):
                in_dim = time_grid_embed.out_dim + geo_feat_dim if l == 0 else hidden_dim*(self.big+1)
                out_dim = 3 + 1 + 1 if l==num_layers_FG-1 else hidden_dim*(self.big+1)  # 3 rgb_FG + 1 sigma_FG + 1 uncertainty_FG
                FG_net.append(nn.Linear(in_dim, out_dim, bias=False))
                if l!=num_layers_FG-1:
                    FG_net.append(nn.ReLU())
            self.FG_net = nn.Sequential(*FG_net)

            ### Actor Network
            ACTOR_net = []
            for l in range(num_layers_ACTOR):
                in_dim = camera_grid_embed.out_dim + time_grid_embed.out_dim if l == 0 else hidden_dim*2
                out_dim = 3 + 1 + 1 if l==num_layers_ACTOR-1 else hidden_dim*2  # 3 rgb_ACTOR + 1 sigma_ACTOR + 1 uncertainty_ACTOR
                ACTOR_net.append(nn.Linear(in_dim, out_dim, bias=False))
                if l!=num_layers_ACTOR-1:
                    ACTOR_net.append(nn.ReLU())
            self.ACTOR_net = nn.Sequential(*ACTOR_net)
    
    def forward(self, x):
        input_pts, input_pts_cam, input_views, input_views_cam = torch.split(x, [self.input_ch, self.input_cam_ch, self.input_ch_views, self.input_ch_views_cam], dim=-1)
        embedded_xyz, keep_mask = self.world_grid_embed(input_pts[...,:3])
        if not self.coarse:
            embedded_time = self.time_grid_embed(input_pts[...,3].unsqueeze(-1))
            embedded_xyz_cam, keep_mask_cam = self.camera_grid_embed(input_pts_cam[...,:3])

        ### Static components
        h = self.BG_sigma_net(embedded_xyz)
        BG_sigma, ray_encoding = h[..., 0], h[..., 1:]
        BG_color = self.BG_color_net(torch.cat([input_views, ray_encoding], dim=-1))
        BG_color = F.sigmoid(BG_color)
        BG_sigma = relu_act(BG_sigma)
        BG_sigma, BG_color = BG_sigma*keep_mask, BG_color*keep_mask[:,None]

        ### Dynamic components
        if not self.coarse:
            h = self.FG_net(torch.cat([embedded_time, ray_encoding], dim=-1))
            FG_color, FG_sigma, FG_uncertainties = h[..., :3], h[..., 3], h[..., 4]
            FG_color = F.sigmoid(FG_color)

            h = self.ACTOR_net(torch.cat([embedded_xyz_cam, embedded_time], dim=-1))
            ACTOR_color, ACTOR_sigma, ACTOR_uncertainties = h[..., :3], h[..., 3], h[..., 4]
            ACTOR_color = F.sigmoid(ACTOR_color)

            FG_sigma, FG_uncertainties = relu_act(FG_sigma), relu_act(FG_uncertainties)
            ACTOR_sigma, ACTOR_uncertainties = relu_act(ACTOR_sigma), relu_act(ACTOR_uncertainties)
            FG_sigma, FG_color = FG_sigma*keep_mask, FG_color*keep_mask[:,None]
            ACTOR_sigma, ACTOR_color = ACTOR_sigma*keep_mask_cam, ACTOR_color*keep_mask_cam[:,None]
        else:
            FG_sigma, FG_color, FG_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), None
            ACTOR_sigma, ACTOR_color, ACTOR_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), None

        # Principled color mixing
        sigma = BG_sigma + FG_sigma + ACTOR_sigma + 1e-9
        color = (BG_sigma/sigma)[:,None] * BG_color + (FG_sigma/sigma)[:,None] * FG_color + (ACTOR_sigma/sigma)[:,None] * ACTOR_color

        if self.use_uncertainties:
            return torch.cat([BG_color, FG_color, ACTOR_color, # :3, 3:6, 6:9
                                BG_sigma.unsqueeze(dim=-1), FG_sigma.unsqueeze(dim=-1), ACTOR_sigma.unsqueeze(dim=-1), # 9, 10, 11
                                FG_uncertainties.unsqueeze(dim=-1), ACTOR_uncertainties.unsqueeze(dim=-1)], -1) # 12, 13
        else:
            return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)


class NeuralDiff_BGFGSeparate(nn.Module):
    '''
    Static Background model uses a low-frequency grid, 
    and Foreground model uses a high-frequency grid.
    '''
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 num_layers_FG=4, num_layers_ACTOR=4,
                 input_ch=4, input_cam_ch=4,
                 input_ch_views=3, input_ch_views_cam=3,
                 use_uncertainties=False,
                 world_grid_embed=None, world_grid_embed_FG=None, camera_grid_embed=None, time_grid_embed=None,
                 big=False,
                 coarse=True):
        super(NeuralDiff_BGFGSeparate, self).__init__()

        self.input_ch, self.input_cam_ch = input_ch, input_cam_ch # it's raw xyzt, so input_ch=4
        self.input_ch_views, self.input_ch_views_cam = input_ch_views, input_ch_views_cam # has embedded views

        self.num_layers, self.num_layers_color, self.hidden_dim = num_layers, num_layers_color, hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_FG, self.num_layers_ACTOR = num_layers_FG, num_layers_ACTOR
        if coarse:
            self.use_uncertainties = False
        else:
            self.use_uncertainties = use_uncertainties
        self.big = big
        self.coarse = coarse # Is this a coarse model?
        
        self.world_grid_embed = world_grid_embed
        if not coarse: # if this is a "fine" model, use dynamic components
            self.camera_grid_embed, self.time_grid_embed = camera_grid_embed, time_grid_embed
            self.world_grid_embed_FG = world_grid_embed_FG # separate high freq grid for FG

        ### Background Network
        # sigma network
        BG_sigma_net = []
        for l in range(num_layers):
            in_dim = world_grid_embed.out_dim if l == 0 else (hidden_dim + 64)*(self.big+1)
            out_dim = 1 + geo_feat_dim if l==num_layers-1 else (hidden_dim + 64)*(self.big+1)   # 1 sigma + 15 SH features for color
            BG_sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l!=num_layers-1:
                BG_sigma_net.append(nn.ReLU())

        # color network
        BG_color_net =  []
        for l in range(num_layers_color):
            in_dim = input_ch_views + geo_feat_dim if l == 0 else hidden_dim*(self.big+1)
            out_dim = 3 if l==num_layers_color-1 else hidden_dim*(self.big+1)  # 3 rgb
            BG_color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l!=num_layers_color-1:
                BG_color_net.append(nn.ReLU())

        self.BG_sigma_net, self.BG_color_net = nn.Sequential(*BG_sigma_net), nn.Sequential(*BG_color_net)

        if not coarse: # if this is a "fine" model, use dynamic components
            ### Foreground Network
            FG_net = []
            for l in range(num_layers_FG):
                in_dim = world_grid_embed_FG.out_dim + time_grid_embed.out_dim if l == 0 else hidden_dim*(self.big+1)
                out_dim = 3 + 1 + 1 if l==num_layers_FG-1 else hidden_dim*(self.big+1)  # 3 rgb_FG + 1 sigma_FG + 1 uncertainty_FG
                FG_net.append(nn.Linear(in_dim, out_dim, bias=False))
                if l!=num_layers_FG-1:
                    FG_net.append(nn.ReLU())
            self.FG_net = nn.Sequential(*FG_net)

            ### Actor Network
            ACTOR_net = []
            for l in range(num_layers_ACTOR):
                in_dim = camera_grid_embed.out_dim + time_grid_embed.out_dim if l == 0 else hidden_dim*2
                out_dim = 3 + 1 + 1 if l==num_layers_ACTOR-1 else hidden_dim*2  # 3 rgb_ACTOR + 1 sigma_ACTOR + 1 uncertainty_ACTOR
                ACTOR_net.append(nn.Linear(in_dim, out_dim, bias=False))
                if l!=num_layers_ACTOR-1:
                    ACTOR_net.append(nn.ReLU())
            self.ACTOR_net = nn.Sequential(*ACTOR_net)
    
    def forward(self, x):
        input_pts, input_pts_cam, input_views, _ = torch.split(x, [self.input_ch, self.input_cam_ch, self.input_ch_views, self.input_ch_views_cam], dim=-1)
        embedded_xyz, keep_mask = self.world_grid_embed(input_pts[...,:3])
        if not self.coarse:
            embedded_xyz_FG, keep_mask_FG = self.world_grid_embed_FG(input_pts[...,:3])
            embedded_time = self.time_grid_embed(input_pts[...,3].unsqueeze(-1))
            embedded_xyz_cam, keep_mask_cam = self.camera_grid_embed(input_pts_cam[...,:3])

        ### Static components
        h = self.BG_sigma_net(embedded_xyz)
        BG_sigma, ray_encoding = h[..., 0], h[..., 1:]
        BG_color = self.BG_color_net(torch.cat([input_views, ray_encoding], dim=-1))
        BG_color = F.sigmoid(BG_color)
        BG_sigma = relu_act(BG_sigma)
        BG_sigma, BG_color = BG_sigma*keep_mask, BG_color*keep_mask[:,None]

        ### Dynamic components
        if not self.coarse:
            h = self.FG_net(torch.cat([embedded_xyz_FG, embedded_time], dim=-1))
            FG_color, FG_sigma, FG_uncertainties = h[..., :3], h[..., 3], h[..., 4]
            FG_color = F.sigmoid(FG_color)

            h = self.ACTOR_net(torch.cat([embedded_xyz_cam, embedded_time], dim=-1))
            ACTOR_color, ACTOR_sigma, ACTOR_uncertainties = h[..., :3], h[..., 3], h[..., 4]
            ACTOR_color = F.sigmoid(ACTOR_color)

            FG_sigma, FG_uncertainties = relu_act(FG_sigma), relu_act(FG_uncertainties)
            ACTOR_sigma, ACTOR_uncertainties = relu_act(ACTOR_sigma), relu_act(ACTOR_uncertainties)
            FG_sigma, FG_color = FG_sigma*keep_mask_FG, FG_color*keep_mask_FG[:,None]
            ACTOR_sigma, ACTOR_color = ACTOR_sigma*keep_mask_cam, ACTOR_color*keep_mask_cam[:,None]
        else:
            FG_sigma, FG_color, FG_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), None
            ACTOR_sigma, ACTOR_color, ACTOR_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), None

        # Principled color mixing
        sigma = BG_sigma + FG_sigma + ACTOR_sigma + 1e-9
        color = (BG_sigma/sigma)[:,None] * BG_color + (FG_sigma/sigma)[:,None] * FG_color + (ACTOR_sigma/sigma)[:,None] * ACTOR_color

        if self.use_uncertainties:
            return torch.cat([BG_color, FG_color, ACTOR_color, # :3, 3:6, 6:9
                                BG_sigma.unsqueeze(dim=-1), FG_sigma.unsqueeze(dim=-1), ACTOR_sigma.unsqueeze(dim=-1), # 9, 10, 11
                                FG_uncertainties.unsqueeze(dim=-1), ACTOR_uncertainties.unsqueeze(dim=-1)], -1) # 12, 13
        else:
            return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)


class BGFG_XYZT(nn.Module):
    '''
    XYZT grid for foreground model 
    '''
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 num_layers_FG=4,
                 input_ch=4, input_cam_ch=4,
                 input_ch_views=3, input_ch_views_cam=3,
                 use_uncertainties=False,
                 static_grid=None, xyzt_grid=None, xyzt_grid_cam=None,
                 coarse=True,
                 use_viewdirs_FG=False,
                 use_actor=False,
                 small_MLPs_dyn=False):
        super(BGFG_XYZT, self).__init__()

        self.input_ch, self.input_cam_ch = input_ch, input_cam_ch # it's raw xyzt, so input_ch=4
        self.input_ch_views, self.input_ch_views_cam = input_ch_views, input_ch_views_cam # has embedded views

        self.num_layers, self.num_layers_color, self.hidden_dim = num_layers, num_layers_color, hidden_dim
        self.small_MLPs_dyn = small_MLPs_dyn
        if small_MLPs_dyn:
            self.num_layers_dyn, self.num_layers_color_dyn = 1, 1
        else:
            self.num_layers_dyn, self.num_layers_color_dyn = num_layers, num_layers_color

        self.geo_feat_dim = geo_feat_dim
        self.num_layers_FG = num_layers_FG
        if coarse:
            self.use_uncertainties = False
        else:
            self.use_uncertainties = use_uncertainties
        self.coarse = coarse # Is this a coarse model?
        self.use_viewdirs_FG = use_viewdirs_FG
        self.use_actor = use_actor

        self.static_grid = static_grid
        if not coarse: # if this is a "fine" model, use dynamic components
            self.xyzt_grid = xyzt_grid # separate high freq grid for FG
            if self.use_actor:
                self.xyzt_grid_cam = xyzt_grid_cam 

        ### Background Network
        self.BG_sigma_net, self.BG_color_net = create_sigma_and_color_MLP(num_layers, num_layers_color,
                                                                    hidden_dim, 53, # 53 is random! bad code, sorry :(
                                                                    self.static_grid.out_dim, # only XYZ
                                                                    input_ch_views, geo_feat_dim)

        if not coarse: # if this is a "fine" model, use dynamic components
            ### Foreground Network
            self.FG_sigma_net, self.FG_color_net = create_sigma_and_color_MLP(self.num_layers_dyn, self.num_layers_color_dyn,
                                                                    hidden_dim, 53, # 53 is random! bad code, sorry :(
                                                                    self.xyzt_grid.out_dim,
                                                                    input_ch_views, geo_feat_dim,
                                                                    use_viewdirs=use_viewdirs_FG)
            if self.use_actor:            
                ### Actor Network
                self.ACTOR_sigma_net, self.ACTOR_color_net = create_sigma_and_color_MLP(self.num_layers_dyn, self.num_layers_color_dyn,
                                                                        hidden_dim, 53, # 53 is random! bad code, sorry :(
                                                                        self.xyzt_grid_cam.out_dim,
                                                                        input_ch_views, geo_feat_dim,
                                                                        use_viewdirs=use_viewdirs_FG)

    def forward(self, x):
        input_pts, input_pts_cam, input_views, input_views_cam = torch.split(x, [self.input_ch, self.input_cam_ch, self.input_ch_views, self.input_ch_views_cam], dim=-1)
        embedded_xyz, keep_mask = self.static_grid(input_pts[...,:3])
        if not self.coarse:
            embedded_xyzt, keep_mask_FG = self.xyzt_grid(input_pts)
            if self.use_actor:
                embedded_xyzt_cam, keep_mask_ACTOR = self.xyzt_grid_cam(input_pts_cam)
        
        ### Static components
        BG_sigma, BG_color, _ = forward_through_MLP(self.BG_sigma_net, self.BG_color_net, \
                                                    embedded_xyz, input_views, \
                                                    self.num_layers, self.num_layers_color)
        BG_color = F.sigmoid(BG_color)
        BG_sigma, BG_color = BG_sigma*keep_mask, BG_color*keep_mask[:,None]

        ### Dynamic components
        if not self.coarse:
            FG_sigma, FG_color, FG_uncertainties = forward_through_MLP(self.FG_sigma_net, self.FG_color_net, \
                                                                    embedded_xyzt, input_views if self.use_viewdirs_FG else None, \
                                                                    self.num_layers_dyn, self.num_layers_color_dyn)
            FG_color = F.sigmoid(FG_color)
            FG_sigma, FG_color = FG_sigma*keep_mask_FG, FG_color*keep_mask_FG[:,None]

            if self.use_actor:
                ACTOR_sigma, ACTOR_color, ACTOR_uncertainties = forward_through_MLP(self.ACTOR_sigma_net, self.ACTOR_color_net, \
                                                                    embedded_xyzt_cam, input_views_cam if self.use_viewdirs_FG else None, \
                                                                    self.num_layers_dyn, self.num_layers_color_dyn)
                ACTOR_color = F.sigmoid(ACTOR_color)
                ACTOR_sigma, ACTOR_color = ACTOR_sigma*keep_mask_ACTOR, ACTOR_color*keep_mask_ACTOR[:,None]
            else:
                ### we don't have actor, but doing this to make the code consistent
                ACTOR_sigma, ACTOR_color = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color) 
                ACTOR_uncertainties = torch.zeros_like(FG_uncertainties)
                ##################            
        else:
            FG_sigma, FG_color, FG_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), None
            ACTOR_sigma, ACTOR_color, ACTOR_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), None

        # Principled color mixing
        sigma = BG_sigma + FG_sigma + ACTOR_sigma + 1e-9
        color = (BG_sigma/sigma)[:,None] * BG_color + (FG_sigma/sigma)[:,None] * FG_color + (ACTOR_sigma/sigma)[:,None] * ACTOR_color

        if self.use_uncertainties:
            return torch.cat([BG_color, FG_color, ACTOR_color, # :3, 3:6, 6:9
                                BG_sigma.unsqueeze(dim=-1), FG_sigma.unsqueeze(dim=-1), ACTOR_sigma.unsqueeze(dim=-1), # 9, 10, 11
                                FG_uncertainties.unsqueeze(dim=-1), ACTOR_uncertainties.unsqueeze(dim=-1), # 12, 13
                                embedded_xyzt], -1) # 14:
        else:
            return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)


class BGFG_OnOffEncoding(nn.Module):
    '''
    Uses OnOffEncoding for time
    '''
    def __init__(self,
                 t_bounds,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 input_ch=4, input_cam_ch=4,
                 input_ch_views=3, input_ch_views_cam=3,
                 use_uncertainties=False,
                 static_grid=None, FG_xyz_grid=None,
                 coarse=True):
        super(BGFG_OnOffEncoding, self).__init__()

        self.input_ch, self.input_cam_ch = input_ch, input_cam_ch # it's raw xyzt, so input_ch=4
        self.input_ch_views, self.input_ch_views_cam = input_ch_views, input_ch_views_cam # has embedded views

        self.num_layers, self.num_layers_color, self.hidden_dim = num_layers, num_layers_color, hidden_dim
        
        self.geo_feat_dim = geo_feat_dim
        self.use_uncertainties = False if coarse else use_uncertainties
        self.coarse = coarse # Is this a coarse model?
        
        self.static_grid = static_grid
        if not coarse: # if this is a "fine" model, use dynamic components
            self.FG_xyzt_encoder = XYZ_TimeOnOff_Encoding(xyz_encoder=FG_xyz_grid, t_bounds=t_bounds)
            
        ### Background Network
        self.BG_sigma_net, self.BG_color_net = create_sigma_and_color_MLP(num_layers, num_layers_color,
                                                                    hidden_dim, 53, # 53 is random! bad code, sorry :(
                                                                    self.static_grid.out_dim, # only XYZ
                                                                    input_ch_views, geo_feat_dim)

        if not coarse: # if this is a "fine" model, use dynamic components
            ### Foreground Network
            self.FG_sigma_net, self.FG_color_net = create_sigma_and_color_MLP(num_layers, num_layers_color,
                                                                    hidden_dim, 53, # 53 is random! bad code, sorry :(
                                                                    self.FG_xyzt_encoder.out_dim,
                                                                    input_ch_views, geo_feat_dim,
                                                                    use_viewdirs=False)
            
    def forward(self, x):
        input_pts, input_pts_cam, input_views, input_views_cam = torch.split(x, [self.input_ch, self.input_cam_ch, self.input_ch_views, self.input_ch_views_cam], dim=-1)
        embedded_xyz, keep_mask = self.static_grid(input_pts[...,:3])
        if not self.coarse:
            embedded_xyzt = self.FG_xyzt_encoder(input_pts)
            
        ### Static components
        BG_sigma, BG_color, _ = forward_through_MLP(self.BG_sigma_net, self.BG_color_net, \
                                                    embedded_xyz, input_views, \
                                                    self.num_layers, self.num_layers_color)
        BG_color = F.sigmoid(BG_color)
        BG_sigma, BG_color = BG_sigma*keep_mask, BG_color*keep_mask[:,None]

        ### Dynamic components
        if not self.coarse:
            FG_sigma, FG_color, FG_uncertainties = forward_through_MLP(self.FG_sigma_net, self.FG_color_net, \
                                                                    embedded_xyzt, None, \
                                                                    self.num_layers, self.num_layers_color)
            FG_color = F.sigmoid(FG_color)

            ### we don't have actor, but doing this to make the code consistent
            ACTOR_sigma, ACTOR_color = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color) 
            ACTOR_uncertainties = torch.zeros_like(FG_uncertainties)
            ##################            
        else:
            FG_sigma, FG_color, FG_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), None
            ACTOR_sigma, ACTOR_color, ACTOR_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), None

        # Principled color mixing
        sigma = BG_sigma + FG_sigma + ACTOR_sigma + 1e-9
        color = (BG_sigma/sigma)[:,None] * BG_color + (FG_sigma/sigma)[:,None] * FG_color + (ACTOR_sigma/sigma)[:,None] * ACTOR_color

        if self.use_uncertainties:
            return torch.cat([BG_color, FG_color, ACTOR_color, # :3, 3:6, 6:9
                                BG_sigma.unsqueeze(dim=-1), FG_sigma.unsqueeze(dim=-1), ACTOR_sigma.unsqueeze(dim=-1), # 9, 10, 11
                                FG_uncertainties.unsqueeze(dim=-1), ACTOR_uncertainties.unsqueeze(dim=-1), # 12, 13
                                embedded_xyzt], -1) # 14:
        else:
            return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)


class BGFG_PiecewiseConst(nn.Module):
    '''
    Uses OnOffEncoding for time
    '''
    def __init__(self,
                 xyzt_bounds,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 input_ch=4, input_cam_ch=4,
                 input_ch_views=3, input_ch_views_cam=3,
                 use_uncertainties=False,
                 static_grid=None,
                 coarse=True,
                 init_temperature=100.0,
                 n_pieces=10):
        super(BGFG_PiecewiseConst, self).__init__()

        self.input_ch, self.input_cam_ch = input_ch, input_cam_ch # it's raw xyzt, so input_ch=4
        self.input_ch_views, self.input_ch_views_cam = input_ch_views, input_ch_views_cam # has embedded views

        self.num_layers, self.num_layers_color, self.hidden_dim = num_layers, num_layers_color, hidden_dim
        
        self.geo_feat_dim = geo_feat_dim
        self.use_uncertainties = False if coarse else use_uncertainties
        self.coarse = coarse # Is this a coarse model?
        
        self.static_grid = static_grid
        if not coarse: # if this is a "fine" model, use dynamic components
            self.FG_xyzt_encoder = XYZ_TimePiecewiseConstant(xyzt_bounds=xyzt_bounds,
                                                             init_temperature=init_temperature,
                                                             n_pieces=n_pieces)
            
        ### Background Network
        self.BG_sigma_net, self.BG_color_net = create_sigma_and_color_MLP(num_layers, num_layers_color,
                                                                    hidden_dim, 53, # 53 is random! bad code, sorry :(
                                                                    self.static_grid.out_dim, # only XYZ
                                                                    input_ch_views, geo_feat_dim)

        if not coarse: # if this is a "fine" model, use dynamic components
            ### Foreground Network
            self.FG_sigma_net, self.FG_color_net = create_sigma_and_color_MLP(num_layers, num_layers_color,
                                                                    hidden_dim, 53, # 53 is random! bad code, sorry :(
                                                                    self.FG_xyzt_encoder.out_dim,
                                                                    input_ch_views, geo_feat_dim,
                                                                    use_viewdirs=False)
            
    def forward(self, x):
        input_pts, input_pts_cam, input_views, input_views_cam = torch.split(x, [self.input_ch, self.input_cam_ch, self.input_ch_views, self.input_ch_views_cam], dim=-1)
        embedded_xyz, keep_mask = self.static_grid(input_pts[...,:3])
        if not self.coarse:
            embedded_xyzt = self.FG_xyzt_encoder(input_pts)
            
        ### Static components
        BG_sigma, BG_color, _ = forward_through_MLP(self.BG_sigma_net, self.BG_color_net, \
                                                    embedded_xyz, input_views, \
                                                    self.num_layers, self.num_layers_color)
        BG_color = F.sigmoid(BG_color)
        BG_sigma, BG_color = BG_sigma*keep_mask, BG_color*keep_mask[:,None]

        ### Dynamic components
        if not self.coarse:
            FG_sigma, FG_color, FG_uncertainties = forward_through_MLP(self.FG_sigma_net, self.FG_color_net, \
                                                                    embedded_xyzt, None, \
                                                                    self.num_layers, self.num_layers_color)
            FG_color = F.sigmoid(FG_color)

            ### we don't have actor, but doing this to make the code consistent
            ACTOR_sigma, ACTOR_color = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color) 
            ACTOR_uncertainties = torch.zeros_like(FG_uncertainties)
            ##################            
        else:
            FG_sigma, FG_color, FG_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), None
            ACTOR_sigma, ACTOR_color, ACTOR_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), None

        # Principled color mixing
        sigma = BG_sigma + FG_sigma + ACTOR_sigma + 1e-9
        color = (BG_sigma/sigma)[:,None] * BG_color + (FG_sigma/sigma)[:,None] * FG_color + (ACTOR_sigma/sigma)[:,None] * ACTOR_color

        if self.use_uncertainties:
            return torch.cat([BG_color, FG_color, ACTOR_color, # :3, 3:6, 6:9
                                BG_sigma.unsqueeze(dim=-1), FG_sigma.unsqueeze(dim=-1), ACTOR_sigma.unsqueeze(dim=-1), # 9, 10, 11
                                FG_uncertainties.unsqueeze(dim=-1), ACTOR_uncertainties.unsqueeze(dim=-1), # 12, 13
                                embedded_xyzt], -1) # 14:
        else:
            return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)


class BGFG_XYZT_Bottleneck(nn.Module):
    '''
    XYZT grid for foreground model 
    Foreground model also uses some information (encoding) from the BG model, which helps in triangulation
    '''
    def __init__(self,
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=2,
                 num_layers_FG=4,
                 input_ch=4, input_cam_ch=4,
                 input_ch_views=3, input_ch_views_cam=3,
                 use_uncertainties=False,
                 static_grid=None, xyzt_grid=None,
                 coarse=True):
        super(BGFG_XYZT_Bottleneck, self).__init__()

        self.input_ch, self.input_cam_ch = input_ch, input_cam_ch # it's raw xyzt, so input_ch=4
        self.input_ch_views, self.input_ch_views_cam = input_ch_views, input_ch_views_cam # has embedded views

        self.num_layers, self.num_layers_color, self.hidden_dim = num_layers, num_layers_color, hidden_dim
        
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_FG = num_layers_FG
        if coarse:
            self.use_uncertainties = False
        else:
            self.use_uncertainties = use_uncertainties
        self.coarse = coarse # Is this a coarse model?
        
        self.static_grid = static_grid
        if not coarse: # if this is a "fine" model, use dynamic components
            self.xyzt_grid = xyzt_grid # separate high freq grid for FG
            
        ### Static components
        self.xyz_encoder = nn.Sequential(
            nn.Linear(static_grid.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.BG_sigma_net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        self.xyz_final_encoder = nn.Linear(hidden_dim, geo_feat_dim)
        self.BG_color_net = nn.Sequential(
            nn.Linear(geo_feat_dim + input_ch_views, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

        ### Dynamic components
        if not coarse:  
            self.FG_encoder = nn.Sequential(
                nn.Linear(xyzt_grid.out_dim + geo_feat_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, hidden_dim//2),
                nn.ReLU()
            )
            self.FG_sigma_net = nn.Sequential(nn.Linear(hidden_dim//2, 1), nn.Softplus())
            self.FG_color_net = nn.Sequential(nn.Linear(hidden_dim//2, 3), nn.Sigmoid())
            self.FG_uncertainty_net = nn.Sequential(nn.Linear(hidden_dim//2, 1), nn.Softplus())

    def forward(self, x):
        input_pts, input_pts_cam, input_views, input_views_cam = torch.split(x, [self.input_ch, self.input_cam_ch, self.input_ch_views, self.input_ch_views_cam], dim=-1)
        embedded_xyz, keep_mask = self.static_grid(input_pts[...,:3])
        if not self.coarse:
            embedded_xyzt, keep_mask_FG = self.xyzt_grid(input_pts)
            
        ### Static components
        xyz_encoding = self.xyz_encoder(embedded_xyz)
        BG_sigma = self.BG_sigma_net(xyz_encoding)

        xyz_encoding_final = self.xyz_final_encoder(xyz_encoding) # size: (B, geo_feat_dim)

        BG_color = self.BG_color_net(torch.cat([xyz_encoding_final, input_views], dim=-1))
        BG_sigma, BG_color = BG_sigma*keep_mask[:,None], BG_color*keep_mask[:,None]

        ### Dynamic components
        if not self.coarse:
            FG_encoding = self.FG_encoder(torch.cat([embedded_xyzt, xyz_encoding_final], dim=-1)) # size: (B, hidden_dim//2)
            FG_sigma = self.FG_sigma_net(FG_encoding)
            FG_color = self.FG_color_net(FG_encoding)
            FG_uncertainties = self.FG_uncertainty_net(FG_encoding)
            FG_sigma, FG_color = FG_sigma*keep_mask_FG[:,None], FG_color*keep_mask_FG[:,None]

            ### we don't have actor, but doing this to make the code consistent
            ACTOR_sigma, ACTOR_color, ACTOR_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), torch.zeros_like(FG_uncertainties)
            ##################
        else:
            FG_sigma, FG_color, FG_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), None
            ACTOR_sigma, ACTOR_color, ACTOR_uncertainties = torch.zeros_like(BG_sigma), torch.zeros_like(BG_color), None

        # Principled color mixing
        sigma = BG_sigma + FG_sigma + ACTOR_sigma + 1e-9
        color = (BG_sigma/sigma) * BG_color + (FG_sigma/sigma) * FG_color + (ACTOR_sigma/sigma) * ACTOR_color

        if self.use_uncertainties:
            return torch.cat([BG_color, FG_color, ACTOR_color, # :3, 3:6, 6:9
                                BG_sigma, FG_sigma, ACTOR_sigma, # 9, 10, 11
                                FG_uncertainties, ACTOR_uncertainties, # 12, 13
                                FG_encoding], -1) # 14:
        else:
            return torch.cat([color, sigma], -1)


# Ray helpers
def get_rays_incameraframe(H, W, K):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    origins = torch.zeros_like(dirs)
    return origins, dirs


def get_rays_incameraframe_np(H, W, K):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    origins = np.zeros_like(dirs)
    return origins, dirs


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
