from ast import If
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from hash_encoding import HashEmbedder, SHEncoder, XYZplusT_HashEmbedder

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2mse_with_uncertainty = lambda x, y, u : torch.mean(((x - y) ** 2)/(2*(u+1e-9)**2) + torch.log((u+1e-9)**2))
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
relu_act = nn.Softplus()

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
    geo_feat_dim
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
            in_dim = input_ch_views + geo_feat_dim
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
        h = torch.cat([embedded_views, geo_feat], dim=-1)
        for l in range(num_layers_color):
            h = color_net[l](h)
            if l != num_layers_color - 1:
                h = F.relu(h, inplace=True)
            
        color = h
        return sigma, color, uncertainties


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


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
                 only_background=False, # we will rarely use this option
                 args=None
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
        self.only_background = only_background

        xyz_bounding_box = args.bounding_box[0][:3], args.bounding_box[1][:3]
        self.BG_embedder = HashEmbedder(bounding_box=xyz_bounding_box, \
                            log2_hashmap_size=args.log2_hashmap_size, \
                            finest_resolution=args.finest_res)
        self.FG_embedder = XYZplusT_HashEmbedder(bounding_box=args.bounding_box, \
                            log2_hashmap_size=args.log2_hashmap_size, \
                            finest_resolution=args.finest_res)
        self.ACTOR_embedder = XYZplusT_HashEmbedder(bounding_box=args.bounding_box_incameraframe, \
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

        ### Actor Network
        self.ACTOR_sigma_net, self.ACTOR_color_net = create_sigma_and_color_MLP(num_layers, num_layers_color,
                                                                    hidden_dim, hidden_dim_color,
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
                                                    self.num_layers, self.num_layers_color)
        ACTOR_sigma, ACTOR_color, ACTOR_uncertainties = forward_through_MLP(self.ACTOR_sigma_net, self.ACTOR_color_net, \
                                                    ACTOR_embedded_xyzt, input_views_cam, \
                                                    self.num_layers, self.num_layers_color)

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
