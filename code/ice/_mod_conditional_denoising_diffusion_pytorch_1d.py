#The code in this file was modified from https://github.com/lucidrains/denoising-diffusion-pytorch

import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import time
import gc
import numpy as np
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm
from denoising_diffusion_pytorch.version import __version__
from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime
import pickle


# added
def get_shuffle_indices(Nt, NE, Ni0):
    s = np.arange(Nt*NE*Ni0).reshape(NE*Ni0, Nt)
    s = np.take_along_axis(s, np.random.randn(*s.shape).argsort(axis=1), axis=1)
    s = s.reshape(-1)
    return s

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# data
#modified
class cDataset1D(Dataset):
    def __init__(self, tensor: Tensor):
        super().__init__()
        self.xtensor = tensor[0].clone()
        self.ctensor = tensor[1].clone()

    def __len__(self):
        return len(self.xtensor)

    def __getitem__(self, idx):
        return (self.xtensor[idx].clone(), self.ctensor[idx].clone())

class Dataset1D(Dataset):
    def __init__(self, tensor: Tensor):
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].clone()

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        
        

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        
        return h + self.res_conv(x)



class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)
        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)



class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v) #b .. batch, h ... heads, i ... queries, d ... dim key/query
        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)



#%%


# model

class cUnet1D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        pars = -1
    ):
        super().__init__()

        self.pars = pars
        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        #modified
        self.class_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)                
        )
        
        
        if not 'use_attn' in pars:
            pars['use_attn'] = True
        if not 'fixed_angle' in self.pars:
            self.pars['fixed_angle'] = False
        if self.pars['fixed_angle'] == False:
            self.class2_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)                
        )
        
        tc_input_dim = 2*time_dim if self.pars['fixed_angle'] == True else 3*time_dim
        self.tc_mlp = nn.Sequential(
            nn.Linear(tc_input_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.tc_mlp2 = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, 1)
        )
        
        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                *[Residual(PreNorm(dim_in, LinearAttention(dim_in))) if self.pars['use_attn'] else None],
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),                
                *[Residual(PreNorm(dim_out, LinearAttention(dim_out))) if self.pars['use_attn'] else None],
                #Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)
        
        
        

    def forward(self, x, tc, x_self_cond = None): #forward
    
        t0, classes = tc[0], tc[1]
        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(t0)
        #these additional embeddings are how we modified the code to also condition on E and theta
        if self.Nc > 1 or self.Nc == -1:   
            c1 = self.class_mlp(classes[:,0]) #E
            if self.pars['fixed_angle'] == False:
                c2 = self.class2_mlp(classes[:,1]) #theta
                t = self.tc_mlp(torch.cat((t,c1,c2),-1)) #combine t, E, theta
            else:
                t = self.tc_mlp(torch.cat((t,c1),-1)) #combine t, E, theta
        
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t) #ResNet block
            h.append(x)

            x = block2(x, t) #ResNet block
            if self.pars['use_attn']:
                x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t) #ResNet block
        if self.pars['use_attn']:
            x = self.mid_attn(x) #Attention layer
        x = self.mid_block2(x, t) #ResNet block

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t) #ResNet block

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t) #ResNet block
            if self.pars['use_attn']:
                x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class cGaussianDiffusion1D(nn.Module): #modified
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        Nc = 1,
        batch_signals = False,
        batch_with_noise = False,
        sub_batch_size = 2
    ):
        super().__init__()
        self.Nc = Nc
        self.batch_signals = batch_signals
        self.batch_with_noise = batch_with_noise
        self.sub_batch_size = sub_batch_size
        
        self.model = model
        self.model.Nc = Nc
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, tc, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False): #modified
        model_output = self.model(x, tc, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        t = tc[0]

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise) #modify?
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start) #modify?

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start) #modify?

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v) #modify?
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start) #modify?

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, tc, x_self_cond = None, clip_denoised = True): #modified
        preds = self.model_predictions(x, tc, x_self_cond)
        x_start = preds.pred_x_start
        
        
        t = tc[0]


        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, tc, x_self_cond = None, clip_denoised = True): #modified
        t = tc[0]
        classes = tc[1]
        
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        
        tc = (batched_times, classes)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, tc = tc, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, classes = False, img = False): #modified
        batch, device = shape[0], self.betas.device

        if type(img) == bool: #choose random seed if not preselected
            img = torch.randn(shape, device=device)
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            tc = (t, classes)
            img, x_start = self.p_sample(img, tc, self_cond)

        img = self.unnormalize(img)
        return img

    #@torch.no_grad()
    def ddim_sample(self, shape, classes, clip_denoised = True, img = False): #modified
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if type(img) == bool:
            img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            tc_cond = (time_cond, classes)
            
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, tc_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            if self.batch_signals == False or img.shape[0]%self.sub_batch_size != 0 or self.batch_with_noise == True:
                noise = torch.randn_like(img)
            else:
                noise = torch.randn_like(img[::self.sub_batch_size,:,:])
                noise = torch.kron(torch.ones(self.sub_batch_size, device=img.device).unsqueeze(1).unsqueeze(1), noise)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = self.unnormalize(img)
        return img

    #@torch.no_grad()
    def sample(self, batch_size = 16, classes = False, img = False):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        
        device = self.betas.device
        if type(classes) == bool:
            if self.Nc == -1:
                classes = torch.tensor(np.random.uniform(0,1,(batch_size,2,)), device = device).float()
            else:
                classes = torch.randint(0, self.Nc, (batch_size,), device = device)
        else:
            classes = classes.to(device)
            
            
        return sample_fn((batch_size, channels, seq_length), classes = classes, img = img), classes

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        if self.batch_signals == False:
            noise = default(noise, lambda: torch.randn_like(x_start))
        else:
            nshape = list(x_start.shape)
            nshape[0] = int(nshape[0]/self.sub_batch_size)
            noise = default(noise[::self.sub_batch_size], lambda: torch.randn(nshape)).to(x_start.device)
            noise = torch.kron(noise, torch.ones(self.sub_batch_size, device=x_start.device).unsqueeze(1).unsqueeze(1))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, tc, noise = None): #modified
        b, c, n = x_start.shape
        t, classes = tc[0], tc[1]
        
        if self.batch_signals == False:
            noise = default(noise, lambda: torch.randn_like(x_start))
        else:
            nshape = list(x_start.shape)
            nshape[0] = int(nshape[0]/self.sub_batch_size)
            noise = default(noise, lambda: torch.randn(nshape)).to(x_start.device)
            noise = torch.kron(noise, torch.ones(self.sub_batch_size, device=x_start.device).unsqueeze(1).unsqueeze(1))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, tc, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, samples, *args, **kwargs): #modified
        img = samples[0]
        classes = samples[1]
        
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        #print('test!!')
        if self.batch_signals == False:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        else:
            t = torch.randint(0, self.num_timesteps, (int(b/self.sub_batch_size),), device=device).long()
            t = t.unsqueeze(1).repeat(1,self.sub_batch_size).reshape(-1)
            #print(t.shape)
        tc = (t, classes)
        
        img = self.normalize(img)
        return self.p_losses(img, tc, *args, **kwargs)

# trainer class

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: cGaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 2500,
        num_samples = 100,#36,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        pars = {},
        writer = -1
    ):
        super().__init__()

        # accelerator
        if pars['device'] == 'cpu':
            self.accelerator = Accelerator(
                split_batches = split_batches,
                mixed_precision = mixed_precision_type if amp else 'no',
                cpu = True
            )
        else:
            self.accelerator = Accelerator(
                split_batches = split_batches,
                mixed_precision = mixed_precision_type if amp else 'no'
            )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps
        
        
        self.pars = pars
        self.writer = None

        # dataset and dataloader
        #modified pin_memory False instead of True
        
        if type(dataset) is not int:
            if pars['batch_signals'] == True:
                self.data = dataset
            else:
                if type(dataset[0]) == torch.Tensor:
                    xn = dataset[0].clone().detach().unsqueeze(1).float()
                    yn = dataset[1].clone().detach().float()
                else:
                    xn = torch.tensor(dataset[0]).unsqueeze(1).float()
                    yn = torch.tensor(dataset[1]).float()
                dataset = cDataset1D((xn, yn))    
                
                dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
                dl = self.accelerator.prepare(dl)
                self.dl = cycle(dl)
            
            self.xn = dataset[:][0].squeeze(1)
            self.yn = dataset[:][1].squeeze(-1)
            

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0
        self.step_dl = 0
        self.pars['steps_dl'] = self.pars['Ns']/(self.batch_size)

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        
        
        
    def reshuffle_dataset(self):
        if self.pars['batch_signals'] == True:
            print('reshuffling dataset!')
            print('')
            Nt = self.pars['Nt']#41
            NE = self.pars['NE']#81
            Ni0 = self.pars['Ni0']
            bs0 = self.pars['sub_batch_size'] #4
            s = get_shuffle_indices(Nt, NE, Ni0)
            buf0 = self.data[0][s]
            buf1 = self.data[1][s]
            buf0 = torch.tensor(buf0.reshape(-1,bs0,896)).float()
            buf1 = torch.tensor(buf1.reshape((-1,bs0,2))).float()
            buf = cDataset1D((buf0, buf1))
        else:
            buf = self.data
        
        
        dl = DataLoader(buf, batch_size = self.batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        dl = self.accelerator.prepare(dl)
        
        if hasattr(self, 'dl') and self.pars['batch_signals'] == True:
            #print('freeing memory!')
            del self.dl_ref.dataset.xtensor, self.dl_ref.dataset.ctensor
            #del self.dl, self.dl_ref, buf, buf0, buf1, s
            gc.collect() 
        
        self.dl_ref = dl
        self.dl = cycle(dl)
        

    @property
    def device(self):
        return self.accelerator.device
    
    def init_writer(self):
        from .utils import save_parameters
        
        from .utils import initialize_writer, init_par
        init_par(self.pars, 'comment', '')
        self.writer = initialize_writer(self.pars['run_dir'], comment0 = self.pars['ds_opt'], comment = self.pars['comment'])
        self.pars['log_dir'] = self.writer.log_dir + '/'        
        
        save_parameters(self.pars, self.pars['log_dir'])    

    def save(self, milestone, folder=''):
        if not self.accelerator.is_local_main_process:
            return
        
        self.writer.close()
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'writer': self.writer,
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__            
        }
        
        if folder != '':
            #torch.save(data, str(folder / f'model-{milestone}.pt'))
            torch.save(data, folder + "final.pt")
        else:
            torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, folder=''):
        accelerator = self.accelerator
        device = accelerator.device   
        
        if type(milestone) == int:
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        else:
            data = torch.load(milestone, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        
        self.writer = data['writer']
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        from .plots import plot_samples, plot_interpolation
        from .utils_load import load_standard_ac
        from .utils_vary_theta import get_vary_E_pars, generate_vary_theta_samples, get_vary_epsilon_pars, get_i_closest_match
        
        accelerator = self.accelerator
        device = accelerator.device
        
        if self.writer == None:
            self.init_writer()

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    #data = next(self.dl).to(device) #modified
                    if (self.step_dl%(self.pars['steps_dl']) == 0 and self.pars['batch_signals'] == True):
                        print('step:', self.step_dl)
                        self.reshuffle_dataset()
                    
                    xdata, cdata = next(self.dl)
                    self.step_dl += 1
                    if self.pars['batch_signals'] == True:
                        xdata = xdata.reshape(-1,896).unsqueeze(1)
                        cdata = cdata.reshape(-1,2)
                    
                    data = (xdata.to(device), cdata.to(device))

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        
                        ##changed
                        loss0 = loss.item()/self.gradient_accumulate_every                            
                        loss = loss / self.gradient_accumulate_every
                        
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()
                
                if self.step%50 == 0:
                    self.writer.add_scalar("loss", loss0, self.step)
                    self.writer.add_scalar("total_loss", total_loss, self.step)
                                   
                
                    

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n)[0], batches))

                        all_samples = torch.cat(all_samples_list, dim = 0)
                        
                        if self.step % (4*self.save_and_sample_every) == 0:
                            torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.png'))
                            self.save(milestone)
                        
                        print(all_samples.shape)
                        fake_samples = plot_samples(all_samples)
                        self.writer.add_figure("fake_samples", fake_samples, self.step)
                        self.writer.flush()
                        
                        from ice.utils_vary_theta import get_closest_match
                        _, deltas = get_i_closest_match(all_samples.squeeze(1).cpu(), self.xn)
                        self.writer.add_scalar("distance to closest real sample", deltas.mean(), self.step)
                    
                    #plot interpolations as a measure of overfit
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        with torch.no_grad():
                            bs = 5
                            Ni = 9
                            self.model.device = self.device
                            ac = load_standard_ac(self.model.seq_length, device)
                            if self.step == self.save_and_sample_every:
                                epsilon0 = torch.randn(bs, self.model.seq_length).repeat(Ni,1).unsqueeze(1).to(self.model.device)
                                Et_vals1, epsilon1 = get_vary_epsilon_pars(self.model.seq_length, E=0.3, theta = self.pars['theta_ref'], bs=bs, Ni=Ni, device=device)
                                Et_vals2, epsilon2 = get_vary_epsilon_pars(self.model.seq_length, E=0.5, theta = self.pars['theta_ref'], bs=bs, Ni=Ni, device=device)
                                Et_vals3, epsilon3 = get_vary_epsilon_pars(self.model.seq_length, E=0.9, theta = self.pars['theta_ref'], bs=bs, Ni=Ni, device=device)
                            
                            Et_vals = get_vary_E_pars(bs=bs, E0=0.9, theta=self.pars['theta_ref'], dE=0.02, Ni=Ni)
                            samples = generate_vary_theta_samples(self.model, ac, Et_vals.to(device), Ni, epsilon0).cpu()
                            legend = ['E=%.3f'%(E) for E in Et_vals[::bs,0]]
                            fig = plot_interpolation(samples, bs, legend)
                            self.writer.add_figure("E 0.9 dE 0.02 interpolation", fig, self.step)
                            self.writer.flush()
                            
                            Et_vals = get_vary_E_pars(bs=bs, E0=0.9, theta=self.pars['theta_ref'], dE=0.1, Ni=Ni)
                            samples = generate_vary_theta_samples(self.model, ac, Et_vals.to(device), Ni, epsilon0).cpu()
                            legend = ['E=%.3f'%(E) for E in Et_vals[::bs,0]]
                            fig = plot_interpolation(samples, bs, legend)
                            self.writer.add_figure("E 0.9 dE = 0.1 interpolation", fig, self.step)
                            self.writer.flush()
                            
                            Et_vals = get_vary_E_pars(bs=bs, E0=0.5, theta=self.pars['theta_ref'], dE=0.02, Ni=Ni)
                            samples = generate_vary_theta_samples(self.model, ac, Et_vals.to(device), Ni, epsilon0).cpu()
                            legend = ['E=%.3f'%(E) for E in Et_vals[::bs,0]]
                            fig = plot_interpolation(samples, bs, legend)
                            self.writer.add_figure("E 0.5 dE 0.02 interpolation", fig, self.step)
                            self.writer.flush()
                            
                            Et_vals = get_vary_E_pars(bs=bs, E0=0.5, theta=self.pars['theta_ref'], dE=0.1, Ni=Ni)
                            samples = generate_vary_theta_samples(self.model, ac, Et_vals.to(device), Ni, epsilon0).cpu()
                            legend = ['E=%.3f'%(E) for E in Et_vals[::bs,0]]
                            fig = plot_interpolation(samples, bs, legend)
                            self.writer.add_figure("E 0.5 dE 0.1 interpolation", fig, self.step)
                            self.writer.flush()
                            
                            
                            #Et_vals, epsilon = get_vary_epsilon_pars(self.model.seq_length, E=0.3, theta = 0.9, bs=bs, Ni=Ni, device=device)
                            samples = generate_vary_theta_samples(self.model, ac, Et_vals1.to(device), Ni, epsilon1).cpu()
                            legend = [r'$\epsilon$=%.2f'%(e/(Ni-1)) for e in range(Ni)]
                            fig = plot_interpolation(samples, bs, legend)
                            self.writer.add_figure("eps interpolation E=0.3 t=0.9", fig, self.step)
                            self.writer.flush()
                            
                            #Et_vals, epsilon = get_vary_epsilon_pars(self.model.seq_length, E=0.5, theta = 0.9, bs=bs, Ni=Ni, device=device)
                            samples = generate_vary_theta_samples(self.model, ac, Et_vals2.to(device), Ni, epsilon2).cpu()
                            legend = [r'$\epsilon$=%.2f'%(e/(Ni-1)) for e in range(Ni)]
                            fig = plot_interpolation(samples, bs, legend)
                            self.writer.add_figure("eps interpolation E=0.5 t=0.9", fig, self.step)
                            self.writer.flush()
                            
                            #Et_vals, epsilon = get_vary_epsilon_pars(self.model.seq_length, E=0.9, theta = 0.9, bs=bs, Ni=Ni, device=device)
                            samples = generate_vary_theta_samples(self.model, ac, Et_vals3.to(device), Ni, epsilon3).cpu()
                            legend = [r'$\epsilon$=%.2f'%(e/(Ni-1)) for e in range(Ni)]
                            fig = plot_interpolation(samples, bs, legend)
                            self.writer.add_figure("eps interpolation E=0.9 t=0.9", fig, self.step)
                            self.writer.flush()

                pbar.update(1)
            
            if self.train_num_steps > 99999:
                self.save('final', folder=self.writer.log_dir + '/')

        accelerator.print('training complete')
        
