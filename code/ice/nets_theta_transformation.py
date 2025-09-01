# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from functools import partial

from ._mod_conditional_denoising_diffusion_pytorch_1d import *
from .ddpm_theta_rel_modules import *
from .utils import *
from .utils_tnet import transform_signal

class Block2(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1, padding_mode='reflect')
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
    
class ResnetBlock2(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8, fdropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block2(dim, dim_out, groups = groups)
        self.block2 = Block2(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        
        #modified
        self.fdropout = fdropout
        self.drop = nn.Dropout(p=self.fdropout)

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        
        #modified
        res = h + self.res_conv(x)
        if self.fdropout > 0:
            return self.drop(res)
        else:
            return res
        #return h + self.res_conv(x)

class CondBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block2(dim, dim_out, groups = groups)
        self.block2 = Block2(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h
        #return h + self.res_conv(x)

class BlockCNN(nn.Module):
    def __init__(self, c0_dim=32, layer_opt='standard'):
        super(BlockCNN, self).__init__()
        self.c0_dim = c0_dim
        
        dim = 16
        time_dim = dim * 4
        
        
        self.blocks = nn.ModuleList([])
        if layer_opt=='standard':
            self.channels = [1, self.c0_dim, 2*self.c0_dim, 4*self.c0_dim, 8*self.c0_dim, 1]
        elif layer_opt=='double':
            self.channels = [1, self.c0_dim, self.c0_dim, 2*self.c0_dim, 2*self.c0_dim, 4*self.c0_dim, 4*self.c0_dim, 8*self.c0_dim, 8*self.c0_dim, 1]
        for j in range(1, len(self.channels)-1):
            print('jj:',j)
            self.blocks.append(CondBlock(self.channels[j-1], self.channels[j], time_emb_dim = time_dim))
        self.convf = nn.Conv1d(in_channels=self.channels[-2], out_channels=self.channels[-1], kernel_size=3, padding=1, padding_mode='reflect')
        
        ######
        ######
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000
        
        sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
        fourier_dim = dim
        
        
        #modified
        self.class_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)                
        )
        
        self.class2_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)                
        )
        
        self.tc_mlp = nn.Sequential(
            nn.Linear(2*time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
    def forward(self, x, E, theta):
        
        theta_emb = self.class_mlp(theta)
        E_emb = self.class2_mlp(E)
        Et_emb = self.tc_mlp(torch.cat((theta_emb,E_emb),-1))
        
        for j in range(len(self.blocks)):
            x = F.relu(self.blocks[j](x, Et_emb))       
        x = F.tanh(self.convf(x))
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.dim = 64
        
        self.clayers = nn.ModuleList([])
        self.channels = [1, self.dim, 2*self.dim, 4*self.dim, 8*self.dim, 1]
        self.channels = [1, self.dim, self.dim, 2*self.dim, 2*self.dim, 4*self.dim, 4*self.dim, 8*self.dim, 8*self.dim, 1]
        for j in range(1, len(self.channels)):
            self.clayers.append(nn.Conv1d(in_channels=self.channels[j-1], out_channels=self.channels[j], kernel_size=3, padding=1, padding_mode='reflect'))
              
        
    def forward(self, x, E=-1, theta=-1):      
        
        for j in range(len(self.clayers)-1):
            x = F.relu(self.clayers[j](x))        
        x = F.tanh(self.clayers[-1](x))
        return x
    
#The code in this class was modified from https://github.com/lucidrains/denoising-diffusion-pytorch
class cUnet1D_theta(nn.Module):
    def __init__(
        self,
        pars,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 1,#3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        fdropout = 0
        #pars = -1
    ):
        super().__init__()

        if not 'use_attn' in pars:
            pars['use_attn'] = True
        if not 'use_block2' in pars:
            pars['use_block2'] = True
        self.pars = pars
        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3, padding_mode = 'reflect')

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if dim < resnet_block_groups:
            resnet_block_groups = dim 
        block_klass = partial(ResnetBlock2, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4
        if self.pars['extract_features'] == True and 'feature_dim' in self.pars:
            feature_dim = self.pars['feature_dim']
        else:
            feature_dim = 0

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.class_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)                
        )
        
        self.class2_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)                
        )
        
        self.tc_mlp = nn.Sequential(
            nn.Linear(2*time_dim + feature_dim, time_dim + feature_dim),
            nn.GELU(),
            nn.Linear(time_dim + feature_dim, time_dim)
        )
        
        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, fdropout = fdropout),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, fdropout = fdropout),
                *[Residual(PreNorm(dim_in, LinearAttention(dim_in))) if pars['use_attn'] else None],
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        time_dim_up = time_dim

        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim_up)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim_up, fdropout = fdropout),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim_up, fdropout = fdropout),
                *[Residual(PreNorm(dim_out, LinearAttention(dim_out))) if pars['use_attn'] else None],
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim_up)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)
        
        
        
        ## optional feature extractor
        self.feature_extractor = nn.Sequential(nn.Conv1d(1,16,4,4),
                                               nn.ReLU(),
                                               nn.Conv1d(16,32,4,4),
                                               nn.ReLU(),
                                               nn.Conv1d(32,64,4,4),
                                               nn.ReLU(),
                                               nn.Flatten(),
                                               nn.Linear(8*64,100),
                                               nn.ReLU(),
                                               nn.Linear(100,50),
                                               nn.ReLU(),
                                               nn.Linear(50,feature_dim),
                                               )
        
        

    def forward(self, ref, E, theta, x_self_cond = None): #forward 
        xref, yref = ref
        
           
        #x, features = x
        if self.pars['extract_features'] == True:
            features = self.feature_extractor(xref[:,512:].unsqueeze(1))
        else:
            features = None
            
        if self.pars['transform_signal'] == True:
            xbase = transform_signal(xref, yref[:,1], theta)
        else:
            xbase = xref
            
        x = xbase.unsqueeze(1)
        #return x.squeeze(), xbase
    
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        

        x = self.init_conv(x)
        r = x.clone()

        theta_emb = self.class_mlp(theta)
        E_emb = self.class2_mlp(E)
        if self.pars['extract_features'] ==True :
            t = self.tc_mlp(torch.cat((theta_emb,E_emb, features), -1))
        else:
            t = self.tc_mlp(torch.cat((theta_emb,E_emb), -1))
        
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t) #ResNet block
            h.append(x)

            if self.pars['use_block2']:
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
            
            if self.pars['use_block2']:
                x = torch.cat((x, h.pop()), dim = 1)
                x = block2(x, t) #ResNet block
                if self.pars['use_attn']:
                    x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x).squeeze(), xbase
