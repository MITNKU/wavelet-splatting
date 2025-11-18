from wipesplat.project_wipes_2d_scale_rot import project_wipes_2d_scale_rot
from wipesplat.rasterize_sum import rasterize_wipes_sum
from pytorch_msssim import SSIM
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from optimizer import Adan
import torch.nn.functional as F

class WIPESImage_RS(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) 
        self.device = kwargs["device"]

        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._scaling = nn.Parameter(torch.rand(self.init_num_points, 2))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self._rotation = nn.Parameter(torch.rand(self.init_num_points, 1))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))

        self.last_size = (self.H, self.W)
        self.background = torch.ones(3, device=self.device)
        self.rotation_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))

        self._normf = nn.Parameter(1e-3 / (self._scaling.detach() + 1e-6), requires_grad=True)

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    @property
    def get_scaling(self):
        return torch.abs(self._scaling+self.bound)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)*2*math.pi
    
    @property
    def get_normf(self):
        return torch.exp(self._normf)
    
    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self._opacity 
    
    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_wipes_2d_scale_rot(self.get_xyz, self.get_scaling, self.get_rotation, self.H, self.W, self.tile_bounds)
        out_img = rasterize_wipes_sum(self.xys, self.get_normf, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)    # [H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def train_iter(self, gt_image, iteration):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
        return loss, psnr

