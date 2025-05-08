import torch
import numpy as np
import torch.nn as nn
import random
import os
import wandb
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_wandb(rank, config, exp_name):
    if rank == 0:
        os.environ["WANDB_API_KEY"] = "8c7b9f8fcc6e76dc62c9ce2fe755802ee90d6de4"
        wandb.init(
            project="nerf",
            entity="suvadityamuk",
            save_code=True,
            name=exp_name,
            config=config,
        )


def cleanup_wandb():
    if wandb.run:
        wandb.finish()


def get_translation(t):
    return torch.from_numpy(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, t],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    )


def get_rotation_phi(phi):
    return torch.from_numpy(
        np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1],
            ]
        )
    )


def get_rotation_theta(theta):
    return torch.from_numpy(
        np.array(
            [
                [np.cos(theta), 0, -np.sin(theta), 0],
                [0, 1, 0, 0],
                [np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
    )


def pose_spherical(theta, phi, t):
    c2w = get_translation(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.from_numpy(
            np.array(
                [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float
            )
        )
        @ c2w
    )
    return c2w


def encode_pos(x: torch.Tensor, POS_ENCODE_DIMS=16):
    pos = [x]
    for i in range(POS_ENCODE_DIMS):
        for func in [torch.sin, torch.cos]:
            pos.append(func(2.0**i * x))
    return torch.cat(pos, dim=-1)


def get_rays(height, width, focal_length, pose):
    h, w = torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy")

    normalized_h = (h - height * 0.5) / focal_length
    normalized_w = (w - width * 0.5) / focal_length

    directions = torch.stack(
        [normalized_h, -normalized_w, -torch.ones_like(normalized_h)], dim=-1
    )

    cam_matrix = pose[:3, :3]
    env_params = pose[:3, -1]

    transformed_dirs = directions.unsqueeze(-2)
    camera_dirs = transformed_dirs * cam_matrix
    ray_dirs = torch.sum(camera_dirs, dim=-1)
    ray_origins = torch.broadcast_to(env_params, ray_dirs.shape)

    return (ray_origins, ray_dirs)


def render_flat_rays(
    ray_origins, ray_dirs, near_bound, far_bound, num_samples, rand=False
):
    t = torch.linspace(near_bound, far_bound, num_samples)
    if rand:
        noise_shape = list(ray_origins.shape[:-1])
        noise_shape.append(num_samples)
        t_noise = torch.rand(noise_shape)
        t_noise_normalized = t_noise * ((far_bound - near_bound) / num_samples)
        t = t + t_noise_normalized

    rays = ray_origins.unsqueeze(-2) + (t.unsqueeze(-1) * ray_dirs.unsqueeze(-2))
    rays_flat = rays.reshape(-1, 3)
    rays_flat = encode_pos(rays_flat)

    return (rays_flat, t)


class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val

    def forward(self, predicted_image, target_image):
        mse = torch.mean((predicted_image - target_image) ** 2)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr


def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    return total_norm**0.5


def l2_norm(model):
    total_norm = 0
    for p in model.parameters():
        total_norm += torch.pow(p, 2).sum()

    return total_norm**0.5
