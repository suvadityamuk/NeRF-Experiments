import wandb
from utils import (
    get_rays,
    render_flat_rays,
    pose_spherical,
    grad_norm,
    PSNRLoss,
    set_seed,
    setup_wandb,
    cleanup_wandb
)
import imageio
import torch
from tqdm.auto import tqdm
import numpy as np
from nerf import NeRF
import os
from dataset import NerfDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from configs import exp5_config

def train(model, train_loader, valid_loader, optimizer, device, config, epochs=100, loss_fn=nn.MSELoss(), lambda_l2=0.1, rank=0):
    psnr_loss = PSNRLoss()

    def log_video(model, epoch_val, focal):
        rgb_frames = []
        batch_flat = []
        batch_t = []

        for idx, theta in tqdm(enumerate(np.linspace(0.0, 360.0, 120, endpoint=False)), total=120):
            c2w = pose_spherical(theta, -30.0, 4.0)
            ray_origins, ray_dirs = get_rays(config["H"], config["W"], focal, c2w)
            rays_flat, t_vals = render_flat_rays(
                ray_origins, ray_dirs, near_bound=2.0, far_bound=6.0, num_samples=config["NUM_SAMPLES"], rand=False
            )
            rays_flat = rays_flat.float().to(device)
            t_vals = t_vals.float().to(device)

            if idx % config["BATCH_SIZE"] == 0 and idx > 0:
                batched_flat = torch.stack(batch_flat, axis=0)
                batch_flat = [rays_flat]
                batched_t = torch.stack(batch_t, axis=0)
                batch_t = [t_vals]

                rgb, _ = model.module.render_rgb_depth(
                    batched_flat, batched_t, rand=False, train=False
                )

                temp_rgb = [torch.clip(255 * img, 0.0, 255.0).int() for img in rgb]
                rgb_frames = rgb_frames + temp_rgb
            else:
                batch_flat.append(rays_flat)
                batch_t.append(t_vals)

        rgb_frames = [frame.to('cpu').numpy().astype(np.uint8) for frame in rgb_frames]
        rgb_video = f"rgb_video_{epoch_val}.mp4"
        imageio.mimwrite(rgb_video, rgb_frames, fps=30, quality=7, macro_block_size=None)
        wandb.log({"rgb_video": wandb.Video(rgb_video, format="mp4")})

    if rank == 0:
        wandb.watch(model, log='all')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_psnr = 0.0

        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), disable=rank != 0)

        for idx, (images, rays, focal) in progress_bar:
            main_focal = focal[0]
            rays_flat, t = rays
            images, rays_flat, t = images.to(device), rays_flat.to(device), t.to(device)

            optimizer.zero_grad()

            rgb, depth_map = model.module.render_rgb_depth(rays_flat, t)
            loss = loss_fn(images, rgb)

            loss.backward()
            gradnorm = grad_norm(model)
            optimizer.step()

            psnr = psnr_loss(rgb, images)
            total_loss += loss.item()
            total_psnr += psnr.item()

            if rank == 0 and idx % 4 == 0:
                wandb.log({
                    "train/loss/mse": loss.item(),
                    "train/loss/psnr": psnr.item(),
                    "train/grad_norm": gradnorm,
                })
            elif rank == 0 and idx == 1 and epoch % 5 == 0:
                pred_img = (rgb[0].detach().cpu().numpy() * 255).astype(np.uint8)
                target_img = (images[0].detach().cpu().numpy() * 255).astype(np.uint8)
                depth_map_img = (depth_map[0].detach().cpu().numpy() * 255).astype(np.uint8)

                wandb.log({
                    "train/sample_pred_image": wandb.Image(pred_img),
                    "train/sample_target_image": wandb.Image(target_img),
                    "train/sample_depth_map": wandb.Image(depth_map_img),
                })

        avg_loss = total_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)

        if rank == 0:
            print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, Avg PSNR = {avg_psnr:.2f}")

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():

                total_valid_loss = 0.0
                total_valid_psnr = 0.0

                for idx, (images, rays, focal) in enumerate(valid_loader):
                    rays_flat, t = rays
                    images, rays_flat, t = images.to(device), rays_flat.to(device), t.to(device)
                    rgb, depth_map = model.module.render_rgb_depth(rays_flat, t)

                    valid_loss = loss_fn(images, rgb)
                    valid_psnr = psnr_loss(rgb, images)
                    total_valid_loss += loss.item()
                    total_valid_psnr += psnr.item()

                    if rank == 0:
                        pred_img = (rgb[0].detach().cpu().numpy() * 255).astype(np.uint8)
                        target_img = (images[0].detach().cpu().numpy() * 255).astype(np.uint8)
                        depth_map_img = (depth_map[0].detach().cpu().numpy() * 255).astype(np.uint8)

                        wandb.log({
                            "valid/loss/mse": valid_loss.item(),
                            "valid/loss/psnr": valid_psnr.item(),
                            "valid/sample_pred_image": wandb.Image(pred_img),
                            "valid/sample_target_image": wandb.Image(target_img),
                            "valid/sample_depth_map": wandb.Image(depth_map_img),
                        })

                        valid_avg_loss = total_valid_loss / len(valid_loader)
                        valid_avg_psnr = total_valid_psnr / len(valid_loader)
                        print(f"Validation on Epoch {epoch+1}:\nValid Avg Loss = {valid_avg_loss:.4f}, Valid Avg PSNR = {valid_avg_psnr:.2f}")

                        # Save checkpoint every 5 epochs (rank 0 only)
                        checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
                        torch.save(model.state_dict(), checkpoint_path)
                        wandb.save(checkpoint_path)

                    break

                if rank == 0:
                    log_video(model, epoch, main_focal)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def execute(rank, world_size, config):
    setup(rank, world_size)
    set_seed(42)
    setup_wandb(rank, config=config, exp_name="nerf-t4x2-ddp-base")

    train_dataset = NerfDataset(
        root_path="train.npz",
        transform=None,
        target_transform=None
    )

    valid_dataset = NerfDataset(
        root_path="valid.npz",
        transform=None,
        target_transform=None
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], sampler=train_sampler, num_workers=os.cpu_count() // world_size, drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config["BATCH_SIZE"], sampler=val_sampler, num_workers=os.cpu_count() // world_size, drop_last=True, pin_memory=True)

    model = NeRF(device=rank, emb_dim=config["EMB_DIM"], config=config)
    model = torch.compile(model, fullgraph=True)
    model = DDP(model.to(rank), device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"], amsgrad=config["AMSGRAD"], fused=config["FUSED_OPTIM"])

    train(model, train_loader, val_loader, optimizer, rank=rank, device=rank, epochs=config["EPOCHS"], config=config)

    cleanup_wandb()
    cleanup()

def run_ddp_training():
    world_size = torch.cuda.device_count()
    assert world_size >= 2, "Requires at least 2 GPUs to run"

    mp.spawn(execute, args=(world_size, exp5_config), nprocs=world_size, join=True)

if __name__ == "__main__":
    run_ddp_training()