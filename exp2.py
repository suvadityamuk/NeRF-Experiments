import wandb
from utils import (
    get_rays,
    render_flat_rays,
    pose_spherical,
    grad_norm,
    PSNRLoss,
    set_seed,
    setup_wandb,
    cleanup_wandb,
)
import imageio
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
from nerf import NeRF
import os
from dataset import NerfDataset
from configs import exp2_config


def train(
    model,
    train_loader,
    valid_loader,
    optimizer,
    device,
    config,
    epochs=100,
    loss_fn=nn.MSELoss(),
):
    psnr_loss = PSNRLoss()
    scaler = torch.amp.GradScaler("cuda")

    wandb.watch(model, log="all")

    def log_video(model, epoch_val, focal):
        rgb_frames = []
        batch_flat = []
        batch_t = []

        for idx, theta in tqdm(
            enumerate(np.linspace(0.0, 360.0, 90, endpoint=False)), total=90
        ):
            c2w = pose_spherical(theta, -30.0, 4.0)
            ray_origins, ray_dirs = get_rays(config["H"], config["W"], focal, c2w)
            rays_flat, t_vals = render_flat_rays(
                ray_origins,
                ray_dirs,
                near_bound=2.0,
                far_bound=6.0,
                num_samples=config["NUM_SAMPLES"],
                rand=False,
            )
            rays_flat = rays_flat.float().to(device)
            t_vals = t_vals.float().to(device)

            if idx % config["BATCH_SIZE"] == 0 and idx > 0:
                batched_flat = torch.stack(batch_flat, axis=0)
                batch_flat = [rays_flat]

                batched_t = torch.stack(batch_t, axis=0)
                batch_t = [t_vals]

                rgb, _ = model.render_rgb_depth(
                    batched_flat, batched_t, rand=False, train=False
                )

                temp_rgb = [torch.clip(255 * img, 0.0, 255.0).int() for img in rgb]

                rgb_frames = rgb_frames + temp_rgb

            else:
                batch_flat.append(rays_flat)
                batch_t.append(t_vals)

        rgb_frames = [frame.to("cpu").numpy().astype(np.uint8) for frame in rgb_frames]
        rgb_video = f"rgb_video_{epoch_val}.mp4"
        imageio.mimwrite(
            rgb_video, rgb_frames, fps=30, quality=7, macro_block_size=None
        )
        wandb.log({"rgb_video": wandb.Video(rgb_video, format="mp4")})

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_psnr = 0.0

        total_valid_loss = 0.0
        total_valid_psnr = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for idx, (images, rays, focal) in progress_bar:
            main_focal = focal[0]
            rays_flat, t = rays
            images, rays_flat, t = images.to(device), rays_flat.to(device), t.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                rgb, depth_map = model.render_rgb_depth(rays_flat, t)
                loss = loss_fn(images, rgb)

            scaler.scale(loss).backward()
            gradnorm = grad_norm(model)
            optimizer.step()

            psnr = psnr_loss(rgb, images)
            total_loss += loss.item()
            total_psnr += psnr.item()

            if idx % 4 == 0:
                wandb.log(
                    {
                        "train/loss/mse": loss.item(),
                        "train/loss/psnr": psnr.item(),
                        "train/grad_norm": gradnorm,
                    }
                )
            elif idx == 1 and epoch % 5 == 0:
                pred_img = (rgb[0].detach().cpu().numpy() * 255).astype(np.uint8)
                target_img = (images[0].detach().cpu().numpy() * 255).astype(np.uint8)
                depth_map_img = (depth_map[0].detach().cpu().numpy() * 255).astype(
                    np.uint8
                )

                wandb.log(
                    {
                        "train/sample_pred_image": wandb.Image(pred_img),
                        "train/sample_target_image": wandb.Image(target_img),
                        "train/sample_depth_map": wandb.Image(depth_map_img),
                    }
                )

        avg_loss = total_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)

        print(
            f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, Avg PSNR = {avg_psnr:.2f}"
        )

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                for idx, (images, rays, focal) in enumerate(valid_loader):
                    rays_flat, t = rays
                    images, rays_flat, t = (
                        images.to(device),
                        rays_flat.to(device),
                        t.to(device),
                    )
                    rgb, depth_map = model.render_rgb_depth(rays_flat, t)
                    valid_loss = loss_fn(images, rgb)
                    valid_psnr = psnr_loss(rgb, images)

                    total_valid_loss += loss.item()
                    total_valid_psnr += psnr.item()

                    pred_img = (rgb[0].detach().cpu().numpy() * 255).astype(np.uint8)
                    target_img = (images[0].detach().cpu().numpy() * 255).astype(
                        np.uint8
                    )
                    depth_map_img = (depth_map[0].detach().cpu().numpy() * 255).astype(
                        np.uint8
                    )

                    wandb.log(
                        {
                            "valid/loss/mse": valid_loss.item(),
                            "valid/loss/psnr": valid_psnr.item(),
                            "valid/sample_pred_image": wandb.Image(pred_img),
                            "valid/sample_target_image": wandb.Image(target_img),
                            "valid/sample_depth_map": wandb.Image(depth_map_img),
                        }
                    )

                    valid_avg_loss = total_valid_loss / len(valid_loader)
                    valid_avg_psnr = total_valid_psnr / len(valid_loader)

                    print(
                        f"Validation on Epoch {epoch+1}:\nValid Avg Loss = {valid_avg_loss:.4f}, Valid Avg PSNR = {valid_avg_psnr:.2f}"
                    )

                    # Save checkpoint every 5 epochs (rank 0 only)
                    checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
                    torch.save(model.state_dict(), checkpoint_path)
                    wandb.save(checkpoint_path)

                    break

                log_video(model, epoch, main_focal)


def execute(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)
    setup_wandb(rank=0, config=config, exp_name="nerf-a100-amp")

    train_dataset = NerfDataset(
        root_path="train.npz", transform=None, target_transform=None
    )

    valid_dataset = NerfDataset(
        root_path="valid.npz", transform=None, target_transform=None
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        num_workers=os.cpu_count(),
        drop_last=False,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config["BATCH_SIZE"],
        num_workers=os.cpu_count(),
        drop_last=False,
        pin_memory=True,
    )

    model = NeRF(device=device, emb_dim=config["EMB_DIM"], config=config)
    model = torch.compile(model, fullgraph=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"],
        amsgrad=config["AMSGRAD"],
        fused=config["FUSED_OPTIM"],
    )

    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        device=device,
        config=config,
        epochs=config["EPOCHS"],
    )

    cleanup_wandb()


if __name__ == "__main__":
    execute(config=exp2_config)
