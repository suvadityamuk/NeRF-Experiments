import numpy as np
import os

def install_dataset():
    """Installs the tiny NeRF dataset and required packages."""
    os.system('wget "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"')
    os.system('pip install wandb[media] torchao -qq')
    print("Dataset and packages installed.")

raw_npz = np.load("tiny_nerf_data.npz")

images = raw_npz["images"]
poses = raw_npz["poses"]
focal = raw_npz["focal"]

valid_split = int(0.9 * len(images))

train_images = images[:valid_split]
train_poses = poses[:valid_split]
train_focal = focal

valid_images = images[valid_split:]
valid_poses = poses[valid_split:]
valid_focal = focal

POS_ENCODE_DIMS = 16
NUM_SAMPLES = 32

# save train and valid in separate npz files
np.savez("train.npz", images=train_images, poses=train_poses, focal=train_focal)
np.savez("valid.npz", images=valid_images, poses=valid_poses, focal=valid_focal)