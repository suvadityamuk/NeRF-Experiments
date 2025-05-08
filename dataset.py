import torch
from utils import get_rays, render_flat_rays
import numpy as np
from configs import NUM_SAMPLES
import torch.nn.functional as F


class NerfDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, transform, target_transform):
        super(NerfDataset, self).__init__()
        self.root_path = root_path
        self.transform = transform
        self.target_transform = target_transform

        raw_npz = np.load(root_path)

        self._raw_images = raw_npz["images"].copy()
        self._raw_poses = raw_npz["poses"].copy()

        self.images = torch.from_numpy(self._raw_images)
        self.poses = torch.from_numpy(self._raw_poses)

        self.height = self.images.shape[1]  # assuming (B, H, W, C)
        self.width = self.images.shape[2]
        self.focal_length = torch.from_numpy(raw_npz["focal"])

        self.images = list(torch.tensor_split(self.images, self.images.shape[0], dim=0))
        self.poses = list(torch.tensor_split(self.poses, self.poses.shape[0], dim=0))

        self.images = [image.squeeze(0) for image in self.images]
        self.poses = [pose.squeeze(0) for pose in self.poses]

    def map_fn(self, pose):
        (ray_origins, ray_dirs) = get_rays(
            height=self.height,
            width=self.width,
            focal_length=self.focal_length,
            pose=pose,
        )
        (rays_flat, t) = render_flat_rays(
            ray_origins=ray_origins,
            ray_dirs=ray_dirs,
            near_bound=2.0,
            far_bound=6.0,
            num_samples=NUM_SAMPLES,
            rand=True,
        )
        return (rays_flat, t)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]
        ray = self.map_fn(pose)
        focal = self.focal_length

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            pose = self.target_transform(pose)

        return image, ray, focal
