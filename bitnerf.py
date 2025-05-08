import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import POS_ENCODE_DIMS, NUM_SAMPLES


class RMSNormLinear(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(RMSNormLinear, self).__init__()
        self.norm = nn.RMSNorm(in_features, device=device)
        self.linear = nn.Linear(in_features, out_features, bias=True, device=device)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x


class BitNeRF(torch.nn.Module):
    def __init__(self, device, config, emb_dim=64, num_layers=8):
        super(BitNeRF, self).__init__()
        self.inp_emb = (2 * 3 * POS_ENCODE_DIMS) + 3

        self.input_layer = nn.Linear(
            in_features=self.inp_emb,
            out_features=emb_dim,
            bias=True,
            device=device,
        )
        input_dim_size = emb_dim
        self.layers = []
        self.config = config
        self.device = device

        for idx in range(num_layers):
            if idx % 4 == 0 and idx > 0:
                in_features = input_dim_size + self.inp_emb
            else:
                in_features = input_dim_size
            layer = RMSNormLinear(
                in_features=in_features, out_features=emb_dim, device=device
            )
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

        self.output_layer = nn.Linear(
            in_features=input_dim_size, out_features=4, bias=True, device=device
        )

    def forward(self, x):
        inputs = x
        x = self.input_layer(x)
        for idx, layer in enumerate(self.layers):
            if idx % 4 == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)
            x = F.relu(layer(x))
        outputs = self.output_layer(x)
        return outputs

    def render_rgb_depth(self, rays_flat, t, rand=True, train=True):
        predictions = self.forward(rays_flat)
        predictions = torch.reshape(
            predictions, (-1, self.config["H"], self.config["W"], NUM_SAMPLES, 4)
        )

        rgb = torch.sigmoid(predictions[..., :-1])
        sigma = F.relu(predictions[..., -1])

        delta = t[..., 1:] - t[..., :-1]
        delta = torch.cat(
            [
                delta,
                torch.broadcast_to(
                    torch.tensor(1e10, device=self.device), delta[..., :1].shape
                ),
            ],
            dim=-1,
        )
        if rand:
            alpha = 1.0 - torch.exp(-sigma * delta)
        else:
            alpha = 1.0 - torch.exp(-sigma * delta[:, None, None, :])

        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        weights = alpha * transmittance

        rendered_rgb = torch.sum(weights[..., None] * rgb, dim=-2)
        if rand:
            depth_map = torch.sum(weights * t, dim=-1)
        else:
            depth_map = torch.sum(weights * t[:, None, None], dim=-1)

        return (rendered_rgb, depth_map)
