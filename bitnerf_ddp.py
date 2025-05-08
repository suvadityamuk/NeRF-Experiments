import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import POS_ENCODE_DIMS, NUM_SAMPLES


class BitNeRFDDP(torch.nn.Module):
    def __init__(self, device, config, emb_dim=64, num_layers=8):
        super(BitNeRFDDP, self).__init__()
        self.inp_emb = (2 * 3 * POS_ENCODE_DIMS) + 3
        self.out_dim = 4
        self.padded_input_dim = ((self.inp_emb + 7) // 8) * 8
        self.padded_output_dim = ((self.out_dim + 7) // 8) * 8

        self.input_layer = nn.Linear(
            in_features=self.padded_input_dim,
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
                layer = nn.Linear(
                    in_features=input_dim_size + self.padded_input_dim,
                    out_features=emb_dim,
                    bias=True,
                    device=device,
                )
            else:
                layer = nn.Linear(
                    in_features=input_dim_size,
                    out_features=emb_dim,
                    bias=True,
                    device=device,
                )

            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

        self.output_layer = nn.Linear(
            in_features=input_dim_size,
            out_features=self.padded_output_dim,
            bias=True,
            device=device,
        )

    def forward(self, x):
        inputs = F.pad(x, (0, self.padded_input_dim - self.inp_emb))
        x = inputs
        x = self.input_layer(x)
        for idx, layer in enumerate(self.layers):
            if idx % 4 == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)
            x = F.relu(layer(x))
        outputs = self.output_layer(x)
        outputs = outputs[:, :, : self.out_dim]
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
