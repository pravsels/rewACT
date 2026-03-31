# Copyright 2024 Tony Z. Zhao, The HuggingFace Inc. team, and Ville Kuosmanen.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .configuration_rewact_rlt import RewACTRLTConfig
from .modeling_rewact import RewACT, RewACTPolicy


class RLTTransformerBlock(nn.Module):
    """Pre-norm transformer block used by the encoder and decoder."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class RLTEncoder(nn.Module):
    """Compress the observation encoder sequence into a single RLT token."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, num_layers: int):
        super().__init__()
        self.rlt_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.layers = nn.ModuleList(
            [RLTTransformerBlock(dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )

    def forward(self, encoder_embeddings: Tensor) -> Tensor:
        batch_size = encoder_embeddings.shape[0]
        query = self.rlt_query.expand(batch_size, -1, -1)
        x = torch.cat([encoder_embeddings, query], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x[:, -1, :]


class RLTDecoder(nn.Module):
    """Autoregressively reconstruct the observation encoder sequence."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [RLTTransformerBlock(dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, rlt_token: Tensor, target_embeddings: Tensor) -> Tensor:
        seq_len = target_embeddings.shape[1]
        decoder_input = torch.cat([rlt_token[:, None, :], target_embeddings[:, :-1, :]], dim=1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=target_embeddings.device),
            diagonal=1,
        )

        x = decoder_input
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask)
        return self.output_proj(x)


class RewACTRLT(RewACT):
    """RewACT variant with an RLT bottleneck on top of encoder_out."""

    config: RewACTRLTConfig

    def __init__(self, config: RewACTRLTConfig):
        super().__init__(config)
        self.rlt_encoder = RLTEncoder(
            dim=config.dim_model,
            num_heads=config.rlt_num_heads,
            mlp_dim=config.rlt_mlp_dim,
            num_layers=config.rlt_num_layers,
        )
        self.rlt_decoder = RLTDecoder(
            dim=config.dim_model,
            num_heads=config.rlt_num_heads,
            mlp_dim=config.rlt_mlp_dim,
            num_layers=config.rlt_num_layers,
        )
        self._reset_rlt_parameters()
        if config.rlt_frozen_backbone:
            self.freeze_rewact_backbone()

    def _reset_rlt_parameters(self) -> None:
        for module in chain(self.rlt_encoder.layers, self.rlt_decoder.layers):
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def freeze_rewact_backbone(self) -> None:
        for name, param in self.named_parameters():
            if name.startswith("rlt_encoder") or name.startswith("rlt_decoder"):
                continue
            param.requires_grad = False

    def compute_rlt_target(self, encoder_out: Tensor) -> Tensor:
        target = encoder_out.transpose(0, 1)
        if self.config.rlt_skip_latent_token:
            if target.shape[1] <= 1:
                raise ValueError("Cannot skip the latent token when encoder_out has no remaining tokens.")
            target = target[:, 1:, :]
        return target

    def extract_rlt_token(self, batch: dict[str, Tensor]) -> Tensor:
        encoder_out, _ = self.compute_encoder_out(
            batch,
            use_action_vae_latent=self.config.rlt_use_action_vae_latent,
        )
        target = self.compute_rlt_target(encoder_out).detach()
        return self.rlt_encoder(target)

    def compute_rlt_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        encoder_out, _ = self.compute_encoder_out(
            batch,
            use_action_vae_latent=self.config.rlt_use_action_vae_latent,
        )
        target = self.compute_rlt_target(encoder_out).detach()
        rlt_token = self.rlt_encoder(target)
        predictions = self.rlt_decoder(rlt_token, target)

        recon_sq = torch.square(predictions - target).sum(dim=-1)
        recon_loss = recon_sq.mean(dim=-1).mean()
        loss = recon_loss * self.config.rlt_loss_weight
        return loss, {
            "rlt_recon_loss": recon_loss.item(),
            "loss": loss.item(),
        }


class RewACTRLTPolicy(RewACTPolicy):
    """Policy wrapper for stage-1 RLT training on top of rewACT."""

    config_class = RewACTRLTConfig
    name = "rewact_rlt"

    def __init__(self, config: RewACTRLTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = RewACTRLT(config)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self._prepare_model_batch(batch)
        return self.model.compute_rlt_loss(batch)

    def get_optim_params(self) -> list[dict[str, list[nn.Parameter]]]:
        """Return only non-empty parameter groups with trainable params."""
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("RewACTRLTPolicy has no trainable parameters.")
        return [{"params": trainable_params}]
