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
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from .configuration_rewact import RewACTConfig


@PreTrainedConfig.register_subclass("rewact_rlt")
@dataclass
class RewACTRLTConfig(RewACTConfig):
    rlt_num_layers: int = 2
    rlt_num_heads: int = 8
    rlt_mlp_dim: int = 2048
    rlt_loss_weight: float = 1.0
    # Stage 1 should default to the inference-style zero latent rather than
    # the action-conditioned VAE latent used during imitation-learning training.
    rlt_use_action_vae_latent: bool = False
    # Exclude the encoder's latent slot from the reconstruction target.
    rlt_skip_latent_token: bool = True
    # Freeze the inherited rewACT backbone so only the RLT modules train.
    rlt_frozen_backbone: bool = True
