import unittest

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot_policy_rewact.configuration_rewact import RewACTConfig
from lerobot_policy_rewact.configuration_rewact_rlt import RewACTRLTConfig
from lerobot_policy_rewact.modeling_rewact import RewACT
from lerobot_policy_rewact.modeling_rewact_rlt import RewACTRLTPolicy


def _make_common_kwargs():
    return {
        "input_features": {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(3,)),
        },
        "output_features": {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        "normalization_mapping": {
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ENV: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        },
        "device": "cpu",
        "chunk_size": 4,
        "n_action_steps": 2,
        "dim_model": 16,
        "n_heads": 4,
        "dim_feedforward": 32,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "latent_dim": 8,
        "n_vae_encoder_layers": 1,
        "dropout": 0.0,
        "use_vae": True,
        "vision_backbone": "resnet18",
        "pretrained_backbone_weights": None,
        "proprio_dropout": 0.0,
    }


def _make_batch(batch_size=2):
    return {
        "observation.state": torch.randn(batch_size, 7),
        "observation.environment_state": torch.randn(batch_size, 3),
    }


class RewACTRLTTest(unittest.TestCase):
    def test_compute_encoder_out_supports_zero_latent_without_actions(self):
        model = RewACT(RewACTConfig(**_make_common_kwargs()))
        model.train()

        encoder_out, encoder_pos = model.compute_encoder_out(
            _make_batch(),
            use_action_vae_latent=False,
        )

        self.assertEqual(encoder_out.shape, (3, 2, 16))
        self.assertEqual(encoder_pos.shape, (3, 1, 16))

    def test_rewact_rlt_policy_forward_uses_encoder_output_without_latent_slot(self):
        config = RewACTRLTConfig(
            **_make_common_kwargs(),
            rlt_num_layers=1,
            rlt_num_heads=4,
            rlt_mlp_dim=32,
            rlt_skip_latent_token=True,
        )
        policy = RewACTRLTPolicy(config)
        policy.train()

        batch = _make_batch()
        loss, metrics = policy.forward(batch)

        self.assertEqual(loss.ndim, 0)
        self.assertIn("rlt_recon_loss", metrics)

        with torch.no_grad():
            encoder_out, _ = policy.model.compute_encoder_out(batch, use_action_vae_latent=False)
            target = policy.model.compute_rlt_target(encoder_out)

        self.assertEqual(target.shape, (2, 2, 16))


if __name__ == "__main__":
    unittest.main()
