import sys
import types
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from utils import apply_sampler_episodes  # noqa: E402


class ApplySamplerEpisodesTest(unittest.TestCase):
    def test_leaves_dataset_episodes_unchanged_when_sampler_config_missing(self):
        cfg = types.SimpleNamespace(dataset=types.SimpleNamespace(episodes=["keep-me"]))

        applied = apply_sampler_episodes(cfg, None)

        self.assertFalse(applied)
        self.assertEqual(cfg.dataset.episodes, ["keep-me"])


if __name__ == "__main__":
    unittest.main()
