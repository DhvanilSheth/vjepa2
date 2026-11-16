# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Do the V-JEPA 3 flags COMPOSE?

Each flag has its own test file proving it works alone.  The ablation ladder
ends in a preset that switches several on together, so this file checks the
combinations: every pairing builds, runs, and trains, and with everything on
no parameter is left without a gradient.

"Every parameter gets a gradient" is checked over ONE image batch plus ONE
video batch, because some parameters are modality-specific by design
(`patch_embed_img` / `img_mod_embed` vs `patch_embed` / `video_mod_embed`).
A parameter with no gradient in either modality would be dead weight -- and
under DistributedDataParallel(static_graph=True) a parameter whose usage
varies between iterations is a crash (tests/distributed/).
"""

import itertools
import unittest

import torch

import app.vjepa_3.models.vision_transformer as vit_3

NUM_FRAMES, CROP = 8, 64
BASE_KWARGS = dict(
    img_size=(CROP, CROP),
    num_frames=NUM_FRAMES,
    use_rope=True,
    img_temporal_dim_size=1,
)

# One non-default setting per flag family.
FLAGS = {
    "rope": dict(rope_theta=[100.0, 10000.0, 10000.0], rope_temporal_scale=0.5),
    "string": dict(use_string=True, string_init_std=0.1),
    "ffn": dict(ffn_type="swiglu"),
    "norm": dict(norm_type="derf"),
}


def _build(*families):
    kwargs = dict(BASE_KWARGS)
    for family in families:
        kwargs.update(FLAGS[family])
    torch.manual_seed(0)
    return vit_3.vit_tiny(**kwargs)


class TestFlagsCompose(unittest.TestCase):
    def test_every_pair_trains(self):
        torch.manual_seed(1)
        video = torch.randn(1, 3, NUM_FRAMES, CROP, CROP)
        for pair in itertools.combinations(FLAGS, 2):
            with self.subTest(flags=pair):
                model = _build(*pair).train()
                out = model(video, training=True)
                self.assertTrue(torch.isfinite(out).all())
                out.sum().backward()

    def test_all_on_every_parameter_gets_a_gradient(self):
        model = _build(*FLAGS).train()
        torch.manual_seed(1)
        video = torch.randn(1, 3, NUM_FRAMES, CROP, CROP)
        image = torch.randn(1, 3, 1, CROP, CROP)
        loss = model(video, training=True).sum() + model(image, training=True).sum()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        dead = [
            name
            for name, p in model.named_parameters()
            if p.grad is None or not torch.isfinite(p.grad).all()
        ]
        self.assertEqual(dead, [], f"parameters with no finite gradient: {dead}")

    def test_all_on_matches_under_activation_checkpointing(self):
        """train.py runs with use_activation_checkpointing: true."""
        torch.manual_seed(1)
        video = torch.randn(1, 3, NUM_FRAMES, CROP, CROP)
        plain = _build(*FLAGS).train()
        ckpt = _build(*FLAGS).train()
        ckpt.use_activation_checkpointing = True
        ckpt.load_state_dict(plain.state_dict())

        out_plain = plain(video, training=True)
        out_ckpt = ckpt(video, training=True)
        torch.testing.assert_close(out_plain, out_ckpt, rtol=1e-5, atol=1e-6)

        out_plain.sum().backward()
        out_ckpt.sum().backward()
        for (name, p), (_, q) in zip(plain.named_parameters(), ckpt.named_parameters()):
            torch.testing.assert_close(
                p.grad, q.grad, rtol=1e-4, atol=1e-5, msg=f"grad mismatch: {name}"
            )


if __name__ == "__main__":
    unittest.main()
