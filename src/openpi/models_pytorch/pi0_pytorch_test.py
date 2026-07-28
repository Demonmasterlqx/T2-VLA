from types import SimpleNamespace

import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class _FakePaliGemma:
    def __init__(self):
        self.paligemma = SimpleNamespace(
            language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation=None))
        )

    def forward(self, **kwargs):
        del kwargs
        return None, "prefix-cache"


def test_sample_actions_aligns_supplied_noise_to_action_projection_dtype():
    model = object.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(action_horizon=1, action_dim=32)
    model.action_in_proj = torch.nn.Linear(32, 4, dtype=torch.float32)
    model.paligemma_with_expert = _FakePaliGemma()
    state = torch.zeros(1, 7)
    model._preprocess_observation = lambda observation, train: (  # noqa: SLF001
        [],
        [],
        None,
        None,
        state,
        None,
    )
    model.embed_prefix = lambda *args: (
        torch.zeros(1, 1, 4),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool),
    )
    model._prepare_attention_masks_4d = lambda mask: mask[:, None, :, :]  # noqa: SLF001
    denoise_dtypes = []

    def denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep):
        del state, prefix_pad_masks, past_key_values, timestep
        denoise_dtypes.append(x_t.dtype)
        return torch.zeros_like(x_t)

    model.denoise_step = denoise_step
    observation = SimpleNamespace(state=state)
    noise = torch.zeros(1, 1, 32, dtype=torch.bfloat16)

    actions = PI0Pytorch.sample_actions(
        model,
        torch.device("cpu"),
        observation,
        noise=noise,
        num_steps=1,
    )

    assert denoise_dtypes == [torch.float32]
    assert actions.dtype == torch.float32
    assert noise.dtype == torch.bfloat16
