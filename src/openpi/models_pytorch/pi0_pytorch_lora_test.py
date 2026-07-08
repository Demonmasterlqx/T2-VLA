import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import _select_action_expert_overlay_keys
from openpi.policies import policy_config


def test_select_action_expert_overlay_keys_filters_paligemma_vlm_key():
    state_dict = {
        "paligemma_with_expert.paligemma.language_model.layers.0.self_attn.q_proj.weight": torch.ones(1),
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight": torch.ones(1),
        "action_in_proj.weight": torch.ones(1),
        "tactile_prefix_encoder.out_proj.weight": torch.ones(1),
    }

    selected = _select_action_expert_overlay_keys(state_dict)

    assert set(selected) == {
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight",
        "action_in_proj.weight",
        "tactile_prefix_encoder.out_proj.weight",
    }
    assert (
        selected["paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight"]
        is state_dict["paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight"]
    )
    assert selected["action_in_proj.weight"] is state_dict["action_in_proj.weight"]
    assert selected["tactile_prefix_encoder.out_proj.weight"] is state_dict["tactile_prefix_encoder.out_proj.weight"]


def test_select_action_expert_overlay_keys_rejects_empty_overlay():
    with pytest.raises(ValueError, match="no action expert"):
        _select_action_expert_overlay_keys({})


def test_select_action_expert_overlay_keys_rejects_unexpected_key():
    with pytest.raises(RuntimeError, match="Unexpected action expert overlay keys"):
        _select_action_expert_overlay_keys({"optimizer.slot": torch.ones(1)})


def test_create_trained_policy_rejects_action_expert_overlay_for_non_pytorch_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="only supported for PyTorch"):
        policy_config.create_trained_policy(
            train_config=object(),
            checkpoint_dir=tmp_path,
            action_expert_merged_safetensors="overlay.safetensors",
        )
