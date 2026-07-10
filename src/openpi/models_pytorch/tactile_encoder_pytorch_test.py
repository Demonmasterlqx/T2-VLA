import pytest
import torch

from openpi.models_pytorch.tactile_encoder_pytorch import TactileTCNEncoder


def _encoder() -> TactileTCNEncoder:
    return TactileTCNEncoder(
        input_dim=6,
        hidden_dim=8,
        output_dim=4,
        history_len=2,
        has_reference_frame=True,
        diff_from_reference=False,
    )


def test_tactile_tcn_encoder_builds_one_prefix_embedding():
    output = _encoder()(torch.zeros(3, 3, 6))

    assert output.shape == (3, 4)


def test_tactile_tcn_encoder_rejects_wrong_feature_width():
    with pytest.raises(ValueError, match="expected input_dim=6"):
        _encoder()(torch.zeros(1, 3, 5))
