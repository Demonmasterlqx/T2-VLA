import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from safetensors.torch import load_file
from safetensors.torch import save_file
import torch

from openpi.policies import policy as policy_module
from openpi.policies import policy_config
from openpi.policies import tabero_dsrl_policy
import openpi.transforms as transforms

ACTOR_SHAPES = {
    "dsrl_action_noise_net.shared_net.0.weight": (128, 192),
    "dsrl_action_noise_net.shared_net.0.bias": (128,),
    "dsrl_action_noise_net.shared_net.1.weight": (128,),
    "dsrl_action_noise_net.shared_net.1.bias": (128,),
    "dsrl_action_noise_net.shared_net.3.weight": (128, 128),
    "dsrl_action_noise_net.shared_net.3.bias": (128,),
    "dsrl_action_noise_net.shared_net.4.weight": (128,),
    "dsrl_action_noise_net.shared_net.4.bias": (128,),
    "dsrl_action_noise_net.shared_net.6.weight": (128, 128),
    "dsrl_action_noise_net.shared_net.6.bias": (128,),
    "dsrl_action_noise_net.shared_net.7.weight": (128,),
    "dsrl_action_noise_net.shared_net.7.bias": (128,),
    "dsrl_action_noise_net.mean_layer.weight": (32, 128),
    "dsrl_action_noise_net.mean_layer.bias": (32,),
    "dsrl_action_noise_net.log_std_layer.weight": (32, 128),
    "dsrl_action_noise_net.log_std_layer.bias": (32,),
    "actor_image_encoder.encoder.0.weight": (32, 3, 3, 3),
    "actor_image_encoder.encoder.0.bias": (32,),
    "actor_image_encoder.encoder.2.weight": (32, 32, 3, 3),
    "actor_image_encoder.encoder.2.bias": (32,),
    "actor_image_encoder.encoder.4.weight": (32, 32, 3, 3),
    "actor_image_encoder.encoder.4.bias": (32,),
    "actor_image_encoder.encoder.6.weight": (32, 32, 3, 3),
    "actor_image_encoder.encoder.6.bias": (32,),
    "actor_image_encoder.bottleneck.1.weight": (64, 32768),
    "actor_image_encoder.bottleneck.1.bias": (64,),
    "actor_image_encoder.bottleneck.2.weight": (64,),
    "actor_image_encoder.bottleneck.2.bias": (64,),
    "actor_state_encoder.encoder.0.weight": (64, 7),
    "actor_state_encoder.encoder.0.bias": (64,),
    "actor_state_encoder.encoder.1.weight": (64,),
    "actor_state_encoder.encoder.1.bias": (64,),
    "actor_tactile_encoder.blocks.0.kernels.0.weight": (64, 396),
    "actor_tactile_encoder.blocks.0.kernels.0.bias": (64,),
    "actor_tactile_encoder.blocks.0.kernels.1.weight": (64, 396),
    "actor_tactile_encoder.blocks.0.kernels.1.bias": (64,),
    "actor_tactile_encoder.blocks.0.kernels.2.weight": (64, 396),
    "actor_tactile_encoder.blocks.0.kernels.2.bias": (64,),
    "actor_tactile_encoder.blocks.0.residual_proj.weight": (64, 396),
    "actor_tactile_encoder.blocks.0.residual_proj.bias": (64,),
    "actor_tactile_encoder.blocks.1.kernels.0.weight": (64, 64),
    "actor_tactile_encoder.blocks.1.kernels.0.bias": (64,),
    "actor_tactile_encoder.blocks.1.kernels.1.weight": (64, 64),
    "actor_tactile_encoder.blocks.1.kernels.1.bias": (64,),
    "actor_tactile_encoder.blocks.1.kernels.2.weight": (64, 64),
    "actor_tactile_encoder.blocks.1.kernels.2.bias": (64,),
    "actor_tactile_encoder.out_proj.weight": (64, 64),
    "actor_tactile_encoder.out_proj.bias": (64,),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, values: dict) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


def _zero_actor_state() -> dict[str, torch.Tensor]:
    return {key: torch.zeros(shape, dtype=torch.bfloat16) for key, shape in ACTOR_SHAPES.items()}


def _actor_metadata(task_id: int, global_step: int) -> dict[str, str]:
    return {
        "format": "tabero_dsrl_t2vla",
        "format_version": "1",
        "task_id": str(task_id),
        "global_step": str(global_step),
        "dtype": "bfloat16",
    }


def _write_bundle(
    root: Path,
    *,
    task_id: int = 0,
    global_step: int = 50,
    training_config: str | None = None,
) -> tuple[Path, Path]:
    bundle_path = root / "bundle"
    bundle_path.mkdir(parents=True)
    base_path = root / "base"
    base_path.mkdir()
    (base_path / "model.safetensors").write_bytes(b"base pytorch checkpoint")
    actor_path = bundle_path / "dsrl_actor.safetensors"
    save_file(
        _zero_actor_state(),
        actor_path,
        metadata=_actor_metadata(task_id, global_step),
    )
    actor_hash = _sha256(actor_path)
    base_hash = _sha256(base_path / "model.safetensors")
    source_hash = "1" * 64
    manifest = {
        "format": "tabero_dsrl_t2vla",
        "format_version": 1,
        "algorithm": "dsrl-sac",
        "task_id": task_id,
        "global_step": global_step,
        "is_final": True,
        "training_config": training_config or f"isaaclab_pi0_dsrl_tacfield_tabero_task{task_id}_firm_8gpu_50step",
        "base_model": str(base_path),
        "base_model_sha256": base_hash,
        "source_checkpoint": (f"/tmp/global_step_{global_step}/actor/model_state_dict/trainable_weights.pt"),
        "source_checkpoint_sha256": source_hash,
        "source_provenance": "/tmp/provenance.env",
        "source_provenance_sha256": "2" * 64,
        "source_config_snapshot": "/tmp/config_snapshot.yaml",
        "source_config_snapshot_sha256": "3" * 64,
        "legacy_source_config": None,
        "legacy_source_config_sha256": None,
        "source_git_commit": "deadbeef",
        "actor_weights": actor_path.name,
        "actor_weights_sha256": actor_hash,
        "actor_manifest_version": 1,
        "actor_tensor_count": 48,
        "actor_parameter_count": 2_311_648,
        "actor_dtype": "bfloat16",
        "observation_contract": {
            "image": {
                "key": "dsrl_raw_image",
                "shape": [256, 256, 3],
                "layout": "HWC",
                "dtype": "uint8",
                "value_range": [0, 255],
                "preprocessing": {
                    "resize": [64, 64],
                    "mode": "bilinear",
                    "align_corners": False,
                    "output_layout": "NCHW",
                    "normalization": "uint8_to_minus_one_one",
                },
            },
            "state": {"key": "state", "shape": [7], "dtype": "float32"},
            "tactile": {
                "key": "tactile_marker_motion",
                "shape": [9, 198, 2],
                "dtype": "float32",
                "encoder_shape": [9, 396],
            },
        },
        "feature_contract": {"order": ["state", "image", "tactile"], "dims": [64, 64, 64], "total_dim": 192},
        "noise_contract": {
            "dim": 32,
            "horizon": 50,
            "deterministic": "tanh(mean)",
            "broadcast_across_horizon": True,
            "pi0_denoise_steps": 10,
        },
        "architecture": {
            "image_size": 64,
            "state_dim": 7,
            "tactile_shape": [9, 198, 2],
            "hidden_dims": [128, 128, 128],
            "feature_dim": 192,
            "noise_dim": 32,
        },
        "artifact_audit": "artifact_audit.json",
    }
    manifest_path = bundle_path / "manifest.json"
    _write_json(manifest_path, manifest)
    audit = {
        "format": "tabero_dsrl_artifact_audit",
        "format_version": 1,
        "status": "passed",
        "task_id": task_id,
        "global_step": global_step,
        "source_checkpoint_sha256": source_hash,
        "base_model_sha256": base_hash,
        "actor_weights_sha256": actor_hash,
        "manifest_sha256": _sha256(manifest_path),
        "checks": {
            "final_checkpoint_path": True,
            "source_metadata": True,
            "source_trainable_manifest": True,
            "actor_manifest": True,
            "actor_dtype": True,
            "actor_finite": True,
            "base_model_sha256": True,
            "formal_provenance": True,
            "output_hashes": True,
        },
    }
    _write_json(bundle_path / "artifact_audit.json", audit)
    return bundle_path, base_path


@pytest.fixture(scope="session")
def valid_bundle(tmp_path_factory):
    return _write_bundle(tmp_path_factory.mktemp("dsrl-bundle"))


@pytest.fixture
def bundle_copy(valid_bundle, tmp_path):
    source_bundle, source_base = valid_bundle
    bundle_path = tmp_path / "bundle"
    base_path = tmp_path / "base"
    shutil.copytree(source_bundle, bundle_path, ignore=shutil.ignore_patterns("*.safetensors"))
    os.link(source_bundle / "dsrl_actor.safetensors", bundle_path / "dsrl_actor.safetensors")
    shutil.copytree(source_base, base_path)
    return bundle_path, base_path


def _rewrite_manifest(bundle_path: Path, update: dict) -> None:
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(update)
    _write_json(manifest_path, manifest)
    audit_path = bundle_path / "artifact_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["manifest_sha256"] = _sha256(manifest_path)
    _write_json(audit_path, audit)


def _remove_manifest_field(bundle_path: Path, field: str) -> None:
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop(field)
    _write_json(manifest_path, manifest)
    audit_path = bundle_path / "artifact_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["manifest_sha256"] = _sha256(manifest_path)
    _write_json(audit_path, audit)


def _replace_actor(bundle_path: Path, state: dict[str, torch.Tensor]) -> None:
    actor_path = bundle_path / "dsrl_actor.safetensors"
    private_copy = bundle_path / ".dsrl_actor.copy"
    shutil.copy2(actor_path, private_copy)
    private_copy.replace(actor_path)
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    save_file(
        state,
        actor_path,
        metadata=_actor_metadata(manifest["task_id"], manifest["global_step"]),
    )
    actor_hash = _sha256(actor_path)
    manifest["actor_weights_sha256"] = actor_hash
    _write_json(manifest_path, manifest)
    audit_path = bundle_path / "artifact_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["actor_weights_sha256"] = actor_hash
    audit["manifest_sha256"] = _sha256(manifest_path)
    _write_json(audit_path, audit)


def test_tabero_dsrl_policy_module_exists():
    assert importlib.util.find_spec("openpi.policies.tabero_dsrl_policy") is not None


def test_bundle_loads_exact_final_actor_and_base_checkpoint(bundle_copy):
    bundle_path, base_path = bundle_copy

    bundle = tabero_dsrl_policy.TaberoDSRLBundle.load(
        bundle_path,
        base_checkpoint_dir=base_path,
    )

    assert bundle.manifest.task_id == 0
    assert bundle.manifest.global_step == 50
    assert bundle.manifest.is_final is True
    actor = bundle.build_actor()
    assert set(actor.state_dict()) == set(ACTOR_SHAPES)
    assert {parameter.dtype for parameter in actor.parameters()} == {torch.bfloat16}


def test_bundle_loads_allowlisted_task5_small4gpu40_profile(tmp_path):
    bundle_path, base_path = _write_bundle(
        tmp_path,
        task_id=5,
        global_step=40,
        training_config=("isaaclab_pi0_dsrl_tacfield_tabero_task5_firm_4gpu_40step_small"),
    )

    bundle = tabero_dsrl_policy.TaberoDSRLBundle.load(
        bundle_path,
        base_checkpoint_dir=base_path,
    )

    assert bundle.manifest.task_id == 5
    assert bundle.manifest.global_step == 40
    assert bundle.manifest.training_config.endswith("task5_firm_4gpu_40step_small")


@pytest.mark.parametrize(
    ("task_id", "global_step", "training_config"),
    [
        (
            0,
            40,
            "isaaclab_pi0_dsrl_tacfield_tabero_task5_firm_4gpu_40step_small",
        ),
        (
            5,
            40,
            "isaaclab_pi0_dsrl_tacfield_tabero_task5_firm_8gpu_50step",
        ),
        (
            5,
            50,
            "isaaclab_pi0_dsrl_tacfield_tabero_task5_firm_4gpu_40step_small",
        ),
        (
            5,
            39,
            "isaaclab_pi0_dsrl_tacfield_tabero_task5_firm_4gpu_40step_small",
        ),
    ],
)
def test_bundle_rejects_non_allowlisted_training_identity(
    tmp_path,
    task_id,
    global_step,
    training_config,
):
    bundle_path, base_path = _write_bundle(
        tmp_path,
        task_id=task_id,
        global_step=global_step,
        training_config=training_config,
    )

    with pytest.raises(ValueError, match="training identity"):
        tabero_dsrl_policy.TaberoDSRLBundle.load(
            bundle_path,
            base_checkpoint_dir=base_path,
        )


def test_bundle_rejects_actor_safetensors_metadata_step_mismatch(bundle_copy):
    bundle_path, base_path = bundle_copy
    actor_path = bundle_path / "dsrl_actor.safetensors"
    state = load_file(actor_path)
    private_copy = bundle_path / ".dsrl_actor.copy"
    shutil.copy2(actor_path, private_copy)
    private_copy.replace(actor_path)
    save_file(state, actor_path, metadata=_actor_metadata(task_id=0, global_step=40))
    actor_hash = _sha256(actor_path)
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["actor_weights_sha256"] = actor_hash
    _write_json(manifest_path, manifest)
    audit_path = bundle_path / "artifact_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["actor_weights_sha256"] = actor_hash
    audit["manifest_sha256"] = _sha256(manifest_path)
    _write_json(audit_path, audit)

    with pytest.raises(ValueError, match="actor metadata global_step"):
        tabero_dsrl_policy.TaberoDSRLBundle.load(
            bundle_path,
            base_checkpoint_dir=base_path,
        )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"format": "other"}, "format"),
        ({"format_version": 2}, "version"),
        ({"algorithm": "ppo"}, "dsrl-sac"),
        ({"task_id": 1}, "Task 0 or 5"),
        ({"global_step": 49}, "global_step.*50"),
        ({"is_final": False}, "is_final.*true"),
    ],
)
def test_bundle_rejects_wrong_final_training_identity(bundle_copy, update, message):
    bundle_path, base_path = bundle_copy
    _rewrite_manifest(bundle_path, update)

    with pytest.raises(ValueError, match=message):
        tabero_dsrl_policy.TaberoDSRLBundle.load(bundle_path, base_checkpoint_dir=base_path)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("actor_manifest_version",), True),
        (("actor_tensor_count",), 48.0),
        (("observation_contract", "image", "value_range", 0), False),
        (("architecture", "image_size"), 64.0),
    ],
)
def test_bundle_rejects_type_coercion_in_exact_manifest_contract(bundle_copy, path, value):
    bundle_path, base_path = bundle_copy
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parent = manifest
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    _write_json(manifest_path, manifest)
    audit_path = bundle_path / "artifact_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["manifest_sha256"] = _sha256(manifest_path)
    _write_json(audit_path, audit)

    with pytest.raises(ValueError, match="does not match.*contract"):
        tabero_dsrl_policy.TaberoDSRLBundle.load(bundle_path, base_checkpoint_dir=base_path)


def test_bundle_rejects_base_checkpoint_content_hash_mismatch(bundle_copy, tmp_path):
    bundle_path, _ = bundle_copy
    other_base = tmp_path / "other-base"
    other_base.mkdir()
    (other_base / "model.safetensors").write_bytes(b"different checkpoint")

    with pytest.raises(ValueError, match="base checkpoint SHA-256 mismatch"):
        tabero_dsrl_policy.TaberoDSRLBundle.load(bundle_path, base_checkpoint_dir=other_base)


def test_bundle_requires_checkpoint_directory_with_model_safetensors(bundle_copy):
    bundle_path, base_path = bundle_copy

    with pytest.raises(ValueError, match="checkpoint directory"):
        tabero_dsrl_policy.TaberoDSRLBundle.load(
            bundle_path,
            base_checkpoint_dir=base_path / "model.safetensors",
        )


def test_bundle_rejects_actor_hash_mismatch(bundle_copy):
    bundle_path, base_path = bundle_copy
    actor_path = bundle_path / "dsrl_actor.safetensors"
    private_copy = bundle_path / ".dsrl_actor.copy"
    shutil.copy2(actor_path, private_copy)
    private_copy.replace(actor_path)
    actor_path.write_bytes(actor_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="actor.*SHA-256 mismatch"):
        tabero_dsrl_policy.TaberoDSRLBundle.load(bundle_path, base_checkpoint_dir=base_path)


def test_bundle_actor_load_is_bound_to_captured_hashed_bytes(bundle_copy, monkeypatch):
    bundle_path, base_path = bundle_copy
    actor_path = bundle_path / "dsrl_actor.safetensors"
    replacement_path = bundle_path / ".replacement_actor"
    original_backup = bundle_path / ".original_actor"
    state = load_file(actor_path)
    save_file({key: torch.ones_like(value) for key, value in state.items()}, replacement_path)

    def during_actor_swap(callback):
        os.replace(actor_path, original_backup)
        os.replace(replacement_path, actor_path)
        try:
            return callback()
        finally:
            os.replace(actor_path, replacement_path)
            os.replace(original_backup, actor_path)

    real_load_file = getattr(tabero_dsrl_policy, "load_file", None)
    if real_load_file is not None:

        def aba_load_file(*args, **kwargs):
            return during_actor_swap(lambda: real_load_file(*args, **kwargs))

        monkeypatch.setattr(tabero_dsrl_policy, "load_file", aba_load_file)
    if hasattr(tabero_dsrl_policy, "load"):
        real_load = tabero_dsrl_policy.load

        def aba_load(*args, **kwargs):
            return during_actor_swap(lambda: real_load(*args, **kwargs))

        monkeypatch.setattr(tabero_dsrl_policy, "load", aba_load)

    bundle = tabero_dsrl_policy.TaberoDSRLBundle.load(bundle_path, base_checkpoint_dir=base_path)

    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in bundle.actor_state.values())
    assert _sha256(actor_path) == bundle.manifest.actor_weights_sha256


def test_bundle_manifest_parse_and_audit_hash_use_same_captured_bytes(bundle_copy, monkeypatch):
    bundle_path, base_path = bundle_copy
    manifest_path = bundle_path / "manifest.json"
    replacement_path = bundle_path / ".replacement_manifest"
    original_backup = bundle_path / ".original_manifest"
    replacement = json.loads(manifest_path.read_text())
    replacement["source_git_commit"] = "replacement"
    _write_json(replacement_path, replacement)
    real_parse_json_object = tabero_dsrl_policy._parse_json_object  # noqa: SLF001
    swaps = 0

    def aba_parse_json_object(content, path, label):
        nonlocal swaps
        if path != manifest_path:
            return real_parse_json_object(content, path, label)
        swaps += 1
        os.replace(manifest_path, original_backup)
        os.replace(replacement_path, manifest_path)
        try:
            return real_parse_json_object(content, path, label)
        finally:
            os.replace(manifest_path, replacement_path)
            os.replace(original_backup, manifest_path)

    monkeypatch.setattr(tabero_dsrl_policy, "_parse_json_object", aba_parse_json_object)

    bundle = tabero_dsrl_policy.TaberoDSRLBundle.load(bundle_path, base_checkpoint_dir=base_path)

    assert swaps == 1
    assert bundle.manifest.values["source_git_commit"] == "deadbeef"


def test_bundle_rejects_failed_audit_even_during_passed_audit_aba(bundle_copy, monkeypatch):
    bundle_path, base_path = bundle_copy
    audit_path = bundle_path / "artifact_audit.json"
    replacement_path = bundle_path / ".replacement_audit"
    original_backup = bundle_path / ".original_audit"
    passed_audit = json.loads(audit_path.read_text())
    _write_json(replacement_path, passed_audit)
    failed_audit = dict(passed_audit)
    failed_audit["status"] = "failed"
    _write_json(audit_path, failed_audit)
    real_parse_json_object = tabero_dsrl_policy._parse_json_object  # noqa: SLF001
    swaps = 0

    def aba_parse_json_object(content, path, label):
        nonlocal swaps
        if path != audit_path:
            return real_parse_json_object(content, path, label)
        swaps += 1
        os.replace(audit_path, original_backup)
        os.replace(replacement_path, audit_path)
        try:
            return real_parse_json_object(content, path, label)
        finally:
            os.replace(audit_path, replacement_path)
            os.replace(original_backup, audit_path)

    monkeypatch.setattr(tabero_dsrl_policy, "_parse_json_object", aba_parse_json_object)

    with pytest.raises(ValueError, match="audit status.*passed"):
        tabero_dsrl_policy.TaberoDSRLBundle.load(bundle_path, base_checkpoint_dir=base_path)
    assert swaps == 1


def test_bundle_rejects_more_than_one_actor_safetensors(bundle_copy):
    bundle_path, base_path = bundle_copy
    os.link(bundle_path / "dsrl_actor.safetensors", bundle_path / "other.safetensors")

    with pytest.raises(ValueError, match="exactly one actor safetensors"):
        tabero_dsrl_policy.TaberoDSRLBundle.load(bundle_path, base_checkpoint_dir=base_path)


def test_bundle_requires_passed_matching_artifact_audit(bundle_copy):
    bundle_path, base_path = bundle_copy
    audit_path = bundle_path / "artifact_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["status"] = "failed"
    _write_json(audit_path, audit)

    with pytest.raises(ValueError, match="audit status.*passed"):
        tabero_dsrl_policy.TaberoDSRLBundle.load(bundle_path, base_checkpoint_dir=base_path)


@pytest.mark.parametrize(
    "field",
    [
        "base_model",
        "source_checkpoint",
        "source_provenance",
        "source_provenance_sha256",
        "source_config_snapshot",
        "source_config_snapshot_sha256",
        "source_git_commit",
    ],
)
def test_bundle_requires_complete_exporter_v1_provenance_manifest(bundle_copy, field):
    bundle_path, base_path = bundle_copy
    _remove_manifest_field(bundle_path, field)

    with pytest.raises(ValueError, match="manifest keyspace mismatch"):
        tabero_dsrl_policy.TaberoDSRLBundle.load(bundle_path, base_checkpoint_dir=base_path)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"format_version": True}, "format_version.*integer 1"),
        ({"checks": {"made_up_check": True}}, "audit checks keyspace mismatch"),
    ],
)
def test_bundle_rejects_noncanonical_artifact_audit(bundle_copy, update, message):
    bundle_path, base_path = bundle_copy
    audit_path = bundle_path / "artifact_audit.json"
    audit = json.loads(audit_path.read_text())
    audit.update(update)
    _write_json(audit_path, audit)

    with pytest.raises(ValueError, match=message):
        tabero_dsrl_policy.TaberoDSRLBundle.load(bundle_path, base_checkpoint_dir=base_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task_id", 5, "audit task_id mismatch"),
        ("global_step", 49, "audit global_step mismatch"),
        ("actor_weights_sha256", "4" * 64, "audit actor_weights_sha256 mismatch"),
        ("base_model_sha256", "5" * 64, "audit base_model_sha256 mismatch"),
        ("manifest_sha256", "6" * 64, "audit manifest_sha256 mismatch"),
    ],
)
def test_bundle_rejects_inconsistent_artifact_audit(bundle_copy, field, value, message):
    bundle_path, base_path = bundle_copy
    audit_path = bundle_path / "artifact_audit.json"
    audit = json.loads(audit_path.read_text())
    audit[field] = value
    _write_json(audit_path, audit)

    with pytest.raises(ValueError, match=message):
        tabero_dsrl_policy.TaberoDSRLBundle.load(bundle_path, base_checkpoint_dir=base_path)


@pytest.mark.parametrize("corruption", ["extra", "shape", "dtype", "nonfinite"])
def test_bundle_rejects_noncanonical_actor_state(bundle_copy, corruption):
    bundle_path, base_path = bundle_copy
    state = load_file(bundle_path / "dsrl_actor.safetensors")
    key = "dsrl_action_noise_net.mean_layer.bias"
    if corruption == "extra":
        state["unexpected"] = torch.zeros(1, dtype=torch.bfloat16)
    elif corruption == "shape":
        state[key] = torch.zeros(31, dtype=torch.bfloat16)
    elif corruption == "dtype":
        state[key] = state[key].float()
    else:
        state[key][0] = torch.nan
    _replace_actor(bundle_path, state)

    with pytest.raises(
        ValueError,
        match={
            "extra": "unexpected",
            "shape": "shape",
            "dtype": "bfloat16",
            "nonfinite": "finite",
        }[corruption],
    ):
        tabero_dsrl_policy.TaberoDSRLBundle.load(bundle_path, base_checkpoint_dir=base_path)


def _raw_observation() -> dict[str, np.ndarray]:
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    return {
        "dsrl_raw_image": image,
        "state": np.linspace(-1.0, 1.0, 7, dtype=np.float32),
        "tactile_marker_motion": np.zeros((9, 198, 2), dtype=np.float32),
    }


def test_actor_preprocess_matches_rlinf_image_state_and_tactile_contract():
    actor = tabero_dsrl_policy.TaberoDSRLActor()
    observation = _raw_observation()

    images, states, tactile = actor.preprocess(observation)

    source = torch.from_numpy(observation["dsrl_raw_image"]).float() / 255.0
    expected_images = torch.nn.functional.interpolate(
        source.permute(2, 0, 1).unsqueeze(0),
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    )
    expected_images = (expected_images * 2.0 - 1.0).unsqueeze(1).to(torch.bfloat16)
    torch.testing.assert_close(images, expected_images)
    torch.testing.assert_close(states, torch.from_numpy(observation["state"])[None].to(torch.bfloat16))
    assert tactile.shape == (1, 9, 396)
    assert tactile.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda obs: obs.pop("tactile_marker_motion"), "tactile_marker_motion"),
        (lambda obs: obs.__setitem__("dsrl_raw_image", np.zeros((255, 256, 3), dtype=np.uint8)), "256, 256, 3"),
        (lambda obs: obs.__setitem__("dsrl_raw_image", np.zeros((256, 256, 3), dtype=np.float32)), "uint8"),
        (lambda obs: obs.__setitem__("state", np.zeros(8, dtype=np.float32)), "shape.*7"),
        (lambda obs: obs.__setitem__("state", np.zeros(7, dtype=np.float64)), "state.*float32"),
        (
            lambda obs: obs.__setitem__("tactile_marker_motion", np.zeros((8, 198, 2), dtype=np.float32)),
            "9, 198, 2",
        ),
        (
            lambda obs: obs.__setitem__("tactile_marker_motion", np.zeros((9, 198, 2), dtype=np.float64)),
            "tactile_marker_motion.*float32",
        ),
    ],
)
def test_actor_rejects_invalid_raw_observation(mutation, message):
    actor = tabero_dsrl_policy.TaberoDSRLActor()
    observation = _raw_observation()
    mutation(observation)

    with pytest.raises((KeyError, ValueError), match=message):
        actor.preprocess(observation)


def test_actor_tactile_encoder_is_two_layer_causal_tcn():
    actor = tabero_dsrl_policy.TaberoDSRLActor()
    encoder = actor.actor_tactile_encoder
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.zero_()
        encoder.blocks[0].kernels[0].weight[0, 0] = 1
        encoder.blocks[0].kernels[1].weight[0, 0] = 10
        encoder.blocks[0].kernels[2].weight[0, 0] = 100
        encoder.blocks[1].kernels[0].weight[0, 0] = 1
        encoder.out_proj.weight.copy_(torch.eye(64, dtype=torch.bfloat16))
    tactile = torch.zeros(1, 9, 396, dtype=torch.bfloat16)
    tactile[0, :, 0] = torch.arange(1, 10, dtype=torch.bfloat16)

    encoded = encoder(tactile)

    block0_last = torch.nn.functional.silu(torch.tensor(789.0, dtype=torch.bfloat16))
    expected = torch.nn.functional.silu(2 * block0_last)
    torch.testing.assert_close(encoded[0, 0], expected)
    torch.testing.assert_close(encoded[0, 1:], torch.zeros(63, dtype=torch.bfloat16))


def test_actor_feature_order_is_state_image_tactile(monkeypatch):
    actor = tabero_dsrl_policy.TaberoDSRLActor()
    monkeypatch.setattr(
        actor.actor_state_encoder,
        "forward",
        lambda value: torch.full((value.shape[0], 64), 1, dtype=torch.bfloat16),
    )
    monkeypatch.setattr(
        actor.actor_image_encoder,
        "forward",
        lambda value: torch.full((value.shape[0], 64), 2, dtype=torch.bfloat16),
    )
    monkeypatch.setattr(
        actor.actor_tactile_encoder,
        "forward",
        lambda value: torch.full((value.shape[0], 64), 3, dtype=torch.bfloat16),
    )

    features = actor.features(_raw_observation())

    torch.testing.assert_close(features[:, :64], torch.ones(1, 64, dtype=torch.bfloat16))
    torch.testing.assert_close(features[:, 64:128], torch.full((1, 64), 2, dtype=torch.bfloat16))
    torch.testing.assert_close(features[:, 128:], torch.full((1, 64), 3, dtype=torch.bfloat16))


def test_actor_deployment_noise_is_bf16_tanh_mean_broadcast_across_horizon(monkeypatch):
    actor = tabero_dsrl_policy.TaberoDSRLActor()
    mean = torch.linspace(-2, 2, 32, dtype=torch.bfloat16)[None]
    monkeypatch.setattr(actor, "mean", lambda observation: mean)

    noise = actor.noise(_raw_observation())

    expected_vector = torch.tanh(mean)
    assert isinstance(noise, torch.Tensor)
    assert noise.dtype == torch.bfloat16
    assert noise.shape == (1, 50, 32)
    torch.testing.assert_close(noise, expected_vector[:, None, :].expand(-1, 50, -1))
    assert noise.stride(1) == 0


def test_dsrl_policy_passes_original_observation_and_same_torch_noise_to_base():
    observation = _raw_observation()
    sentinel_noise = torch.zeros(1, 50, 32, dtype=torch.bfloat16)
    actor = SimpleNamespace(noise=lambda raw: sentinel_noise if raw is observation else None)

    class CapturingPolicy:
        metadata: ClassVar[dict[str, str]] = {"name": "base"}

        def infer(self, raw, *, noise):
            assert raw is observation
            assert noise is sentinel_noise
            return {"actions": "sentinel"}

    policy = tabero_dsrl_policy.TaberoDSRLPolicy(CapturingPolicy(), actor)

    result = policy.infer(observation)

    assert result == {"actions": "sentinel"}
    assert policy.metadata is CapturingPolicy.metadata


class _NoiseCapturingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received_noise = None

    def sample_actions(self, device, observation, **kwargs):
        del device, observation
        self.received_noise = kwargs["noise"]
        return torch.zeros(1, 1, 1)


def test_policy_preserves_torch_noise_dtype_without_numpy_roundtrip(monkeypatch):
    model = _NoiseCapturingModel()
    monkeypatch.setattr(policy_module._model.Observation, "from_dict", staticmethod(lambda inputs: inputs))  # noqa: SLF001
    base_policy = policy_module.Policy(model, is_pytorch=True, pytorch_device="cpu")
    noise = torch.zeros(1, 50, 32, dtype=torch.bfloat16)

    base_policy.infer({"state": np.zeros(7, dtype=np.float32)}, noise=noise)

    assert model.received_noise is noise
    assert model.received_noise.dtype == torch.bfloat16


def test_policy_keeps_existing_numpy_noise_conversion_and_2d_batching(monkeypatch):
    model = _NoiseCapturingModel()
    monkeypatch.setattr(policy_module._model.Observation, "from_dict", staticmethod(lambda inputs: inputs))  # noqa: SLF001
    base_policy = policy_module.Policy(model, is_pytorch=True, pytorch_device="cpu")
    noise = np.zeros((50, 32), dtype=np.float32)

    base_policy.infer({"state": np.zeros(7, dtype=np.float32)}, noise=noise)

    assert isinstance(model.received_noise, torch.Tensor)
    assert model.received_noise.shape == (1, 50, 32)
    assert model.received_noise.dtype == torch.float32


def test_policy_rejects_torch_noise_for_jax_model_with_clear_error(monkeypatch):
    class FakeJaxModel:
        def sample_actions(self, rng, observation, **kwargs):
            del rng, observation, kwargs
            return np.zeros((1, 1, 1), dtype=np.float32)

    monkeypatch.setattr(policy_module.nnx_utils, "module_jit", lambda function: function)
    base_policy = policy_module.Policy(FakeJaxModel())

    with pytest.raises(TypeError, match="torch.*JAX"):
        base_policy.infer(
            {"state": np.zeros(7, dtype=np.float32)},
            noise=torch.zeros(1, 50, 32),
        )


class _FakeLoadedPI0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = SimpleNamespace(to_bfloat16_for_selected_params=lambda precision: None)

    def sample_actions(self, *args, **kwargs):
        del args, kwargs
        return torch.zeros(1, 1, 1)


def _fake_train_config(tmp_path, loaded_model=None):
    loaded_model = loaded_model or _FakeLoadedPI0()
    model_config = SimpleNamespace(
        load_pytorch=lambda train_config, path: loaded_model,
        load=lambda params: (_ for _ in ()).throw(AssertionError("JAX load must not run")),
    )
    data_config = SimpleNamespace(
        asset_id=None,
        data_transforms=transforms.Group(),
        model_transforms=transforms.Group(),
        use_quantile_norm=False,
    )
    return SimpleNamespace(
        name="pi0_test",
        model=model_config,
        data=SimpleNamespace(create=lambda assets_dirs, model: data_config),
        assets_dirs=tmp_path,
        policy_metadata={"source": "base"},
    )


def test_create_trained_policy_wraps_normal_pytorch_policy_with_dsrl_actor(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").touch()
    loaded_model = _FakeLoadedPI0()
    train_config = _fake_train_config(tmp_path, loaded_model)
    actor = SimpleNamespace(noise=lambda obs: None)
    calls = []

    def from_bundle(cls, bundle_path, **kwargs):
        del cls
        calls.append((bundle_path, kwargs))
        return actor

    monkeypatch.setattr(tabero_dsrl_policy.TaberoDSRLActor, "from_bundle", classmethod(from_bundle))

    result = policy_config.create_trained_policy(
        train_config,
        tmp_path,
        norm_stats={},
        pytorch_device="cpu",
        dsrl_bundle_path=tmp_path / "dsrl",
    )

    assert isinstance(result, tabero_dsrl_policy.TaberoDSRLPolicy)
    assert result._base_policy._model is loaded_model  # noqa: SLF001
    assert result._actor is actor  # noqa: SLF001
    assert calls == [
        (
            tmp_path / "dsrl",
            {
                "base_checkpoint_dir": tmp_path,
                "base_model_sha256": hashlib.sha256(b"").hexdigest(),
                "device": "cpu",
            },
        )
    ]


def test_policy_config_import_does_not_require_fcntl():
    script = """
import builtins

real_import = builtins.__import__

def import_without_fcntl(name, *args, **kwargs):
    if name == "fcntl":
        raise ImportError("simulated missing fcntl")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_fcntl
import openpi.policies.policy_config
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_stable_dsrl_checkpoint_reports_missing_fcntl(tmp_path, monkeypatch):
    weight_path = tmp_path / "model.safetensors"
    weight_path.touch()
    real_import = builtins.__import__

    def import_without_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("simulated missing fcntl")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_fcntl)

    with (
        pytest.raises(ValueError, match="requires fcntl"),
        policy_config._stable_dsrl_checkpoint(weight_path),  # noqa: SLF001
    ):
        pass


def test_create_trained_policy_loads_dsrl_base_from_stable_fd_and_rejects_path_swap(tmp_path, monkeypatch):
    weight_path = tmp_path / "model.safetensors"
    original_bytes = b"original checkpoint"
    replacement_bytes = b"replacement checkpoint"
    weight_path.write_bytes(original_bytes)
    replacement_path = tmp_path / "replacement.safetensors"
    replacement_path.write_bytes(replacement_bytes)
    original_backup = tmp_path / "original.safetensors"
    loaded_model = _FakeLoadedPI0()
    train_config = _fake_train_config(tmp_path, loaded_model)
    observed = []

    def load_pytorch(train_config, path):
        del train_config
        os.replace(weight_path, original_backup)
        os.replace(replacement_path, weight_path)
        observed.append(Path(path).read_bytes())
        return loaded_model

    train_config.model.load_pytorch = load_pytorch
    monkeypatch.setattr(
        tabero_dsrl_policy.TaberoDSRLActor,
        "from_bundle",
        classmethod(lambda cls, *args, **kwargs: SimpleNamespace(noise=lambda obs: None)),
    )

    with pytest.raises(ValueError, match="base checkpoint changed during load"):
        policy_config.create_trained_policy(
            train_config,
            tmp_path,
            norm_stats={},
            pytorch_device="cpu",
            dsrl_bundle_path="dsrl",
        )

    assert observed == [original_bytes]


def test_create_trained_policy_detects_in_place_base_checkpoint_aba(tmp_path, monkeypatch):
    weight_path = tmp_path / "model.safetensors"
    original_bytes = b"original checkpoint"
    weight_path.write_bytes(original_bytes)
    loaded_model = _FakeLoadedPI0()
    train_config = _fake_train_config(tmp_path, loaded_model)

    def load_pytorch(train_config, path):
        del train_config
        assert Path(path).read_bytes() == original_bytes
        weight_path.write_bytes(b"temporary replacement checkpoint")
        weight_path.write_bytes(original_bytes)
        return loaded_model

    train_config.model.load_pytorch = load_pytorch
    monkeypatch.setattr(
        policy_config,
        "_checkpoint_stat_fingerprint",
        lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, 0, 0),
    )
    monkeypatch.setattr(
        tabero_dsrl_policy.TaberoDSRLActor,
        "from_bundle",
        classmethod(lambda cls, *args, **kwargs: SimpleNamespace(noise=lambda obs: None)),
    )

    with pytest.raises(ValueError, match="base checkpoint changed during load"):
        policy_config.create_trained_policy(
            train_config,
            tmp_path,
            norm_stats={},
            pytorch_device="cpu",
            dsrl_bundle_path="dsrl",
        )


def test_create_trained_policy_rejects_rlt_and_dsrl_together(tmp_path, monkeypatch):
    monkeypatch.setattr(
        policy_config.download,
        "maybe_download",
        lambda path: (_ for _ in ()).throw(AssertionError("must reject before loading")),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        policy_config.create_trained_policy(
            _fake_train_config(tmp_path),
            tmp_path,
            norm_stats={},
            rlt_bundle_path="rlt",
            dsrl_bundle_path="dsrl",
        )


def test_create_trained_policy_rejects_dsrl_denoise_step_override(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").touch()
    monkeypatch.setattr(
        tabero_dsrl_policy.TaberoDSRLActor,
        "from_bundle",
        classmethod(lambda cls, *args, **kwargs: SimpleNamespace(noise=lambda obs: None)),
    )

    with pytest.raises(ValueError, match="num_steps.*10"):
        policy_config.create_trained_policy(
            _fake_train_config(tmp_path),
            tmp_path,
            norm_stats={},
            pytorch_device="cpu",
            sample_kwargs={"num_steps": 5},
            dsrl_bundle_path="dsrl",
        )


def test_create_trained_policy_rejects_model_safetensors_directory(tmp_path):
    (tmp_path / "model.safetensors").mkdir()

    with pytest.raises(ValueError, match="requires an explicit PyTorch.*model.safetensors"):
        policy_config.create_trained_policy(
            _fake_train_config(tmp_path),
            tmp_path,
            norm_stats={},
            dsrl_bundle_path="dsrl",
        )


@pytest.mark.parametrize("bundle_kwargs", [{"rlt_bundle_path": "rlt"}, {"dsrl_bundle_path": "dsrl"}])
def test_create_trained_policy_bundle_requires_explicit_pytorch_model_safetensors(tmp_path, bundle_kwargs):
    with pytest.raises(ValueError, match="requires an explicit PyTorch.*model.safetensors"):
        policy_config.create_trained_policy(
            _fake_train_config(tmp_path),
            tmp_path,
            norm_stats={},
            **bundle_kwargs,
        )
