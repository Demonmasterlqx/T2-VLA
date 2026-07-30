"""Tabero DSRL-SAC actor inference for PyTorch PI0 serving."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from safetensors.torch import load
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

DSRL_ACTOR_MANIFEST_V1: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
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
)
DSRL_BUNDLE_MANIFEST_KEYS_V1 = frozenset(
    {
        "format",
        "format_version",
        "algorithm",
        "task_id",
        "global_step",
        "is_final",
        "training_config",
        "base_model",
        "base_model_sha256",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_provenance",
        "source_provenance_sha256",
        "source_config_snapshot",
        "source_config_snapshot_sha256",
        "legacy_source_config",
        "legacy_source_config_sha256",
        "source_git_commit",
        "actor_weights",
        "actor_weights_sha256",
        "actor_manifest_version",
        "actor_tensor_count",
        "actor_parameter_count",
        "actor_dtype",
        "observation_contract",
        "feature_contract",
        "noise_contract",
        "architecture",
        "artifact_audit",
    }
)
DSRL_ARTIFACT_AUDIT_KEYS_V1 = frozenset(
    {
        "format",
        "format_version",
        "status",
        "task_id",
        "global_step",
        "source_checkpoint_sha256",
        "base_model_sha256",
        "actor_weights_sha256",
        "manifest_sha256",
        "checks",
    }
)
DSRL_ARTIFACT_AUDIT_CHECKS_V1 = frozenset(
    {
        "final_checkpoint_path",
        "source_metadata",
        "source_trainable_manifest",
        "actor_manifest",
        "actor_dtype",
        "actor_finite",
        "base_model_sha256",
        "formal_provenance",
        "output_hashes",
    }
)
DSRL_TRAINING_IDENTITIES_V1 = frozenset(
    {
        (
            0,
            50,
            "isaaclab_pi0_dsrl_tacfield_tabero_task0_firm_8gpu_50step",
            True,
        ),
        (
            5,
            50,
            "isaaclab_pi0_dsrl_tacfield_tabero_task5_firm_8gpu_50step",
            True,
        ),
        (
            5,
            40,
            "isaaclab_pi0_dsrl_tacfield_tabero_task5_firm_4gpu_40step_small",
            True,
        ),
        (
            0,
            10,
            "isaaclab_pi0_dsrl_tacfield_tabero_task0_firm_8gpu_50step",
            False,
        ),
    }
)
DSRL_ACTOR_METADATA_KEYS_V1 = frozenset({"format", "format_version", "task_id", "global_step", "dtype"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_artifact(path: Path, label: str) -> tuple[bytes, str]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Tabero DSRL {label} could not be captured: {path}") from error
    return content, hashlib.sha256(content).hexdigest()


def _require_artifact_unchanged(path: Path, expected_hash: str, label: str) -> None:
    try:
        actual_hash = _sha256(path)
    except OSError as error:
        raise ValueError(f"Tabero DSRL {label} changed during bundle load: {path}") from error
    if actual_hash != expected_hash:
        raise ValueError(
            f"Tabero DSRL {label} changed during bundle load: expected {expected_hash}, got {actual_hash}."
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 hex digest.")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{label} must be a SHA-256 hex digest.")
    return normalized


def _parse_json_object(content: bytes, path: Path, label: str) -> dict[str, Any]:
    try:
        values = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Tabero DSRL {label} is not valid JSON: {path}") from error
    if not isinstance(values, dict):
        raise ValueError(f"Tabero DSRL {label} must be a JSON object.")
    return values


def _parse_safetensors_metadata(content: bytes, path: Path) -> dict[str, str]:
    if len(content) < 8:
        raise ValueError(f"Tabero DSRL actor weights have an invalid safetensors header: {path}")
    header_length = int.from_bytes(content[:8], byteorder="little", signed=False)
    header_end = 8 + header_length
    if header_length == 0 or header_end > len(content):
        raise ValueError(f"Tabero DSRL actor weights have an invalid safetensors header: {path}")
    try:
        header = json.loads(content[8:header_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Tabero DSRL actor weights have an invalid safetensors header: {path}") from error
    metadata = header.get("__metadata__") if isinstance(header, dict) else None
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError("Tabero DSRL actor metadata must be a string mapping.")
    return metadata


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], expected_value) for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _require_exact_contract(values: Mapping[str, Any], key: str, expected: Any) -> None:
    if not _strict_json_equal(values.get(key), expected):
        raise ValueError(f"Tabero DSRL manifest {key} does not match the version-1 inference contract.")


@dataclasses.dataclass(frozen=True)
class TaberoDSRLManifest:
    format: str
    format_version: int
    algorithm: str
    task_id: int
    global_step: int
    is_final: bool
    training_config: str
    base_model_sha256: str
    source_checkpoint_sha256: str
    actor_weights: str
    actor_weights_sha256: str
    artifact_audit: str
    values: Mapping[str, Any] = dataclasses.field(repr=False)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TaberoDSRLManifest:
        actual_keys = set(values)
        if actual_keys != DSRL_BUNDLE_MANIFEST_KEYS_V1:
            raise ValueError(
                "Tabero DSRL manifest keyspace mismatch; "
                f"missing={sorted(DSRL_BUNDLE_MANIFEST_KEYS_V1 - actual_keys)}; "
                f"unexpected={sorted(actual_keys - DSRL_BUNDLE_MANIFEST_KEYS_V1)}."
            )
        if values.get("format") != "tabero_dsrl_t2vla":
            raise ValueError(f"Unsupported Tabero DSRL bundle format: {values.get('format')!r}.")
        if type(values.get("format_version")) is not int or values["format_version"] != 1:
            raise ValueError(f"Unsupported Tabero DSRL bundle version: {values.get('format_version')!r}.")
        if values.get("algorithm") != "dsrl-sac":
            raise ValueError("Tabero DSRL bundle algorithm must be 'dsrl-sac'.")
        task_id = values.get("task_id")
        if type(task_id) is not int or task_id not in {0, 5}:
            raise ValueError(f"Tabero DSRL bundle task_id must be Task 0 or 5; got {task_id!r}.")
        global_step = values.get("global_step")
        if type(global_step) is not int:
            raise ValueError(f"Tabero DSRL bundle global_step must be an integer; got {global_step!r}.")
        is_final = values.get("is_final")
        if type(is_final) is not bool:
            raise ValueError(f"Tabero DSRL bundle is_final must be a boolean; got {is_final!r}.")
        training_config = values.get("training_config")
        if not isinstance(training_config, str) or not training_config:
            raise ValueError("Tabero DSRL bundle training_config must be a non-empty string.")
        training_identity = (task_id, global_step, training_config, is_final)
        if training_identity not in DSRL_TRAINING_IDENTITIES_V1:
            final_identity = (task_id, global_step, training_config, True)
            if is_final is False and final_identity in DSRL_TRAINING_IDENTITIES_V1:
                raise ValueError("Tabero DSRL bundle is_final must be true for this training identity.")
            raise ValueError(
                "Tabero DSRL bundle training identity is not allowlisted: "
                f"task_id={task_id!r}, global_step={global_step!r}, "
                f"training_config={training_config!r}, is_final={is_final!r}."
            )
        for key, expected in {
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
            "feature_contract": {
                "order": ["state", "image", "tactile"],
                "dims": [64, 64, 64],
                "total_dim": 192,
            },
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
        }.items():
            _require_exact_contract(values, key, expected)

        required_strings = (
            "training_config",
            "base_model",
            "source_checkpoint",
            "source_provenance",
            "source_config_snapshot",
            "source_git_commit",
            "actor_weights",
            "artifact_audit",
        )
        for key in required_strings:
            if not isinstance(values.get(key), str) or not values[key]:
                raise ValueError(f"Tabero DSRL manifest {key} must be a non-empty string.")
        for key in (
            "source_provenance_sha256",
            "source_config_snapshot_sha256",
        ):
            _require_sha256(values[key], key)
        legacy_source = values["legacy_source_config"]
        legacy_source_hash = values["legacy_source_config_sha256"]
        if legacy_source is None:
            if legacy_source_hash is not None:
                raise ValueError("legacy_source_config_sha256 must be null when legacy_source_config is null.")
        elif not isinstance(legacy_source, str) or not legacy_source:
            raise ValueError("legacy_source_config must be null or a non-empty string.")
        else:
            _require_sha256(legacy_source_hash, "legacy_source_config_sha256")
        return cls(
            format=values["format"],
            format_version=values["format_version"],
            algorithm=values["algorithm"],
            task_id=task_id,
            global_step=global_step,
            is_final=values["is_final"],
            training_config=values["training_config"],
            base_model_sha256=_require_sha256(values.get("base_model_sha256"), "base_model_sha256"),
            source_checkpoint_sha256=_require_sha256(
                values.get("source_checkpoint_sha256"), "source_checkpoint_sha256"
            ),
            actor_weights=values["actor_weights"],
            actor_weights_sha256=_require_sha256(values.get("actor_weights_sha256"), "actor_weights_sha256"),
            artifact_audit=values["artifact_audit"],
            values=MappingProxyType(dict(values)),
        )


def _bundle_filename(root: Path, filename: str, label: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != filename:
        raise ValueError(f"Tabero DSRL {label} must be a filename inside the bundle root.")
    return root / relative


def _validate_actor_state(state: Mapping[str, torch.Tensor]) -> None:
    expected_keys = set(DSRL_ACTOR_MANIFEST_V1)
    actual_keys = set(state)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    shapes = {
        key: {"expected": DSRL_ACTOR_MANIFEST_V1[key], "actual": tuple(state[key].shape)}
        for key in expected_keys & actual_keys
        if tuple(state[key].shape) != DSRL_ACTOR_MANIFEST_V1[key]
    }
    if missing or unexpected or shapes:
        raise ValueError(
            "Tabero DSRL actor manifest mismatch; "
            f"missing={missing}; unexpected={unexpected}; shape_mismatches={shapes}."
        )
    wrong_dtype = {key: str(value.dtype) for key, value in state.items() if value.dtype != torch.bfloat16}
    if wrong_dtype:
        raise ValueError(f"Tabero DSRL actor tensors must all use bfloat16; got {wrong_dtype}.")
    nonfinite = sorted(key for key, value in state.items() if not torch.isfinite(value).all().item())
    if nonfinite:
        raise ValueError(f"Tabero DSRL actor tensors must all be finite; keys={nonfinite}.")


@dataclasses.dataclass(frozen=True)
class TaberoDSRLBundle:
    root: Path
    manifest: TaberoDSRLManifest
    actor_state: Mapping[str, torch.Tensor] = dataclasses.field(repr=False)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        base_checkpoint_dir: str | Path,
        base_model_sha256: str | None = None,
    ) -> TaberoDSRLBundle:
        root = Path(path).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Tabero DSRL bundle directory not found: {root}")
        manifest_path = root / "manifest.json"
        manifest_content, manifest_hash = _capture_artifact(manifest_path, "manifest")
        manifest_values = _parse_json_object(manifest_content, manifest_path, "manifest")
        manifest = TaberoDSRLManifest.from_dict(manifest_values)

        base_dir = Path(base_checkpoint_dir).expanduser()
        if not base_dir.is_dir():
            raise ValueError(f"Tabero DSRL base checkpoint directory is required; got {base_dir}.")
        base_weights = base_dir / "model.safetensors"
        if not base_weights.is_file():
            raise FileNotFoundError(f"Tabero DSRL base checkpoint model.safetensors not found: {base_weights}")
        actual_base_hash = (
            _sha256(base_weights)
            if base_model_sha256 is None
            else _require_sha256(base_model_sha256, "base_model_sha256")
        )
        if actual_base_hash != manifest.base_model_sha256:
            raise ValueError(
                "Tabero DSRL base checkpoint SHA-256 mismatch: "
                f"expected {manifest.base_model_sha256}, got {actual_base_hash}."
            )

        actor_path = _bundle_filename(root, manifest.actor_weights, "actor_weights")
        actor_files = sorted(root.glob("*.safetensors"))
        if actor_files != [actor_path]:
            raise ValueError(
                "Tabero DSRL bundle must contain exactly one actor safetensors file; "
                f"got {[candidate.name for candidate in actor_files]}."
            )
        if not actor_path.is_file():
            raise FileNotFoundError(f"Tabero DSRL actor weights not found: {actor_path}")
        actor_content, actual_actor_hash = _capture_artifact(actor_path, "actor weights")
        if actual_actor_hash != manifest.actor_weights_sha256:
            raise ValueError(
                "Tabero DSRL actor weights SHA-256 mismatch: "
                f"expected {manifest.actor_weights_sha256}, got {actual_actor_hash}."
            )
        actor_metadata = _parse_safetensors_metadata(actor_content, actor_path)
        expected_actor_metadata = {
            "format": "tabero_dsrl_t2vla",
            "format_version": "1",
            "task_id": str(manifest.task_id),
            "global_step": str(manifest.global_step),
            "dtype": "bfloat16",
        }
        if set(actor_metadata) != DSRL_ACTOR_METADATA_KEYS_V1:
            raise ValueError(
                "Tabero DSRL actor metadata keyspace mismatch; "
                f"missing={sorted(DSRL_ACTOR_METADATA_KEYS_V1 - set(actor_metadata))}; "
                f"unexpected={sorted(set(actor_metadata) - DSRL_ACTOR_METADATA_KEYS_V1)}."
            )
        for key, expected in expected_actor_metadata.items():
            if actor_metadata[key] != expected:
                raise ValueError(
                    f"Tabero DSRL actor metadata {key} mismatch: expected {expected!r}, got {actor_metadata[key]!r}."
                )

        audit_path = _bundle_filename(root, manifest.artifact_audit, "artifact_audit")
        audit_content, audit_hash = _capture_artifact(audit_path, "artifact audit")
        audit = _parse_json_object(audit_content, audit_path, "artifact audit")
        actual_audit_keys = set(audit)
        if actual_audit_keys != DSRL_ARTIFACT_AUDIT_KEYS_V1:
            raise ValueError(
                "Tabero DSRL artifact audit keyspace mismatch; "
                f"missing={sorted(DSRL_ARTIFACT_AUDIT_KEYS_V1 - actual_audit_keys)}; "
                f"unexpected={sorted(actual_audit_keys - DSRL_ARTIFACT_AUDIT_KEYS_V1)}."
            )
        if audit.get("format") != "tabero_dsrl_artifact_audit":
            raise ValueError("Tabero DSRL artifact audit format is unsupported.")
        if type(audit.get("format_version")) is not int or audit["format_version"] != 1:
            raise ValueError("Tabero DSRL artifact audit format_version must be integer 1.")
        if audit.get("status") != "passed":
            raise ValueError(f"Tabero DSRL artifact audit status must be passed; got {audit.get('status')!r}.")
        for key in ("task_id", "global_step"):
            if type(audit.get(key)) is not int:
                raise ValueError(f"Tabero DSRL artifact audit {key} must be an integer.")
        audit_matches = {
            "task_id": manifest.task_id,
            "global_step": manifest.global_step,
            "source_checkpoint_sha256": manifest.source_checkpoint_sha256,
            "base_model_sha256": manifest.base_model_sha256,
            "actor_weights_sha256": manifest.actor_weights_sha256,
            "manifest_sha256": manifest_hash,
        }
        for key, expected in audit_matches.items():
            if audit.get(key) != expected:
                raise ValueError(
                    f"Tabero DSRL artifact audit {key} mismatch: expected {expected!r}, got {audit.get(key)!r}."
                )
        checks = audit.get("checks")
        if not isinstance(checks, dict) or set(checks) != DSRL_ARTIFACT_AUDIT_CHECKS_V1:
            actual_check_keys = set(checks) if isinstance(checks, dict) else set()
            raise ValueError(
                "Tabero DSRL artifact audit checks keyspace mismatch; "
                f"missing={sorted(DSRL_ARTIFACT_AUDIT_CHECKS_V1 - actual_check_keys)}; "
                f"unexpected={sorted(actual_check_keys - DSRL_ARTIFACT_AUDIT_CHECKS_V1)}."
            )
        if any(value is not True for value in checks.values()):
            raise ValueError("Tabero DSRL artifact audit checks must all be true.")

        try:
            actor_state = load(actor_content)
        except Exception as error:
            raise ValueError(f"Tabero DSRL actor weights could not be loaded: {actor_path}") from error
        _validate_actor_state(actor_state)
        _require_artifact_unchanged(manifest_path, manifest_hash, "manifest")
        _require_artifact_unchanged(audit_path, audit_hash, "artifact audit")
        _require_artifact_unchanged(actor_path, actual_actor_hash, "actor weights")
        _require_artifact_unchanged(base_weights, actual_base_hash, "base checkpoint")
        return cls(root=root, manifest=manifest, actor_state=MappingProxyType(actor_state))

    def build_actor(self, device: str | torch.device = "cpu") -> TaberoDSRLActor:
        actor = TaberoDSRLActor()
        actor.load_state_dict(dict(self.actor_state), strict=True)
        actor.to(device=device)
        actor.eval()
        return actor


class _GaussianPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(128, 32)
        self.log_std_layer = nn.Linear(128, 32)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared_net(features)
        return self.mean_layer(hidden), self.log_std_layer(hidden)


class _ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(nn.Flatten(), nn.Linear(32768, 64), nn.LayerNorm(64), nn.Tanh())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_images, channels, height, width = images.shape
        encoded = self.encoder(images.reshape(batch_size, num_images * channels, height, width))
        return self.bottleneck(encoded)


class _StateEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(7, 64), nn.LayerNorm(64), nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)


class _TactileTCNBlock(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(nn.Linear(input_dim, 64) for _ in range(3))
        self.residual_proj = nn.Linear(input_dim, 64) if input_dim != 64 else None

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        batch_size, steps, input_dim = tactile.shape
        padding = torch.zeros(batch_size, 2, input_dim, dtype=tactile.dtype, device=tactile.device)
        padded = torch.cat([padding, tactile], dim=1)
        result = None
        for index, kernel in enumerate(self.kernels):
            start = 2 - index
            projected = kernel(padded[:, start : start + steps, :])
            result = projected if result is None else result + projected
        residual = tactile if self.residual_proj is None else self.residual_proj(tactile)
        return F.silu(result + residual)


class _TactileEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TactileTCNBlock(396), _TactileTCNBlock(64)])
        self.out_proj = nn.Linear(64, 64)

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        hidden = tactile[:, :9, :]
        for block in self.blocks:
            hidden = block(hidden)
        return self.out_proj(hidden[:, -1, :])


class TaberoDSRLActor(nn.Module):
    """Exact BF16 actor architecture exported by RLinf DSRL-SAC."""

    def __init__(self) -> None:
        super().__init__()
        self.dsrl_action_noise_net = _GaussianPolicy()
        self.actor_image_encoder = _ImageEncoder()
        self.actor_state_encoder = _StateEncoder()
        self.actor_tactile_encoder = _TactileEncoder()
        self.to(dtype=torch.bfloat16)

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        *,
        base_checkpoint_dir: str | Path,
        base_model_sha256: str | None = None,
        device: str | torch.device = "cpu",
    ) -> TaberoDSRLActor:
        return TaberoDSRLBundle.load(
            bundle_path,
            base_checkpoint_dir=base_checkpoint_dir,
            base_model_sha256=base_model_sha256,
        ).build_actor(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def preprocess(
        self,
        observation: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = ("dsrl_raw_image", "state", "tactile_marker_motion")
        missing = [key for key in required if key not in observation]
        if missing:
            raise KeyError(f"Tabero DSRL raw observation is missing required keys: {missing}.")

        image = torch.as_tensor(observation["dsrl_raw_image"])
        if tuple(image.shape) != (256, 256, 3):
            raise ValueError(f"Tabero DSRL dsrl_raw_image expected HWC shape (256, 256, 3); got {tuple(image.shape)}.")
        if image.dtype != torch.uint8:
            raise ValueError(f"Tabero DSRL dsrl_raw_image must use uint8; got {image.dtype}.")

        state = torch.as_tensor(observation["state"])
        if tuple(state.shape) != (7,):
            raise ValueError(f"Tabero DSRL state expected shape (7,); got {tuple(state.shape)}.")
        if state.dtype != torch.float32:
            raise ValueError(f"Tabero DSRL state must use float32; got {state.dtype}.")

        tactile = torch.as_tensor(observation["tactile_marker_motion"])
        if tuple(tactile.shape) != (9, 198, 2):
            raise ValueError(
                f"Tabero DSRL tactile_marker_motion expected shape (9, 198, 2); got {tuple(tactile.shape)}."
            )
        if tactile.dtype != torch.float32:
            raise ValueError(f"Tabero DSRL tactile_marker_motion must use float32; got {tactile.dtype}.")

        image = image.to(device=self.device, dtype=torch.float32)
        image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
        image = F.interpolate(image, size=(64, 64), mode="bilinear", align_corners=False)
        image = (image * 2.0 - 1.0).unsqueeze(1).to(dtype=torch.bfloat16)
        state = state.to(device=self.device, dtype=torch.bfloat16).unsqueeze(0)
        tactile = tactile.to(device=self.device, dtype=torch.bfloat16).reshape(1, 9, 396)
        return image, state, tactile

    @torch.no_grad()
    def features(self, observation: Mapping[str, Any]) -> torch.Tensor:
        image, state, tactile = self.preprocess(observation)
        return torch.cat(
            [
                self.actor_state_encoder(state),
                self.actor_image_encoder(image),
                self.actor_tactile_encoder(tactile),
            ],
            dim=-1,
        )

    @torch.no_grad()
    def mean(self, observation: Mapping[str, Any]) -> torch.Tensor:
        mean, _ = self.dsrl_action_noise_net(self.features(observation))
        return mean

    @torch.no_grad()
    def noise(self, observation: Mapping[str, Any]) -> torch.Tensor:
        deterministic = torch.tanh(self.mean(observation))
        return deterministic[:, None, :].expand(-1, 50, -1)

    def forward(self, observation: Mapping[str, Any]) -> torch.Tensor:
        return self.noise(observation)


class TaberoDSRLPolicy:
    """Generate DSRL noise from raw inputs before delegating to normal Policy transforms."""

    def __init__(self, base_policy: Any, actor: TaberoDSRLActor) -> None:
        self._base_policy = base_policy
        self._actor = actor

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        noise = self._actor.noise(obs)
        return self._base_policy.infer(obs, noise=noise)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._base_policy.metadata
