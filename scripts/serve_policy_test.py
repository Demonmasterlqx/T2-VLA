from scripts import serve_policy


def test_checkpoint_policy_forwards_rlt_bundle(monkeypatch):
    sentinel = object()
    calls = []

    monkeypatch.setattr(serve_policy._config, "get_config", lambda name: f"config:{name}")  # noqa: SLF001

    def create_trained_policy(config, checkpoint_dir, **kwargs):
        calls.append((config, checkpoint_dir, kwargs))
        return sentinel

    monkeypatch.setattr(
        serve_policy._policy_config,  # noqa: SLF001
        "create_trained_policy",
        create_trained_policy,
    )
    args = serve_policy.Args(
        default_prompt="test prompt",
        rlt_bundle="/tmp/rlt-bundle",
        policy=serve_policy.Checkpoint(config="pi0_lora_tacfield_tabero", dir="/tmp/base"),
    )

    result = serve_policy.create_policy(args)

    assert result is sentinel
    assert calls == [
        (
            "config:pi0_lora_tacfield_tabero",
            "/tmp/base",
            {
                "default_prompt": "test prompt",
                "rlt_bundle_path": "/tmp/rlt-bundle",
            },
        )
    ]
