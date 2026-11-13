import hashlib
import json
from pathlib import Path

import pytest

MODEL_HASHES = {
    "toto-2.0-22m": "9cd503d82df3aa71747862688f47a31c1d0a4b80f898df6e046189016eaa21dd",
    "toto-2.0-4m": "316660d5afb47943e531f39242e0b02ca0b8bb73be5709dfe07ca80dfce9805e",
}


@pytest.fixture(scope="module", params=tuple(MODEL_HASHES))
def toto_model_dir(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    name: object = request.param
    assert isinstance(name, str)
    weights = Path.home() / ".cache/promcast/models" / name / "model.safetensors"
    with weights.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == MODEL_HASHES[name]

    # Exact configuration fields from the section 6 revisions, recorded in ADR 0007.
    width, layers, heads, feedforward = (
        (512, 6, 8, 1368) if name.endswith("22m") else (256, 4, 4, 688)
    )
    config = {
        "attn_bias": True,
        "d_ff": feedforward,
        "d_model": width,
        "dropout_p": 0.0,
        "heads_per_group": 1,
        "layer_group_size": layers,
        "mlp_bias": False,
        "norm_eps": 0.0001,
        "norm_include_weight": False,
        "num_groups": heads,
        "num_heads": heads,
        "num_layers": layers,
        "num_output_patches": 1,
        "num_variate_layers_per_group": 1,
        "patch_size": 32,
        "per_dim_scale": True,
        "pre_norm": True,
        "qk_dim": 64,
        "qk_norm": False,
        "qk_norm_include_weight": False,
        "residual_attn_ratio": 5.136215466577748,
        "residual_mult": 0.75,
        "use_xpos": True,
        "v_dim": 64,
        "variate_layer_first": False,
    }
    local_dir = tmp_path_factory.mktemp(name)
    (local_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (local_dir / "model.safetensors").symlink_to(weights)
    return local_dir
