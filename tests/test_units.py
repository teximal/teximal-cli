"""
Checks that need no model, on any machine:  python -m pytest tests -q
"""
import json, os, tempfile

import pytest

from teximal import Fort
from teximal.evaluate import ece, fit_temperature, macro_f1, soften
from teximal.hub import config, default_backend
from teximal.models.fort.api import _shown


def folder(**cfg):
    d = tempfile.mkdtemp()
    if cfg:
        json.dump(cfg, open(os.path.join(d, "teximal.json"), "w"))
    return d


def test_config_normalizes_the_family():
    cfg = config(folder(family="Fort-1", model="teximal/fort-1-2b", backends=["mlx", "torch"]))
    assert cfg["family"] == "fort" and cfg["backends"] == ["mlx", "torch"]


def test_older_folders_are_mlx_only():
    assert config(folder(family="fort"))["backends"] == ["mlx"]


def test_not_a_model_folder():
    with pytest.raises(FileNotFoundError):
        config(folder())


def test_mlx_only_build_points_to_the_full_repo():
    d = folder(family="fort", model="teximal/fort-1-2b-mlx-4bit", backends=["mlx"])
    with pytest.raises(ValueError, match="use teximal/fort-1-2b here"):
        Fort(d, backend="torch")


def test_backend_override():
    before = os.environ.get("TEXIMAL_BACKEND")
    os.environ["TEXIMAL_BACKEND"] = "torch"
    try:
        assert default_backend() == "torch"
    finally:
        if before is None:
            del os.environ["TEXIMAL_BACKEND"]
        else:
            os.environ["TEXIMAL_BACKEND"] = before


def test_options_as_shown():
    assert _shown({"sci_tech": "science and technology", "sports": ""}) == ["sci tech: science and technology", "sports"]


def test_calibration_helpers():
    assert ece([1.0, 1.0], [True, True]) == 0
    assert abs(sum(soften({"a": 0.9, "b": 0.1}, 2.0).values()) - 1) < 1e-9
    assert macro_f1(["a", "b"], ["a", "b"], ["a", "b"]) == 1.0
    sure_but_coin_flip = [({"a": 0.99, "b": 0.01}, "a" if i % 2 else "b") for i in range(20)]
    assert fit_temperature(sure_but_coin_flip) > 1          # overconfident: the fitted temperature softens


def test_fast_convolutions_match_transformers():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import transformers.models.qwen3_5.modeling_qwen3_5 as Q
    from teximal.models.fort.engine_torch import _fast_convs
    _fast_convs()                    # patches only when its probe matches transformers' own functions
    assert getattr(Q.causal_conv1d_fn, "teximal", False) and getattr(Q.causal_conv1d_update, "teximal", False)


def test_cache_states_of_any_layout():
    mx = pytest.importorskip("mlx.core")
    from teximal.models.fort.engine import _copy, _repeat
    a = mx.ones((1, 2, 3))
    for st in ([a, None], (a, a), ([a, a], None, None), (a, a, 7)):     # mlx-lm 0.31, then 0.32 (offsets, Nones)
        c, r = _copy(st), _repeat(st, 4)
        assert type(c) is type(st) and type(r) is type(st)
        flat = lambda s: [x for y in s for x in (y if isinstance(y, list) else [y])]
        for x, y, z in zip(flat(st), flat(c), flat(r)):
            if isinstance(x, mx.array):
                assert y.shape == x.shape and z.shape == (4,) + x.shape[1:]
            else:
                assert x == y == z
