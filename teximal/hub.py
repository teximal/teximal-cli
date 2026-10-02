"""
Finding Teximal models: a local folder, or a Hugging Face repo (downloaded once, then cached). Every Teximal
model folder carries a teximal.json saying which family it belongs to, how to run it, and which backends
(MLX, PyTorch) can.
"""
import json, os, platform, sys

FAMILIES = {}                                   # family name -> class, filled in by teximal.models


def default_backend():
    """MLX on Apple silicon when it is installed; PyTorch everywhere else. TEXIMAL_BACKEND overrides."""
    if os.environ.get("TEXIMAL_BACKEND"):
        return os.environ["TEXIMAL_BACKEND"]
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import mlx.core  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    return "torch"


def resolve(model):
    """A local folder, or a repo id ("teximal/fort-1-0.8b", or just "fort-1-0.8b"), as a local path."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model if "/" in model else f"teximal/{model}")


def config(path):
    for name in ("teximal.json", "fort.json"):    # fort.json: release candidates packed before 2026-10-02
        p = os.path.join(path, name)
        if os.path.exists(p):
            cfg = json.load(open(p))
            cfg["family"] = str(cfg.get("family", "fort")).lower().split("-")[0]    # "Fort-1" -> "fort"
            cfg.setdefault("backends", ["mlx"])  # packed before 2026-10-02: MLX layout only
            return cfg
    raise FileNotFoundError(f"{path} has no teximal.json: not a Teximal model folder")


def load(model, **kwargs):
    """The right class for any Teximal model, read from its teximal.json."""
    path = resolve(model)
    family = config(path)["family"]
    if family not in FAMILIES:
        raise ValueError(f"{model}: family {family!r} is not supported by this version of teximal")
    return FAMILIES[family](path, **kwargs)


def pull(model):
    return resolve(model)


def cached():
    """Teximal models in the local Hugging Face cache: (repo id, size in GB, path)."""
    from huggingface_hub import scan_cache_dir
    try:
        repos = scan_cache_dir().repos
    except Exception:                           # no cache yet
        return []
    return sorted((r.repo_id, r.size_on_disk / 1e9, str(r.repo_path)) for r in repos
                  if r.repo_type == "model" and r.repo_id.startswith("teximal/"))
