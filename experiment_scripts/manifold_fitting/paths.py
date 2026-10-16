"""Discover native experiment directories, including their repetition field."""
from pathlib import Path


def model_directories(root: Path, model_type: str = "beta_vae"):
    """Yield (path, dataset, dimension) for actual six-field experiments."""
    for path in sorted(Path(root).iterdir()):
        fields = path.name.split("__")
        if path.is_dir() and len(fields) == 6 and fields[1] == model_type:
            yield path, fields[0], int(fields[3])
