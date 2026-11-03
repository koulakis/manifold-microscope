"""Sphinx configuration; building the guides does not import the package."""
from pathlib import Path
import re

project = "Manifold Microscope"
author = "Marios Koulakis and Constantin Seibold"
copyright = "2026, Marios Koulakis and Constantin Seibold"
version_match = re.search(
    r'^version = "([^"]+)"',
    (Path(__file__).parents[2] / "pyproject.toml").read_text(),
    re.MULTILINE
)
assert version_match is not None
release = version_match.group(1)

extensions = []
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = []
nitpicky = True
html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release}"
html_theme_options = {"navigation_depth": 3}
