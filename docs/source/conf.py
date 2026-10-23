"""Sphinx configuration; building the guides does not import the package."""
from pathlib import Path
import re

project = "Manifold Microscope"
author = "Marios Koulakis and Constantin Seibold"
copyright = "2026, Marios Koulakis and Constantin Seibold"
release = re.search(r'^version = "([^"]+)"',
                    (Path(__file__).parents[2] / "pyproject.toml").read_text(),
                    re.MULTILINE).group(1)

extensions = []
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = []
nitpicky = True
html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release}"
html_theme_options = {"navigation_depth": 3}
