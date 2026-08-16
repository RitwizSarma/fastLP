"""Sphinx configuration for the fastLP documentation."""

from fastlp import __version__

project = "fastLP"
author = "Ritwiz Sarma"
copyright = "2026, Ritwiz Sarma"
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = [
    "_build",
    "claude_response.md",
    "deep-research-report.md",
    "fastlp_implementation_plan.md",
    "fastlp_step_by_step_comparison.md",
    "main_thesis.md",
    "main_thesis_long.md",
    "next_steps.md",
    "pypi_checklist.md",
    "regression_estimation_handbook.md",
]

html_theme = "pydata_sphinx_theme"
html_title = f"fastLP {release}"
html_favicon = "_static/favicon.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_sidebars = {"**": []}
html_theme_options = {
    "github_url": "https://github.com/RitwizSarma/fastLP",
    "show_toc_level": 2,
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
