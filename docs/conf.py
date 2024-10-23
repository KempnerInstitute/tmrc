import os
import sys
sys.path.insert(0, os.path.abspath('../src'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TMRC'
copyright = '2024, Research and Engineering at Kempner'
author = 'Research and Engineering at Kempner'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',     # Automatically document from docstrings
    'sphinx.ext.napoleon',    # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',    # Add links to highlighted source code
    'sphinx.ext.autosummary', # Generate summary tables for modules/classes
    'myst_parser',            # Support for Markdown files (if using Markdown)
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # Use the Read the Docs theme

html_static_path = ['_static']