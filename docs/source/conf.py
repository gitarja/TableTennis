import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))
# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'TableTennis'
copyright = '2023, Prasetia Putra'
author = 'Prasetia Putra'
release = 'Beta 0.1'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}


intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
