# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import mock
import os
import sys
#from dotenv import load_dotenv
"""
MOCK_MODULES = ["dlib","numpy","tensorflow","paramiko","pandas","matplotlib","imutils","matplotlib.pyplot","dotenv","'tensorflow.keras","cv2","scipy","sklearn","mtcnn","sklearn.model_selection","scipy.signal","matplotlib.figure"]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()
"""
autodoc_mock_imports = ["tensorboard","dlib","numpy","tensorflow","paramiko","pandas","matplotlib","imutils","matplotlib.pyplot","dotenv","'tensorflow.keras","cv2","scipy","sklearn","mtcnn","sklearn.model_selection","scipy.signal","matplotlib.figure"]

sys.path.insert(0, os.path.abspath('../..'))
#load_dotenv("env_file/.env_credentials")
#load_dotenv("env_file/.env_path")


# -- Project information -----------------------------------------------------

project = 'Intelligent Fatigue Detection System'
copyright = '2021, Lambert'
author = 'Lambert'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions =  ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']