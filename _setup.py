"""Path bootstrap. Importing this module adds the project root to sys.path
and chdirs to it, so scripts in experiments/ and scripts/ can:
  - import src.eml_tree etc.
  - read 'results/foo.json' and write 'figures/foo.png' relative to root.
"""
import os, sys
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
