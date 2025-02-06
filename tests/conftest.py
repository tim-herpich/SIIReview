# tests/conftest.py
import sys
import os

# Compute the absolute path to the project root (one directory up)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
