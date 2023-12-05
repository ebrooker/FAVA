"""Top-level package for fava."""

__author__ = """Ezra Brooker"""
__email__ = 'ebrooker@fsu.edu'
__version__ = '0.1.0'

__all__ = ["load", "cross_correlation"]

from pathlib import Path
from typing import List, Optional

from fava.temporary import Temporary
from .analysis import *
from .mesh import *
from .plot import *

def load_mesh(filename: str | Path, fields: Optional[List[str]]=None):
    return Temporary.load_mesh(filename, fields)

def load(filename: str | Path):
    
    obj = Temporary()
    obj.load(filename)
    return obj
