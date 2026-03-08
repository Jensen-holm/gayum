from . import dists
from . import terms
from .gam import GAM

__all__ = ['GAM', 'dists', 'terms']

def version() -> str:
    return '0.1.0'
