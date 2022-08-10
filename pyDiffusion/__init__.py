""" 
This scrip generates the namespace for libDiffusion
"""

try:
    import npquad
except ImportError as ie:
    raise type(ie)(
        str(ie) + ", npquad is available at "
        "http://www.github.com/SimonsGlass/numpy_quad"
        )

__all__ = ["FirstPassagePDF", "DiffusionPDF", "DiffusionTimeCDF", "DiffusionPositionPDF", "fileIO", "quadMath", "FirstPassageDriver"]

from .pydiffusionCDF import DiffusionPositionCDF, DiffusionTimeCDF
from .pydiffusionPDF import DiffusionPDF
from .pyfirstPassagePDF import FirstPassagePDF
from .pyfirstPassageDriver import FirstPassageDriver
from . import fileIO
from . import quadMath