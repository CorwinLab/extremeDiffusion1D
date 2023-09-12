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

__all__ = [
    "FirstPassagePDF", 
    "pyFirstPassageNumba", 
    "DiffusionPDF", 
    "DiffusionTimeCDF", 
    "DiffusionPositionPDF", 
    "fileIO", 
    "quadMath", 
    "FirstPassageDriver", 
    "FirstPassageEvolve",
    "pydiffusion2D",
    "pyDiffusionND", 
    "pycontinuous1D"]

from .pydiffusionCDF import DiffusionPositionCDF, DiffusionTimeCDF
from .pydiffusionPDF import DiffusionPDF
from .pyfirstPassagePDF import FirstPassagePDF
from .pyfirstPassageDriver import FirstPassageDriver
from .pyfirstPassageEvolve import FirstPassageEvolve
from .pyscatteringC import ScatteringModel
from . import fileIO
from . import quadMath
from . import pyfirstPassageNumba
from . import pydiffusion2D
from . import pydiffusionND
from . import pyscattering
from . import pymultijumpRW
from . import pycontinuous1D