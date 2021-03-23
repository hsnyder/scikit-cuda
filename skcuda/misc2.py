#!/usr/bin/env python

"""
Miscellaneous PyCUDA functions.
"""


from __future__ import absolute_import, division

import atexit
import numbers
from string import Template

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.elementwise as elementwise
import pycuda.reduction as reduction
import pycuda.scan as scan
import pycuda.tools as tools
from pycuda.tools import context_dependent_memoize, dtype_to_ctype
from pycuda.compiler import SourceModule
from pytools import memoize
import numpy as np

from . import cuda

import sys
if sys.version_info < (3,):
    range = xrange

isdoubletype = lambda x : True if x == np.float64 or \
               x == np.complex128 else False
isdoubletype.__doc__ = """
Check whether a type has double precision.

Parameters
----------
t : numpy float type
    Type to test.

Returns
-------
result : bool
    Result.
"""

def get_compute_capability(dev):
    """
    Get the compute capability of the specified device.

    Retrieve the compute capability of the specified CUDA device and
    return it as a floating point value.

    Parameters
    ----------
    d : pycuda.driver.Device
        Device object to examine.

    Returns
    -------
    c : float
        Compute capability.
    """

    return np.float('.'.join([str(i) for i in
                                 dev.compute_capability()]))

def get_current_device():
    """
    Get the device in use by the current context.

    Returns
    -------
    d : pycuda.driver.Device
        Device in use by current context.
    """

    return drv.Device(cuda.cudaGetDevice())


