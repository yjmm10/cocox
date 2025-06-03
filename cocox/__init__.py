"""Top-level package for COCOX."""

__author__ = """liferecords"""
__email__ = 'yjmm10@yeah.net'
__version__ = '0.1.0'


from cocox.base import COCO
from cocox.utils import CCX
from cocox.cocox import COCOX
from cocox.utils.callback import *

__all__ = ['COCO', 'CCX', 'COCOX', 'STATIC_DATA']