"""
Misc utility funcs
"""

import pandas as pd
import numpy as np

import os, argparse, yaml, pickle

def str2bool(v):
    """
    Converts strings to booleans (e.g., 't' -> True)
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')