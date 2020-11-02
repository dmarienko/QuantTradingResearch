"""
  Just simple starter script to injecting local libraries into notebook.
"""
import os, sys, os.path as path

if len(sys.argv) > 1:
    project = sys.argv[1]
else:
    raise ValueError("Project name must be specified !")

c_path = os.getcwd()
if project in c_path:
    __project_path = path.join(path.abspath(c_path[:c_path.find(project)]), project)
    sys.path.insert(0, __project_path)
    del c_path, project 
else:
    raise ValueError(f"Can't find path for specified project '{project}'")
    
from tools.charting.plot_helpers import *
from tools.analysis.timeseries import *
from tools.analysis.data import make_forward_returns_matrix, permutate_params, retain_columns_and_join
from tools.analysis.tools import (
    scols, srows, drop_duplicated_indexes, apply_to_frame, ohlc_resample, roll
)
from tools.utils.utils import mstruct, red, green, yellow, blue, magenta, cyan, white, dict2struct

from tqdm.notebook import tqdm

import seaborn as sns
import pandas as dp
import numpy as np

# if second agrument specified we use it as theme name
if len(sys.argv) > 2:
    setup_mpl_theme(sys.argv[2])