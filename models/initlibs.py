"""
  Just simple starter script to injecting local libraries into notebook.
"""
import sys, os.path as p; sys.path.insert(0, p.abspath('../'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tools.charting.plot_helpers import *

if len(sys.argv) > 1:
    setup_mpl_theme(sys.argv[1])