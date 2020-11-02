import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools.charting.plot_helpers import sbp

import scipy.stats as stats
import seaborn as sns


def cmp_to_norm(xs, xranges=None):
    """
    Compare distribution from xs against normal using estimated mean and std
    """
    _m, _s = np.mean(xs), np.std(xs)
    fit = stats.norm.pdf(sorted(xs), _m, _s)  #this is a fitting indeed

    sbp(12,1)
    plt.plot(sorted(xs), fit, 'r--', lw=2, label='N(%.2f, %.2f)' % (_m, _s))
    plt.legend(loc='upper right')
    
    sns.kdeplot(xs, color='g', label='Data', shade=True)
    if xranges is not None and len(xranges) > 1: 
        plt.xlim(xranges)
    plt.legend(loc='upper right')

    sbp(12,2) 
    stats.probplot(xs, dist="norm", sparams=(_m, _s), plot=plt)
