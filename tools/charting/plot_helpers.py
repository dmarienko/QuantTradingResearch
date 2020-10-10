"""
   Misc graphics handy utilitites to be used in interactive analysis
"""
import numpy as np
import pandas as pd
import itertools as it
from typing import List, Tuple, Union

try:
    import matplotlib
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except:
    print("Can't import matplotlib modules in charting modlue")

from tools.analysis.tools import isscalar
from tools.charting.mpl_finance import ohlc_plot


def setup_mpl_theme(theme='dark'):
    from cycler import cycler
    import matplotlib
    
    DARK_MPL_THEME = [
        ('backend', 'module://ipykernel.pylab.backend_inline'),
        ('interactive', True),
        ('lines.color', '#5050f0'),
        ('text.color', '#d0d0d0'),
        ('axes.facecolor', '#000000'),
        ('axes.edgecolor', '#404040'),
        ('axes.grid', True),
        ('axes.labelsize', 'large'),
        ('axes.labelcolor', 'green'),
        ('axes.prop_cycle', cycler('color', ['#449AcD', 'g', '#f62841', 'y', '#088487', '#E24A33', '#f01010'])),
        ('legend.fontsize', 'small'),
        ('legend.fancybox', False),
        ('legend.edgecolor', '#305030'),
        ('legend.shadow', False),
        ('lines.antialiased', True),
        ('lines.linewidth', 0.8), # reduced line width
        ('patch.linewidth', 0.5),
        ('patch.antialiased', True),
        ('xtick.color', '#909090'),
        ('ytick.color', '#909090'),
        ('xtick.labelsize', 'small'),
        ('ytick.labelsize', 'small'),
        ('grid.color', '#404040'),
        ('grid.linestyle', '--'),
        ('grid.linewidth', 0.5),
        ('grid.alpha', 0.8),
        ('figure.figsize', [8.0, 5.0]),
        ('figure.dpi', 80.0),
        ('figure.facecolor', '#000000'),
        ('figure.edgecolor', (1, 1, 1, 0)),
        ('figure.subplot.bottom', 0.125)
    ]

    LIGHT_MPL_THEME = [
        ('backend', 'module://ipykernel.pylab.backend_inline'),
        ('interactive', True),
        ('lines.color', '#101010'),
        ('text.color', '#303030'),
        ('lines.antialiased', True),
        ('lines.linewidth', 1),
        ('patch.linewidth', 0.5),
        ('patch.facecolor', '#348ABD'),
        ('patch.edgecolor', '#eeeeee'),
        ('patch.antialiased', True),
        ('axes.facecolor', '#fafafa'),
        ('axes.edgecolor', '#d0d0d0'),
        ('axes.linewidth', 1),
        ('axes.titlesize', 'x-large'),
        ('axes.labelsize', 'large'),
        ('axes.labelcolor', '#555555'),
        ('axes.axisbelow', True),
        ('axes.grid', True),
        ('axes.prop_cycle', cycler('color', ['#6792E0', '#27ae60', '#c44e52', '#975CC3', '#ff914d', '#77BEDB',
                                             '#303030', '#4168B7', '#93B851', '#e74c3c', '#bc89e0', '#ff711a',
                                             '#3498db', '#6C7A89'])),
        ('legend.fontsize', 'small'),
        ('legend.fancybox', False),
        ('xtick.color', '#707070'),
        ('ytick.color', '#707070'),
        ('grid.color', '#606060'),
        ('grid.linestyle', '--'),
        ('grid.linewidth', 0.5),
        ('grid.alpha', 0.3),
        ('figure.figsize', [8.0, 5.0]),
        ('figure.dpi', 80.0),
        ('figure.facecolor', '#ffffff'),
        ('figure.edgecolor', '#ffffff'),
        ('figure.subplot.bottom', 0.1)
    ]
    
    t = DARK_MPL_THEME if 'dark' in theme.lower() else LIGHT_MPL_THEME
    for (k, v) in t:
        matplotlib.rcParams[k] = v
        

def fig(w=16, h=5, dpi=96, facecolor=None, edgecolor=None, num=None):
    """
    Simple helper for creating figure
    """
    return plt.figure(num=num, figsize=(w, h), dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)


def subplot(shape, loc, rowspan=1, colspan=1):
    """
    Some handy grid splitting for plots. Example for 2x2:
    
    >>> subplot(22, 1); plt.plot([-1,2,-3])
    >>> subplot(22, 2); plt.plot([1,2,3])
    >>> subplot(22, 3); plt.plot([1,2,3])
    >>> subplot(22, 4); plt.plot([3,-2,1])

    same as following

    >>> subplot((2,2), (0,0)); plt.plot([-1,2,-3])
    >>> subplot((2,2), (0,1)); plt.plot([1,2,3])
    >>> subplot((2,2), (1,0)); plt.plot([1,2,3])
    >>> subplot((2,2), (1,1)); plt.plot([3,-2,1])

    :param shape: scalar (like matlab subplot) or tuple
    :param loc: scalar (like matlab subplot) or tuple
    :param rowspan: rows spanned
    :param colspan: columns spanned
    """
    if isscalar(shape):
        if 0 < shape < 100:
            shape = (max(shape // 10, 1), max(shape % 10, 1))
        else:
            raise ValueError("Wrong scalar value for shape. It should be in range (1...99)")

    if isscalar(loc):
        nm = max(shape[0], 1) * max(shape[1], 1)
        if 0 < loc <= nm:
            x = (loc - 1) // shape[1]
            y = loc - 1 - shape[1] * x
            loc = (x, y)
        else:
            raise ValueError("Wrong scalar value for location. It should be in range (1...%d)" % nm)

    return plt.subplot2grid(shape, loc=loc, rowspan=rowspan, colspan=colspan)


def sbp(shape, loc, r=1, c=1):
    """
    Just shortcut for subplot(...) function

    :param shape: scalar (like matlab subplot) or tuple
    :param loc: scalar (like matlab subplot) or tuple
    :param r: rows spanned
    :param c: columns spanned
    :return:
    """
    return subplot(shape, loc, rowspan=r, colspan=c)


def plot_trends(trends, uc='w--', dc='c--', lw=0.7, ms=5, fmt='%H:%M'):
    """
    Plot find_movements function output as trend lines on chart

    >>> from ira.analysis import find_movements
    >>>
    >>> tx = pd.Series(np.random.randn(500).cumsum() + 100, index=pd.date_range('2000-01-01', periods=500))
    >>> trends = find_movements(tx, np.inf, use_prev_movement_size_for_percentage=False,
    >>>                    pcntg=0.02,
    >>>                    t_window=np.inf, drop_weekends_crossings=False,
    >>>                    drop_out_of_market=False, result_as_frame=True, silent=True)
    >>> plot_trends(trends)

    :param trends: find_movements output
    :param uc: up trends line spec ('w--')
    :param dc: down trends line spec ('c--')
    :param lw: line weight (0.7)
    :param ms: trends reversals marker size (5)
    :param fmt: time format (default is '%H:%M')
    """
    if not trends.empty:
        u, d = trends.UpTrends.dropna(), trends.DownTrends.dropna()
        plt.plot([u.index, u.end], [u.start_price, u.end_price], uc, lw=lw, marker='.', markersize=ms);
        plt.plot([d.index, d.end], [d.start_price, d.end_price], dc, lw=lw, marker='.', markersize=ms);

        from matplotlib.dates import num2date
        import datetime
        ax = plt.gca()
        ax.set_xticklabels([datetime.date.strftime(num2date(x), fmt) for x in ax.get_xticks()])

setup_mpl_theme('dark')