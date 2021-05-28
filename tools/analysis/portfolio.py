import pandas as pd
import numpy as np

from scipy import stats
from scipy.stats import norm
from typing import Union

from datetime import timedelta

# Most used annualization factors
YEARLY = 1
MONTHLY = 12
WEEKLY = 52
DAILY = 252
HOURLY = DAILY*6.5
MINUTELY = HOURLY*60
HOURLY_FX = DAILY*24
MINUTELY_FX = HOURLY_FX*60


def sharpe_ratio(returns, risk_free=0.0, periods=DAILY):
    """
    Calculates the Sharpe ratio.

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param risk_free: constant risk-free return throughout the period
    :param periods: Defines the periodicity of the 'returns' data for purposes of annualizing.
                     Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :return: Sharpe ratio
    """
    if len(returns) < 2:
        return np.nan

    returns_risk_adj = returns - risk_free
    returns_risk_adj = returns_risk_adj[~np.isnan(returns_risk_adj)]

    if np.std(returns_risk_adj, ddof=1) == 0:
        return np.nan

    return np.mean(returns_risk_adj) / np.std(returns_risk_adj, ddof=1) * np.sqrt(periods)