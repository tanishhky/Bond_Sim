"""Nothing is left independent.

    correlation  full-sample, rolling, and regime-split correlations for every
                 pair; lead/lag scan; Granger causality with FDR control
    regimes      regime labels (quantile or HMM) on any driver, regime-consistency
                 tests across pairs (adjusted Rand), and regime lead/lag
"""
from .correlation import CorrelationAnalyzer, PairResult
from .regimes import RegimeDetector, adjusted_rand_index

__all__ = ["CorrelationAnalyzer", "PairResult", "RegimeDetector", "adjusted_rand_index"]
