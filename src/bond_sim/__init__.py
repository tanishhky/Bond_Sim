"""Bond_Sim: US sovereign-debt doom-loop research engine.

Layers (each importable on its own):
    calendar    the shared monthly grid every array is indexed on
    data        point-in-time loaders (FRED/ALFRED vintages, Treasury Fiscal Data)
    engine      bond-level cash flow book and scenario discounting
    analysis    correlations, regimes, lead/lag, Granger with FDR control
    sim         rate paths, doom-loop feedback, sector shocks, policies, recovery
"""
__version__ = "0.2.0"
