"""Point-in-time data layer.

Two providers, one contract: every loader takes ``as_of`` and returns only
what was publicly known on that date.

    fred        FRED/ALFRED, full vintage history per series (decision 0002)
    fiscaldata  Treasury Fiscal Data: auction ledger, MSPD outstanding detail,
                interest expense, average rates, debt to the penny
    pit         the VintageFrame that makes "as it was known then" a one-liner
"""
from .pit import VintageFrame
from .fred import SERIES, FredClient, load_macro_panel
from .fiscaldata import (FiscalDataClient, load_auctions, load_mspd_marketable, load_mspd_summary,
                         load_interest_expense, load_avg_interest_rates, load_debt_to_penny)

__all__ = ["VintageFrame", "SERIES", "FredClient", "load_macro_panel", "FiscalDataClient",
           "load_auctions", "load_mspd_marketable", "load_mspd_summary", "load_interest_expense",
           "load_avg_interest_rates", "load_debt_to_penny"]
