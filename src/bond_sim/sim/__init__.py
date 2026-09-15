"""Controlled Monte Carlo of the fiscal doom loop.

    macro          quarterly levels VAR: estimated joint dynamics and shock covariance
    states         block factors -> latent Markov chain, bootstrapped residuals, shocks
    macroblock     one interface over both, so the debt engine runs on either
    premium        endogenous fiscal risk premium (none / linear / threshold)
    doomloop       the state machine: monthly debt accounting x quarterly macro block
    sustainability identity-based diagnostics: pb*, g*, decomposition, feasible pb, trigger
    admissible     reject impossible paths, never clip; report every rejection
    policy         the five counterfactuals as interventions on the same draws
    labor          sector employment responses estimated from history
    recovery       half-lives from historical episodes and simulated paths
    setup          InitialState read off data
"""
from .macro import MacroVAR, MacroSpec, build_quarterly_state
from .states import LatentStateModel, ForcedTransition, FactorImpulse, Draws
from .macroblock import VARBlock, StateBlock
from .premium import RiskPremiumModel, LinearPremium, ThresholdPremium, NoPremium, premium_from_config
from .doomloop import DoomLoopSimulator, SimResult, InitialState
from .sustainability import (FeasiblePB, FiscalReaction, fit_fiscal_reaction, decompose, stabilizing_growth,
                             stabilizing_primary_balance, debt_limit, evaluate_paths, evaluate_history)
from .policy import Policy, StatusQuo, NoLayoffMandate, Austerity, Monetization, GrowthLed, policy_from_name
from .labor import LaborModel
from .recovery import RecoveryModel
from . import admissible

__all__ = ["MacroVAR", "MacroSpec", "build_quarterly_state", "LatentStateModel", "ForcedTransition", "FactorImpulse",
           "Draws", "VARBlock", "StateBlock", "RiskPremiumModel", "LinearPremium", "ThresholdPremium", "NoPremium",
           "premium_from_config", "DoomLoopSimulator", "SimResult", "InitialState", "FeasiblePB", "FiscalReaction",
           "fit_fiscal_reaction", "decompose", "stabilizing_growth", "stabilizing_primary_balance", "debt_limit",
           "evaluate_paths", "evaluate_history", "Policy", "StatusQuo", "NoLayoffMandate", "Austerity",
           "Monetization", "GrowthLed", "policy_from_name", "LaborModel", "RecoveryModel", "admissible"]
