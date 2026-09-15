"""Controlled Monte Carlo of the fiscal doom loop.

    macro      quarterly fiscal-macro VAR: the estimated joint dynamics and
               residual covariance of rates, growth, primary balance, and
               unemployment (this is where "no two parameters are independent"
               is enforced by estimation, not assumption)
    premium    endogenous fiscal risk premium models (Phase 4's open call P-01)
    doomloop   the state machine: monthly debt accounting from the real bond
               book, quarterly macro block, premium feedback into rates
    policy     the five counterfactuals as interventions on the same shocks
    labor      sector employment responses estimated from history (Phase 5)
    recovery   half-lives from historical episodes and from simulated paths
"""
from .macro import MacroVAR, MacroSpec
from .premium import RiskPremiumModel, LinearPremium, ThresholdPremium, NoPremium, premium_from_config
from .doomloop import DoomLoopSimulator, SimResult, InitialState
from .policy import Policy, StatusQuo, NoLayoffMandate, Austerity, Monetization, GrowthLed, policy_from_name
from .labor import LaborModel
from .recovery import RecoveryModel

__all__ = ["MacroVAR", "MacroSpec", "RiskPremiumModel", "LinearPremium", "ThresholdPremium", "NoPremium",
           "premium_from_config", "DoomLoopSimulator", "SimResult", "InitialState", "Policy", "StatusQuo",
           "NoLayoffMandate", "Austerity", "Monetization", "GrowthLed", "policy_from_name", "LaborModel",
           "RecoveryModel"]
