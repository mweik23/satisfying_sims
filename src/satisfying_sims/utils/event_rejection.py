
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Callable
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from satisfying_sims.core.recording import EventContext

@dataclass(frozen=True)
class RejectConfig:
    lam0: float # for lam<lam0, p=1
    lam_max: float # p*lam -> lam_max as lam->infty
    k: float = 2.0 # sharpness of transition
    gamma: float = 1.0 #overall power of lam after break
    p_min: float = 0.0
    p_max: float = 1.0
    
    @classmethod
    def from_args(cls, args) -> "RejectConfig":
        kwargs={}
        for f in fields(cls):
            name = f.name
            if hasattr(args, name):
                kwargs[name] = getattr(args, name)
        return cls(**kwargs)

def make_keep_prob(cfg: RejectConfig) -> Callable:
    def keep_prob(snap: EventContext, rule_events: list[str]) -> float:
        lam = sum(snap.rates.get(ev_type, 0.0) for ev_type in rule_events)
        return probability(lam, cfg)
    return keep_prob

def probability(lam: float, cfg:RejectConfig):
    p = 1.0 / (1.0 - (cfg.lam0/cfg.lam_max) ** cfg.k + (lam / cfg.lam_max) ** cfg.k) ** (cfg.gamma/cfg.k)  # example
    return np.clip(p, cfg.p_min, cfg.p_max)

def make_keep_prob_gnrl(cfg: RejectConfig) -> Callable:
    def keep_prob(lam) -> float:
        return probability(lam, cfg)
    return keep_prob