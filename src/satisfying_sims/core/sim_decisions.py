from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SimAction(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"
    RESTART = "restart"


@dataclass
class SimDecision:
    action: SimAction
    reason: str = ""


@dataclass
class SimStopRestartPolicy:
    """
    Handles BOTH:
      - end/stop conditions: n_steps, max_bodies
      - restart conditions:
          * ensure >= num_bodies_1 by tmin_1
          * once >= num_bodies_1 is first achieved at time t1 (>= tmin_1),
            ensure >= num_bodies_2 by (t1 + delta_t21_min)

    Intended usage:
        policy = SimStopRestartPolicy(...)
        policy.reset()  # per run / per seed
        for step in ...:
            # step world, build snapshot/recording
            decision = policy.decide(step=step, t=world.time, n_bodies=world.n_bodies)
            if decision.action == SimAction.RESTART: ...
            if decision.action == SimAction.STOP: ...
    """

    # stop conditions
    n_steps: Optional[int] = None
    max_bodies: Optional[int] = None
    multi_bodies_required: bool = False
    # restart conditions (phase 1)
    tmin_1: float = 0.0
    num_bodies_1: Optional[int] = None
    tmax_1: Optional[float] = None

    # restart conditions (phase 2)
    delta_t21_min: Optional[float] = None
    num_bodies_2: Optional[int] = None
    
    #max allowed fractional difference between body counts
    max_frac_diff: Optional[float] = None
    max_frac_diff_thresh: Optional[int] = None

    # internal state (set by reset / during run)
    _t1_reached: Optional[float] = None
    _phase1_failed: bool = False
    _phase2_failed: bool = False
    min_duration: Optional[float] = 55.0

    def reset(self) -> None:
        """Call at the start of each simulation run (new seed/world)."""
        self._t1_reached = None
        self._phase1_failed = False
        self._phase2_failed = False
    def decide(self, *, step: int, t: float, body_counts: dict[str, int]) -> SimDecision:
        """
        Return what the simulation should do *after* observing the current state.
        Priority: RESTART > STOP > CONTINUE
        """

        # ---------- Restart logic (priority) ----------
        #check if any body counts go to one
        
        n_bodies = sum(body_counts.values())
        if self.multi_bodies_required:
            for count in body_counts.values():
                if count < 2:
                    #check if min duration has been reached
                    if self.min_duration is not None and t < self.min_duration:
                        return SimDecision(
                            action=SimAction.RESTART,
                            reason=f"restart: body count of a species became less than 2",
                        )
                    else:
                        #end simulation
                        return SimDecision(
                            action=SimAction.STOP,
                            reason=f"stop: body count of a species became less than 2",
                        )
        if len(body_counts)==2:
            counts = list(body_counts.values())
            frac_diff = abs(counts[0]-counts[1])/max(counts) if max(counts)>0 else 0.0
            if self.max_frac_diff is not None and frac_diff > self.max_frac_diff and n_bodies>self.max_frac_diff_thresh:
                if self.min_duration is not None and t < self.min_duration:
                    return SimDecision(
                        action=SimAction.RESTART,
                        reason=(
                            f"restart: fractional difference {frac_diff:.3f} "
                            f"exceeded max_frac_diff {self.max_frac_diff:.3f}"
                        ),
                    )
                else:
                    return SimDecision(
                        action=SimAction.STOP,
                        reason=(
                            f"stop: fractional difference {frac_diff:.3f} "
                            f"exceeded max_frac_diff {self.max_frac_diff:.3f}"
                        ),
                    )
        # Phase 1: before tmin_1, we must not have num_bodies_1
        if self.num_bodies_1 is not None:
            if t < self.tmin_1 and n_bodies >= self.num_bodies_1:
                self._phase1_failed = True
                return SimDecision(
                    action=SimAction.RESTART,
                    reason=(
                        f"phase1 failed: at t={t:.3f} (>= tmin_1={self.tmin_1:.3f}), "
                        f"n_bodies={n_bodies} < num_bodies_1={self.num_bodies_1}"
                    ),
                )
            #we must have num_bodies_1 by tmax_1 if tmax_1 is set
            if self.tmax_1 is not None:
                if t > self.tmax_1 and n_bodies < self.num_bodies_1:
                    self._phase1_failed = True
                    return SimDecision(
                        action=SimAction.RESTART,
                        reason=(
                            f"phase1 failed: at t={t:.3f} (> tmax_1={self.tmax_1:.3f}), "
                            f"n_bodies={n_bodies} < num_bodies_1={self.num_bodies_1}"
                        ),
                    )

            # Track the FIRST time we achieve num_bodies_1, but only once we're
            # past tmin_1 (matches your docstring intent: milestone at/after tmin_1).
            if self._t1_reached is None and t >= self.tmin_1 and n_bodies >= self.num_bodies_1:
                self._t1_reached = t

        # Phase 2: after reaching phase1 milestone at t1, ensure num_bodies_2 by t1 + delta
        if (
            self.num_bodies_2 is not None
            and self.delta_t21_min is not None
            and self._t1_reached is not None
        ):
            deadline2 = self._t1_reached + self.delta_t21_min
            if t < deadline2 and n_bodies >= self.num_bodies_2:
                self._phase2_failed = True
                return SimDecision(
                    action=SimAction.RESTART,
                    reason=(
                        f"phase2 failed: at t={t:.3f} (>= t1+delta={deadline2:.3f}, "
                        f"t1={self._t1_reached:.3f}, delta={self.delta_t21_min:.3f}), "
                        f"n_bodies={n_bodies} < num_bodies_2={self.num_bodies_2}"
                    ),
                )

        # ---------- Stop logic ----------
        if self.n_steps is not None and step >= self.n_steps:
            return SimDecision(
                action=SimAction.STOP,
                reason=f"stop: step {step} >= n_steps {self.n_steps}",
            )

        if self.max_bodies is not None and n_bodies >= self.max_bodies:
            return SimDecision(
                action=SimAction.STOP,
                reason=f"stop: n_bodies {n_bodies} >= max_bodies {self.max_bodies}",
            )

        return SimDecision(action=SimAction.CONTINUE)

    @property
    def t1_reached(self) -> Optional[float]:
        """For debugging/telemetry."""
        return self._t1_reached
