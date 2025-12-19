from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from satisfying_sims.core import Body

def velocity_cm(a: Body, b: Body) -> np.ndarray:
    
    # Center-of-mass velocity
    vel_a = np.asarray(a.vel, dtype=float)
    vel_b = np.asarray(b.vel, dtype=float)
    m_a = a.mass
    m_b = b.mass
    m_tot = m_a + m_b

    if m_tot > 0.0:
        v_cm = (m_a * vel_a + m_b * vel_b) / m_tot
    else:
        v_cm = 0.5 * (vel_a + vel_b)
    return v_cm   