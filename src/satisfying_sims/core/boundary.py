# src/satisfying_sims/core/boundary.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, List
from .shapes import Body  # see below
from .events import HitWallEvent, BaseEvent
from satisfying_sims.utils.random import rng
from dataclasses import dataclass, field
import numpy as np
# at top of file
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

_TWO_PI = 2.0 * np.pi
PointsFn = Callable[[float], np.ndarray | list[list[float]]]

@dataclass(frozen=True)
class Contact:
    point: np.ndarray      # closest point on wall
    normal: np.ndarray     # unit normal (pointing "out of wall" toward the body)
    depth: float           # penetration depth (>= 0 means overlapping)
    t_param: float | None  # optional: arc-length / segment param for debugging

@dataclass
class Wall(ABC):
    """
    Base class for open or closed wall geometry.

    Semantics:
      - two-sided (one_sided=False): valid_position means "not overlapping the wall"
      - one-sided (one_sided=True): valid_position means "on the allowed side and not overlapping"
        where the allowed side is defined by the wall's normal (after applying normal_sign).
    """
    one_sided: bool = False
    normal_sign: float = 1.0          # +1 or -1 to flip normal orientation
    constrains_domain: bool = False   # whether wall participates in Boundary.contains()
    wall_idx: int = 0                   # optional index for event reporting 

    @abstractmethod
    def closest_point(self, pos: np.ndarray) -> tuple[np.ndarray, float | None]:
        """Return (closest_point_on_wall, param)."""

    @abstractmethod
    def normal_at(self, param: float | None, closest: np.ndarray) -> np.ndarray:
        """
        Return a *unit* normal defining the "positive/allowed" side when one_sided=True.
        For two-sided walls, this is mainly used when dist ~ 0 for signed_distance.
        """

    def signed_distance(self, pos: np.ndarray) -> float:
        cp, param = self.closest_point(pos)
        n = self.normal_sign * self.normal_at(param, cp)
        n = n / (np.linalg.norm(n) + 1e-12)
        return float(np.dot(pos - cp, n))

    def distance(self, pos: np.ndarray) -> float:
        cp, _ = self.closest_point(pos)
        return float(np.linalg.norm(pos - cp))

    def valid_position(self, pos: np.ndarray, radius: float = 0.0) -> bool:
        if self.one_sided:
            return self.signed_distance(pos) >= radius
        return self.distance(pos) >= radius
    
    def contact_circle(self, pos: np.ndarray, radius: float) -> Contact | None:
        cp, param = self.closest_point(pos)
        delta = pos - cp
        dist = float(np.linalg.norm(delta))
        # If center is exactly on wall, fall back to wall normal
        if dist > 1e-12:
            n = delta / dist
        else:
            n = self.normal_at(param, cp)
            n = n / (np.linalg.norm(n) + 1e-12)

        n = self.normal_sign * n

        # one-sided: only if approaching from normal side
        if self.one_sided:
            # If point is "behind" wall (negative side), ignore collision
            if float(np.dot(pos - cp, n)) < 0.0:
                return None

        depth = radius - dist
        if depth <= 0.0:
            return None

        return Contact(point=cp, normal=n, depth=depth, t_param=param)
    
    def resolve_collision(self, body: Body, restitution: float = 0.5, t: float | None = None) -> tuple[list[HitWallEvent], bool]:
        c = self.contact_circle(body.pos, body.collider.bounding_radius())
        if c is None:
            return [], False
        ev, bounced = resolve_wall_contact(body, c, restitution=restitution, t=t, wall_idx=self.wall_idx)
        if ev is not None:
            events = [ev]
        else:
            events = []
        return events, bounced
    
def build_linear(base, velocity, min_pos=None, max_pos=None):
    def f(t):
        pos = base + velocity * t
        if min_pos is not None or max_pos is not None:
            pos = np.clip(pos, min_pos, max_pos)
        return pos
    return f 
        
@dataclass
class PolylineWall(Wall):
    """
    Wall defined by a polyline (piecewise linear path).

    points: (N,2) arraylike with N>=2
    closed: if True, last point connects back to first.

    Param encoding:
      - closest_point() returns param = i + u
        where i is segment index and u in [0,1] is fractional position along segment.
    """
    points: np.ndarray | list[list[float]] | list[tuple[float, float]] | None = None
    closed: bool = False
    color: str = "none"
    priority: float = 1.0  # for rendering order; higher renders on top
    spawn_region_points: np.ndarray | list[list[float]] | list[tuple[float, float]] | None = None
    spawn_vel_dir: np.ndarray | list[float] | tuple[float, float] | None = None
    points_fn: PointsFn | None = None
    animate: dict | None = None  # e.g. {'velocities': [[vx1, vy1], [vx2, vy2], ...]}

    def __post_init__(self) -> None:
        self.set_points(self.points)
        if self.closed and self.points is not None and len(self.points) >= 2:
            self._normalize_closed_points()
            self._ensure_ccw()
        self.points_init = self.points.copy()
        if self.animate is not None:
            velocities = self.animate.get('velocities', None)
            if velocities is not None:
                vel_array = np.asarray(velocities, dtype=float)
                if vel_array.shape != self.points.shape:
                    raise ValueError("animate.velocities must match shape of points")
                self.points_fn = build_linear(self.points_init, vel_array)
        self.spawn_region = PolylineWall(
            points=self.spawn_region_points, 
            normal_sign=-1.0,      # spawn region is inside the closed wall, so flip normal           
            closed=True) if self.spawn_region_points is not None else None
        self.spawn_vel_dir = np.asarray(self.spawn_vel_dir, dtype=float) if self.spawn_vel_dir is not None else None
    
    def get_state(self) -> dict:
        return {"points": self.points.copy()}
    def _ensure_ccw(self):
        pts = np.asarray(self.points, dtype=float)
        area = 0.5 * np.sum(
            pts[:, 0] * np.roll(pts[:, 1], -1)
            - pts[:, 1] * np.roll(pts[:, 0], -1)
        )
        if area < 0:  # CW → reverse
            self.points = pts[::-1]
    import numpy as np

    def _normalize_closed_points(points: np.ndarray, *, tol: float = 1e-9) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must be (N,2); got {pts.shape}")

        if pts.shape[0] < 3:
            raise ValueError("closed wall needs at least 3 points")

        # If user repeats p0 at end, remove it
        if np.linalg.norm(pts[-1] - pts[0]) <= tol:
            pts = pts[:-1]

        # Check unique vertex count
        # (simple approach; you can do more sophisticated de-dup if you want)
        if np.unique(pts, axis=0).shape[0] < 3:
            raise ValueError("closed wall needs at least 3 distinct vertices")

        return pts
                
    def _iter_segments(self):
        """
        Yield (segment_index, a, b) for each segment.

        For closed polylines:
        - Implicitly closes last->first.
        - If points repeat the first vertex at the end, that duplicate is ignored.
        """
        pts = np.asarray(self.points, dtype=float)
        n = len(pts)
        if n < 2:
            return

        n_eff = n
        if self.closed and n >= 3 and np.allclose(pts[0], pts[-1], rtol=0.0, atol=1e-9):
            n_eff = n - 1

        seg_count = n_eff if self.closed else (n_eff - 1)
        if seg_count <= 0:
            return

        for i in range(seg_count):
            a, b = self._get_segment(i)
            yield i, a, b


    def _get_segment(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (a,b) for segment i."""
        pts = np.asarray(self.points, dtype=float)
        n = len(pts)

        if n < 2:
            raise ValueError("PolylineWall needs at least 2 points")

        # Effective vertex count: ignore duplicated closing vertex if present
        n_eff = n
        if self.closed and n >= 3 and np.allclose(pts[0], pts[-1], rtol=0.0, atol=1e-9):
            n_eff = n - 1

        if self.closed:
            if n_eff < 3:
                raise ValueError("Closed PolylineWall needs at least 3 distinct vertices")
            # segment i is (pts[i], pts[i+1]) with wrap, but only over n_eff vertices
            i = i % n_eff
            a = pts[i]
            b = pts[(i + 1) % n_eff]
        else:
            # clamp to valid segment range
            i = max(0, min(i, n_eff - 2))
            a = pts[i]
            b = pts[i + 1]

        return a, b
    
    def _normalize_closed_points(self, *, tol: float = 1e-9) -> None:
        """
        Normalize self.points for a closed wall:
        - drop duplicated closing vertex if present
        - enforce minimum vertex count
        """
        pts = np.asarray(self.points, dtype=float)

        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must be (N,2); got {pts.shape}")

        if pts.shape[0] < 3:
            raise ValueError("closed wall needs at least 3 points")

        # Drop duplicate closing vertex
        if np.allclose(pts[0], pts[-1], rtol=0.0, atol=tol):
            pts = pts[:-1]

        if np.unique(pts, axis=0).shape[0] < 3:
            raise ValueError("closed wall needs at least 3 distinct vertices")

        self.points = pts

    
    def set_points(self, points: np.ndarray | list[list[float]] | list[tuple[float, float]]) -> None:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            raise ValueError("points must be (N,2) with N>=2")
        self.points = pts
        
    def update(self, t: float) -> None:
        if self.points_fn is not None:
            self.set_points(self.points_fn(t))

    def closest_point(self, pos: np.ndarray) -> tuple[np.ndarray, float | None]:
        p = np.asarray(pos, dtype=float)

        best_cp: np.ndarray | None = None
        best_d2: float = float("inf")
        best_param: float | None = None

        for i, a, b in self._iter_segments():
            ab = b - a
            ab2 = float(np.dot(ab, ab))

            if ab2 < 1e-12:
                # degenerate segment
                u = 0.0
                cp = a
            else:
                u = float(np.dot(p - a, ab) / ab2)
                u = max(0.0, min(1.0, u))
                cp = a + u * ab

            d = p - cp
            d2 = float(np.dot(d, d))
            if d2 < best_d2:
                best_d2 = d2
                best_cp = cp
                best_param = float(i) + u

        # For completeness; should never be None if N>=2
        assert best_cp is not None
        return best_cp, best_param

    def normal_at(self, param: float | None, closest: np.ndarray) -> np.ndarray:
        """
        Define the wall's normal using the segment direction:

          d = (b - a)
          n = perp(d) = (-dy, dx)

        The returned normal is unit length. For closed polylines, vertex ordering
        controls which side is "positive" before applying normal_sign.

        Note: This is a *geometric* normal for the segment, not the "radial" normal.
        """
        if param is None:
            return np.array([0.0, 1.0], dtype=float)

        i = int(np.floor(param))
        a, b = self._get_segment(i)
        d = b - a

        # If segment is degenerate, try a neighbor
        if float(np.dot(d, d)) < 1e-12:
            if self.closed or i > 0:
                a2, b2 = self._get_segment(i - 1)
                d2 = b2 - a2
                if float(np.dot(d2, d2)) >= 1e-12:
                    d = d2
            if float(np.dot(d, d)) < 1e-12 and (self.closed or i < len(self.points) - 2):
                a3, b3 = self._get_segment(i + 1)
                d3 = b3 - a3
                if float(np.dot(d3, d3)) >= 1e-12:
                    d = d3

        n = np.array([-d[1], d[0]], dtype=float)
        n /= (np.linalg.norm(n) + 1e-12)
        return n
    
    def get_bounding_box(self) -> tuple[float, float, float, float]:
        xs = self.points[:, 0]
        ys = self.points[:, 1]
        return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    
    def sample_spawn_coords(self, vel_mag: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        if self.spawn_region_points is None:
            raise ValueError("No spawn region defined for this wall")
        # sample position uniformly within bounding box
        xmin, xmax, ymin, ymax = self.spawn_region.get_bounding_box()
        while True:            
            x = rng("physics").uniform(xmin, xmax)
            y = rng("physics").uniform(ymin, ymax)
            pos = np.array([x, y], dtype=float)
            if self.spawn_region.valid_position(pos):
                break

        vel = self.spawn_vel_dir if self.spawn_vel_dir is not None else np.zeros(2, dtype=float)
        if np.linalg.norm(vel) > 1e-12:
            vel = (vel / np.linalg.norm(vel)) * vel_mag
        return pos, vel
    

def _wrap_angle(theta: float) -> float:
    """Map angle to [0, 2pi)."""
    return float(theta % _TWO_PI)


def _angle_diff_ccw(a: float, b: float) -> float:
    """
    Smallest CCW delta taking a -> b, in [0, 2pi).
    """
    return _wrap_angle(b - a)


def _clamp_angle_to_arc(theta: float, a0: float, a1: float) -> float:
    """
    Clamp theta to the *CCW* arc from a0 to a1.

    We interpret the arc as moving CCW starting at a0 and ending at a1.
    If a1 == a0, the arc is treated as a full circle only if the user explicitly
    sets that intent (handled elsewhere); otherwise it's a zero-length arc.
    """
    t = _wrap_angle(theta)
    s = _wrap_angle(a0)
    e = _wrap_angle(a1)

    arc_len = _angle_diff_ccw(s, e)  # in [0, 2pi)
    # If arc_len == 0, it's either "degenerate" or "full circle".
    # We treat it as degenerate by default; user can set delta_angle=2pi for full circle.
    if arc_len <= 1e-12:
        return s

    rel = _angle_diff_ccw(s, t)  # how far CCW from start to theta
    if rel <= arc_len:
        return t
    # Outside arc: closer endpoint in angular metric along CCW arc boundary
    # Candidates are start(s) and end(e). The clamp is whichever endpoint is "closer"
    # in Euclidean distance once projected to the circle; but we don't know radius here.
    # For angle-only clamp, choose nearer along the circular angle in either direction:
    # distance to start = min( rel, 2pi-rel ) but since rel>arc_len, it might be nearer to end.
    # Use angle distance on circle:
    d_to_start = min(rel, _TWO_PI - rel)
    d_to_end = min(abs(rel - arc_len), _TWO_PI - abs(rel - arc_len))
    return s if d_to_start <= d_to_end else e


def _arc_contains_angle(theta: float, a0: float, a1: float) -> bool:
    """Return True if theta lies on CCW arc [a0 -> a1]."""
    t = _wrap_angle(theta)
    s = _wrap_angle(a0)
    e = _wrap_angle(a1)
    arc_len = _angle_diff_ccw(s, e)
    if arc_len <= 1e-12:
        return False
    return _angle_diff_ccw(s, t) <= arc_len + 1e-12


@dataclass
class CircularArcWall(Wall):
    """
    Circular arc wall defined by:
      - center (2,)
      - radius > 0
      - angle_start, delta_angle (radians)

    Arc semantics:
      - The arc runs CCW from angle_start to angle_start + delta_angle.
      - For a full circle, set delta_angle = 2*pi (or any +k*2*pi).
        (If delta_angle == 0 exactly, this is treated as a degenerate arc.)

    Param encoding:
      - closest_point() returns param = theta_clamped (the angle of the closest point).
        (Useful for debugging and normal evaluation.)

    Animation:
      animate can specify constant velocities for any defining parameter, e.g.
        animate = {
          "center_velocity": [vx, vy],
          "radius_velocity": vr,
          "angle_start_velocity": va0,
          "delta_angle_velocity": vda,
          # optional clamps:
          "center_min": [xmin, ymin],
          "center_max": [xmax, ymax],
          "radius_min": rmin,
          "radius_max": rmax,
        }
      You can provide any subset of these keys.

    Spawn region:
      - Provide spawn_region_points just like PolylineWall, and we build a closed
        PolylineWall spawn_region (with normal_sign flipped) for uniform-in-AABB rejection sampling.
    """
    #state variables
    #-------------------
    center: np.ndarray | list[float] | tuple[float, float] = (0.0, 0.0)
    radius: float = 1.0
    angle_start: float = 0.0
    delta_angle: float = _TWO_PI
    #-------------------
    #static for now; could add animation later if desired
    color: str = "none"
    linewidth: float = 1.0
    priority: float = 1.0
    
    # Spawn region (same pattern as PolylineWall)
    spawn_region_points: np.ndarray | list[list[float]] | list[tuple[float, float]] | None = None
    spawn_vel_dir: np.ndarray | list[float] | tuple[float, float] | None = None

    # Animation plumbing
    center_fn: callable | None = None
    radius_fn: callable | None = None
    angle_start_fn: callable | None = None
    delta_angle_fn: callable | None = None
    animate: dict | None = None

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(2)
        self.radius = float(self.radius)
        self.angle_start = float(self.angle_start)
        self.delta_angle = float(self.delta_angle)

        if self.radius < 0.0:
            raise ValueError("radius must be >= 0")

        # Save initial values for linear animation
        self._center_init = self.center.copy()
        self._radius_init = float(self.radius)
        self._a0_init = float(self.angle_start)
        self._da_init = float(self.delta_angle)
        self.similarity_invariant = True  # if False, angle_start and delta_angle can change. Will check this condition below
        if self.animate is not None:
            # center velocity
            cv = self.animate.get("center_velocity", None)
            if cv is not None:
                cv = np.asarray(cv, dtype=float).reshape(2)
                cmin = self.animate.get("center_min", None)
                cmax = self.animate.get("center_max", None)
                if cmin is not None:
                    cmin = np.asarray(cmin, dtype=float).reshape(2)
                if cmax is not None:
                    cmax = np.asarray(cmax, dtype=float).reshape(2)
                self.center_fn = build_linear(self._center_init, cv, min_pos=cmin, max_pos=cmax)

            # radius velocity
            rv = self.animate.get("radius_velocity", None)
            if rv is not None:
                rv = float(rv)
                rmin = self.animate.get("radius_min", None)
                rmax = self.animate.get("radius_max", None)

                def r_fn(t: float, base=self._radius_init, vel=rv, rmin=rmin, rmax=rmax):
                    r = base + vel * t
                    if rmin is not None:
                        r = max(float(rmin), float(r))
                    if rmax is not None:
                        r = min(float(rmax), float(r))
                    return float(r)

                self.radius_fn = r_fn

            # angle velocities
            a0v = self.animate.get("angle_start_velocity", None)
            if a0v is not None:
                a0v = float(a0v)

                def a0_fn(t: float, base=self._a0_init, vel=a0v):
                    return float(base + vel * t)

                self.angle_start_fn = a0_fn

            dav = self.animate.get("delta_angle_velocity", None)
            if dav is not None:
                dav = float(dav)
                if abs(dav) > 1e-12:
                    self.similarity_invariant = False
                def da_fn(t: float, base=self._da_init, vel=dav):
                    return float(base + vel * t)

                self.delta_angle_fn = da_fn

        # Spawn region wall (same as PolylineWall)
        self.spawn_region = (
            PolylineWall(
                points=self.spawn_region_points,
                normal_sign=-1.0,  # spawn region is inside the closed wall
                closed=True,
            )
            if self.spawn_region_points is not None
            else None
        )
        self.spawn_vel_dir = (
            np.asarray(self.spawn_vel_dir, dtype=float).reshape(2)
            if self.spawn_vel_dir is not None
            else None
        )

    def get_state(self) -> dict:
        return {
            "center": self.center.copy(),
            "radius": float(self.radius),
            "angle_start": float(self.angle_start),
            "delta_angle": float(self.delta_angle),
        }
    # ---- External-ish helpers (parallel-ish to PolylineWall) ----

    def set_center(self, center: np.ndarray | list[float] | tuple[float, float]) -> None:
        self.center = np.asarray(center, dtype=float).reshape(2)

    def set_radius(self, radius: float) -> None:
        r = float(radius)
        if r < 0.0:
            raise ValueError("radius must be >= 0")
        self.radius = r

    def set_angles(self, angle_start: float, delta_angle: float) -> None:
        self.angle_start = float(angle_start)
        self.delta_angle = float(delta_angle)

    def update(self, t: float) -> None:
        if self.center_fn is not None:
            self.set_center(self.center_fn(t))
        if self.radius_fn is not None:
            self.set_radius(self.radius_fn(t))
        if self.angle_start_fn is not None:
            self.angle_start = float(self.angle_start_fn(t))
        if self.delta_angle_fn is not None:
            self.delta_angle = float(self.delta_angle_fn(t))

    # ---- Geometry ----

    def _is_full_circle(self) -> bool:
        # Treat as full circle if delta_angle is an integer multiple of 2pi (within tol)
        if abs(self.delta_angle) < 1e-12:
            return False
        k = self.delta_angle / _TWO_PI
        return abs(k - round(k)) < 1e-9

    def _closest_on_circle_angle(self, p: np.ndarray) -> float:
        v = p - self.center
        return float(np.arctan2(v[1], v[0]))

    def closest_point(self, pos: np.ndarray) -> tuple[np.ndarray, float | None]:
        p = np.asarray(pos, dtype=float).reshape(2)

        # Degenerate radius => treat wall as a single point at center
        if self.radius <= 1e-12:
            return self.center.copy(), None

        theta = self._closest_on_circle_angle(p)

        # Full circle case: clamp does nothing
        if self._is_full_circle():
            theta_c = theta
        else:
            theta_c = _clamp_angle_to_arc(theta, self.angle_start, self.angle_start + self.delta_angle)

        cp = self.center + self.radius * np.array([np.cos(theta_c), np.sin(theta_c)], dtype=float)

        # If arc is degenerate (span ~ 0), cp computed at start angle.
        # But we still want true closest point among endpoints for numerical stability:
        if not self._is_full_circle():
            s = float(self.angle_start)
            e = float(self.angle_start + self.delta_angle)

            ps = self.center + self.radius * np.array([np.cos(s), np.sin(s)], dtype=float)
            pe = self.center + self.radius * np.array([np.cos(e), np.sin(e)], dtype=float)

            d2_cp = float(np.dot(p - cp, p - cp))
            d2_ps = float(np.dot(p - ps, p - ps))
            d2_pe = float(np.dot(p - pe, p - pe))

            # If theta was outside arc, the clamp likely returned an endpoint;
            # but in weird wrapping cases, just pick the best among these.
            if d2_ps < d2_cp or d2_pe < d2_cp:
                if d2_ps <= d2_pe:
                    return ps, s
                else:
                    return pe, e

        return cp, float(theta_c)

    def normal_at(self, param: float | None, closest: np.ndarray) -> np.ndarray:
        """
        Use radial normal: n = (closest - center)/radius (unit length).
        This points "outward" from the circle center.
        """
        cp = np.asarray(closest, dtype=float).reshape(2)
        v = cp - self.center
        nv = float(np.linalg.norm(v))
        if nv <= 1e-12:
            return np.array([0.0, 1.0], dtype=float)
        return v / nv

    # ---- Optional utilities (handy for spawn-region sampling, debug, rendering bounds) ----

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        """
        Tight-ish AABB for the arc:
          - endpoints
          - any cardinal angles (0, pi/2, pi, 3pi/2) that lie on the arc
        Full circle => center +/- radius.
        """
        c = self.center
        r = float(self.radius)

        if r <= 1e-12:
            x, y = float(c[0]), float(c[1])
            return x, x, y, y

        if self._is_full_circle():
            return float(c[0] - r), float(c[0] + r), float(c[1] - r), float(c[1] + r)

        angles = [float(self.angle_start), float(self.angle_start + self.delta_angle)]
        for a in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
            if _arc_contains_angle(a, self.angle_start, self.angle_start + self.delta_angle):
                angles.append(float(a))

        pts = np.stack([c + r * np.array([np.cos(a), np.sin(a)], dtype=float) for a in angles], axis=0)
        xs = pts[:, 0]
        ys = pts[:, 1]
        return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())

    def sample_spawn_coords(self, vel_mag: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Same behavior as PolylineWall: sample uniformly in the spawn_region AABB
        with rejection using spawn_region.valid_position(pos).
        """
        if self.spawn_region is None:
            raise ValueError("No spawn region defined for this wall")

        xmin, xmax, ymin, ymax = self.spawn_region.get_bounding_box()
        while True:
            x = rng("physics").uniform(xmin, xmax)
            y = rng("physics").uniform(ymin, ymax)
            pos = np.array([x, y], dtype=float)
            if self.spawn_region.valid_position(pos):
                break

        vel = self.spawn_vel_dir if self.spawn_vel_dir is not None else np.zeros(2, dtype=float)
        n = float(np.linalg.norm(vel))
        if n > 1e-12:
            vel = (vel / n) * float(vel_mag)
        return pos, vel


    
def resolve_wall_contact(body, contact: Contact, restitution: float, t: float, wall_idx: int) -> tuple[BaseEvent | None, bool]:
    # push out
    body.pos = body.pos + contact.normal * contact.depth

    vn = float(np.dot(body.vel, contact.normal))
    if vn < 0.0:
        old_vel = body.vel.copy()
        body.vel = body.vel - (1.0 + restitution) * vn * contact.normal
        impulse = body.mass * float(np.linalg.norm(old_vel - body.vel))
        ev = HitWallEvent(t=t, body_id=body.id, body_theme_id = body.theme_id or "unknown", norm_vec=contact.normal, impulse=impulse, wall_idx=wall_idx)
        return ev, True

    return None, False

@dataclass
class _Candidate:
    wall: object                 # Wall or Boundary
    contact: Contact
    priority: float
    order: int                   # stable tie-break
    wall_idx: int                # for events


def _priority_of(w: object) -> float:
    return float(getattr(w, "priority", 0.0))


def _wall_idx_of(w: object) -> int:
    return int(getattr(w, "wall_idx", 0))

class Boundary(ABC):
    wall_idx: int
    
    def contact_circle(self, pos: np.ndarray, radius: float) -> Contact | None:
        return None
    
    @abstractmethod
    def resolve_collision(
        self,
        body: "Body",
        restitution: float,
        t: float,
    ) -> List[BaseEvent]:
        """
        Mutate the body's position/velocity if it collides with the boundary.
        Return any events describing what happened.
        """
        ...
    @abstractmethod
    def contains(self, pos: np.ndarray, radius: float = 0.0) -> bool:
        """
        Return True if a (possibly extended) point is fully inside the domain.

        `radius` lets you check "does this circle of radius r fit inside?".
        """
        ...

    @abstractmethod
    def sample_position(
        self,
        radius: float = 0.0,
        policy: str = 'uniform'
    ) -> np.ndarray:
        """
        Sample a random position inside the domain (optionally padded by `radius`).
        """
        ...

    # Optional helpers for renderers / presets

    def bounds(self) -> tuple[float, float, float, float]:
        """
        Optionally provide (xmin, xmax, ymin, ymax) for camera setup / plotting.
        Default raises if not meaningful.
        """
        raise NotImplementedError

@dataclass
class EllipseBoundary(Boundary):
    a: float  # semi-major axis
    b: float  # semi-minor axis
    color: str = "none"
    priority: float = 1.0
    wall_idx: int = field(init=False, default=-1)

    def contact_circle(self, pos: np.ndarray, radius: float) -> Contact | None:
        x, y = float(pos[0]), float(pos[1])
        a, b = float(self.a), float(self.b)

        # inward normal (same as your resolve_collision)
        n = -np.array([x / (a * a), y / (b * b)], dtype=float)
        nn = float(np.linalg.norm(n))
        if nn < 1e-12:
            n = np.array([0.0, 1.0], dtype=float)
        else:
            n /= nn

        # point on the circle closest to boundary along -n
        p_closest = np.array([x, y], dtype=float) - n * radius

        val = (p_closest[0] ** 2) / (a * a) + (p_closest[1] ** 2) / (b * b)
        if val <= 1.0:
            return None

        # depth proxy (good enough for ranking); you can refine later
        depth = radius * (val - 1.0)

        # contact point: use that closest point (or a better projection later)
        cp = p_closest

        return Contact(point=cp, normal=n, depth=float(depth), t_param=None)

    def resolve_collision(
        self,
        body: Body,
        restitution: float,
        t: float,
    ) -> tuple[List[BaseEvent], bool]:
        # Simple approach: reflect velocity if outside ellipse
        events: List[BaseEvent] = []
        pos = body.pos
        vel = body.vel
        # Check if outside ellipse
        norm_vec = -np.array([pos[0] / (self.a ** 2), pos[1] / (self.b ** 2)])
        norm_vec /= np.linalg.norm(norm_vec)
        r = body.collider.bounding_radius()
        hit_wall=False
        pos_closest = pos - norm_vec * r
        if (pos_closest[0] ** 2) / (self.a ** 2) + (pos_closest[1] ** 2) / (self.b ** 2) > 1.0:
            hit_wall=True
            old_vel = vel.copy()
            vel -= (1 + restitution) * np.dot(vel, norm_vec) * norm_vec
            impulse = body.mass * np.linalg.norm(old_vel - vel)
            events.append(
                HitWallEvent(
                    t=t, 
                    body_id=body.id, 
                    body_theme_id=body.theme_id or "unknown", 
                    norm_vec=norm_vec, 
                    impulse=impulse,
                    wall_idx=self.wall_idx
                )
            )
        return events, hit_wall

    def contains(self, pos: np.ndarray, radius: float = 0.0) -> bool:
        ar = self.a - radius
        br = self.b - radius
        if ar <= 0.0 or br <= 0.0:
            return False
        x, y = float(pos[0]), float(pos[1])
        return (x*x)/(ar*ar) + (y*y)/(br*br) <= 1.0
    def get_bounding_box(self) -> tuple[float, float, float, float]:
        return -self.a, self.a, -self.b, self.b
    def get_largest_box_inside(self) -> BoxBoundary:
        # The largest axis-aligned box that fits inside the ellipse has width 2a/sqrt(2) and height 2b/sqrt(2)
        return -self.a/np.sqrt(2), self.a/np.sqrt(2), -self.b/np.sqrt(2), self.b/np.sqrt(2)
    def sample_position(
        self,
        radius: float = 0.0,
        policy: str = 'uniform',
        **kwargs
    ) -> np.ndarray:
        if policy == 'uniform':
            while True:
                x = rng("physics").uniform(-self.a + radius, self.a - radius)
                y_limit = self.b * np.sqrt(1 - (x ** 2) / (self.a ** 2))
                y = rng("physics").uniform(-y_limit + radius, y_limit - radius)
                if self.contains(np.array([x, y]), radius=radius):
                    return np.array([x, y], dtype=float)
        elif policy=='exact':
            coords = kwargs.get("coords", np.zeros(2))
            out = np.array(coords)
            if self.contains(np.array(out), radius=radius):
                return out
            else:
                raise ValueError(f"Exact sampling policy provided out-of-bounds coords: {out}")
        else:
            raise ValueError(f"Unknown sampling policy: {policy}")
        
    def bounds(self) -> tuple[float, float, float, float]:
        return -self.a, self.a, -self.b, self.b
    
    def plot(self, ax=None, delta=5, **kwargs):
        """
        Plot the box boundary using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new figure and axes are created.
        **kwargs :
            Extra keyword arguments passed to Rectangle, e.g.
            edgecolor, linewidth, linestyle.

        Returns
        -------
        ax or (fig, ax)
            If ax was provided, returns the same Axes.
            If ax was None, returns (fig, ax).
        """
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True

        # Rectangle from (0, 0) to (width, height)
        ell = Ellipse(
            xy=(0, 0),   # center
            width=2*self.a,      # 2a
            height=2*self.b,     # 2b
            angle=0,         # degrees, CCW
            **kwargs
        )

        ax.add_patch(ell)
        '''
        ax.spines["bottom"].set_color("gray")
        ax.spines["left"].set_color("gray")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(colors="gray")
        '''
        ell.set_clip_on(False)
        ax.set_xlim(-delta-self.a, self.a + delta)
        ax.set_ylim(-delta-self.b, self.b + delta)
        ax.set_aspect("equal", adjustable="box")

        if created_fig:
            return fig, ax
        return ax

@dataclass
class BoxBoundary(Boundary):
    width: float
    height: float
    color: str = "none"
    priority: float = 1.0
    wall_idx: int = field(init=False, default=-1)
    
    def contact_circle(self, pos: np.ndarray, radius: float) -> Contact | None:
        x, y = float(pos[0]), float(pos[1])
        w, h = float(self.width), float(self.height)

        d_left   = radius - x
        d_right  = radius - (w - x)
        d_top    = radius - y
        d_bottom = radius - (h - y)

        best = None  # (depth, normal, cp)

        if d_left > 0:
            best = (d_left, np.array([1.0, 0.0]), np.array([0.0, y]))
        if d_right > 0:
            cand = (d_right, np.array([-1.0, 0.0]), np.array([w, y]))
            if best is None or cand[0] > best[0]: best = cand
        if d_top > 0:
            cand = (d_top, np.array([0.0, 1.0]), np.array([x, 0.0]))
            if best is None or cand[0] > best[0]: best = cand
        if d_bottom > 0:
            cand = (d_bottom, np.array([0.0, -1.0]), np.array([x, h]))
            if best is None or cand[0] > best[0]: best = cand

        if best is None:
            return None

        depth, n, cp = best
        n = n / (np.linalg.norm(n) + 1e-12)
        return Contact(point=cp.astype(float), normal=n.astype(float), depth=float(depth), t_param=None)

    def resolve_collision(
        self,
        body: Body,
        restitution: float,
        t: float,
    ) -> List[BaseEvent]:
        events: List[BaseEvent] = []
        # delegate to collider’s bounding radius / AABB as needed
        # for now, assume circle-like
        r = body.collider.bounding_radius()
        pos = body.pos
        vel = body.vel
        hit_wall = False
        # Left wall
        if pos[0] - r < 0.0:
            hit_wall = True
            old_vx = vel[0]
            pos[0] = r
            vel[0] = -restitution * vel[0]
            impulse = body.mass * abs(old_vx - vel[0])
            events.append(
                HitWallEvent(
                    t=t, 
                    body_id=body.id, 
                    body_theme_id=body.theme_id or "unknown", 
                    norm_vec=np.array([1.0, 0.0]),
                    impulse=impulse,
                    wall_idx=self.wall_idx
                )
            )
        # Right wall
        if pos[0] + r > self.width:
            hit_wall = True
            old_vx = vel[0]
            pos[0] = self.width - r
            vel[0] = -restitution * vel[0]
            impulse = body.mass * abs(old_vx - vel[0])
            events.append(
                HitWallEvent(
                    t=t, 
                    body_id=body.id, 
                    body_theme_id=body.theme_id or "unknown", 
                    norm_vec=np.array([-1.0, 0.0]), 
                    impulse=impulse,
                    wall_idx=self.wall_idx
                )
            )
        # Top wall
        if pos[1] - r < 0.0:
            hit_wall = True
            old_vy = vel[1]
            pos[1] = r
            vel[1] = -restitution * vel[1]
            impulse = body.mass * abs(old_vy - vel[1])
            events.append(
                HitWallEvent(
                    t=t, 
                    body_id=body.id, 
                    body_theme_id=body.theme_id or "unknown", 
                    norm_vec=np.array([0.0, 1.0]), 
                    impulse=impulse,
                    wall_idx=self.wall_idx
                )
            )

        # Bottom wall
        if pos[1] + r > self.height:
            hit_wall = True
            old_vy = vel[1]
            pos[1] = self.height - r
            vel[1] = -restitution * vel[1]
            impulse = body.mass * abs(old_vy - vel[1])
            events.append(
                HitWallEvent(
                    t=t, 
                    body_id=body.id, 
                    body_theme_id=body.theme_id or "unknown", 
                    norm_vec=np.array([0.0, -1.0]), 
                    impulse=impulse,
                    wall_idx=self.wall_idx
                )
            )
            
        return events, hit_wall
    
    def contains(self, pos: np.ndarray, radius: float = 0.0) -> bool:
        x, y = float(pos[0]), float(pos[1])
        return (
            x - radius >= 0.0
            and x + radius <= self.width
            and y - radius >= 0.0
            and y + radius <= self.height
        )
    def get_bounding_box(self) -> tuple[float, float, float, float]:
        return 0.0, self.width, 0.0, self.height
    
    def get_largest_box_inside(self) -> BoxBoundary:
        # The largest axis-aligned box that fits inside the ellipse has width 2a/sqrt(2) and height 2b/sqrt(2)
        return 0.0, self.width, 0.0, self.height

    def sample_position(
        self,
        radius: float = 0.0,
        policy: str = 'uniform',
        **kwargs
    ) -> np.ndarray:
        if policy == 'center':
            return np.array([self.width / 2.0, self.height / 2.0], dtype=float)
        elif policy == 'uniform':
            min_x = 0.0 + radius
            max_x = self.width - radius
            min_y = 0.0 + radius
            max_y = self.height - radius
        elif policy == 'left half':
            min_x = 0.0 + radius
            max_x = (self.width / 2.0) - radius
            min_y = 0.0 + radius
            max_y = self.height - radius
        elif policy == 'right half':
            min_x = (self.width / 2.0) + radius
            max_x = self.width - radius
            min_y = 0.0 + radius
            max_y = self.height - radius
        elif policy == 'gaussian':
            mu = kwargs.get('mu', [self.width / 2.0, self.height / 2.0])
            sigma = kwargs.get('sigma', [self.width / 8.0, self.height / 8.0])
        elif policy == 'exact':
            coords = kwargs.get("coords", np.zeros(2))
        else:
            raise ValueError(f"Unknown sampling policy: {policy}")
        
        if policy == 'gaussian':
            out = rng("physics").normal(
                loc=mu,
                scale=sigma,
            )
        elif policy=='exact':
            out = np.array(coords)
        else:
            x = rng("physics").uniform(min_x, max_x)
            y = rng("physics").uniform(min_y, max_y)
            out = np.array([x, y], dtype=float)
        return out

    def bounds(self) -> tuple[float, float, float, float]:
        return 0.0, self.width, 0.0, self.height
    
    def plot(self, ax=None, delta=5, **kwargs):
        """
        Plot the box boundary using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new figure and axes are created.
        **kwargs :
            Extra keyword arguments passed to Rectangle, e.g.
            edgecolor, linewidth, linestyle.

        Returns
        -------
        ax or (fig, ax)
            If ax was provided, returns the same Axes.
            If ax was None, returns (fig, ax).
        """
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True

        # Rectangle from (0, 0) to (width, height)
        rect = Rectangle(
            (0.0, 0.0),
            self.width,
            self.height,
            **kwargs,
        )
        ax.add_patch(rect)
        '''
        ax.spines["bottom"].set_color("gray")
        ax.spines["left"].set_color("gray")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(colors="gray")
        '''
        rect.set_clip_on(False)
        ax.set_xlim(-delta, self.width + delta)
        ax.set_ylim(-delta, self.height + delta)
        ax.set_aspect("equal", adjustable="box")

        if created_fig:
            return fig, ax
        return ax
    
@dataclass
class WallBoundary(Boundary):
    outer: Boundary
    walls: list[Wall]
    
    def __post_init__(self) -> None:
        #create priority ranked ordering of walls
        self.all_walls = self.walls + [self.outer]
        self.all_walls.sort(key=lambda w: w.priority, reverse=True)

    def set_points(self, wall_index: int, points: np.ndarray) -> None:
        wall = self.walls[wall_index]
        if not isinstance(wall, PolylineWall):
            raise ValueError("set_points can only be used on PolylineWall instances.")
        wall.set_points(points)
        
    def contains(self, pos: np.ndarray, radius: float = 0.0) -> bool:
        if not self.outer.contains(pos, radius=radius):
            return False
        for w in self.walls:
            if w.constrains_domain and not w.valid_position(pos, radius=radius):
                return False
        return True

    def sample_position(self, radius: float = 0.0, policy: str = 'uniform', **kwargs) -> np.ndarray:
        for _ in range(500):
            p = self.outer.sample_position(radius=radius, policy=policy, **kwargs)
            if self.contains(p, radius=radius):
                return p
        # fallback: return something even if constraints are tight
        return self.outer.sample_position(radius=radius, policy=policy, **kwargs)
    
    def resolve_collision(self, body: Body, restitution: float, t: float) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        r = body.collider.bounding_radius()

        any_bounce = False

        # a few iterations: resolve one best contact each time
        for _ in range(4):
            best_c = None          # Contact
            best_wall_idx = None
            best_key = None        # (depth, priority, -order)

            for order, w in enumerate(self.all_walls):
                pr = float(getattr(w, "priority", 0.0))
                widx = int(getattr(w, "wall_idx", 0))

                # both Boundary and Wall now have contact_circle (duck-typed)
                c = w.contact_circle(body.pos, r)
                if c is None:
                    continue

                # OPTIONAL CLIP: inner walls only collide where the contact point lies in the outer domain
                if w is not self.outer:
                    if not self.outer.contains(c.point, radius=0.0):
                        continue

                key = (c.depth, pr, -order)
                if best_key is None or key > best_key:
                    best_key = key
                    best_c = c
                    best_wall_idx = widx

            if best_c is None:
                break

            ev, bounced = resolve_wall_contact(
                body,
                best_c,
                restitution=restitution,
                t=t,
                wall_idx=best_wall_idx,
            )
            if ev is not None:
                events.append(ev)
            if bounced:
                any_bounce = True

            if best_c.depth < 1e-9:
                break

        if any_bounce:
            body.update_rotation(collision=True)

        return events

    
    def bounds(self) -> tuple[float, float, float, float]:
        return self.outer.bounds()

    def is_same_side(self, pos_a: np.ndarray, pos_b: np.ndarray) -> bool:
        for w in self.walls:
            d_a = w.signed_distance(pos_a)
            d_b = w.signed_distance(pos_b)
            if d_a * d_b < 0.0:
                return False
        return True
    
    def get_aspect_ratio(self) -> float:
        xmin, xmax, ymin, ymax = self.bounds()
        return (xmax - xmin) / (ymax - ymin)
    
    def plot(self, ax=None, delta=0, **kwargs):
        """
        Plot the outer boundary and inner walls using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new figure and axes are created.
        **kwargs :
            Extra keyword arguments passed to Rectangle for outer boundary,
            e.g. edgecolor, linewidth, linestyle.

        Returns
        -------
        ax or (fig, ax)
            If ax was provided, returns the same Axes.
            If ax was None, returns (fig, ax).
        """
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True

        # Plot outer boundary
        if isinstance(self.outer, BoxBoundary) or isinstance(self.outer, EllipseBoundary):
            self.outer.plot(ax=ax, delta=delta, **kwargs)
        else:
            raise NotImplementedError("Plotting only implemented for BoxBoundary or EllipseBoundary outer.")
        '''
        # Plot inner walls
        for w in self.walls:
            if isinstance(w, PolylineWall):
                pts = w.points
                ax.plot(pts[:, 0], pts[:, 1], color=kwargs.get('edgecolor', 'white'), linewidth=kwargs.get('linewidth', 1))
            else:
                raise NotImplementedError("Plotting only implemented for PolylineWall inner walls.")
        '''
        if created_fig:
            return fig, ax
        return ax
