# src/satisfying_sims/core/boundary.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, List
from .shapes import Body  # see below
from .events import HitWallEvent, BaseEvent
from satisfying_sims.utils.random import rng
from dataclasses import dataclass
import numpy as np
# at top of file
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

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
            n = self.normal_at(param, pos, cp)
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
    points: np.ndarray | list[list[float]] | list[tuple[float, float]] = None
    closed: bool = False
    
    points_fn: PointsFn | None = None
    animate: dict | None = None  # e.g. {'velocities': [[vx1, vy1], [vx2, vy2], ...]}

    def __post_init__(self) -> None:
        self.set_points(self.points)
        self.points_init = self.points.copy()
        if self.animate is not None:
            velocities = self.animate.get('velocities', None)
            if velocities is not None:
                vel_array = np.asarray(velocities, dtype=float)
                if vel_array.shape != self.points.shape:
                    raise ValueError("animate.velocities must match shape of points")
                self.points_fn = build_linear(self.points_init, vel_array, min_pos=np.array([0.0, -1.0]))
        
    def _get_segment(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (a,b) for segment i."""
        pts = self.points
        n = len(pts)
        if self.closed:
            a = pts[i % n]
            b = pts[(i + 1) % n]
        else:
            # clamp to valid segment range
            i = max(0, min(i, n - 2))
            a = pts[i]
            b = pts[i + 1]
        return a, b
    def set_points(self, points: np.ndarray | list[list[float]] | list[tuple[float, float]]) -> None:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            raise ValueError("points must be (N,2) with N>=2")
        self.points = pts

    def update(self, t: float) -> None:
        if self.points_fn is not None:
            self.set_points(self.points_fn(t))
            
    def _iter_segments(self):
        pts = self.points
        n = len(pts)
        seg_count = n if self.closed else (n - 1)
        for i in range(seg_count):
            a, b = self._get_segment(i)
            yield i, a, b

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
    
def resolve_wall_contact(body, contact: Contact, restitution: float, t: float):
    # push out
    body.pos = body.pos + contact.normal * contact.depth

    vn = float(np.dot(body.vel, contact.normal))
    if vn < 0.0:
        old_vel = body.vel.copy()
        body.vel = body.vel - (1.0 + restitution) * vn * contact.normal
        impulse = body.mass * float(np.linalg.norm(old_vel - body.vel))
        ev = HitWallEvent(t=t, body_id=body.id, body_theme_id = body.theme_id or "unknown", norm_vec=contact.normal, impulse=impulse)
        return ev, True

    return None, False

class Boundary(ABC):
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

    def resolve_collision(
        self,
        body: Body,
        restitution: float,
        t: float,
    ) -> List[BaseEvent]:
        # Simple approach: reflect velocity if outside ellipse
        events: List[BaseEvent] = []
        pos = body.pos
        vel = body.vel
        norm_vec = -np.array([pos[0] / (self.a ** 2), pos[1] / (self.b ** 2)])
        norm_vec /= np.linalg.norm(norm_vec)
        # Check if outside ellipse
        if (pos[0] ** 2) / (self.a ** 2) + (pos[1] ** 2) / (self.b ** 2) > 1.0:
            old_vel = vel.copy()
            vel -= (1 + restitution) * np.dot(vel, norm_vec) * norm_vec
            impulse = body.mass * np.linalg.norm(old_vel - vel)
            events.append(
                HitWallEvent(
                    t=t, 
                    body_id=body.id, 
                    body_theme_id=body.theme_id or "unknown", 
                    norm_vec=norm_vec, 
                    impulse=impulse
                )
            )
        return events

    def contains(self, pos: np.ndarray, radius: float = 0.0) -> bool:
        ar = self.a - radius
        br = self.b - radius
        if ar <= 0.0 or br <= 0.0:
            return False
        x, y = float(pos[0]), float(pos[1])
        return (x*x)/(ar*ar) + (y*y)/(br*br) <= 1.0

    def sample_position(
        self,
        radius: float = 0.0,
        policy: str = 'uniform'
    ) -> np.ndarray:
        while True:
            x = rng("physics").uniform(-self.a + radius, self.a - radius)
            y_limit = self.b * np.sqrt(1 - (x ** 2) / (self.a ** 2))
            y = rng("physics").uniform(-y_limit + radius, y_limit - radius)
            if self.contains(np.array([x, y]), radius=radius):
                return np.array([x, y], dtype=float)

    def bounds(self) -> tuple[float, float, float, float]:
        return -self.a, self.a, -self.b, self.b

@dataclass
class BoxBoundary(Boundary):
    width: float
    height: float
    
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
                    impulse=impulse
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
                    impulse=impulse
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
                    impulse=impulse
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
                    impulse=impulse
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
    
    def plot(self, ax=None, delta=0, **kwargs):
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

        new_events, did_bounce = self.outer.resolve_collision(body, restitution=restitution, t=t)
        events += new_events
        for _ in range(2):
            hit_any = False
            for w in self.walls:
                c = w.contact_circle(body.pos, r)
                if c is None:
                    continue

                ev, bounced = resolve_wall_contact(body, c, restitution=restitution, t=t)
                if ev is not None:
                    events.append(ev)
                did_bounce = did_bounce or bounced
                hit_any = True

            if not hit_any:
                break

        if did_bounce:
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
        if isinstance(self.outer, BoxBoundary):
            self.outer.plot(ax=ax, delta=delta, **kwargs)
        else:
            raise NotImplementedError("Plotting only implemented for BoxBoundary outer.")

        # Plot inner walls
        for w in self.walls:
            if isinstance(w, PolylineWall):
                pts = w.points
                ax.plot(pts[:, 0], pts[:, 1], color=kwargs.get('edgecolor', 'white'), linewidth=kwargs.get('linewidth', 1))
            else:
                raise NotImplementedError("Plotting only implemented for PolylineWall inner walls.")

        if created_fig:
            return fig, ax
        return ax
