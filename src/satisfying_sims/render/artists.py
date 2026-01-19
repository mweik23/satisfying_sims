from __future__ import annotations

from typing import Protocol, Callable, Dict, Tuple, Any
import numpy as np
import matplotlib.transforms as mtransforms

from satisfying_sims.core.recording import (
    WallStaticSnapshot as WallStatic,
    WallStateSnapshot as WallState,
)

# -----------------------------
# Public protocol
# -----------------------------

class WallArtist(Protocol):
    # NOTE: update should take WallState, not WallStatic. We keep t optional.
    def update(self, state: WallState, t: float | None = None) -> None: ...
    def remove(self) -> None: ...


# -----------------------------
# Style resolution
# -----------------------------

def resolve_style(user_style: dict | None, *, defaults: dict) -> dict:
    """
    Merge user style with defaults.
    - Missing keys → default
    - Explicit None → default

    This matches your "renderer owns defaults" policy.
    """
    style = {} if user_style is None else dict(user_style)
    for k, v in defaults.items():
        if k not in style or style[k] is None:
            style[k] = v
    return style


# -----------------------------
# Geometry registry
# -----------------------------

GeometryFn = Callable[[WallStatic, WallState, float | None], Tuple[np.ndarray, np.ndarray]]
GEOMETRY_BUILDERS: Dict[str, GeometryFn] = {}

def register_geometry(kind: str):
    def deco(fn: GeometryFn):
        GEOMETRY_BUILDERS[kind] = fn
        return fn
    return deco

def build_wall_geometry(kind: str, wall_static: WallStatic, wall_state: WallState, t: float | None) -> tuple[np.ndarray, np.ndarray]:
    try:
        fn = GEOMETRY_BUILDERS[kind]
    except KeyError:
        raise ValueError(f"No geometry builder registered for wall kind '{kind}'")
    return fn(wall_static, wall_state, t)


# -----------------------------
# Artist registry
# -----------------------------

ArtistBuilder = Callable[[Any, WallStatic, dict], WallArtist]  # ax is matplotlib Axes; keep Any to avoid importing types
ARTIST_BUILDERS: dict[str, ArtistBuilder] = {}

def register_wall_artist(kind: str):
    def deco(fn: ArtistBuilder):
        ARTIST_BUILDERS[kind] = fn
        return fn
    return deco


def build_wall_artist(ax, spec: WallStatic, default_style: dict, plot_layer=0) -> WallArtist:
    """
    Create a wall artist ONCE (during figure init).

    - Resolves style here so all artists get concrete values (no None/missing semantics inside artists).
    - Chooses a specialized artist if registered, otherwise falls back to a generic sampled-line artist.
    """
    if plot_layer != spec.layer:
        return None # Don't build an artist for this wall if it's not on the current plot layer
    # Resolve style once at build time. Artists just consume concrete style.
    resolved_default = dict(default_style) if default_style is not None else {}
    # Allow spec.style to be absent or None.
    spec_style = getattr(spec, "style", None)
    resolved_style = resolve_style(spec_style, defaults=resolved_default)

    builder = ARTIST_BUILDERS.get(spec.kind, None)
    if builder is not None:
        return builder(ax, spec, resolved_style)

    return build_sampled_line_artist(ax, spec, resolved_style)


def build_sampled_line_artist(ax, spec: WallStatic, resolved_style: dict) -> WallArtist:
    """
    Generic artist for any wall that can be represented as a polyline each frame.
    Allocates a Line2D once; update() rebuilds geometry and calls set_data().
    """
    # Draw something reasonable initially:
    # - Prefer init_state if present (it should be a WallStateSnapshot or equivalent).
    init_state = getattr(spec, "init_state", None)

    if init_state is not None:
        xs, ys = build_wall_geometry(spec.kind, spec, init_state, t=0.0)
    else:
        xs, ys = np.array([], dtype=float), np.array([], dtype=float)

    (line,) = ax.plot(xs, ys, **resolved_style)
    line.set_clip_on(False)

    class _SampledArtist:
        def update(self, state: WallState, t: float | None = None) -> None:
            xs2, ys2 = build_wall_geometry(spec.kind, spec, state, t)
            line.set_data(xs2, ys2)

        def remove(self) -> None:
            line.remove()

    return _SampledArtist()


# -----------------------------
# Specific artists
# -----------------------------

@register_wall_artist("CircularArcWall")
def build_circular_arc_artist(ax, spec: WallStatic, resolved_style: dict) -> WallArtist:
    """
    Transform-based arc artist:
      - Build a unit arc in local coords once using delta_angle.
      - Each frame, apply similarity transform (scale+rotate+translate) to the artist.

    Assumes:
      spec.init_state contains "delta_angle" and optionally "n".
      state.wall_attrs contains "center", "radius", "theta_start".
    """
    init = getattr(spec, "init_state", None)
    if init is None:
        raise ValueError("CircularArcWall artist requires spec.init_state")

    # Accept both dict-like and attribute-like init_state
    init_attrs = getattr(init, "wall_attrs", init)  # WallStateSnapshot might have wall_attrs; else assume dict

    if "delta_angle" not in init_attrs:
        raise ValueError("CircularArcWall init_state must include 'delta_angle'")

    delta = float(init_attrs["delta_angle"])
    n = int(init_attrs.get("n", getattr(spec, "sample_count", 256)))

    th = np.linspace(0.0, delta, n)
    x = np.cos(th)
    y = np.sin(th)

    (line,) = ax.plot(x, y, **resolved_style)  # local unit arc
    line.set_clip_on(False)
    class _ArcArtist:
        def __init__(self):
            self._last = None
        def update(self, state: WallState, t: float | None = None) -> None:
            p = state.wall_attrs  # per your snapshots
            cx, cy = p["center"]
            r = float(p["radius"])
            theta0 = float(p["angle_start"])
            key = (cx, cy, r, theta0)
            if key == self._last:
                return  # no change, skip transform update

            # Similarity transform: unit arc -> world
            T = (
                mtransforms.Affine2D()
                .scale(r, r)
                .rotate(theta0)
                .translate(float(cx), float(cy))
                + ax.transData
            )
            line.set_transform(T)
            self._last = key

        def remove(self) -> None:
            line.remove()

    return _ArcArtist()


@register_wall_artist("PolylineWall")
def build_polyline_artist(ax, spec: WallStatic, resolved_style: dict) -> WallArtist:
    """
    Simple polyline artist: allocate once, set_data each frame.
    Uses init_state for initial draw, wall_attrs for per-frame points.
    """
    init = getattr(spec, "init_state", None)
    if init is None:
        # If we don't have init_state, start empty and rely on update().
        pts = np.zeros((0, 2), dtype=float)
    else:
        init_attrs = getattr(init, "wall_attrs", init)
        pts = np.asarray(init_attrs.get("points", []), dtype=float)

    (line,) = ax.plot(pts[:, 0] if pts.size else [], pts[:, 1] if pts.size else [], **resolved_style)
    line.set_clip_on(False)
    class _PolylineArtist:
        def __init__(self):
            self._last = None
        def update(self, state: WallState, t: float | None = None) -> None:
            pts2 = np.asarray(state.wall_attrs["points"], dtype=float)
            if np.array_equal(pts2, self._last):
                return  # no change, skip set_data
            self._last = pts2
            line.set_data(pts2[:, 0], pts2[:, 1])

        def remove(self) -> None:
            line.remove()

    return _PolylineArtist()


# -----------------------------
# Geometry builders
# -----------------------------

@register_geometry("PolylineWall")
def build_polyline_geometry(ws: WallStatic, st: WallState, t: float | None = None):
    # Prefer wall_attrs (your snapshot format). Fall back to attributes.
    if hasattr(st, "wall_attrs"):
        pts = np.asarray(st.wall_attrs.get("points", []), dtype=float)
    else:
        pts = np.asarray(getattr(st, "points", []), dtype=float)

    if pts.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return pts[:, 0], pts[:, 1]


@register_geometry("CircularArcWall")
def build_circular_arc_geometry(ws: WallStatic, st: WallState, t: float | None = None):
    """
    Sample the arc directly in world coords.

    Uses your snapshot conventions if available:
      st.wall_attrs: center, radius, theta_start, delta_angle OR theta_end
      ws.sample_count (or ws.init_state["n"]) controls sampling count.

    This is used by the generic sampled-line artist (and can also be useful for debugging).
    """
    if hasattr(st, "wall_attrs"):
        p = st.wall_attrs
        c = np.asarray(p["center"], dtype=float).reshape(2)
        r = float(p["radius"])
        a0 = float(p.get("theta_start", p.get("angle_start", 0.0)))
        if "theta_end" in p or "angle_end" in p:
            a1 = float(p.get("theta_end", p.get("angle_end")))
        else:
            a1 = a0 + float(p["delta_angle"])
    else:
        c = np.asarray(getattr(st, "center"), dtype=float).reshape(2)
        r = float(getattr(st, "radius"))
        a0 = float(getattr(st, "theta_start", getattr(st, "angle_start")))
        a1 = float(getattr(st, "theta_end", getattr(st, "angle_end", a0 + float(getattr(st, "delta_angle")))))

    # Sampling count precedence:
    #   1) ws.sample_count
    #   2) ws.init_state["n"] (if present)
    n = int(getattr(ws, "sample_count", 256))
    init = getattr(ws, "init_state", None)
    if init is not None:
        init_attrs = getattr(init, "wall_attrs", init)
        if isinstance(init_attrs, dict) and "n" in init_attrs:
            n = int(init_attrs["n"])

    # ensure CCW (wrap)
    if a1 < a0:
        a1 += 2.0 * np.pi

    th = np.linspace(a0, a1, n)
    xs = c[0] + r * np.cos(th)
    ys = c[1] + r * np.sin(th)
    return xs, ys
