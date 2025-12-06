# src/satisfying_sims/core/boundary.py
from abc import ABC, abstractmethod
from typing import List
from .shapes import Body  # see below
from .events import HitWallEvent, BaseEvent
from dataclasses import dataclass
import numpy as np
# at top of file
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
        
        # Left wall
        if pos[0] - r < 0.0:
            old_vx = vel[0]
            pos[0] = r
            vel[0] = -restitution * vel[0]
            impulse = body.mass * abs(old_vx - vel[0])
            events.append(HitWallEvent(t=t, body_id=body.id, norm_vec=np.array([1.0, 0.0]), impulse=impulse))

        # Right wall
        if pos[0] + r > self.width:
            old_vx = vel[0]
            pos[0] = self.width - r
            vel[0] = -restitution * vel[0]
            impulse = body.mass * abs(old_vx - vel[0])
            events.append(HitWallEvent(t=t, body_id=body.id, norm_vec=np.array([-1.0, 0.0]), impulse=impulse))

        # Top wall
        if pos[1] - r < 0.0:
            old_vy = vel[1]
            pos[1] = r
            vel[1] = -restitution * vel[1]
            impulse = body.mass * abs(old_vy - vel[1])
            events.append(HitWallEvent(t=t, body_id=body.id, norm_vec=np.array([0.0, 1.0]), impulse=impulse))

        # Bottom wall
        if pos[1] + r > self.height:
            old_vy = vel[1]
            pos[1] = self.height - r
            vel[1] = -restitution * vel[1]
            impulse = body.mass * abs(old_vy - vel[1])
            events.append(HitWallEvent(t=t, body_id=body.id, norm_vec=np.array([0.0, -1.0]), impulse=impulse))
        return events
    
    
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
            fill=False,
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