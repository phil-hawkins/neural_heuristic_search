try:
    from gym.envs.classic_control import rendering
except:
    print("unable to import rendering library, running headless")
import math


class View():
    viewport_w = 800
    viewport_h = 600
    scale = 80.
    unit_per_m = 1
    body_colours = {
        'node' : ((121/256, 109/256, 214/256), (35/256, 22/256, 138/256)),
        'target' : (0, 0, 0),
        'strut' : (0, 0, 0),
        'support' : ((93/256, 125/256, 57/256), (57/256, 82/256, 29/256))
    }
    node_radius = 0.1
    pin_h = node_radius * 3
    pin_w = pin_h / 2

    def __init__(self):
        self._viewer = rendering.Viewer(
                self.viewport_w, 
                self.viewport_h
            )
        self._viewer.set_bounds(
            -self.viewport_w/(self.scale*2.), 
            self.viewport_w/(self.scale*2.), 
            -self.viewport_h/(self.scale*2.), 
            self.viewport_h/(self.scale*2.))
        self._window_still_open = True

    def show(self, state):
        # show target
        t = rendering.Transform(translation=state.target_geometry())
        xc, yc = x1, y1 = x2, y2 = state.target_geometry()
        o = (3*self.node_radius)
        x1 -= o
        y1 -= o
        x2 += o
        y2 += o
        self._viewer.draw_line((xc, y1), (xc, y2), color=self.body_colours['target'])
        self._viewer.draw_line((x1, yc), (x2, yc), color=self.body_colours['target'])

        # show spokes
        for epos in state.edge_geometry():
            start, end = epos
            self._viewer.draw_line(start, end, color=self.body_colours['strut']).add_attr(rendering.LineWidth(1))
            
        # show hubs
        pin_shape = [(0, 0), (-self.pin_w, -self.pin_h), (self.pin_w, -self.pin_h)]
        hub_shape = []
        for i in range(6):
            ang = (2*i+1) * math.pi / 6
            hub_shape.append((math.cos(ang)*self.node_radius, math.sin(ang)*self.node_radius))
        for n in state.node_geometry():
            npos, pinned = n
            t = rendering.Transform(translation=npos)
            if pinned:
                # draw pin
                self._viewer.draw_polygon(v=pin_shape, color=self.body_colours['support'][0]).add_attr(t)
                self._viewer.draw_polygon(v=pin_shape, color=self.body_colours['support'][1], filled=False, linewidth=2).add_attr(t)
            # draw hub
            self._viewer.draw_polygon(v=hub_shape, color=self.body_colours['node'][0]).add_attr(t)
            self._viewer.draw_polygon(v=hub_shape, color=self.body_colours['node'][1], filled=False, linewidth=2).add_attr(t)

        self._window_still_open = self._viewer.render()

    @property
    def window_still_open(self):
        return self._window_still_open

