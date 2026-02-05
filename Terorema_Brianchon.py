import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

EPS = 1e-6

# ------------------ Geometría ------------------

def tangent_ellipse(a, b, x0, y0):
    return x0/(a*a), y0/(b*b), -1

def intersect_lines(L1, L2):
    A1, B1, C1 = L1
    A2, B2, C2 = L2
    D = A1*B2 - A2*B1
    if abs(D) < EPS:
        return None
    x = (B1*C2 - B2*C1) / D
    y = (A2*C1 - A1*C2) / D
    return np.array([x, y])

def intersect_segments(P1, P2, P3, P4):
    def det(a, b):
        return a[0]*b[1] - a[1]*b[0]

    xdiff = (P1[0]-P2[0], P3[0]-P4[0])
    ydiff = (P1[1]-P2[1], P3[1]-P4[1])

    div = det(xdiff, ydiff)
    if abs(div) < EPS:
        return None

    d = (det(P1, P2), det(P3, P4))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y])

# ------------------ App ------------------

class BrianchonApp:

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        self.a, self.b = 4, 2

        self.points = []
        self.tangents = []
        self.intersections = []
        self.brianchon_point = None

        self.drag_index = None

        self.ax.set_aspect('equal')
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-6, 6)

        self.init_sliders()

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

        self.redraw()
        plt.show()

    # ------------------ UI ------------------

    def init_sliders(self):
        ax_a = plt.axes([0.15, 0.1, 0.65, 0.03])
        ax_b = plt.axes([0.15, 0.05, 0.65, 0.03])

        self.slider_a = Slider(ax_a, 'a', 1, 8, valinit=self.a)
        self.slider_b = Slider(ax_b, 'b', 1, 6, valinit=self.b)

        self.slider_a.on_changed(self.update_conic)
        self.slider_b.on_changed(self.update_conic)

    # ------------------ Eventos ------------------

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        # Buscar punto cercano
        for i, (px, py) in enumerate(self.points):
            if np.hypot(px-x, py-y) < 0.3:
                self.drag_index = i
                return

        # Añadir punto (máx 6)
        if len(self.points) < 6 and event.button == 1:
            t = np.arctan2(y/self.b, x/self.a)
            self.points.append((self.a*np.cos(t), self.b*np.sin(t)))
            self.redraw()

        # Borrar
        if event.button == 3:
            for i, (px, py) in enumerate(self.points):
                if np.hypot(px-x, py-y) < 0.3:
                    self.points.pop(i)
                    self.redraw()
                    return

    def on_motion(self, event):
        if self.drag_index is None or event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        t = np.arctan2(y/self.b, x/self.a)
        self.points[self.drag_index] = (self.a*np.cos(t), self.b*np.sin(t))
        self.redraw()

    def on_release(self, event):
        self.drag_index = None

    # ------------------ Lógica ------------------

    def update_conic(self, val):
        self.a = self.slider_a.val
        self.b = self.slider_b.val
        self.redraw()

    def compute_tangents(self):
        self.tangents = [tangent_ellipse(self.a, self.b, x, y) for x, y in self.points]

    def compute_intersections(self):
        self.intersections = []
        n = len(self.tangents)
        for i in range(n):
            P = intersect_lines(self.tangents[i], self.tangents[(i+1) % n])
            if P is not None and np.linalg.norm(P) < 50:
                self.intersections.append(tuple(P))

    def compute_brianchon(self):
        if len(self.intersections) != 6:
            self.brianchon_point = None
            return
        I = self.intersections
        P = intersect_segments(I[0], I[3], I[1], I[4])
        self.brianchon_point = tuple(P) if P is not None else None

    # ------------------ Dibujo ------------------

    def redraw(self):
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-6, 6)

        t = np.linspace(0, 2*np.pi, 400)
        self.ax.plot(self.a*np.cos(t), self.b*np.sin(t), 'k')

        self.compute_tangents()
        for A, B, C in self.tangents:
            xs = np.linspace(-10, 10, 2)
            ys = (-A*xs - C)/B
            self.ax.plot(xs, ys, 'r--', alpha=0.6)

        for x, y in self.points:
            self.ax.plot(x, y, 'ro', markersize=7)

        self.compute_intersections()
        for x, y in self.intersections:
            self.ax.plot(x, y, 'bo')

        for i in range(len(self.intersections)):
            p1 = self.intersections[i]
            p2 = self.intersections[(i+1) % len(self.intersections)]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')

        self.compute_brianchon()
        if self.brianchon_point:
            bx, by = self.brianchon_point

            # Punto de Brianchon
            self.ax.plot(bx, by, 'go', markersize=10, zorder=5)

            # Segmentos concurrentes (Brianchon)
            if len(self.intersections) == 6:
                pairs = [(0, 3), (1, 4), (2, 5)]
                for i, j in pairs:
                    p1 = self.intersections[i]
                    p2 = self.intersections[j]
                    self.ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        color='purple',
                        linewidth=2,
                        alpha=0.9,
                        zorder=4
                    )


        self.fig.canvas.draw_idle()

BrianchonApp()