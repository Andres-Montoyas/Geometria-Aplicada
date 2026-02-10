import numpy as np
import matplotlib.pyplot as plt

# =====================
# Utilidades geométricas
# =====================

def intersect_lines(A, B, C, D):
    """Intersección de las rectas AB y CD"""
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D

    den = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if abs(den) < 1e-9:
        return None

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / den
    return np.array([px, py])

def project_to_ellipse(x, y, a=3, b=2, n=1000):
    t = np.linspace(0, 2*np.pi, n)
    xs = a * np.cos(t)
    ys = b * np.sin(t)

    pts = np.column_stack((xs, ys))
    dists = np.linalg.norm(pts - np.array([x, y]), axis=1)
    return pts[np.argmin(dists)]


def fit_line(points):
    """Recta por mínimos cuadrados"""
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b


# =====================
# Clase principal
# =====================

class PascalDemo:
    def __init__(self):
        self.a = 3
        self.b = 2

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")

        # control de zoom: 1.0 = zoom por defecto
        self.zoom = 1.0
        self.base_lim = 5
        self.ax.set_xlim(-self.base_lim * self.zoom, self.base_lim * self.zoom)
        self.ax.set_ylim(-self.base_lim * self.zoom, self.base_lim * self.zoom)

        self.points = []
        self.dragging = None

        self.draw_conic()

        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_scroll = self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        plt.show()

    def draw_conic(self):
        t = np.linspace(0, 2*np.pi, 400)
        x = self.a * np.cos(t)
        y = self.b * np.sin(t)
        self.ax.plot(x, y, 'k')


    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        for i, p in enumerate(self.points):
            if np.linalg.norm(p - np.array([event.xdata, event.ydata])) < 0.2:
                self.dragging = i
                return

        if len(self.points) < 6:
            p = project_to_ellipse(event.xdata, event.ydata, self.a, self.b)
            self.points.append(p)
            self.redraw()

    def on_release(self, event):
        self.dragging = None

    def on_motion(self, event):
        if self.dragging is None or event.inaxes != self.ax:
            return
        self.points[self.dragging] = project_to_ellipse(event.xdata, event.ydata, self.a, self.b)

        self.redraw()

    def plot_infinite_line(self, A, B, **kwargs):
        """Dibuja la recta definida por A-B extendida a los límites actuales del eje."""
        x1, y1 = A
        x2, y2 = B
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        if abs(x2 - x1) < 1e-9:
            xs = np.array([x1, x1])
            ys = np.array([ymin, ymax])
        else:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            xs = np.linspace(xmin, xmax, 200)
            ys = m * xs + b

        self.ax.plot(xs, ys, **kwargs)

    def redraw(self):
        self.ax.cla()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-self.base_lim * self.zoom, self.base_lim * self.zoom)
        self.ax.set_ylim(-self.base_lim * self.zoom, self.base_lim * self.zoom)

        self.draw_conic()

        if len(self.points) == 6:
            P = self.points

            edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)]
            for i, j in edges:
                # dibujar la recta extendida (infinite) en estilo dashed
                self.plot_infinite_line(P[i], P[j], color='black', linestyle='--', linewidth=1, alpha=0.9)

            inter_pairs = [((0,1),(3,4)), ((1,2),(4,5)), ((2,3),(5,0))]
            inters = []

            for (i,j),(k,l) in inter_pairs:
                I = intersect_lines(P[i], P[j], P[k], P[l])
                if I is not None:
                    inters.append(I)
                    self.ax.scatter(I[0], I[1], color="red", s=60)

            if len(inters) == 3:
                inters = np.array(inters)
                m, b = fit_line(inters)
                xs = np.linspace(-5, 5, 200)
                ys = m*xs + b
                self.ax.plot(xs, ys, 'r--')

        for p in self.points:
            self.ax.scatter(p[0], p[1], color="blue", s=60)

        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        """Zoom in/out con la rueda del ratón."""
        # event.step >0 scroll up -> zoom in (mas cercano), step<0 zoom out (mas alejado)
        factor = 0.9 if event.step > 0 else 1.1
        # limitar rango de zoom
        new_zoom = self.zoom * factor
        if 0.2 <= new_zoom <= 10:
            self.zoom = new_zoom
            self.redraw()


# =====================
# Ejecutar
# =====================

PascalDemo()
