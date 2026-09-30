import math
import random
import tkinter as tk

# ---------------------------------------------------------------- model

def sigmoid(x):
    if x < -40: return 0.0
    if x > 40: return 1.0
    return 1.0 / (1.0 + math.exp(-x))

XOR = [((0, 0), 0.0), ((0, 1), 1.0), ((1, 0), 1.0), ((1, 1), 0.0)]

class Net:
    """2-2-1 netwerk met handgeschreven gradient descent."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.p = [random.uniform(-1, 1) for _ in range(9)]
        # index: 0=w1 1=w2 2=b1 | 3=w3 4=w4 5=b2 | 6=w5 7=w6 8=b3
        self.steps = 0
        self.loss = None
        self.history = {n: [] for n in ("h1", "h2", "out")}

    def forward(self, x1, x2, p=None):
        p = p or self.p
        h1 = sigmoid(p[0] * x1 + p[1] * x2 + p[2])
        h2 = sigmoid(p[3] * x1 + p[4] * x2 + p[5])
        y = sigmoid(p[6] * h1 + p[7] * h2 + p[8])
        return h1, h2, y

    def compute_loss(self):
        tot = 0.0
        for (x1, x2), t in XOR:
            y = self.forward(x1, x2)[2]
            tot += (y - t) ** 2
        return tot / len(XOR)

    def train_step(self, lr):
        grad = [0.0] * 9
        for (x1, x2), t in XOR:
            p = self.p
            z1 = p[0] * x1 + p[1] * x2 + p[2]; h1 = sigmoid(z1)
            z2 = p[3] * x1 + p[4] * x2 + p[5]; h2 = sigmoid(z2)
            z3 = p[6] * h1 + p[7] * h2 + p[8]; y = sigmoid(z3)
            # loss = gemiddelde van (y - t)^2  ->  dL/dy = 2(y - t) / 4
            d3 = (2 * (y - t) / len(XOR)) * y * (1 - y)
            grad[6] += d3 * h1
            grad[7] += d3 * h2
            grad[8] += d3
            d1 = d3 * p[6] * h1 * (1 - h1)
            d2 = d3 * p[7] * h2 * (1 - h2)
            grad[0] += d1 * x1
            grad[1] += d1 * x2
            grad[2] += d1
            grad[3] += d2 * x1
            grad[4] += d2 * x2
            grad[5] += d2
        for i in range(9):
            self.p[i] -= lr * grad[i]
        self.steps += 1
        self.loss = self.compute_loss()
        # onthoud de "route" van elke neuron
        self.history["h1"].append((self.p[0], self.p[1], self.p[2]))
        self.history["h2"].append((self.p[3], self.p[4], self.p[5]))
        self.history["out"].append((self.p[6], self.p[7], self.p[8]))

# ---------------------------------------------------------------- stijl

BG     = "#15181d"
PANEL  = "#1d2129"
INK    = "#e8e6e3"
MUTED  = "#9a958b"
ACCENT = "#e8723f"
GOOD   = "#4caf7d"
GRIDLN = "#2a2f3a"
LINE_W = "#7fb8a4"   # positief gewicht
LINE_A = "#e8723f"   # negatief gewicht

NODE_R = 24
NODES = {"in0": (70, 90), "in1": (70, 210),
         "h1": (200, 90), "h2": (200, 210),
         "out": (320, 150)}
EDGES = [("in0", "h1", 0), ("in1", "h1", 1), ("in0", "h2", 3), ("in1", "h2", 4),
         ("h1", "out", 6), ("h2", "out", 7)]
NEURON_PARAMS = {"h1": (0, "hidden 1"), "h2": (3, "hidden 2"), "out": (6, "output")}

# ---------------------------------------------------------------- app

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XOR trainen — gradient descent live")
        self.configure(bg=BG)
        self.net = Net()
        self.inputs = [0, 1]
        self.hover = None
        self.training = False
        self.stuck = 0
        self.best = None
        self._build()
        self._tick_draw()

    # ---------- lay-out ----------
    def _build(self):
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=14, pady=(12, 2))

        self.btn_train = tk.Button(bar, text="▶ Train", width=9, relief="flat",
                                   bg=ACCENT, fg="#15181d",
                                   font=("Segoe UI", 10, "bold"),
                                   command=self._toggle_train)
        self.btn_train.pack(side="left")
        tk.Button(bar, text="Reset", width=7, relief="flat", bg=PANEL, fg=INK,
                  command=self._reset).pack(side="left", padx=6)

        tk.Label(bar, text="  leersnelheid", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left")
        self.lr = tk.Scale(bar, from_=0.5, to=20, resolution=0.5,
                           orient="horizontal", length=110, bg=BG, fg=INK,
                           troughcolor=GRIDLN, highlightthickness=0,
                           showvalue=False)
        self.lr.set(10.0)
        self.lr.pack(side="left")
        tk.Label(bar, text="snelheid", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=(10, 0))
        self.speed = tk.Scale(bar, from_=1, to=40, orient="horizontal",
                              length=90, bg=BG, fg=INK, troughcolor=GRIDLN,
                              highlightthickness=0, showvalue=False)
        self.speed.set(8)
        self.speed.pack(side="left")

        self.status = tk.StringVar()
        tk.Label(bar, textvariable=self.status, bg=BG, fg=ACCENT,
                 font=("Consolas", 11, "bold")).pack(side="right")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=6)

        self.cv = tk.Canvas(body, width=400, height=300, bg=BG,
                            highlightthickness=0)
        self.cv.pack(side="left")
        self.cv.bind("<Motion>", self._on_motion)
        self.cv.bind("<Leave>", lambda e: self._set_hover(None))
        self.cv.bind("<Button-1>", self._on_click)

        side = tk.Frame(body, bg=BG)
        side.pack(side="left", fill="both", expand=True, padx=(14, 0))

        self.hover_label = tk.StringVar(value="beweeg over een neuron…")
        tk.Label(side, textvariable=self.hover_label, bg=BG, fg=INK,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.plot = tk.Canvas(side, width=380, height=170, bg=PANEL,
                              highlightthickness=0)
        self.plot.pack(anchor="w", pady=6)
        self.plot_legend = tk.Label(side, text="", bg=BG, fg=MUTED,
                                    font=("Consolas", 9), justify="left")
        self.plot_legend.pack(anchor="w")

        tk.Label(side,
                 text="klik op een input-neuron om 0/1 te wisselen —\n"
                      "de activatie loopt meteen door het netwerk",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9), justify="left"
                 ).pack(anchor="w", pady=(8, 0))

    # ---------- training ----------
    def _toggle_train(self):
        self.training = not self.training
        self.btn_train.config(text="⏸ Pauze" if self.training else "▶ Train")
        if self.training:
            self._train_loop()

    def _train_loop(self):
        if not self.training:
            return
        for _ in range(int(self.speed.get())):
            self.net.train_step(self.lr.get())
        # zit het model vast in een slecht punt (geen verbetering
        # meer terwijl de loss hoog is)? start dan opnieuw
        if self.best is None or self.net.loss < self.best - 0.0005:
            self.best = self.net.loss
            self.stuck = 0
        else:
            self.stuck += 1
            if self.stuck > 100:
                self.net.reset()
                self.best = None
                self.stuck = 0
        if self.net.loss < 0.01:
            self.training = False
            self.btn_train.config(text="▶ Train")
        self._tick_draw()
        if self.training:
            self.after(30, self._train_loop)

    def _reset(self):
        self.training = False
        self.btn_train.config(text="▶ Train")
        self.net.reset()
        self.stuck = 0
        self.best = None
        self._tick_draw()

    # ---------- interactie ----------
    def _on_motion(self, e):
        for name, (x, y) in NODES.items():
            if (e.x - x) ** 2 + (e.y - y) ** 2 <= (NODE_R + 6) ** 2:
                self._set_hover(name)
                return
        self._set_hover(None)

    def _on_click(self, e):
        for i, name in enumerate(("in0", "in1")):
            x, y = NODES[name]
            if (e.x - x) ** 2 + (e.y - y) ** 2 <= (NODE_R + 6) ** 2:
                self.inputs[i] = 1 - self.inputs[i]
                self._tick_draw()
                return

    def _set_hover(self, name):
        if name != self.hover:
            self.hover = name
            self._draw_plot()

    # ---------- tekenen ----------
    def _tick_draw(self):
        self.cv.delete("all")
        h1, h2, y = self.net.forward(*self.inputs)
        acts = {"in0": self.inputs[0], "in1": self.inputs[1],
                "h1": h1, "h2": h2, "out": y}
        # verbindingen: dikte ~ |gewicht|, kleur ~ teken
        for a, b, idx in EDGES:
            xa, ya = NODES[a]; xb, yb = NODES[b]
            w = self.net.p[idx]
            width = 1 + int(min(abs(w), 12))
            color = LINE_W if w >= 0 else LINE_A
            self.cv.create_line(xa, ya, xb, yb, fill=color, width=width)
        # neuronen
        for name, (x, yc) in NODES.items():
            act = acts[name]
            k = int(29 + act * 200)
            fill = f"#{k:02x}{k:02x}{k:02x}"
            outline = ACCENT if name == self.hover else GRIDLN
            self.cv.create_oval(x - NODE_R, yc - NODE_R, x + NODE_R,
                                yc + NODE_R, fill=fill, outline=outline,
                                width=2 if name == self.hover else 1)
            label = (f"{self.inputs[0]}" if name == "in0"
                     else f"{self.inputs[1]}" if name == "in1"
                     else f"{act:.2f}")
            self.cv.create_text(x, yc, text=label,
                                font=("Consolas", 11, "bold"),
                                fill="#15181d" if act > 0.5 else INK)
        self.cv.create_text(70, 40, text="inputs (klik!)",
                            fill=MUTED, font=("Segoe UI", 9))
        self.cv.create_text(200, 40, text="hidden",
                            fill=MUTED, font=("Segoe UI", 9))
        self.cv.create_text(320, 40, text="output",
                            fill=MUTED, font=("Segoe UI", 9))
        # status
        loss = self.net.loss if self.net.loss is not None else self.net.compute_loss()
        self.status.set(f"stap {self.net.steps:>5}   loss = {loss:.4f}")
        if self.hover:
            self._draw_plot()

    def _draw_plot(self):
        self.plot.delete("all")
        W, H = 380, 170
        self.plot.create_rectangle(0, 0, W, H, fill=PANEL, outline="")
        name = self.hover
        if not name or name not in NEURON_PARAMS:
            self.hover_label.set("beweeg over een neuron…")
            self.plot_legend.config(text="")
            return
        base, desc = NEURON_PARAMS[name]
        hist = self.net.history[name]
        self.hover_label.set(f"{desc} — geprobeerde waarden per trainingsstap")
        if len(hist) < 2:
            self.plot.create_text(W / 2, H / 2,
                                  text="nog geen trainingsstappen — druk op Train",
                                  fill=MUTED, font=("Segoe UI", 10))
            self.plot_legend.config(text="")
            return
        series = list(zip(*hist))          # 3 lijsten: w_a, w_b, bias
        labels = ("w in-1", "w in-2", "bias")
        colors = ("#7fb8a4", "#c9a227", ACCENT)
        lo = min(min(s) for s in series)
        hi = max(max(s) for s in series)
        if hi - lo < 1e-6:
            hi = lo + 1
        pad = 10
        for s, lab, col in zip(series, labels, colors):
            n = len(s)
            pts = []
            for i, v in enumerate(s):
                px = pad + i * (W - 2 * pad) / (n - 1)
                py = H - pad - (v - lo) / (hi - lo) * (H - 2 * pad)
                pts += [px, py]
            self.plot.create_line(*pts, fill=col, width=2)
            self.plot.create_oval(pts[-2] - 3, pts[-1] - 3, pts[-2] + 3,
                                  pts[-1] + 3, fill=col, outline="")
        self.plot.create_text(pad, 8, text=f"max {hi:+.2f}", fill=MUTED,
                              font=("Consolas", 8), anchor="w")
        self.plot.create_text(pad, H - 6, text=f"min {lo:+.2f}", fill=MUTED,
                              font=("Consolas", 8), anchor="w")
        cur = hist[-1]
        self.plot_legend.config(
            text="   ".join(f"{c} {l}={v:+.2f}"
                            for c, l, v in zip(("■", "■", "■"), labels, cur)))

if __name__ == "__main__":
    random.seed()
    App().mainloop()
