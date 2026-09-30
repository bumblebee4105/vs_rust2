import math
import tkinter as tk

# ---------------------------------------------------------------- model

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

TARGETS = {
    "AND": {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
    "OR":  {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
    "XOR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
}

# bekende werkende gewichten voor XOR met 2-2-1 netwerk (voor de hint-knop)
# h1 = OR-poort, h2 = AND-poort, output = h1 EN NIET h2
XOR_SOLUTION = dict(w1=8.0, w2=8.0, b1=-4.0,
                    w3=8.0, w4=8.0, b2=-12.0,
                    w5=8.0, w6=-8.0, b3=-4.0)

class Net:
    """Eén neuron (y = sigmoid(w1*x1 + w2*x2 + b)), of 2-2-1 met hidden layer."""

    def __init__(self):
        self.params = dict(w1=0.0, w2=0.0, b=0.0, b1=0.0,
                           w3=0.0, w4=0.0, b2=0.0,
                           w5=0.0, w6=0.0, b3=0.0)

    def forward(self, x1, x2, hidden):
        p = self.params
        if not hidden:
            return sigmoid(p["w1"] * x1 + p["w2"] * x2 + p["b"])
        h1 = sigmoid(p["w1"] * x1 + p["w2"] * x2 + p["b1"])
        h2 = sigmoid(p["w3"] * x1 + p["w4"] * x2 + p["b2"])
        return sigmoid(p["w5"] * h1 + p["w6"] * h2 + p["b3"])

    def loss(self, target, hidden):
        tot = 0.0
        for (x1, x2), t in TARGETS[target].items():
            fout = self.forward(x1, x2, hidden) - t
            tot += fout * fout
        return tot / 4.0

# ---------------------------------------------------------------- kleuren (donker, passend bij de python-box)

BG      = "#15181d"
PANEL   = "#1d2129"
INK     = "#e8e6e3"
MUTED   = "#9a958b"
ACCENT  = "#e8723f"
GOOD    = "#4caf7d"
BAD     = "#e05b4b"
GRIDLN  = "#2a2f3a"

CELLS = 26                 # resolutie van de heatmap

# ---------------------------------------------------------------- app

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XOR live — de klas bouwt het netwerk")
        self.configure(bg=BG)
        self.net = Net()
        self.target = tk.StringVar(value="XOR")
        self.hidden = tk.BooleanVar(value=False)
        self.sliders = {}
        self._build()
        self.update_all()

    # ---------- lay-out ----------
    def _build(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=14, pady=(12, 4))

        tk.Label(top, text="Poort:", bg=BG, fg=MUTED).pack(side="left")
        for name in ("AND", "OR", "XOR"):
            tk.Radiobutton(top, text=name, value=name, variable=self.target,
                           bg=BG, fg=INK, selectcolor=BG, activebackground=BG,
                           command=self.update_all).pack(side="left", padx=4)

        tk.Checkbutton(top, text="hidden layer (2 neuronen)", variable=self.hidden,
                       bg=BG, fg=INK, selectcolor=BG, activebackground=BG,
                       command=self._rebuild_sliders).pack(side="left", padx=(26, 0))

        tk.Button(top, text="Hint: gouden waarden", bg=PANEL, fg=ACCENT,
                  relief="flat", command=self._hint).pack(side="right")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=6)

        # heatmap
        self.map_size = 380
        self.map = tk.Canvas(body, width=self.map_size, height=self.map_size,
                             bg=BG, highlightthickness=0)
        self.map.pack(side="left", padx=(0, 14))
        self.map_cells = [[self.map.create_rectangle(0, 0, 0, 0, outline="")
                           for _ in range(CELLS)] for _ in range(CELLS)]
        self.map.bind("<Configure>", lambda e: self._draw_map())

        # rechterkolom
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(right, text="Gewichten & bias — roep maar!", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w")

        self.slider_frame = tk.Frame(right, bg=BG)
        self.slider_frame.pack(fill="x", pady=4)
        self._rebuild_sliders()

        self.loss_var = tk.StringVar()
        tk.Label(right, textvariable=self.loss_var, bg=BG, fg=ACCENT,
                 font=("Consolas", 20, "bold")).pack(anchor="w", pady=(8, 0))

        self.table = tk.Canvas(right, width=340, height=118, bg=PANEL,
                               highlightthickness=0)
        self.table.pack(anchor="w", pady=(10, 0))

    def _rebuild_sliders(self):
        for w in self.slider_frame.winfo_children():
            w.destroy()
        self.sliders.clear()
        if self.hidden.get():
            spec = [("w1", "w1"), ("w2", "w2"), ("b1", "bias h1"),
                    ("w3", "w3"), ("w4", "w4"), ("b2", "bias h2"),
                    ("w5", "w5"), ("w6", "w6"), ("b3", "bias y")]
        else:
            spec = [("w1", "w1"), ("w2", "w2"), ("b", "bias")]
        for i, (key, label) in enumerate(spec):
            f = tk.Frame(self.slider_frame, bg=BG)
            f.grid(row=i % 3, column=i // 3, sticky="w", padx=(0, 16), pady=2)
            tk.Label(f, text=label, bg=BG, fg=MUTED, width=7, anchor="w",
                     font=("Consolas", 9)).pack(side="left")
            val = self.net.params.get(key, 0.0)
            s = tk.Scale(f, from_=-8, to=8, resolution=0.5, orient="horizontal",
                         length=120, bg=BG, fg=INK, troughcolor=GRIDLN,
                         highlightthickness=0, showvalue=False,
                         command=lambda v, k=key: self._slide(k, float(v)))
            s.set(val)
            s.pack(side="left")
            self.sliders[key] = s

    def _hint(self):
        for key, val in XOR_SOLUTION.items():
            self.net.params[key] = val
        self.hidden.set(True)
        self._rebuild_sliders()
        self.update_all()

    def _slide(self, key, val):
        self.net.params[key] = val
        self.update_all()

    # ---------- tekenen ----------
    def update_all(self):
        self._draw_map()
        self._draw_table()
        loss = self.net.loss(self.target.get(), self.hidden.get())
        verdict = "  ✓ onder 0,10!" if loss < 0.10 else ""
        self.loss_var.set(f"loss = {loss:.4f}{verdict}")

    def _draw_map(self):
        size = self.map_size
        cell = size / CELLS
        hidden = self.hidden.get()
        for r in range(CELLS):
            y = r / (CELLS - 1)
            for c in range(CELLS):
                x = c / (CELLS - 1)
                out = self.net.forward(x, y, hidden)
                # kleur: grijs -> oranje naarmate de output naar 1 gaat
                k = int(30 + out * 200)
                col = f"#{k:02x}{int(30 + out * 84):02x}{int(38 + out * 24):02x}"
                self.map.itemconfig(self.map_cells[r][c], fill=col,
                                    outline=GRIDLN)
                self.map.coords(self.map_cells[r][c],
                                c * cell, r * cell, (c + 1) * cell, (r + 1) * cell)
        # de vier hoekpunten van de waarheidstabel er bovenop
        t = TARGETS[self.target.get()]
        for (x, y), target_val in t.items():
            px, py = x * size, y * size
            out = self.net.forward(x, y, hidden)
            kleur = GOOD if abs(out - target_val) < 0.5 else BAD
            r = 11
            self.map.create_oval(px - r, py - r, px + r, py + r,
                                 outline=kleur, width=3, fill="")
            self.map.create_text(px, py - r - 9, text=str(target_val),
                                 fill=INK, font=("Segoe UI", 10, "bold"))
        self.map.create_text(size / 2, size - 8, text="x1  →",
                             fill=MUTED, font=("Consolas", 10))
        self.map.create_text(12, size / 2, text="x2", fill=MUTED,
                             font=("Consolas", 10))

    def _draw_table(self):
        self.table.delete("all")
        hidden = self.hidden.get()
        t = TARGETS[self.target.get()]
        self.table.create_text(60, 16, text="x1 x2", fill=MUTED,
                               font=("Consolas", 10, "bold"))
        self.table.create_text(150, 16, text="output", fill=MUTED,
                               font=("Consolas", 10, "bold"))
        self.table.create_text(250, 16, text="target", fill=MUTED,
                               font=("Consolas", 10, "bold"))
        self.table.create_line(10, 28, 330, 28, fill=GRIDLN)
        for i, ((x1, x2), tv) in enumerate(sorted(t.items())):
            y = 48 + i * 18
            out = self.net.forward(x1, x2, hidden)
            kleur = GOOD if abs(out - tv) < 0.5 else BAD
            self.table.create_text(60, y, text=f" {x1}  {x2} ", fill=INK,
                                   font=("Consolas", 11))
            self.table.create_text(150, y, text=f"{out:.3f}", fill=kleur,
                                   font=("Consolas", 11))
            self.table.create_text(250, y, text=f"{float(tv):.0f}", fill=INK,
                                   font=("Consolas", 11))

if __name__ == "__main__":
    App().mainloop()
