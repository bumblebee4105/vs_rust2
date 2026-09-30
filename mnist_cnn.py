"""
mnist_cnn.py — Demo bij slide 12: "handschrift herkennen met een CNN"

Teken een cijfer (0 t/m 9) en het netwerk voorspelt direct welk cijfer
het is, met zekerheid per cijfer. Via "Toon filters" kun je zien waar
het eerste filter-op-het-kijken zich op concentreert (Adam Harley-stijl).

Starten:    python mnist_cnn.py
Vereist:    pip install torch torchvision pillow

Eerste start: het model wordt dan automatisch één keer getraind op de
MNIST-dataset (± 1 à 3 minuten op de cpu) en opgeslagen als
mnist_cnn.pth naast dit bestand. Elke volgende start laadt het model
meteen. Tip: draai dit een dag van tevoren een keer, zodat de klas
meteen kan tekenen.
"""

import math
import os
import random
import sys
import tkinter as tk
from tkinter import messagebox

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms
except ImportError:
    root = tk.Tk(); root.withdraw()
    messagebox.showerror("Ontbrekende libraries",
        "Deze demo heeft torch en torchvision nodig.\n\n"
        "Installeer ze met:\n\n    pip install torch torchvision")
    sys.exit(1)

from PIL import Image, ImageDraw, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "mnist_cnn.pth")
DATA = os.path.join(HERE, "mnist_data")
CANVAS = 280                     # teken-canvas is 280x280 (28x28 x10)

# ---------------------------------------------------------------- model

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)      # 28 -> 24
        self.conv2 = nn.Conv2d(16, 32, 5)     # 12 -> 8
        self.fc = nn.Linear(32 * 4 * 4, 10)   # na 2x pool: 4x4

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 24 -> 12
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 8 -> 4
        return self.fc(x.view(-1, 32 * 4 * 4))

def train_once(model):
    """Train het netwerk één epoch op MNIST en sla het op."""
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST(DATA, train=True, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    print("Eerste start: even trainen op MNIST (eenmalig)...")
    for batch, (x, y) in enumerate(loader):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if batch % 100 == 0:
            print(f"  batch {batch:>4}/{len(loader)}  loss {loss.item():.4f}")
    torch.save(model.state_dict(), CKPT)
    print(f"Klaar — model opgeslagen als {CKPT}")

def load_model():
    model = CNN()
    if os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location="cpu"))
        print("Model geladen uit", CKPT)
    else:
        train_once(model)
    model.eval()
    return model

# ---------------------------------------------------------------- app

BG, PANEL, INK, MUTED = "#15181d", "#1d2129", "#e8e6e3", "#9a958b"
ACCENT, GOOD, GRIDLN = "#e8723f", "#4caf7d", "#2a2f3a"

class App(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("MNIST CNN — teken een cijfer")
        self.configure(bg=BG)
        self.model = model
        self.photo = None
        self._build()
        self.new_canvas()

    # ---------- lay-out ----------
    def _build(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=12)

        left = tk.Frame(body, bg=BG)
        left.pack(side="left")
        self.cv = tk.Canvas(left, width=CANVAS, height=CANVAS, bg="#000000",
                            highlightthickness=0)
        self.cv.pack()
        self.cv.bind("<B1-Motion>", self._paint)
        self.cv.bind("<Button-1>", self._paint)
        tk.Label(left, text="teken hier (wit op zwart, zoals MNIST)",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(pady=(4, 0))

        btns = tk.Frame(left, bg=BG)
        btns.pack(fill="x", pady=(8, 0))
        for text, cmd in (("Voorspel", self.predict),
                          ("Wis", self.new_canvas),
                          ("Willekeurig voorbeeld", self.random_example),
                          ("Toon filters", self.show_filters)):
            tk.Button(btns, text=text, relief="flat", bg=PANEL, fg=INK,
                      font=("Segoe UI", 9), command=cmd).pack(
                      side="left", padx=(0, 6))

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(18, 0))

        self.guess = tk.Label(right, text="?", bg=BG, fg=ACCENT,
                              font=("Segoe UI", 64, "bold"))
        self.guess.pack(anchor="w")
        self.conf = tk.Label(right, text="teken een cijfer en druk op Voorspel",
                             bg=BG, fg=MUTED, font=("Segoe UI", 10))
        self.conf.pack(anchor="w", pady=(0, 8))

        self.bars = []
        self.bar_labels = []
        for d in range(10):
            row = tk.Frame(right, bg=BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=str(d), bg=BG, fg=INK, width=2,
                     font=("Consolas", 10)).pack(side="left")
            cv = tk.Canvas(row, width=200, height=14, bg=BG,
                           highlightthickness=0)
            cv.pack(side="left")
            lbl = tk.Label(row, text="", bg=BG, fg=MUTED, width=7,
                           anchor="w", font=("Consolas", 9))
            lbl.pack(side="left", padx=(6, 0))
            self.bars.append(cv)
            self.bar_labels.append(lbl)

    # ---------- teken-canvas ----------
    def new_canvas(self):
        self.cv.delete("all")
        self.img = Image.new("L", (28, 28), 0)
        self.draw = ImageDraw.Draw(self.img)
        self.last = None

    def _paint(self, e):
        x, y = e.x, e.y
        if self.last:
            lx, ly = self.last
            # tussenliggende punten, anders worden snelle strepen onderbroken
            steps = max(abs(x - lx), abs(y - ly)) // 3 + 1
            for i in range(1, steps + 1):
                px = lx + (x - lx) * i / steps
                py = ly + (y - ly) * i / steps
                r = 10
                self.cv.create_oval(px - r, py - r, px + r, py + r,
                                    fill="#ffffff", outline="")
                self.draw.ellipse([(px - 5) / 10, (py - 5) / 10,
                                   (px + 5) / 10, (py + 5) / 10], fill=255)
        self.last = (x, y)

    def _show_tensor(self, t):
        """Toon een 28x28 tensor (0..1) op het teken-canvas."""
        self.new_canvas()
        arr = (t.squeeze().numpy() * 255).astype("uint8")
        pil = Image.fromarray(arr).resize((CANVAS, CANVAS), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(pil)
        self.cv.create_image(0, 0, anchor="nw", image=self.photo)

    def random_example(self):
        tf = transforms.Compose([transforms.ToTensor()])
        testset = datasets.MNIST(DATA, train=False, download=True, transform=tf)
        t, label = testset[random.randrange(len(testset))]
        self._show_tensor(t)
        self.predict(true_label=label)

    # ---------- voorspellen ----------
    def predict(self, true_label=None):
        t = transforms.Normalize((0.1307,), (0.3081,))(
            transforms.ToTensor()(self.img)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(t), dim=1)[0]
        best = int(probs.argmax())
        self.guess.config(text=str(best))
        pct = 100 * float(probs[best])
        extra = f"  (dit was echt een {true_label})" if true_label is not None else ""
        self.conf.config(text=f"zekerheid {pct:.1f}%{extra}",
                         fg=GOOD if pct > 80 else MUTED)
        for d, (bar, lbl) in enumerate(zip(self.bars, self.bar_labels)):
            bar.delete("all")
            w = int(200 * float(probs[d]))
            col = ACCENT if d == best else "#3a4150"
            bar.create_rectangle(0, 1, w, 13, fill=col, outline="")
            lbl.config(text=f"{100 * float(probs[d]):5.1f}%")

    # ---------- filter-visualisatie ----------
    def show_filters(self):
        t = transforms.Normalize((0.1307,), (0.3081,))(
            transforms.ToTensor()(self.img)).unsqueeze(0)
        with torch.no_grad():
            act = F.relu(self.model.conv1(t))[0]        # (16, 24, 24)
        w = self.model.conv1.weight.detach()            # (16, 1, 5, 5)
        top = tk.Toplevel(self)
        top.title("Waar kijkt het model naar? — conv1 filters + activaties")
        top.configure(bg=BG)
        S = 5
        for i in range(16):
            fr = tk.Frame(top, bg=BG, padx=6, pady=6)
            fr.grid(row=i // 4, column=i % 4)
            f = w[i, 0]
            f = ((f - f.min()) / (f.max() - f.min() + 1e-9))
            fpil = Image.fromarray((f.numpy() * 255).astype("uint8"))
            fpil = fpil.resize((S * 12, S * 12), Image.NEAREST)
            a = act[i]
            a = a / (a.max() + 1e-9)
            apil = Image.fromarray((a.numpy() * 255).astype("uint8"))
            apil = apil.resize((S * 12, S * 12), Image.NEAREST)
            im1 = ImageTk.PhotoImage(fpil); im2 = ImageTk.PhotoImage(apil)
            cv = tk.Canvas(fr, width=S * 12 * 2 + 8, height=S * 12 + 20,
                           bg=BG, highlightthickness=0)
            cv.pack()
            cv.create_image(0, 0, anchor="nw", image=im1)
            cv.create_image(S * 12 + 8, 0, anchor="nw", image=im2)
            cv.create_text(0, S * 12 + 8, anchor="nw", text=f"filter {i}",
                           fill=MUTED, font=("Consolas", 8))
            cv.im1, cv.im2 = im1, im2   # referenties vasthouden

if __name__ == "__main__":
    model = load_model()
    App(model).mainloop()
