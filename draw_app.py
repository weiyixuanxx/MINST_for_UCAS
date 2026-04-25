import argparse
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps, ImageTk

from data import MNIST_MEAN, MNIST_STD
from model import MNISTCNN


def pick_font(root: tk.Tk, size: int, weight: str = "normal") -> tuple[str, int, str]:
    families = set(tkfont.families(root))
    candidates = [
        "Segoe UI",
        "Helvetica Neue",
        "Arial",
        "Ubuntu",
        "TkDefaultFont",
    ]
    for name in candidates:
        if name in families:
            return (name, size, weight)
    return ("TkDefaultFont", size, weight)


def load_model(checkpoint_path: str, device: torch.device) -> MNISTCNN:
    model = MNISTCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def autocrop_white_border(img: Image.Image, threshold: int = 250) -> Image.Image:
    arr = np.array(img)
    mask = arr < threshold
    if not np.any(mask):
        return img.copy()

    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    return img.crop((x_min, y_min, x_max + 1, y_max + 1))


def make_mnist_input_from_canvas(
    canvas_img: Image.Image,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    original = canvas_img.copy()
    cropped = autocrop_white_border(original)

    w, h = cropped.size
    if w == 0 or h == 0:
        mnist = Image.new("L", (28, 28), color=0)
        return original, cropped, mnist

    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cropped.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    inverted = ImageOps.invert(resized)

    mnist = Image.new("L", (28, 28), color=0)
    offset_x = (28 - new_w) // 2
    offset_y = (28 - new_h) // 2
    mnist.paste(inverted, (offset_x, offset_y))
    return original, cropped, mnist


class DrawApp:
    def __init__(self, root: tk.Tk, model: MNISTCNN, device: torch.device):
        self.root = root
        self.model = model
        self.device = device

        self.canvas_size = 360
        self.brush_size = 18
        self.preview_size = 220
        self.preview_title_font = pick_font(root, 12, "bold")
        self.body_font = pick_font(root, 11)
        self.result_font = pick_font(root, 28, "bold")

        self.root.title("Digit Recognizer")
        self.root.geometry("1100x760")
        self.root.minsize(980, 680)
        self.root.configure(bg="#eef2f7")

        self._build_styles()
        self._build_layout()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self._preview_refs = [None, None, None]
        self._clear_preview()

    def _build_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#eef2f7")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Title.TLabel", background="#eef2f7", foreground="#1f2a37")
        style.configure("Hint.TLabel", background="#ffffff", foreground="#64748b")
        style.configure("Primary.TButton", font=self.body_font, padding=(14, 8))
        style.configure("Secondary.TButton", font=self.body_font, padding=(14, 8))

    def _build_layout(self):
        root_container = ttk.Frame(self.root, style="App.TFrame", padding=14)
        root_container.pack(fill="both", expand=True)

        header = ttk.Label(
            root_container,
            text="Digit Recognizer",
            style="Title.TLabel",
            font=pick_font(self.root, 20, "bold"),
        )
        header.pack(anchor="w", pady=(2, 10))

        main = ttk.Frame(root_container, style="App.TFrame")
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left_card = ttk.Frame(main, style="Card.TFrame", padding=14)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        right_card = ttk.Frame(main, style="Card.TFrame", padding=14)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        canvas_title = tk.Label(
            left_card,
            text="Drawing Area",
            bg="#ffffff",
            fg="#334155",
            font=self.preview_title_font,
        )
        canvas_title.pack(anchor="w", pady=(0, 8))

        self.canvas = tk.Canvas(
            left_card,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
            cursor="cross",
            highlightthickness=1,
            highlightbackground="#d0d7e2",
        )
        self.canvas.pack(anchor="w")

        controls = ttk.Frame(left_card, style="Card.TFrame")
        controls.pack(anchor="w", pady=(12, 6))
        ttk.Button(
            controls, text="Predict", style="Primary.TButton", command=self.predict
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            controls, text="Clear", style="Secondary.TButton", command=self.clear
        ).pack(side=tk.LEFT)

        hint = ttk.Label(
            left_card,
            text="Tip: draw one digit near the center for more stable prediction.",
            style="Hint.TLabel",
            font=self.body_font,
        )
        hint.pack(anchor="w", pady=(8, 2))

        result_title = tk.Label(
            right_card,
            text="Result",
            bg="#ffffff",
            fg="#334155",
            font=self.preview_title_font,
        )
        result_title.pack(anchor="w", pady=(0, 8))

        self.result_label = tk.Label(
            right_card,
            text="Prediction: - | Confidence: -",
            bg="#ffffff",
            fg="#0f172a",
            anchor="w",
            font=self.result_font,
        )
        self.result_label.pack(fill="x", pady=(0, 14))

        preview_wrap = tk.Frame(right_card, bg="#ffffff")
        preview_wrap.pack(fill="both", expand=True)

        self.preview_labels = []
        self.preview_titles = ["Original", "Cropped", "Model Input (28x28)"]
        for idx, title in enumerate(self.preview_titles):
            col = tk.Frame(preview_wrap, bg="#ffffff")
            col.grid(row=0, column=idx, padx=6, sticky="n")

            tk.Label(
                col,
                text=title,
                bg="#ffffff",
                fg="#475569",
                font=pick_font(self.root, 11, "bold"),
            ).pack(pady=(0, 6))

            box = tk.Frame(
                col,
                bg="#f8fafc",
                width=self.preview_size,
                height=self.preview_size,
                highlightthickness=1,
                highlightbackground="#d0d7e2",
            )
            box.pack()
            box.pack_propagate(False)

            label = tk.Label(box, bg="#f8fafc")
            label.pack(fill="both", expand=True)
            self.preview_labels.append(label)

    def paint(self, event):
        x, y = event.x, event.y
        r = self.brush_size // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw.ellipse((x - r, y - r, x + r, y + r), fill=0, outline=0)

    def _clear_preview(self):
        for label in self.preview_labels:
            label.config(image="", bg="#f8fafc")
        self._preview_refs = [None, None, None]

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Prediction: - | Confidence: -")
        self._clear_preview()

    def update_preview(self, imgs: list[Image.Image]):
        for i, img in enumerate(imgs):
            if i < 2:
                vis = img.resize(
                    (self.preview_size, self.preview_size),
                    resample=Image.Resampling.BILINEAR,
                )
            else:
                vis = img.resize(
                    (self.preview_size, self.preview_size),
                    resample=Image.Resampling.NEAREST,
                )
            tk_img = ImageTk.PhotoImage(vis)
            self.preview_labels[i].config(image=tk_img, bg="white")
            self._preview_refs[i] = tk_img

    def predict(self):
        original, cropped, mnist_input = make_mnist_input_from_canvas(self.image)
        arr = np.array(mnist_input, dtype=np.float32) / 255.0

        if float(arr.max()) < 1e-6:
            messagebox.showinfo("Notice", "Please draw one digit before prediction.")
            return

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)
        tensor = (tensor - MNIST_MEAN) / MNIST_STD

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(probs.argmax().item())
            conf = float(probs[pred].item() * 100.0)

        self.result_label.config(text=f"Prediction: {pred} | Confidence: {conf:.1f}%")
        self.update_preview([original, cropped, mnist_input])


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST draw-and-predict app")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/mnist_cnn.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    try:
        model = load_model(args.checkpoint, device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint: {args.checkpoint}. "
            "Please train first or pass a valid checkpoint path."
        ) from exc

    root = tk.Tk()
    app = DrawApp(root, model=model, device=device)
    root.mainloop()


if __name__ == "__main__":
    main()
