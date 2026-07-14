import cv2
import tkinter as tk
from tkinter import Frame, Label, messagebox
from tkinter import ttk


class CapturePanel:
    def __init__(
        self,
        parent,
        app,
        save_image,
    ):
        self.parent = parent
        self.app = app
        self.save_image = save_image
        self.crop_image_var = tk.BooleanVar(value=True)
        self.vignette_image_var = tk.BooleanVar(value=False)
        self.filter_image_var = tk.BooleanVar(value=False)
        self.frame = self._build_panel()

    def _build_panel(self):
        frame = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=215
        )
        frame.place(relx=1.0, rely=0.0, anchor="ne")

        background = Frame(
            frame,
            bg="white",
            width=200,
            height=213
        )
        background.place(x=2, y=0)

        title = Label(
            frame,
            text="Capture",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title.place(relx=0.5, y=10, anchor="n")

        style = ttk.Style()
        style.configure("Save.TButton", background="white")
        style.configure("Save.TButton", relief="flat")

        self.capture_image_button = ttk.Button(
            background,
            text="Save Image",
            style="Save.TButton",
            command=self.save_capture_image
        )
        self.capture_image_button.place(relx=0.5, y=55, anchor="center")
        self.app.register_button(self.capture_image_button)

        style.configure("Capture.TCheckbutton", background="white")

        self.crop_image_checkbox = ttk.Checkbutton(
            background,
            text="Save cropped image",
            variable=self.crop_image_var,
            style="Capture.TCheckbutton",
        )
        self.crop_image_checkbox.place(relx=0.5, y=85, anchor="center")

        self.vignette_image_checkbox = ttk.Checkbutton(
            background,
            text="Apply vignette filter",
            variable=self.vignette_image_var,
            style="Capture.TCheckbutton",
        )
        self.vignette_image_checkbox.place(relx=0.5, y=115, anchor="center")

        self.capture_map_button = ttk.Button(
            background,
            text="Save Map",
            style="Save.TButton",
            command=self.save_map
        )
        self.capture_map_button.place(relx=0.5, y=150, anchor="center")
        self.app.register_button(self.capture_map_button)

        self.filter_image_checkbox = ttk.Checkbutton(
            background,
            text="Apply chip filter",
            variable=self.filter_image_var,
            style="Capture.TCheckbutton",
        )
        self.filter_image_checkbox.place(relx=0.5, y=185, anchor="center")

        return frame

    def _save_with_notification(self, image_name, image=None, **save_options):
        try:
            filepath = self.save_image(image=image, **save_options)
            if filepath is None:
                raise RuntimeError(
                    f"No {image_name.lower()} was available to save."
                )
        except FileNotFoundError as exc:
            messagebox.showwarning(
                "Vignette Filter Unavailable",
                str(exc),
                parent=self.parent,
            )
            return None
        except Exception as exc:
            messagebox.showerror(
                f"Save {image_name} Error",
                f"Could not save the {image_name.lower()}:\n\n{exc}",
                parent=self.parent,
            )
            return None

        messagebox.showinfo(
            f"{image_name} Saved",
            f"{image_name} saved to:\n{filepath}",
            parent=self.parent,
        )
        return filepath

    def save_capture_image(self):
        return self._save_with_notification(
            "Image",
            crop=self.crop_image_var.get(),
            apply_vignette=self.vignette_image_var.get(),
            apply_chip_filter=self.filter_image_var.get(),
        )

    def save_map(self):
        try:
            true_map = self.app.get_true_map()
            true_map_bgr = cv2.cvtColor(true_map, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            messagebox.showerror(
                "Save Map Error",
                f"Could not prepare the map for saving:\n\n{exc}",
                parent=self.parent,
            )
            return None

        return self._save_with_notification("Map", image=true_map_bgr)
