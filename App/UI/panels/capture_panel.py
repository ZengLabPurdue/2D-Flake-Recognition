import cv2
from tkinter import Frame, Label
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
        self.frame = self._build_panel()

    def _build_panel(self):
        frame = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=120
        )
        frame.place(relx=1.0, rely=0.0, anchor="ne")

        background = Frame(
            frame,
            bg="white",
            width=200,
            height=118
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
            command=self.save_image
        )
        self.capture_image_button.place(relx=0.5, y=55, anchor="center")
        self.app.buttons.append(self.capture_image_button)

        self.capture_map_button = ttk.Button(
            background,
            text="Save Map",
            style="Save.TButton",
            command=self.save_map
        )
        self.capture_map_button.place(relx=0.5, y=90, anchor="center")
        self.app.buttons.append(self.capture_map_button)

        return frame

    def save_map(self):
        true_map = self.app.get_true_map()
        true_map_bgr = cv2.cvtColor(true_map, cv2.COLOR_RGB2BGR)
        self.save_image(image=true_map_bgr)