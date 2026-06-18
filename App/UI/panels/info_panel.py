from tkinter import Frame, Label

class InfoPanel:
    def __init__(self, parent):
        self.parent = parent
        self.frame = self._build_panel()

    def _build_panel(self):
        panel = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=87
        )
        panel.place(relx=1.0, rely=0.0, anchor="ne")

        background = Frame(
            panel,
            bg="white",
            width=200,
            height=85
        )
        background.place(x=2, y=0)

        title_label = Label(
            panel,
            text="Info",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title_label.place(relx=0.5, y=10, anchor="n")

        self.camera_fps_label = Label(
            panel,
            text="Camera FPS: 0 fps",
            bg="white",
            fg="black"
        )
        self.camera_fps_label.place(relx=0.5, y=35, anchor="n")

        self.app_fps_label = Label(
            panel,
            text="App FPS: 0 fps",
            bg="white",
            fg="black"
        )
        self.app_fps_label.place(relx=0.5, y=55, anchor="n")

        return panel

    def update_fps(
        self,
        camera_fps=None,
        app_fps=None,
    ):
        if camera_fps is not None:
            self.camera_fps_label.config(text=f"Camera FPS: {camera_fps:.2f} fps")

        if app_fps is not None:
            self.app_fps_label.config(text=f"App FPS: {app_fps:.2f} fps")

        self.frame.update_idletasks()