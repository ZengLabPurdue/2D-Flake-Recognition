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
            height=112,
        )
        panel.place(relx=1.0, rely=0.0, anchor="ne")

        background = Frame(
            panel,
            bg="white",
            width=200,
            height=110,
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

        self.render_label = Label(
            panel,
            text="Display: waiting",
            bg="white",
            fg="black",
        )
        self.render_label.place(relx=0.5, y=75, anchor="n")

        return panel

    def update_fps(
        self,
        camera_fps=None,
        app_fps=None,
        render_ms=None,
        render_backend=None,
    ):
        if camera_fps is not None:
            self._set_text(
                self.camera_fps_label,
                f"Camera FPS: {camera_fps:.2f} fps",
            )

        if app_fps is not None:
            self._set_text(
                self.app_fps_label,
                f"App FPS: {app_fps:.2f} fps",
            )

        if render_ms is not None:
            backend = render_backend or "CPU"
            self._set_text(
                self.render_label,
                f"Display: {render_ms:.1f} ms ({backend})",
            )

    @staticmethod
    def _set_text(label, text):
        if label.cget("text") != text:
            label.config(text=text)
