from tkinter import Frame, Label

class ScanInfoPanel:
    def __init__(self, parent):
        self.parent = parent
        self.frame = self._build_panel()

    def _build_panel(self):
        panel = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=147
        )
        panel.place(relx=1.0, rely=0.0, anchor="ne")

        background = Frame(
            panel,
            bg="white",
            width=200,
            height=145
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

        self.scan_type_label = Label(
            panel,
            text="Scan: None",
            bg="white",
            fg="black"
        )
        self.scan_type_label.place(relx=0.5, y=35, anchor="n")

        self.stage_label = Label(
            panel,
            text="Stage: Not Started",
            bg="white",
            fg="black"
        )
        self.stage_label.place(relx=0.5, y=55, anchor="n")

        self.progress_label = Label(
            panel,
            text="Progress: Not Started",
            bg="white",
            fg="black"
        )
        self.progress_label.place(relx=0.5, y=75, anchor="n")

        self.stage_time_label = Label(
            panel,
            text="Stage Time Elapsed: Not Started",
            bg="white",
            fg="black"
        )
        self.stage_time_label.place(relx=0.5, y=95, anchor="n")

        self.total_time_label = Label(
            panel,
            text="Total Time Elapsed: Not Started",
            bg="white",
            fg="black"
        )
        self.total_time_label.place(relx=0.5, y=115, anchor="n")

        return panel

    def update_status(
        self,
        scan_type=None,
        stage=None,
        progress=None,
        stage_elapsed_time=None,
        total_elapsed_time=None,
    ):
        if scan_type is not None:
            self.scan_type_label.config(text=f"Scan: {scan_type}")

        if stage is not None:
            self.stage_label.config(text=f"Stage: {stage}")

        if progress is not None:
            self.progress_label.config(text=f"Stage Progress: {progress}")

        if stage_elapsed_time is not None:
            self.stage_time_label.config(text=f"Stage Time Elapsed: {stage_elapsed_time}")

        if total_elapsed_time is not None:
            self.total_time_label.config(text=f"Total Time Elapsed: {total_elapsed_time}")

        self.frame.update_idletasks()