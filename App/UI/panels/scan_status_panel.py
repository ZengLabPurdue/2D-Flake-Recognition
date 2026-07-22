import time

from tkinter import Frame, Label, messagebox
from tkinter import ttk

class ScanStatusPanel:
    def __init__(self, parent, root=None, ui_dispatch=None):
        self.parent = parent
        self.root = root or parent.winfo_toplevel()
        self.ui_dispatch = ui_dispatch or (
            lambda callback, *args, **kwargs: callback(*args, **kwargs)
        )
        self.stop_callback = None
        self._scan_running = False
        self._scan_started_at = None
        self._final_total_elapsed = None
        self._total_timer_job = None
        self._progress_text = "Not Started"
        self._processing_text = None
        self.frame = self._build_panel()

    def _build_panel(self):
        panel = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=225
        )
        panel.place(relx=1.0, rely=0.0, anchor="ne")

        self.background = Frame(
            panel,
            bg="white",
            width=200,
            height=223
        )
        self.background.place(x=2, y=0)

        title_label = Label(
            panel,
            text="Scan Info",
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
            text="(Imaging) Not Started",
            bg="white",
            fg="black",
            justify="center",
            wraplength=188,
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

        style = ttk.Style()
        style.configure(
            "Normal.TButton",
            font="TkDefaultFont",
            background="white",
            relief="flat",
        )

        self.stop_button = ttk.Button(
            panel,
            text="Stop Scan",
            command=self._request_stop,
            style="Normal.TButton",
        )
        self.stop_button.place(relx=0.5, y=142, anchor="n")
        self.stop_button.state(["disabled"])

        panel.after_idle(self._refresh_panel_layout)

        return panel

    def _refresh_panel_layout(self):
        self.frame.update_idletasks()
        row_gap = 1
        self.stage_time_label.place(
            relx=0.5,
            y=self.progress_label.winfo_y() + self.progress_label.winfo_height() + row_gap,
            anchor="n",
        )
        self.frame.update_idletasks()
        self.total_time_label.place(
            relx=0.5,
            y=self.stage_time_label.winfo_y() + self.stage_time_label.winfo_height() + row_gap,
            anchor="n",
        )
        self.frame.update_idletasks()
        self.stop_button.place(
            relx=0.5,
            y=self.total_time_label.winfo_y() + self.total_time_label.winfo_height() + 8,
            anchor="n",
        )
        self.frame.update_idletasks()
        panel_height = self.stop_button.winfo_y() + self.stop_button.winfo_height() + 15
        self.frame.configure(height=panel_height)
        self.background.configure(height=max(1, panel_height - 2))

    def _refresh_progress_text(self):
        parts = [f"(Imaging) {self._progress_text}"]
        if self._processing_text:
            parts.append(f"(Processing) {self._processing_text}")
        self.progress_label.config(text=" | ".join(parts))
        self._refresh_panel_layout()

    def set_stop_callback(self, callback):
        self.stop_callback = callback

    def set_scan_running(self, running):
        self.ui_dispatch(self._set_scan_running_ui, bool(running))

    def _set_scan_running_ui(self, running):
        if running:
            if not self._scan_running:
                self._scan_started_at = time.monotonic()
                self._final_total_elapsed = None
            self._scan_running = True
            self.stop_button.configure(text="Stop Scan")
            self.stop_button.state(["!disabled"])
            self._schedule_total_timer()
        else:
            if self._final_total_elapsed is not None:
                self.total_time_label.config(
                    text=(
                        "Total Time Elapsed: "
                        f"{self._final_total_elapsed}"
                    )
                )
            elif self._scan_started_at is not None:
                self.total_time_label.config(
                    text=f"Total Time Elapsed: {self._elapsed_string()}"
                )
            self._scan_running = False
            if self._total_timer_job is not None:
                self.root.after_cancel(self._total_timer_job)
                self._total_timer_job = None
            self.stop_button.configure(text="Stop Scan")
            self.stop_button.state(["disabled"])

    def _schedule_total_timer(self):
        if self._total_timer_job is None and self._scan_running:
            self._total_timer_job = self.root.after(250, self._tick_total_time)

    def _tick_total_time(self):
        self._total_timer_job = None
        if not self._scan_running:
            return
        self.total_time_label.config(
            text=f"Total Time Elapsed: {self._elapsed_string()}"
        )
        self._schedule_total_timer()

    def _elapsed_string(self):
        elapsed = (
            time.monotonic() - self._scan_started_at
            if self._scan_started_at is not None
            else 0.0
        )
        return time.strftime("%H:%M:%S", time.gmtime(max(0.0, elapsed)))

    def _request_stop(self):
        if self.stop_callback is None:
            return
        if not messagebox.askyesno(
            "Stop Scan",
            "Stop the scan after the current stage movement, image capture, "
            "or processing step finishes?",
            parent=self.root,
        ):
            return
        self.stop_button.configure(text="Stopping...")
        self.stop_button.state(["disabled"])
        self.stop_callback()

    def update_status(
        self,
        scan_type=None,
        stage=None,
        progress=None,
        stage_elapsed_time=None,
        total_elapsed_time=None,
        processing_state=None,
    ):
        self.ui_dispatch(
            self._apply_status,
            {
                "scan_type": scan_type,
                "stage": stage,
                "progress": progress,
                "stage_elapsed_time": stage_elapsed_time,
                "total_elapsed_time": total_elapsed_time,
                "processing_state": processing_state,
            },
        )

    def _apply_status(self, status):
        scan_type = status["scan_type"]
        stage = status["stage"]
        progress = status["progress"]
        stage_elapsed_time = status["stage_elapsed_time"]
        total_elapsed_time = status["total_elapsed_time"]
        processing_state = status["processing_state"]

        if scan_type is not None:
            self.scan_type_label.config(text=f"Scan: {scan_type}")

        if stage is not None:
            self.stage_label.config(text=f"Stage: {stage}")

        if progress is not None:
            self._progress_text = progress

        if stage_elapsed_time is not None:
            self.stage_time_label.config(text=f"Stage Time Elapsed: {stage_elapsed_time}")

        if total_elapsed_time is not None:
            if stage in {"Complete", "Stopped", "Error"}:
                self._final_total_elapsed = total_elapsed_time
                self.total_time_label.config(
                    text=f"Total Time Elapsed: {total_elapsed_time}"
                )
            elif not self._scan_running:
                self.total_time_label.config(
                    text=f"Total Time Elapsed: {total_elapsed_time}"
                )

        if processing_state is not None:
            self._processing_text = (
                None if processing_state == "Not required" else processing_state
            )

        if progress is not None or processing_state is not None:
            self._refresh_progress_text()
        else:
            self._refresh_panel_layout()
