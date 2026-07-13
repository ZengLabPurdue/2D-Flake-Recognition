from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import Frame, Label, filedialog, messagebox
from tkinter import ttk

from Scanning.contour_extractor import get_region_from_point
from Scanning.scan_profile import (
    ProfileClassDraft,
    ScanProfileError,
    ScanProfileStore,
    ScanSearchProfile,
    build_region_overlay,
)


class ScanProfilePanel:
    """Right-side editor and viewer for scan search profiles."""

    IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]

    def __init__(self, parent, root, app, profile_store: ScanProfileStore):
        self.parent = parent
        self.root = root
        self.app = app
        self.profile_store = profile_store

        self.mode = "create"
        self.class_edit_active = False
        self.selected_class_index: int | None = None
        self.editing_class_index: int | None = None
        self.pending_new_class = False
        self._updating_class_list = False
        self.draft_classes: list[ProfileClassDraft] = []
        self.loaded_profile: ScanSearchProfile | None = None

        self.current_source_path: Path | None = None
        self.current_image_bgr: np.ndarray | None = None
        self.current_region_mask: np.ndarray | None = None
        self.current_seed_point: tuple[int, int] | None = None
        self.current_threshold: int | None = None

        self.profile_name_var = tk.StringVar()
        self.minimum_size_var = tk.StringVar()
        self.maximum_size_var = tk.StringVar()
        self.class_name_var = tk.StringVar()
        self.threshold_var = tk.StringVar(value="5")
        self.class_count_var = tk.StringVar(value="Classes: 0")
        self.status_var = tk.StringVar(value="Create or load a profile from the Scan menu.")

        self.frame = self._build_panel()
        self.frame.place_forget()
        self.status_var.trace_add("write", self._schedule_footer_resize)
        self.class_name_var.trace_add("write", self._on_class_name_changed)
        self.app.img_label.bind("<Button-1>", self._on_image_click, add="+")

    def _build_panel(self):
        panel = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=605,
        )
        panel.place(relx=0.0, rely=0.0, anchor="nw")

        background = Frame(panel, bg="white", width=200, height=598)
        background.place(x=2, y=0)

        style = ttk.Style()
        style.configure(
            "Normal.TButton",
            font="TkDefaultFont",
            background="white",
            relief="flat",
        )

        title = Label(
            background,
            text="Scan Search Profile",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13),
        )
        title.place(relx=0.5, y=10, anchor="n")

        Label(background, text="Profile name", bg="white", fg="black").place(
            x=8, y=43
        )
        self.profile_name_entry = ttk.Entry(
            background,
            textvariable=self.profile_name_var,
        )
        self.profile_name_entry.place(relx=0.5, y=65, anchor="n", width=184)

        self.class_count_label = Label(
            background,
            textvariable=self.class_count_var,
            bg="white",
            fg="black",
        )
        self.class_count_label.place(x=8, y=98)

        list_frame = Frame(background, bg="white", width=184, height=95)
        list_frame.place(relx=0.5, y=120, anchor="n")
        list_frame.pack_propagate(False)

        self.class_list = tk.Listbox(
            list_frame,
            height=5,
            width=23,
            exportselection=False,
        )
        class_scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.class_list.yview,
        )
        self.class_list.configure(yscrollcommand=class_scrollbar.set)
        self.class_list.pack(side="left", fill="both", expand=True)
        class_scrollbar.pack(side="right", fill="y")
        self.class_list.bind("<<ListboxSelect>>", self._on_class_selected)

        class_button_row = Frame(background, bg="white", width=184, height=30)
        class_button_row.place(relx=0.5, y=225, anchor="n")
        class_button_row.pack_propagate(False)

        self.new_class_button = ttk.Button(
            class_button_row,
            text="New class",
            width=10,
            style="Normal.TButton",
            command=self.new_class,
        )
        self.new_class_button.place(relx=0.25, rely=0.5, anchor="center")

        self.edit_class_button = ttk.Button(
            class_button_row,
            text="Edit class",
            width=10,
            style="Normal.TButton",
            command=self.edit_selected_class,
        )
        self.edit_class_button.place(relx=0.75, rely=0.5, anchor="center")

        remove_button_row = Frame(background, bg="white", width=184, height=30)
        remove_button_row.place(relx=0.5, y=260, anchor="n")
        remove_button_row.pack_propagate(False)

        self.remove_class_button = ttk.Button(
            remove_button_row,
            text="Remove class",
            width=12,
            style="Normal.TButton",
            command=self.remove_selected_class,
        )
        self.remove_class_button.place(relx=0.5, rely=0.5, anchor="center")

        Label(background, text="Class name", bg="white", fg="black").place(
            x=8, y=300
        )
        self.class_name_entry = ttk.Entry(
            background,
            textvariable=self.class_name_var,
        )
        self.class_name_entry.place(relx=0.5, y=322, anchor="n", width=184)

        self.load_image_button = ttk.Button(
            background,
            text="Load image",
            style="Normal.TButton",
            command=self.load_class_image,
        )
        self.load_image_button.place(relx=0.5, y=355, anchor="n")

        tolerance_row = Frame(background, bg="white", width=184, height=28)
        tolerance_row.place(relx=0.5, y=393, anchor="n")
        tolerance_row.pack_propagate(False)

        tolerance_controls = Frame(tolerance_row, bg="white")
        tolerance_controls.place(relx=0.5, rely=0.5, anchor="center")

        Label(tolerance_controls, text="Tolerance", bg="white", fg="black").pack(
            side="left", padx=(0, 5)
        )
        self.threshold_spinbox = ttk.Spinbox(
            tolerance_controls,
            from_=0,
            to=255,
            textvariable=self.threshold_var,
            width=5,
        )
        self.threshold_spinbox.pack(side="left")

        Label(
            background,
            text="Size (um, optional)",
            bg="white",
            fg="black",
        ).place(x=8, y=431)

        size_row = Frame(background, bg="white", width=184, height=23)
        size_row.place(relx=0.5, y=453, anchor="n")
        size_row.pack_propagate(False)

        Label(size_row, text="Min", bg="white", fg="black").place(x=0, y=2)
        self.minimum_size_entry = ttk.Entry(
            size_row,
            textvariable=self.minimum_size_var,
            width=6,
        )
        self.minimum_size_entry.place(x=28, y=0)

        Label(size_row, text="Max", bg="white", fg="black").place(x=94, y=2)
        self.maximum_size_entry = ttk.Entry(
            size_row,
            textvariable=self.maximum_size_var,
            width=6,
        )
        self.maximum_size_entry.place(x=126, y=0)

        self.confirm_region_button = ttk.Button(
            background,
            text="Confirm region",
            style="Normal.TButton",
            command=self.confirm_region,
        )
        self.confirm_region_button.place(relx=0.5, y=486, anchor="n")

        status_group = Frame(background, bg="white")
        status_group.place(relx=0.5, y=524, anchor="n", width=184)

        self.status_label = Label(
            status_group,
            textvariable=self.status_var,
            bg="white",
            fg="#333333",
            justify="center",
            wraplength=184,
        )
        self.status_label.pack(fill="x")

        self.save_profile_button = ttk.Button(
            status_group,
            text="Save profile",
            style="Normal.TButton",
            command=self.save_profile,
        )
        self.save_profile_button.pack(pady=(10, 0))

        self.panel_frame = panel
        self.panel_background = background
        self.status_group = status_group
        self.status_group_y = 524
        self._resize_panel_to_footer()

        self.profile_edit_widgets = (
            self.profile_name_entry,
            self.new_class_button,
            self.edit_class_button,
            self.remove_class_button,
            self.save_profile_button,
        )
        self.class_edit_widgets = (
            self.class_name_entry,
            self.threshold_spinbox,
            self.minimum_size_entry,
            self.maximum_size_entry,
            self.load_image_button,
            self.confirm_region_button,
        )
        return panel

    def start_create(self):
        self.mode = "create"
        self.class_edit_active = False
        self.selected_class_index = None
        self.editing_class_index = None
        self.pending_new_class = False
        self.loaded_profile = None
        self.draft_classes.clear()
        self.profile_name_var.set("")
        self.minimum_size_var.set("")
        self.maximum_size_var.set("")
        self.class_name_var.set("")
        self.threshold_var.set("5")
        self._clear_pending_image()
        self._set_editing_enabled(True)
        self._refresh_class_list()
        self.status_var.set("Name the profile, then click New class to begin.")

        self.app.set_view("Create Search Profile")
        self.show()
        self.app.display_image_message("Open an image to create a class.")
        self.profile_name_entry.focus_set()

    def choose_and_load_profile(self):
        self.editing_class_index = None
        self.selected_class_index = None
        self.pending_new_class = False
        self._set_class_editing_enabled(False)
        self.app.set_view("Load Search Profile")
        self.show()
        self.app.display_image_message("Open a profile, then select a class.")
        self.profile_store.profiles_dir.mkdir(parents=True, exist_ok=True)
        load_folder = messagebox.askyesnocancel(
            "Load Scan Search Profile",
            "Load a profile folder?\n\n"
            "Yes: select the folder containing profile.json\n"
            "No: select profile.json directly",
        )
        if load_folder is None:
            return

        if load_folder:
            selected = filedialog.askdirectory(
                title="Select Scan Search Profile Folder",
                initialdir=self.profile_store.profiles_dir,
            )
        else:
            selected = filedialog.askopenfilename(
                title="Select Scan Search Profile JSON",
                initialdir=self.profile_store.profiles_dir,
                filetypes=[("Scan search profile", "profile.json"), ("JSON", "*.json")],
            )
        if not selected:
            return

        try:
            profile = self.profile_store.load_profile(selected)
        except ScanProfileError as exc:
            messagebox.showerror("Invalid Scan Search Profile", str(exc))
            return

        self.show_loaded_profile(profile)
        self.app.set_active_scan_profile(profile)
        messagebox.showinfo(
            "Profile Loaded",
            f"Loaded '{profile.name}' with {len(profile.classes)} class(es).",
        )

    def show_loaded_profile(self, profile: ScanSearchProfile):
        self.mode = "edit"
        self.class_edit_active = False
        self.selected_class_index = None
        self.editing_class_index = None
        self.pending_new_class = False
        self.loaded_profile = profile
        try:
            self.draft_classes = [
                self.profile_store.profile_class_to_draft(profile_class)
                for profile_class in profile.classes
            ]
        except ScanProfileError as exc:
            self.draft_classes.clear()
            messagebox.showerror("Profile Edit Error", str(exc))
        self.profile_name_var.set(profile.name)
        self.minimum_size_var.set("")
        self.maximum_size_var.set("")
        self.class_name_var.set("")
        self._clear_pending_image()
        self._set_editing_enabled(True)
        self._refresh_class_list()
        self.status_var.set(f"Editing {profile.name}. Select a class or create a new one.")

        self.app.set_view("Load Search Profile")
        self.show()
        self.app.display_image_message("Select a class to edit or create a new class.")

    def load_class_image(self):
        if not self.class_edit_active:
            self._show_create_or_edit_class_prompt()
            return

        selected = filedialog.askopenfilename(
            title="Select an Image Containing the Flake",
            filetypes=self.IMAGE_TYPES,
        )
        if not selected:
            return

        source_path = Path(selected)
        try:
            encoded = np.frombuffer(source_path.read_bytes(), dtype=np.uint8)
            image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except OSError:
            image_bgr = None

        if image_bgr is None:
            messagebox.showerror("Image Error", f"Could not read image:\n{source_path}")
            return

        self.current_source_path = source_path
        self.current_image_bgr = image_bgr
        self.current_region_mask = None
        self.current_seed_point = None
        self.current_threshold = None
        if not self.class_name_var.get().strip():
            self.class_name_var.set(self._next_default_class_name())

        self.app.set_view(
            "Load Search Profile" if self.mode == "edit" else "Create Search Profile"
        )
        self.root.update_idletasks()
        self._display_bgr(image_bgr)
        self.status_var.set("Click a point inside the flake in the main image.")

    def confirm_region(self):
        if not self.class_edit_active:
            self._show_create_or_edit_class_prompt()
            return

        if self.current_image_bgr is None or self.current_source_path is None:
            messagebox.showwarning("No Image", "Load a class image first.")
            return
        if self.current_region_mask is None or self.current_seed_point is None:
            messagebox.showwarning("No Region", "Click inside the flake to select a region first.")
            return

        class_name = self.class_name_var.get().strip()
        if not class_name:
            class_name = self._next_default_class_name()
            self.class_name_var.set(class_name)

        try:
            selected_threshold = self._get_threshold()
            minimum_size, maximum_size = self._get_size_requirement()
        except ScanProfileError as exc:
            messagebox.showwarning("Invalid Class Requirement", str(exc))
            return
        if self.current_threshold is None or selected_threshold != self.current_threshold:
            messagebox.showwarning(
                "Select Region Again",
                "The tolerance changed after the preview. Click inside the flake again.",
            )
            return

        draft = ProfileClassDraft(
            name=class_name,
            source_path=self.current_source_path,
            image_bgr=self.current_image_bgr.copy(),
            region_mask=self.current_region_mask.copy(),
            seed_point=self.current_seed_point,
            threshold=self.current_threshold,
            minimum_size_um=minimum_size,
            maximum_size_um=maximum_size,
        )

        existing_index = self._draft_index_for_name(class_name)
        if self.editing_class_index is not None:
            if existing_index is not None and existing_index != self.editing_class_index:
                messagebox.showwarning(
                    "Duplicate Class Name",
                    f"A class named '{class_name}' already exists.",
                )
                return
            self.draft_classes[self.editing_class_index] = draft
        elif existing_index is None:
            self.draft_classes.append(draft)
        else:
            replace = messagebox.askyesno(
                "Replace Class",
                f"Replace the existing '{class_name}' class?",
            )
            if not replace:
                return
            self.draft_classes[existing_index] = draft

        self.pending_new_class = False
        self._refresh_class_list()
        self.selected_class_index = None
        self.editing_class_index = None
        self._set_class_editing_enabled(False)
        self.status_var.set(f"Confirmed {class_name}.")

    def edit_selected_class(self):
        if self.mode not in ("create", "edit"):
            return

        selection = self.class_list.curselection()
        if not selection:
            messagebox.showinfo("Edit Class", "Select a class to edit first.")
            return

        index = int(selection[0])
        if self.pending_new_class and index == len(self.draft_classes):
            self._set_class_editing_enabled(True)
            return
        if not 0 <= index < len(self.draft_classes):
            return

        self.selected_class_index = index
        self.editing_class_index = index
        self._set_class_editing_enabled(True)
        self.status_var.set(f"Editing class: {self.draft_classes[index].name}")
        self.class_name_entry.focus_set()

    def remove_selected_class(self):
        if self.mode not in ("create", "edit"):
            return
        selection = self.class_list.curselection()
        if not selection:
            return

        index = int(selection[0])
        if self.pending_new_class and index == len(self.draft_classes):
            self.pending_new_class = False
            self.selected_class_index = None
            self.editing_class_index = None
            self._set_class_editing_enabled(False)
            self._clear_pending_image()
            self.class_name_var.set("")
            self.minimum_size_var.set("")
            self.maximum_size_var.set("")
            self._refresh_class_list()
            self.status_var.set("Cancelled the unconfirmed class.")
            self.app.display_image_message("Select a class or create a new class.")
            return

        removed = self.draft_classes.pop(index)
        self.selected_class_index = None
        self.editing_class_index = None
        self.pending_new_class = False
        self._set_class_editing_enabled(False)
        self._clear_pending_image()
        self.class_name_var.set(self._next_default_class_name())
        self.minimum_size_var.set("")
        self.maximum_size_var.set("")
        self._refresh_class_list()
        self.status_var.set(f"Removed class: {removed.name}")
        self.app.display_image_message("Open an image or select a class.")

    def new_class(self):
        if self.mode not in ("create", "edit"):
            return
        self.class_list.selection_clear(0, tk.END)
        self.selected_class_index = None
        self.editing_class_index = None
        self.pending_new_class = True
        self.threshold_var.set("5")
        self.minimum_size_var.set("")
        self.maximum_size_var.set("")
        self._clear_pending_image()
        self._set_class_editing_enabled(True)
        self.class_name_var.set(self._next_default_class_name())
        self._refresh_class_list(select_pending=True)
        self.status_var.set("Load an image for the new class.")
        self.app.display_image_message("Open an image to create this class.")
        self.class_name_entry.focus_set()

    def save_profile(self):
        if self.class_edit_active:
            messagebox.showinfo(
                "Confirm Class",
                "Confirm the class region before saving the profile.",
            )
            return

        try:
            profile = self.profile_store.save_profile(
                self.profile_name_var.get(),
                self.draft_classes,
            )
        except FileExistsError as exc:
            overwrite = messagebox.askyesno(
                "Replace Profile",
                f"A profile already exists at:\n{exc.args[0]}\n\nReplace it?",
            )
            if not overwrite:
                return
            try:
                profile = self.profile_store.save_profile(
                    self.profile_name_var.get(),
                    self.draft_classes,
                    overwrite=True,
                )
            except (OSError, ScanProfileError) as save_exc:
                messagebox.showerror("Profile Save Error", str(save_exc))
                return
        except (OSError, ScanProfileError) as exc:
            messagebox.showerror("Profile Save Error", str(exc))
            return

        self.app.set_active_scan_profile(profile)
        self.show_loaded_profile(profile)
        messagebox.showinfo(
            "Profile Saved",
            f"Saved and activated '{profile.name}' at:\n{profile.path}",
        )

    def _on_image_click(self, event):
        if self.mode not in ("create", "edit"):
            return
        if self.app.get_view() not in ("Create Search Profile", "Load Search Profile"):
            return
        if not self.class_edit_active:
            self._show_create_or_edit_class_prompt()
            return
        if self.current_image_bgr is None:
            messagebox.showinfo(
                "Load Image",
                "Load an image before selecting a seed point.",
            )
            return

        seed_point = self.app.display_to_image_point(event.x, event.y)
        if seed_point is None:
            self.status_var.set("Click inside the displayed image, not the gray border.")
            return

        try:
            threshold = self._get_threshold()
            _, _, region_mask, contour = get_region_from_point(
                image_bgr=self.current_image_bgr,
                seed_point=seed_point,
                threshold=threshold,
                connectivity=8,
            )
        except (cv2.error, ScanProfileError, ValueError) as exc:
            messagebox.showerror("Region Selection Error", str(exc))
            return

        if contour is None or not np.any(region_mask > 0):
            self.status_var.set("No region was found. Try another point or tolerance.")
            return

        self.current_seed_point = seed_point
        self.current_region_mask = region_mask
        self.current_threshold = threshold
        preview = build_region_overlay(
            self.current_image_bgr,
            region_mask,
            seed_point,
        )
        self._display_bgr(preview)
        self.status_var.set("Region selected. Confirm or click again.")

    def _on_class_selected(self, event=None):
        if self._updating_class_list:
            return
        selection = self.class_list.curselection()
        if not selection:
            return
        index = int(selection[0])

        if self.pending_new_class and index == len(self.draft_classes):
            self.editing_class_index = None
            self._set_class_editing_enabled(True)
            return

        if self.pending_new_class:
            self.pending_new_class = False
            self.class_list.delete(tk.END)
            self.class_count_var.set(f"Classes: {len(self.draft_classes)}")

        if self.editing_class_index is not None and self.editing_class_index != index:
            previous_index = self.editing_class_index
            previous_name = self.draft_classes[previous_index].name
            self._updating_class_list = True
            try:
                self.class_list.delete(previous_index)
                self.class_list.insert(previous_index, f"  {previous_name}")
                self.class_list.selection_set(index)
                self.class_list.activate(index)
            finally:
                self._updating_class_list = False

        if index >= len(self.draft_classes):
            return
        draft = self.draft_classes[index]
        self.selected_class_index = index
        self.editing_class_index = None
        self._set_class_editing_enabled(False)
        self.class_name_var.set(draft.name)
        self.threshold_var.set(str(draft.threshold))
        self.minimum_size_var.set(
            "" if draft.minimum_size_um is None else f"{draft.minimum_size_um:g}"
        )
        self.maximum_size_var.set(
            "" if draft.maximum_size_um is None else f"{draft.maximum_size_um:g}"
        )
        self.current_source_path = draft.source_path
        self.current_image_bgr = draft.image_bgr.copy()
        self.current_region_mask = draft.region_mask.copy()
        self.current_seed_point = draft.seed_point
        self.current_threshold = draft.threshold
        preview = build_region_overlay(
            self.current_image_bgr,
            self.current_region_mask,
            self.current_seed_point,
        )
        self._display_bgr(preview)
        self.status_var.set(f"Selected {draft.name}. Click Edit class to change it.")

    def _display_bgr(self, image_bgr: np.ndarray):
        self.app.display_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    def _refresh_class_list(
        self,
        select_name: str | None = None,
        *,
        select_pending: bool = False,
    ):
        classes = self.draft_classes
        self._updating_class_list = True
        try:
            self.class_list.delete(0, tk.END)
            for profile_class in classes:
                self.class_list.insert(tk.END, f"  {profile_class.name}")

            if self.pending_new_class and self.mode in ("create", "edit"):
                pending_name = self.class_name_var.get().strip() or "(unnamed class)"
                self.class_list.insert(tk.END, f"  {pending_name}")

            displayed_count = len(classes) + int(self.pending_new_class)
            self.class_count_var.set(f"Classes: {displayed_count}")
            if select_name is not None:
                for index, profile_class in enumerate(classes):
                    if profile_class.name.casefold() == select_name.casefold():
                        self.class_list.selection_set(index)
                        self.class_list.activate(index)
                        break
            elif select_pending and self.pending_new_class:
                pending_index = len(classes)
                self.class_list.selection_set(pending_index)
                self.class_list.activate(pending_index)
        finally:
            self._updating_class_list = False

    def _set_editing_enabled(self, enabled: bool):
        for widget in self.profile_edit_widgets:
            widget.state(["!disabled"] if enabled else ["disabled"])
        self._set_class_editing_enabled(False)

    def _set_class_editing_enabled(self, enabled: bool):
        self.class_edit_active = enabled
        for widget in self.class_edit_widgets:
            widget.state(["!disabled"] if enabled else ["disabled"])

    def _get_threshold(self) -> int:
        try:
            threshold = int(self.threshold_var.get())
        except ValueError as exc:
            raise ScanProfileError("Tolerance must be a whole number from 0 to 255.") from exc
        if not 0 <= threshold <= 255:
            raise ScanProfileError("Tolerance must be between 0 and 255.")
        return threshold

    def _get_size_requirement(self) -> tuple[float | None, float | None]:
        values = []
        for label, raw_value in (
            ("Minimum size", self.minimum_size_var.get().strip()),
            ("Maximum size", self.maximum_size_var.get().strip()),
        ):
            if not raw_value:
                values.append(None)
                continue
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ScanProfileError(f"{label} must be a number of micrometers.") from exc
            values.append(value)

        minimum, maximum = values
        return self.profile_store.validate_size_requirement(minimum, maximum)

    def _on_class_name_changed(self, *_):
        if not self.class_edit_active or not hasattr(self, "class_list"):
            return

        if self.editing_class_index is not None:
            list_index = self.editing_class_index
        elif self.pending_new_class:
            list_index = len(self.draft_classes)
        else:
            return

        if list_index >= self.class_list.size():
            return

        display_name = self.class_name_var.get().strip() or "(unnamed class)"
        self._updating_class_list = True
        try:
            self.class_list.delete(list_index)
            self.class_list.insert(list_index, f"  {display_name}")
            self.class_list.selection_set(list_index)
            self.class_list.activate(list_index)
        finally:
            self._updating_class_list = False

    def _schedule_footer_resize(self, *_):
        self.root.after_idle(self._resize_panel_to_footer)

    def _resize_panel_to_footer(self):
        if not hasattr(self, "status_group"):
            return
        self.status_group.update_idletasks()
        panel_height = self.status_group_y + self.status_group.winfo_reqheight() + 10
        self.panel_frame.configure(height=panel_height)
        self.panel_background.configure(height=panel_height - 2)

    def _draft_index_for_name(self, class_name: str) -> int | None:
        normalized_name = class_name.casefold()
        for index, draft in enumerate(self.draft_classes):
            if draft.name.casefold() == normalized_name:
                return index
        return None

    def _next_default_class_name(self) -> str:
        existing_names = {draft.name.casefold() for draft in self.draft_classes}
        class_number = len(self.draft_classes) + 1
        while f"class {class_number}".casefold() in existing_names:
            class_number += 1
        return f"Class {class_number}"

    @staticmethod
    def _show_create_or_edit_class_prompt():
        messagebox.showinfo(
            "Create or Edit Class",
            "Create new class or edit existing class before choosing a seed point",
        )

    def show(self):
        if hasattr(self.app, "view_scans_panel"):
            self.app.view_scans_panel.hide()
        self.frame.place(relx=0.0, rely=0.0, anchor="nw")

    def hide(self):
        self.frame.place_forget()

    def _clear_pending_image(self):
        self.current_source_path = None
        self.current_image_bgr = None
        self.current_region_mask = None
        self.current_seed_point = None
        self.current_threshold = None
