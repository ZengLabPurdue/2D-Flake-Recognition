import os
import subprocess
import sys
from pathlib import Path
from tkinter import Tk, StringVar
from tkinter import ttk

home_dir = os.path.dirname(os.path.abspath(__file__))
scanner_path = Path(home_dir) / "app.py"
os.chdir(home_dir)

# Hello roommate

def on_close():
    prior_args_final = prior_args.get() or "0"
    olympius_args_final = olympius_args.get() or "0"

    with open("port.config", "w") as f:
        f.write(f"{prior_args_final}\n")
        f.write(f"{olympius_args_final}\n")

    root.destroy()

    python_exe = Path(sys.executable)
    script = scanner_path

    cmd = [str(python_exe), str(script), str(prior_args_final), str(olympius_args_final)]
    subprocess.run(cmd, check=True)

try:
    with open("port.config", "r") as f:
        prior_args_value = f.readline().strip()
        olympius_args_value = f.readline().strip()
except FileNotFoundError:
    prior_args_value = ""

root = Tk()
root.title("Launch")
root.resizable(False, False)

frame = ttk.Frame(root, padding=15)
frame.grid(row=0, column=0)

prior_label = ttk.Label(frame, text="Prior COM Port:")
prior_label.grid(row=0, column=0, padx=(0,10), pady=(0,10), sticky="e")

prior_args = StringVar(value=prior_args_value)
prior_entry = ttk.Entry(frame, textvariable=prior_args, width=10)
prior_entry.grid(row=0, column=1, pady=(0,10), sticky="w")
prior_entry.focus()

olympius_label = ttk.Label(frame, text="Olympius COM Port:")
olympius_label.grid(row=1, column=0, padx=(0,10), pady=(0,10), sticky="e")

olympius_args = StringVar(value=olympius_args_value)
olympius_entry = ttk.Entry(frame, textvariable=olympius_args, width=10)
olympius_entry.grid(row=1, column=1, pady=(0,10), sticky="w")
olympius_entry.focus()

confirm_btn = ttk.Button(frame, text="Launch", command=on_close)
confirm_btn.grid(row=2, column=0, columnspan=2, pady=(5,0))

root.bind("<Return>", lambda e: on_close())

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()
