#!/usr/bin/env python3
"""
A small utility to set the `imageData` field to `null` (Python `None`) in every
JSON file inside a user‑selected folder.

How it works
------------
1. Launches a minimal Tkinter GUI with a single **Select Folder** button.
2. The user chooses a directory that contains one or more `.json` files.
3. For each file found:
   • The JSON is loaded.
   • If it has an `imageData` key whose value is *not* `null`, the value is
     replaced with `None`.
   • The modified object is written back to disk in place (same filename,
     pretty‑printed with two‑space indentation and UTF‑8 encoding).
4. A summary dialog reports how many files were processed and how many were
   actually changed.

‣ Tested with Python 3.8+.
‣ No third‑party libraries required.
"""
from __future__ import annotations  # enables postponed evaluation of type hints (PEP 563)

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# ──────────────────────────────────────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────────────────────────────────────

def overwrite_image_data(folder: Path) -> None:
    """Replace the *imageData* value with ``null`` in every JSON file found."""
    json_files = list(folder.glob("*.json"))

    if not json_files:
        messagebox.showinfo("No JSON files", "No .json files were found in the selected folder.")
        return

    modified_count = 0

    for fp in json_files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Change only if the key exists *and* is not already null/None
            if "imageData" in data and data["imageData"] is not None:
                data["imageData"] = None

                with fp.open("w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                modified_count += 1
        except Exception as exc:
            # Non‑fatal per‑file error; let the user know but keep going.
            messagebox.showwarning(
                "Error processing file",
                f"Could not process '{fp.name}':\n{exc}"
            )

    messagebox.showinfo(
        "Operation complete",
        f"Processed {len(json_files)} file(s).\nModified {modified_count} file(s)."
    )

# ──────────────────────────────────────────────────────────────────────────────
# GUI helpers
# ──────────────────────────────────────────────────────────────────────────────

def select_folder() -> None:
    """Prompt the user to pick a directory and then start processing."""
    path = filedialog.askdirectory(title="Select folder containing JSON files")
    if path:
        folder_var.set(path)
        overwrite_image_data(Path(path))

# ──────────────────────────────────────────────────────────────────────────────
# Tkinter plumbing
# ──────────────────────────────────────────────────────────────────────────────

root = tk.Tk()
root.title("ImageData Nullifier")
root.resizable(False, False)

folder_var = tk.StringVar(value="")

main_frame = tk.Frame(root, padx=20, pady=20)
main_frame.pack()

folder_label = tk.Label(main_frame, text="Selected folder:")
folder_label.grid(row=0, column=0, sticky="w")

folder_entry = tk.Entry(main_frame, textvariable=folder_var, width=50, state="readonly")
folder_entry.grid(row=0, column=1, padx=(6, 0))

select_button = tk.Button(main_frame, text="Select Folder", command=select_folder)
select_button.grid(row=0, column=2, padx=6)

root.mainloop()
