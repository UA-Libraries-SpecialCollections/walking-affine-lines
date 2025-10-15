#!/usr/bin/env python3
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\rename_files_for_ingest.py"


"""
Copy and rename .txt files with an incremental number using Tk folder pickers.

- Prompts for a source folder (where the .txt files are now).
- Prompts for a destination folder (where the renamed copies should go).
- Copies each .txt file and names it like: u9999_1234567_0000001.txt, ...0000002.txt, etc.
- If the destination already contains files with the same prefix, numbering continues
  after the highest existing number to avoid overwriting.

Tested with Python 3.10+.
"""

from __future__ import annotations

import shutil
import sys
import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


# ===== Customize these if needed =====
PREFIX = "u9999_1234567_"   # The filename prefix before the numeric counter
PAD_WIDTH = 7               # e.g., 0000001 -> width 7
EXT = ".txt"                # Only .txt files are processed and produced
# =====================================


def pick_directory(title: str) -> Path | None:
    path_str = filedialog.askdirectory(title=title, mustexist=True)
    if not path_str:
        return None
    return Path(path_str)


def next_start_number(dest_dir: Path, prefix: str, pad_width: int, ext: str) -> int:
    """
    Look for existing files like '<prefix><NNN...>.ext' in dest_dir and
    return max(existing) + 1, or 1 if none exist.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{{pad_width}}}){re.escape(ext)}$", re.IGNORECASE)
    max_n = 0
    for p in dest_dir.iterdir():
        if p.is_file():
            m = pattern.match(p.name)
            if m:
                try:
                    n = int(m.group(1))
                    if n > max_n:
                        max_n = n
                except ValueError:
                    pass
    return max_n + 1 if max_n > 0 else 1


def main() -> int:
    root = tk.Tk()
    root.withdraw()  # hide the main Tk window

    try:
        src = pick_directory("Select the SOURCE folder containing .txt files")
        if src is None:
            messagebox.showinfo("Cancelled", "No source folder selected. Exiting.")
            return 0

        dst = pick_directory("Select the DESTINATION folder for renamed copies")
        if dst is None:
            messagebox.showinfo("Cancelled", "No destination folder selected. Exiting.")
            return 0

        if not src.exists() or not src.is_dir():
            messagebox.showerror("Error", f"Source is not a directory:\n{src}")
            return 1
        if not dst.exists() or not dst.is_dir():
            messagebox.showerror("Error", f"Destination is not a directory:\n{dst}")
            return 1

        # Gather .txt files (non-recursive), case-insensitive on extension
        txt_files = sorted(
            [p for p in src.iterdir() if p.is_file() and p.suffix.lower() == EXT.lower()]
        )

        if not txt_files:
            messagebox.showwarning(
                "No .txt files",
                f"No '{EXT}' files found in:\n{src}"
            )
            return 0

        start_n = next_start_number(dst, PREFIX, PAD_WIDTH, EXT)
        count = 0
        errors: list[str] = []

        # Copy each file with the new incremental name
        for i, src_path in enumerate(txt_files, start=start_n):
            new_name = f"{PREFIX}{i:0{PAD_WIDTH}d}{EXT}"
            dst_path = dst / new_name
            try:
                shutil.copy2(src_path, dst_path)
                count += 1
                print(f"Copied: {src_path.name} -> {dst_path.name}")
            except Exception as e:
                errors.append(f"{src_path.name} -> {new_name}: {e}")

        # Show summary
        if errors:
            message = (
                f"Copied {count} file(s) to:\n{dst}\n\n"
                f"Some files failed to copy ({len(errors)}):\n- " +
                "\n- ".join(errors[:10]) +
                ("\n- ..." if len(errors) > 10 else "")
            )
            messagebox.showwarning("Completed with errors", message)
        else:
            last_n = start_n + count - 1
            messagebox.showinfo(
                "Done",
                f"Copied {count} file(s) from:\n{src}\n\nto:\n{dst}\n\n"
                f"New names: {PREFIX}{start_n:0{PAD_WIDTH}d}{EXT} "
                f"through {PREFIX}{last_n:0{PAD_WIDTH}d}{EXT}"
            )

        return 0

    finally:
        try:
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())