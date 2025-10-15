#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\generate_document_delta_manifold.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu



# figure_pickle.py provides a way to save matplotlib figure graph objects for later display without 
# the need to re compute display data



# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


from __future__ import annotations

from typing import Any, Optional
import io
import os
import pickle
import threading

# ---------------------------
# Figure type sniffers
# ---------------------------
def _is_matplotlib_figure(obj: Any) -> bool:
    try:
        import matplotlib.figure as mpf  # lazy import
        return isinstance(obj, mpf.Figure)
    except Exception:
        return False

def _is_plotly_figure(obj: Any) -> bool:
    try:
        import plotly.graph_objs as go  # lazy import
        return isinstance(obj, go.Figure)
    except Exception:
        return False


# ---------------------------
# Core save / load
# ---------------------------
def dump_figure_to_pickle(fig: Any, path: str) -> str:
    """
    Save `fig` to a .pkl at `path`.
      * Matplotlib: pickles the Figure directly; on failure, stores PNG bytes in the pickle.
      * Plotly: stores JSON (pio.to_json) in the pickle for robust reloads.
    Returns the absolute path written.
    """
    abspath = os.path.abspath(path)

    if _is_matplotlib_figure(fig):
        # Try direct pickle first
        try:
            with open(abspath, "wb") as f:
                pickle.dump(("matplotlib", fig), f, protocol=pickle.HIGHEST_PROTOCOL)
            return abspath
        except Exception:
            # Fallback: PNG bytes payload inside a pickle
            try:
                import matplotlib.pyplot as plt  # noqa
                import matplotlib
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150)
                payload = {"kind": "matplotlib_png", "png": buf.getvalue()}
                with open(abspath, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                return abspath
            except Exception as ex:
                raise RuntimeError(f"Could not serialize Matplotlib figure: {ex}") from ex

    if _is_plotly_figure(fig):
        try:
            import plotly.io as pio
            js = pio.to_json(fig, pretty=False)
            with open(abspath, "wb") as f:
                pickle.dump(("plotly_json", js), f, protocol=pickle.HIGHEST_PROTOCOL)
            return abspath
        except Exception:
            # As a last resort, pickle the object itself
            with open(abspath, "wb") as f:
                pickle.dump(("plotly", fig), f, protocol=pickle.HIGHEST_PROTOCOL)
            return abspath

    # Unknown type: store as-is
    with open(abspath, "wb") as f:
        pickle.dump(("object", fig), f, protocol=pickle.HIGHEST_PROTOCOL)
    return abspath


def load_figure_from_pickle(path: str) -> Any:
    """
    Load a figure (or payload) previously saved by dump_figure_to_pickle().
      - ("matplotlib", Figure) -> returns the Figure
      - {"kind": "matplotlib_png", "png": ...} -> returns the dict payload (PNG bytes)
      - ("plotly_json", json) -> returns a Plotly Figure (pio.from_json)
      - ("plotly", Figure) -> returns the Figure
      - else -> returns the stored object
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, tuple) and len(obj) == 2:
        kind, payload = obj
        if kind == "matplotlib":
            return payload
        if kind == "plotly_json":
            import plotly.io as pio
            return pio.from_json(payload)
        if kind == "plotly":
            return payload
        return payload

    if isinstance(obj, dict) and obj.get("kind") == "matplotlib_png":
        return obj  # contains PNG bytes for a non-picklable MPL figure

    return obj


# ---------------------------
# Tk helpers
# ---------------------------
def _ensure_tk_root(parent=None):
    """Return an existing Tk root or create a hidden one; None if Tk is unavailable."""
    try:
        import tkinter as tk
        root = parent or tk._get_default_root()
        if root is None:
            root = tk.Tk()
            root.withdraw()
        return root
    except Exception:
        return None

def _asksave_pkl(default_name: str, parent=None) -> Optional[str]:
    try:
        from tkinter import filedialog
        return filedialog.asksaveasfilename(
            parent=parent,
            title="Save plot (.pkl)",
            defaultextension=".pkl",
            initialfile=default_name,
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        ) or None
    except Exception:
        return None

def _guess_current_matplotlib_fig():
    try:
        import matplotlib.pyplot as plt
        return plt.gcf()
    except Exception:
        return None


# ---------------------------
# Public: popup "Save plot?"
# ---------------------------
def prompt_save_figure_pkl(
    fig: Any = None,
    default_name: str = "plot.pkl",
    parent=None
) -> Optional[str]:
    """
    Show a small Tk window: 'Save plot?' with 'Save' and 'Cancel'.
    On 'Save', open a Save-As dialog and write the .pkl.

    If `fig` is None, we try to grab the current Matplotlib figure.
    For Plotly figures, you should pass `fig` explicitly.
    Returns the saved path, or None if cancelled.
    """
    root = _ensure_tk_root(parent)
    if root is None:
        return None  # Tk not available

    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception:
        return None

    # Modal popup
    win = tk.Toplevel(root)
    win.title("Save plot?")
    win.resizable(False, False)
    ttk.Label(win, text="Save plot?").grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 6))

    result = {"path": None}

    def on_save():
        target = _asksave_pkl(default_name, win)
        if not target:
            win.destroy()
            return
        try:
            _fig = fig if fig is not None else _guess_current_matplotlib_fig()
            if _fig is None:
                raise RuntimeError("No figure was provided and no Matplotlib figure is active.")
            p = dump_figure_to_pickle(_fig, target)
            messagebox.showinfo("Saved", f"Saved to:\n{p}", parent=win)
            result["path"] = p
        except Exception as ex:
            messagebox.showerror("Error", f"Could not save:\n{ex}", parent=win)
        finally:
            win.destroy()

    def on_cancel():
        win.destroy()

    ttk.Button(win, text="Save",   command=on_save).grid(  row=1, column=0, padx=(12, 6), pady=(0, 12), sticky="e")
    ttk.Button(win, text="Cancel", command=on_cancel).grid(row=1, column=1, padx=(6, 12),  pady=(0, 12), sticky="w")

    # Center & modal
    try:
        win.update_idletasks()
        x = (win.winfo_screenwidth() - win.winfo_reqwidth()) // 2
        y = (win.winfo_screenheight() - win.winfo_reqheight()) // 3
        win.geometry(f"+{x}+{y}")
        win.transient(root)
        win.grab_set()
        win.wait_window()
    except Exception:
        pass

    return result["path"]


# ---------------------------
# Optional: Matplotlib button
# ---------------------------
def attach_matplotlib_save_button(
    fig: Any = None,
    label: str = "Save .pkl",
    default_name: str = "plot.pkl",
    parent=None
):
    """
    Add a 'Save .pkl' button to a Matplotlib figure window that opens the popup.
    Returns the Button object so callers may also keep a reference if they want.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        _fig = fig or plt.gcf()

        # Place the button; keep it out of layout so tight_layout/constrained_layout won't resize around it
        ax_btn = _fig.add_axes([0.26, 0.92, 0.12, 0.03])
        try:
            ax_btn.set_in_layout(False)
        except Exception:
            pass
        # Avoid toolbar navigation interfering inside this tiny Axes
        try:
            ax_btn.set_navigate(False)
        except Exception:
            pass

        btn = Button(ax_btn, label)

        def _onclick(_evt, _fig=_fig):
            # prompt_save_figure_pkl is from the same module
            prompt_save_figure_pkl(fig=_fig, default_name=default_name, parent=parent)

        cid = btn.on_clicked(_onclick)

        # ---- IMPORTANT: keep a strong reference so GC doesn't kill the widget ----
        if not hasattr(_fig, "_pkl_save_widgets"):
            _fig._pkl_save_widgets = []  # type: ignore[attr-defined]
        _fig._pkl_save_widgets.append({"ax": ax_btn, "btn": btn, "cid": cid})  # type: ignore[attr-defined]

        return btn
    except Exception:
        return None
