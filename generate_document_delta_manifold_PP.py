#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\generate_document_delta_manifold_PP.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu


# This script processes item level text files into a sentence level segmentation, 
# loads an LDA model, dictionary, and interpretive categories file
# creates a pair-wise displacement vector tensor of each documents clustered text
# uses these displacement vector transformation matricies and principal component directions as directional manifold morphisms for assessing internal document cluster directionality between documents Endpoints
# visualizes document similarity, and returns pair-wise topic language sets for similar cluster text
# analyzes and exports tabular data detailing dataset morphism matches
# analyzes and plots distribution of all matching pair-wise clusters across the two axis of alignment
# arranges items by morphism feature alignment similarity and uses this to re-weight topic assignment by collecting segment text across the flow of the clustered morphisms. renderes this in a sankey flow diagram


# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.



# ---- multiprocessing / GUI guard (must be first) ----
import os, multiprocessing as mp

IS_MAIN_PROCESS = (mp.current_process().name == "MainProcess")

# Make workers headless so they cannot open GUI backends
if not IS_MAIN_PROCESS:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

def log(*args, **kwargs):
    """Print only from the main process."""
    if IS_MAIN_PROCESS:
        print(*args, **kwargs)


from backrooms import load_document_texts_by_prefix, Timer

from arrangement_endpoint import build_categorical_arrangement_from_cdm_tuple, sankey_docs_left
from topic_flow_labels import annotate_tree_with_topic_flow
from morphism_shapes import extract_shapes_from_cdm_dict, cluster_shapes, doc_membership, build_hierarchy_all, _kmeans_np

from figure_pickle import (
    attach_matplotlib_save_button,
    _is_matplotlib_figure,
    _is_plotly_figure,
    dump_figure_to_pickle,
    load_figure_from_pickle,
    prompt_save_figure_pkl,
    _guess_current_matplotlib_fig,
    _asksave_pkl,
    _ensure_tk_root
)

from topic_modeling import (
    mk_delta_manifold,
    build_cluster_delta_matrix,
    build_topic_embeddings_for_doc,
    assess_topic_frequency_across_clusters,
    visualize_documents_with_directional_overlap,
    analyze_morphism_match_field,
    output_analysis,
    plot_morphism_match_field_3d
)
import re
import numpy as np
import threading, sys


import itertools
import nltk

import umap.umap_ as umap #umap-learn

import matplotlib.pyplot as plt

import gensim.downloader as api
from gensim.models import LdaModel
from gensim.corpora import Dictionary

from sentence_transformers import SentenceTransformer

import pickle

import tkinter as tk
from tkinter import Tk, ttk, filedialog, simpledialog, TclError, messagebox

# ----          Top Level         ----
# ---- worker context & functions ----
from concurrent.futures import ProcessPoolExecutor, as_completed

_worker_ctx = {"LDA": None, "DICT": None, "NUM_TOPICS": None, "SBERT_M": None}

def _init_worker(lda_path, dict_path, num_topics, device: str | None = None):
    #Runs once in each worker.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    print("Starting new CPU core process")
    from gensim.models import LdaModel
    from gensim.corpora import Dictionary
    from sentence_transformers import SentenceTransformer
    _worker_ctx["LDA"] = LdaModel.load(lda_path)
    _worker_ctx["DICT"] = Dictionary.load(dict_path)
    _worker_ctx["NUM_TOPICS"] = int(num_topics)
    _worker_ctx["SBERT_M"] = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def manifoldit(item_id, item_text):
    """Per-document build step (runs in workers)."""
    from topic_modeling import mk_delta_manifold, build_cluster_delta_matrix

    # Reuse the worker's cached model for this document
    delta_matrix, cluster_order, labels, segments, embeddings, k = mk_delta_manifold(item_id, item_text, _worker_ctx["SBERT_M"])

    if k < 3 or len(set(labels)) < 3:
        # Not enough structure to form deltas; return a marker
        return (item_id, None, None)

    CDM = build_cluster_delta_matrix(segments, embeddings, labels, _worker_ctx["LDA"], _worker_ctx["DICT"], _worker_ctx["NUM_TOPICS"], n_clusters=k)
    return (item_id, CDM, segments)

def main():
    print("Initializing WaAffL Iron")
    # ==== Cluster Text Reader Helpers ============================================
    # A tiny tools window with a "Cluster text Reader" button (disabled by default).
    # Call `set_cluster_text_reader_sources(document_delta_dict, segments_by_doc, parent=root)`
    # after your data are loaded/built to enable the button.

    # Globals that survive your prompt window lifecycle
    _CTR_TOOLS = {"win": None, "btn": None, "readers": []}
    _CTR_SOURCES = {"doc_delta": None, "segments": None}

    def _ensure_cluster_tools_window(parent):
        """Create (or reuse) a small tools window that hosts the Reader button."""
        import tkinter as tk
        from tkinter import ttk
        if _CTR_TOOLS["win"] is not None and _CTR_TOOLS["win"].winfo_exists():
            return _CTR_TOOLS["win"]

        win = tk.Toplevel(parent)
        win.title("Manifold Tools")
        win.resizable(False, False)

        frm = ttk.Frame(win, padding=(10, 8))
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(0, weight=1)

        def _launch_reader():
            _open_cluster_text_reader(parent=win)

        btn = ttk.Button(frm, text="Open new cluster text reader",
                         command=_launch_reader, state="disabled")
        btn.grid(row=0, column=0, padx=6, pady=6, sticky="ew")

        _CTR_TOOLS["win"] = win
        _CTR_TOOLS["btn"] = btn
        return win

    def set_cluster_text_reader_sources(document_delta_dict, segments_by_doc, parent=None):
        """
        Provide the data the reader needs and enable the button.
        - document_delta_dict[item_id] = (delta_matrix, cluster_order, labels,
                                          cluster_topic_distributions, cluster_embeddings, cluster_dirs)
        - segments_by_doc[item_id] = [full segment text ...]
        """
        _CTR_SOURCES["doc_delta"] = document_delta_dict
        _CTR_SOURCES["segments"] = segments_by_doc

        try:
            import tkinter as tk
            top = parent if parent is not None else (globals().get("root", None) or tk._get_default_root())
            if top is not None:
                _ensure_cluster_tools_window(top)
            if _CTR_TOOLS["btn"] is not None:
                _CTR_TOOLS["btn"].configure(state="normal")
        except Exception as ex:
            print(f"[CTR] Unable to enable Cluster text Reader: {ex}")

    def _open_cluster_text_reader(parent=None):
        """Open a NEW Cluster text Reader window (requires sources)."""
        import tkinter as tk
        from tkinter import ttk, messagebox

        doc_delta = _CTR_SOURCES["doc_delta"]
        segs      = _CTR_SOURCES["segments"]

        if not isinstance(doc_delta, dict) or not isinstance(segs, dict) or not doc_delta:
            messagebox.showwarning(
                "Cluster text Reader",
                "Data not loaded yet. Load/build your manifold and segments first.",
                parent=parent
            )
            return

        # ---- create a NEW reader window every call ----
        idx = len(_CTR_TOOLS["readers"]) + 1
        win = tk.Toplevel(parent)
        win.title(f"Cluster text Reader #{idx}")
        win.geometry("1000x640")
        win.rowconfigure(1, weight=1)
        win.columnconfigure(0, weight=1)

        # cascade position (avoid perfect overlap of multiple windows)
        try:
            base_x = parent.winfo_rootx() if parent else 100
            base_y = parent.winfo_rooty() if parent else 100
            offset = 40 * (idx - 1)
            win.geometry(f"+{base_x + offset}+{base_y + offset}")
        except Exception:
            pass

        # register & cleanup on close
        _CTR_TOOLS["readers"].append(win)
        def _on_destroy(event=None, w=win):
            try:
                _CTR_TOOLS["readers"].remove(w)
            except ValueError:
                pass
        win.bind("<Destroy>", _on_destroy)

        # --- Top controls: item_id then pair-wise cluster ------------------------
        top = ttk.Frame(win, padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Item ID:").grid(row=0, column=0, sticky="w")
        item_var = tk.StringVar()

        # Use display mapping so non-string keys work in the UI
        _id_display_to_key = {str(k): k for k in doc_delta.keys()}
        item_ids = sorted(list(_id_display_to_key.keys()))

        item_cb = ttk.Combobox(top, textvariable=item_var, values=item_ids, state="readonly")
        item_cb.grid(row=0, column=1, sticky="ew", padx=(6, 6))

        ttk.Label(top, text="Pair-wise cluster:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        pair_var = tk.StringVar()
        pair_cb = ttk.Combobox(top, textvariable=pair_var, state="disabled", width=24)
        pair_cb.grid(row=0, column=3, sticky="w", padx=(6, 0))

        # --- Reading area: two panes --------------------------------------------
        paned = ttk.Panedwindow(win, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        def _mk_reader(parent_frame, title):
            frm = ttk.Frame(parent_frame)
            ttk.Label(frm, text=title).pack(anchor="w")
            txt = tk.Text(frm, wrap="word", height=28)
            yscroll = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
            txt.configure(yscrollcommand=yscroll.set)
            txt.pack(side="left", fill="both", expand=True)
            yscroll.pack(side="right", fill="y")
            return frm, txt

        left_frm, left_txt = _mk_reader(paned, "SRC cluster segments")
        right_frm, right_txt = _mk_reader(paned, "DST cluster segments")
        paned.add(left_frm, weight=1)
        paned.add(right_frm, weight=1)

        # --- Helpers -------------------------------------------------------------
        def _clusters_for_item(iid):
            """
            Return (sorted_clusters, labels_list) for the chosen item_id.
            Each bundle is a 6-tuple; labels are at index 2 (see your script).  :contentReference[oaicite:2]{index=2}
            """
            try:
                bundle = doc_delta[iid]
                labels = bundle[2]
                if hasattr(labels, "tolist"):
                    labels = labels.tolist()
                clusters = sorted({int(c) for c in labels})
                return clusters, labels
            except Exception as ex:
                print(f"[CTR] Unable to get clusters for {iid}: {ex}")
                return [], []

        pair_display_to_tuple = {}

        def _update_pairs_for_item(*_):
            iid = _id_display_to_key.get(item_var.get())
            if iid is None:
                pair_cb.configure(state="disabled", values=[])
                return
            clusters, _ = _clusters_for_item(iid)
            pairs = []
            pair_display_to_tuple.clear()
            for s in clusters:
                for d in clusters:
                    if s == d:
                        continue
                    disp = f"C{s} \u2192 C{d}"   # e.g., "C2 → C4"
                    pair_display_to_tuple[disp] = (s, d)
                    pairs.append(disp)
            pair_cb.configure(state=("readonly" if pairs else "disabled"), values=pairs)
            if pairs:
                pair_var.set(pairs[0])
                _on_pair_changed()

        def _on_pair_changed(*_):
            iid = _id_display_to_key.get(item_var.get())
            disp = pair_var.get()
            if iid is None or disp not in pair_display_to_tuple:
                return
            s, d = pair_display_to_tuple[disp]

            # Fetch labels & segments for this item_id
            bundle = doc_delta[iid]
            labels = bundle[2]
            if hasattr(labels, "tolist"):
                labels = labels.tolist()
            segments = segs.get(iid, [])

            # Gather full segment text for each cluster side
            src_texts = [segments[i] for i, lab in enumerate(labels) if int(lab) == int(s)]
            dst_texts = [segments[i] for i, lab in enumerate(labels) if int(lab) == int(d)]

            # Render
            left_txt.configure(state="normal"); left_txt.delete("1.0", "end")
            right_txt.configure(state="normal"); right_txt.delete("1.0", "end")
            left_txt.insert("1.0", "\n\n".join(src_texts) if src_texts else "(no segments)")
            right_txt.insert("1.0", "\n\n".join(dst_texts) if dst_texts else "(no segments)")
            left_txt.configure(state="normal")
            right_txt.configure(state="normal")

        item_cb.bind("<<ComboboxSelected>>", _update_pairs_for_item)
        pair_cb.bind("<<ComboboxSelected>>", _on_pair_changed)

        # Prefill first item if available
        if item_ids:
            item_var.set(item_ids[0])
            _update_pairs_for_item()

        try:
            win.lift()
        except Exception:
            pass

    # ==== End Cluster Text Reader Helpers ========================================



    # Hide the main tkinter window
    root = Tk()
    #root.withdraw()

    timer1 = Timer()
    timer1.start()


    def load_topic_labels(topic_list_path: str, expected_size: int | None = None) -> list[str]:
        """
        Read a topic list file and return a list where index = topic id, value = topic label.
        Supports both tab-delimited 'id<TAB>label' and 'Topic <id>: <label>' formats.
        If expected_size is given, output is padded/truncated to that length (missing = "").
        """
        labels_dict: dict[int, str] = {}
        max_id = -1

        def _parse_line(line: str):
            line = line.strip()
            if not line:
                return None
            # Prefer strict tab-delimited: "<id>\t<label...>"
            if "\t" in line:
                left, right = line.split("\t", 1)
                m = re.search(r"(\d+)", left)
                if not m:
                    return None
                tid = int(m.group(1))
                return tid, right.strip()
            # Fallback: "Topic <id>: <label>" or "<id>: <label>"
            m = re.match(r"^\s*(?:Topic\s*)?(\d+)\s*[:\-]\s*(.+?)\s*$", line)
            if m:
                return int(m.group(1)), m.group(2).strip()
            return None

        # Open with utf-8, fallback to utf-16 if needed
        try:
            f = open(topic_list_path, "r", encoding="utf-8")
        except UnicodeDecodeError:
            f = open(topic_list_path, "r", encoding="utf-16")

        with f:
            for ln, raw in enumerate(f, 1):
                parsed = _parse_line(raw)
                if not parsed:
                    continue
                tid, label = parsed
                labels_dict[tid] = label
                if tid > max_id:
                    max_id = tid

        # Decide final size
        size = expected_size if expected_size is not None else (max_id + 1 if max_id >= 0 else 0)
        labels = [""] * size

        # Fill known labels; expand if a tid exceeds expected_size
        for tid, lab in labels_dict.items():
            if tid < len(labels):
                labels[tid] = lab
            else:
                # Expand list to accommodate larger topic id
                labels.extend([""] * (tid - len(labels) + 1))
                labels[tid] = lab

        return labels

    def load_document_delta_dict() -> dict | None:
        """Pick a previously saved .pkl file and return the dict."""

        path = filedialog.askopenfilename(
            title="Load Document Delta Dict",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if not path:
            print("Load cancelled.")
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as ex:
            print(f"Error loading file: {ex}")
            return None

    def save_document_delta_dict(doc_delta: dict) -> str | None:
        """
        Save the in-memory `document_delta_dict` to a file for later reuse.

        The saved object is exactly the dict that `visualize_documents_with_directional_overlap`
        expects: {doc_id: (delta_matrix, cluster_order, labels, cluster_topic_distributions,
                           cluster_embeddings, cluster_dirs)}.  No conversion is performed,
        so you can `pickle.load` it later and pass it straight to the viz function.

        Returns:
            The file path saved to (str) on success, or None if the user cancels.
        """

        if not isinstance(doc_delta, dict) or len(doc_delta) == 0:
            print("Nothing to save: `document_delta_dict` is empty.")
            return None

        path = filedialog.asksaveasfilename(
            title="Save Document Delta Dict",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if not path:
            print("Save cancelled.")
            return None

        try:
            with open(path, "wb") as f:
                pickle.dump(doc_delta, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved document delta data to: {path}")
            return path
        except Exception as ex:
            print(f"Error saving file: {ex}")
            return None

    def save_segments_by_doc(doc_seg: dict) -> str | None:
        """
        Save the in-memory `segments_by_doc` dict to a file for later reuse.

        The saved object is exactly the dict that `output_analysis`
        expects: {doc_id: segments}.  No conversion is performed,
        so you can `pickle.load` it later and pass it straight to the analysis function.

        Returns:
            The file path saved to (str) on success, or None if the user cancels.
        """

        if not isinstance(doc_seg, dict) or len(doc_seg) == 0:
            print("Nothing to save: `segments_by_doc` is empty.")
            return None

        path = filedialog.asksaveasfilename(
            title="Save Segments by Document Dict",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if not path:
            print("Save cancelled.")
            return None

        try:
            with open(path, "wb") as f:
                pickle.dump(doc_seg, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved document segments data to: {path}")
            return path
        except Exception as ex:
            print(f"Error saving file: {ex}")
            return None

    def _choose_csv_save_path(default_name: str = "morphism_matches.csv", parent_window=None) -> str | None:
        """
        Show a topmost Save-As dialog and return the chosen CSV path (or None if cancelled).
        We parent the dialog to a live Tk window to avoid the 'silent cancel' issue.
        """

        # Derive a stable parent: prefer the provided window, then global root, then Tk default root
        try:
            parent = parent_window if parent_window is not None else (globals().get("root") or tk._get_default_root())
        except Exception:
            parent = None

        # Nudge parent on top so the dialog can't appear behind other Toplevels
        try:
            if parent is not None:
                parent.update_idletasks()
                parent.lift()
                parent.attributes("-topmost", True)
                parent.after(200, lambda: parent.attributes("-topmost", False))
        except Exception:
            pass

        path = filedialog.asksaveasfilename(
            parent=parent,
            title="Save morphism matches Tab-Delimited file",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        return path or None

    def _choose_html_save_path(default_name: str = "collection_arrangement.html", parent_window=None) -> str | None:
        """
        Show a topmost Save-As dialog and return the chosen html path (or None if cancelled).
        We parent the dialog to a live Tk window to avoid the 'silent cancel' issue.
        """

        # Derive a stable parent: prefer the provided window, then global root, then Tk default root
        try:
            parent = parent_window if parent_window is not None else (globals().get("root") or tk._get_default_root())
        except Exception:
            parent = None

        # Nudge parent on top so the dialog can't appear behind other Toplevels
        try:
            if parent is not None:
                parent.update_idletasks()
                parent.lift()
                parent.attributes("-topmost", True)
                parent.after(200, lambda: parent.attributes("-topmost", False))
        except Exception:
            pass

        path = filedialog.asksaveasfilename(
            parent=parent,
            title="Save collection arrangement HTML",
            defaultextension=".html",
            initialfile=default_name,
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )
        return path or None

    def prompt_manifold_action(
        default_mode="build",
        default_num_topics=100,
        default_top_n_topics=3,
        default_top_m_keywords=7,
        default_cos_threshold=0.50,
        default_srcdst_threshold=0.90,
        default_endpoint="visualize",  # "visualize" | "analyze" | "arrange"
        default_doc_id=None,           # optional analysis focus doc_id
        parent=None,
        verbose=False
    ):
        """
        Modal configuration dialog for manifold workflow.

        Returns:
            dict with keys:
              - mode: "build" | "load" | "compare"
              - num_topics, top_n_topics, top_m_keywords, cos_threshold, srcdst_threshold
              - endpoint: "visualize" | "analyze" | "arrange"
              - doc_id: str or None
              - files: {
                    "manifold_pkl": str,
                    "compare_manifold_pkl": str,    # <-- NEW
                    "manifold_segments_pkl": str,
                    "compare_segments_pkl": str,
                    "lda_model": str,
                    "lda_dict": str,
                    "lda_labels": str
                }
            or None if cancelled.
        """
        import os
        import threading
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox

        # ---------- helpers ----------
        def _center_on_screen(win):
            win.update_idletasks()
            try:
                sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
                ww, wh = win.winfo_reqwidth(), win.winfo_reqheight()
                x = max(0, (sw - ww) // 2)
                y = max(0, (sh - wh) // 3)
                win.geometry(f"+{x}+{y}")
            except Exception:
                pass

        def _parse_int(name, s, min_val=1, max_val=None):
            try:
                v = int(str(s).strip())
                if v < min_val:
                    raise ValueError
                if max_val is not None and v > max_val:
                    raise ValueError
                return v, None
            except Exception:
                rng = f"≥ {min_val}" if max_val is None else f"between {min_val} and {max_val}"
                return None, f"“{name}” must be an integer {rng}."

        def _parse_float01(name, s):
            try:
                v = float(str(s).strip())
                if not (0.0 <= v <= 1.0):
                    raise ValueError
                return v, None
            except Exception:
                return None, f"“{name}” must be a number between 0.0 and 1.0."

        def _exists_or_error(path, label):
            if not path:
                return f"Please choose {label}."
            if not os.path.isfile(path):
                return f"The selected {label} does not exist:\n{path}"
            return None
        
        def _plot_pkl_picker():
            """Pick a previously saved plot.pkl file and return the graph."""
            plt_path = filedialog.askopenfilename(
                title="Load matplotlib object",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if not plt_path:
                print("Load cancelled.")
                return None
            try:
                fig = load_figure_from_pickle(plt_path)
                fig.show()
            except Exception as ex:
                print(f"Error loading plot: {ex}")
            return None

        # ---------- guard ----------
        if threading.current_thread() is not threading.main_thread():
            if verbose:
                print("[prompt] Tk must run on the main thread; returning defaults.")
            return {
                "mode": default_mode,
                "num_topics": default_num_topics,
                "top_n_topics": default_top_n_topics,
                "top_m_keywords": default_top_m_keywords,
                "cos_threshold": default_cos_threshold,
                "srcdst_threshold": default_srcdst_threshold,
                "endpoint": default_endpoint,
                "doc_id": default_doc_id,
                "files": {
                    "manifold_pkl": "",
                    "compare_manifold_pkl": "",
                    "manifold_segments_pkl": "",
                    "compare_segments_pkl": "",
                    "lda_model": "",
                    "lda_dict": "",
                    "lda_labels": "",
                }
            }

        # ---------- root / parent ----------
        _parent = parent
        created_root = False
        try:
            if _parent is None:
                _parent = tk.Tk()
                _parent.withdraw()
                created_root = True
        except Exception as ex:
            print(f"[prompt] unable to create root: {ex}")
            return None

        result = {"v": None}

        # ---------- build window ----------
        win = tk.Toplevel(_parent)
        
        # -- Cluster Text Reader: ensure tools window exists (button disabled for now)
        try:
            _ensure_cluster_tools_window(_parent)
        except Exception as _ex:
            print(f"[CTR] Tools window init failed: {_ex}")
        
        win.withdraw()
        win.title("Manifold Processing Configuration")
        win.resizable(False, False)
        try:
            win.transient(_parent)
        except Exception:
            pass

        outer = ttk.Frame(win, padding=(14, 12))
        outer.grid(row=0, column=0, sticky="nsew")

        row = 0
        
        # --- Load ---
        load_frame = ttk.LabelFrame(outer, text="Load")
        load_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1
        
        ttk.Button(load_frame, text="Load plot", command=_plot_pkl_picker).grid(row=0, column=0, pady=(0, 2))
        
        # --- Mode ---
        mode_frame = ttk.LabelFrame(outer, text="Mode")
        mode_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1

        mode_var = tk.StringVar(value=default_mode)
        rb_build   = ttk.Radiobutton(mode_frame, text="Build new manifold", value="build",   variable=mode_var)
        rb_load    = ttk.Radiobutton(mode_frame, text="Load saved document manifolds", value="load",    variable=mode_var)
        rb_compare = ttk.Radiobutton(mode_frame, text="Compare with another manifold (.pkl)", value="compare", variable=mode_var)

        rb_build.grid(row=0, column=0, sticky="w", padx=8, pady=4)
        rb_load.grid(row=1, column=0, sticky="w", padx=8, pady=4)
        rb_compare.grid(row=2, column=0, sticky="w", padx=8, pady=4)

        # --- Parameters ---
        params = ttk.LabelFrame(outer, text="Parameters")
        params.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1

        labels = [
            ("num_topics (int ≥ 1)", str(default_num_topics)),
            ("top_n_topics (int ≥ 1)", str(default_top_n_topics)),
            ("top_m_keywords (int ≥ 1)", str(default_top_m_keywords)),
            ("cos_threshold (0–1)", f"{default_cos_threshold}"),
            ("srcdst_threshold (0–1)", f"{default_srcdst_threshold}"),
        ]
        entries = {}
        for r, (lab, val) in enumerate(labels):
            ttk.Label(params, text=lab).grid(row=r, column=0, sticky="w", padx=8, pady=4)
            e = ttk.Entry(params, width=18)
            e.insert(0, val)
            e.grid(row=r, column=1, sticky="w", padx=(6, 8), pady=4)
            entries[lab] = e

        # --- Endpoint + doc_id (optional) ---
        end_frame = ttk.LabelFrame(outer, text="Endpoint")
        end_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1

        endpoint_var = tk.StringVar(value=default_endpoint)

        # Existing endpoints
        ttk.Radiobutton(
            end_frame,
            text="Visualize (visualize_documents_with_directional_overlap)",
            variable=endpoint_var,
            value="visualize"
        ).grid(row=0, column=0, sticky="w", padx=8, pady=4)

        ttk.Radiobutton(
            end_frame,
            text="Analyze (analyze_morphism_match_field)",
            variable=endpoint_var,
            value="analyze"
        ).grid(row=1, column=0, sticky="w", padx=8, pady=4)

        # NEW endpoint: arrange
        ttk.Radiobutton(
            end_frame,
            text="Arrange (build_categorical_arrangement_from_cdm_tuple)",
            variable=endpoint_var,
            value="arrange"
        ).grid(row=2, column=0, sticky="w", padx=8, pady=4)

        # doc_id only applies to Analyze; leave disabled for Visualize/Arrange
        doc_id_var = tk.StringVar(value="" if default_doc_id is None else str(default_doc_id))
        ttk.Label(end_frame, text="doc_id (optional, for analysis)").grid(row=3, column=0, sticky="w", padx=8, pady=(0, 6))
        ent_doc_id = ttk.Entry(end_frame, textvariable=doc_id_var, width=36)
        ent_doc_id.grid(row=3, column=1, sticky="w", padx=(6, 8), pady=(0, 6))

        # --- Files / pickers ---
        files_frame = ttk.LabelFrame(outer, text="Files")
        files_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1

        def add_picker(row_idx, label_text, var, browse_cmd):
            ttk.Label(files_frame, text=label_text).grid(row=row_idx, column=0, sticky="w", padx=8, pady=4)
            ent = ttk.Entry(files_frame, textvariable=var, width=64)
            ent.grid(row=row_idx, column=1, sticky="w", padx=(6, 6), pady=4)
            btn = ttk.Button(files_frame, text="Browse…", command=browse_cmd)
            btn.grid(row=row_idx, column=2, sticky="w", padx=(0, 8), pady=4)
            return ent, btn

        # Variables
        var_pkl          = tk.StringVar()  # base manifold (.pkl)
        var_cmp_pkl      = tk.StringVar()  # NEW: compare manifold (.pkl)
        var_mdl          = tk.StringVar()  # LDA model
        var_dct          = tk.StringVar()  # dictionary (build only)
        var_lbl          = tk.StringVar()  # topic labels (optional)
        var_seg_pkl      = tk.StringVar()  # manifold segments (.pkl) — Load
        var_cmp_seg_pkl  = tk.StringVar()  # compare segments (.pkl)  — Compare

        # Browsers
        def browse_pkl():
            path = filedialog.askopenfilename(
                title="Select manifold (.pkl)",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if path: var_pkl.set(path)

        def browse_compare_pkl():
            path = filedialog.askopenfilename(
                title="Select compare manifold (.pkl)",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if path: var_cmp_pkl.set(path)

        def browse_lda_model():
            path = filedialog.askopenfilename(
                title="Select Gensim LDA model (saved via LdaModel.save)",
                filetypes=[("Gensim model", "*.gensim *.model *.ldamodel"), ("All files", "*.*")]
            )
            if path: var_mdl.set(path)

        def browse_dictionary():
            path = filedialog.askopenfilename(
                title="Select Gensim Dictionary",
                filetypes=[("Gensim dictionary", "*.id2word *.dict *.dict.gz"), ("All files", "*.*")]
            )
            if path: var_dct.set(path)

        def browse_labels():
            path = filedialog.askopenfilename(
                title="Select topic labels list (TXT/TSV)",
                filetypes=[("Text/TSV files", "*.txt *.tsv"), ("All files", "*.*")]
            )
            if path: var_lbl.set(path)

        def browse_segments_pkl():
            path = filedialog.askopenfilename(
                title="Select manifold segments (.pkl)",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if path: var_seg_pkl.set(path)

        def browse_compare_segments_pkl():
            path = filedialog.askopenfilename(
                title="Select compare segments (.pkl)",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if path: var_cmp_seg_pkl.set(path)

        # Row assembly
        rf = 0
        ent_pkl,      btn_pkl      = add_picker(rf, "Manifold (.pkl):",                 var_pkl,         browse_pkl);               rf += 1
        ent_seg,      btn_seg      = add_picker(rf, "Manifold segments (.pkl):",        var_seg_pkl,     browse_segments_pkl);      rf += 1
        ent_cmp_pkl,  btn_cmp_pkl  = add_picker(rf, "Compare manifold (.pkl):",         var_cmp_pkl,     browse_compare_pkl);       rf += 1
        ent_cmp_seg,  btn_cmp_seg  = add_picker(rf, "Compare segments (.pkl):",         var_cmp_seg_pkl, browse_compare_segments_pkl); rf += 1
        ent_mdl,      btn_mdl      = add_picker(rf, "LDA model:",                       var_mdl,         browse_lda_model);         rf += 1
        ent_dct,      btn_dct      = add_picker(rf, "Dictionary:",                      var_dct,         browse_dictionary);        rf += 1
        ent_lbl,      btn_lbl      = add_picker(rf, "Topic labels:",                    var_lbl,         browse_labels);            rf += 1



        # Enable/disable utilities
        def set_state(widget, enabled):
            try:
                widget.configure(state=("normal" if enabled else "disabled"))
            except Exception:
                pass

        def update_endpoint(*_):
            enable = (endpoint_var.get() == "analyze")
            set_state(ent_doc_id, enable)

        endpoint_var.trace_add("write", update_endpoint)
        update_endpoint()

        # Mode-driven enable/disable
        def update_mode(*_):
            m = mode_var.get()
            # Base manifold: needed in load & compare
            set_state(ent_pkl, m in ("load", "compare")); set_state(btn_pkl, m in ("load", "compare"))
            # Compare manifold: compare only
            set_state(ent_cmp_pkl, m == "compare"); set_state(btn_cmp_pkl, m == "compare")
            # LDA model: always needed
            set_state(ent_mdl, True); set_state(btn_mdl, True)
            # Dictionary: build only
            set_state(ent_dct, m == "build"); set_state(btn_dct, m == "build")
            # Labels: always optional
            set_state(ent_lbl, True); set_state(btn_lbl, True)
            # Manifold segments: Load only (enable also in Compare if you wish)
            set_state(ent_seg, m in ("load", "compare")); set_state(btn_seg, m in ("load", "compare"))
            # Compare segments: Compare only
            set_state(ent_cmp_seg, m == "compare"); set_state(btn_cmp_seg, m == "compare")

        mode_var.trace_add("write", update_mode)
        update_mode()

        # --- Buttons ---
        btns = ttk.Frame(outer)
        btns.grid(row=row, column=0, sticky="e")
        row += 1

        def _on_cancel(event=None):
            result["v"] = None
            try:
                win.destroy()
            except Exception:
                pass

        def _on_continue(event=None):
            # Parse params
            v_num_topics, err = _parse_int("num_topics", entries["num_topics (int ≥ 1)"].get(), 1, None)
            if err: return messagebox.showerror("Invalid input", err, parent=win)
            v_top_n_topics, err = _parse_int("top_n_topics", entries["top_n_topics (int ≥ 1)"].get(), 1, None)
            if err: return messagebox.showerror("Invalid input", err, parent=win)
            v_top_m_keywords, err = _parse_int("top_m_keywords", entries["top_m_keywords (int ≥ 1)"].get(), 1, None)
            if err: return messagebox.showerror("Invalid input", err, parent=win)
            v_cos_threshold, err = _parse_float01("cos_threshold", entries["cos_threshold (0–1)"].get())
            if err: return messagebox.showerror("Invalid input", err, parent=win)
            v_srcdst_threshold, err = _parse_float01("srcdst_threshold", entries["srcdst_threshold (0–1)"].get())
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            # Mode & file requirements
            mode = mode_var.get()
            pkl_path     = var_pkl.get().strip()
            cmp_pkl_path = var_cmp_pkl.get().strip()
            mdl_path     = var_mdl.get().strip()
            dict_path    = var_dct.get().strip()
            lbl_path     = var_lbl.get().strip()
            seg_pkl      = var_seg_pkl.get().strip()
            cmp_seg_pkl  = var_cmp_seg_pkl.get().strip()

            if mode == "load":
                err1 = _exists_or_error(pkl_path, "a manifold .pkl file")
                if err1: return messagebox.showerror("Missing file", err1, parent=win)
                err2 = _exists_or_error(mdl_path, "an LDA model")
                if err2: return messagebox.showerror("Missing file", err2, parent=win)

            elif mode == "build":
                err2 = _exists_or_error(mdl_path, "an LDA model")
                if err2: return messagebox.showerror("Missing file", err2, parent=win)
                err3 = _exists_or_error(dict_path, "a Gensim dictionary")
                if err3: return messagebox.showerror("Missing file", err3, parent=win)

            elif mode == "compare":
                # Require base manifold + compare manifold + LDA model
                err1 = _exists_or_error(pkl_path, "a base manifold .pkl file")
                if err1: return messagebox.showerror("Missing file", err1, parent=win)
                err1b = _exists_or_error(cmp_pkl_path, "a compare manifold .pkl file")
                if err1b: return messagebox.showerror("Missing file", err1b, parent=win)
                err2 = _exists_or_error(mdl_path, "an LDA model")
                if err2: return messagebox.showerror("Missing file", err2, parent=win)

            # Gather result
            raw_doc_id = doc_id_var.get().strip()
            result["v"] = {
                "mode": mode,
                "num_topics": v_num_topics,
                "top_n_topics": v_top_n_topics,
                "top_m_keywords": v_top_m_keywords,
                "cos_threshold": v_cos_threshold,
                "srcdst_threshold": v_srcdst_threshold,
                "endpoint": endpoint_var.get(),
                "doc_id": (raw_doc_id if raw_doc_id else None),
                "files": {
                    "manifold_pkl": pkl_path,
                    "compare_manifold_pkl": cmp_pkl_path,     # <-- NEW
                    "manifold_segments_pkl": seg_pkl,
                    "compare_segments_pkl": cmp_seg_pkl,
                    "lda_model": mdl_path,
                    "lda_dict": dict_path,
                    "lda_labels": lbl_path,
                }
            }
            try:
                win.destroy()
            except Exception:
                pass

        ttk.Button(btns, text="Continue", command=_on_continue).grid(row=0, column=0, pady=(0, 2))
        ttk.Button(btns, text="Cancel",   command=_on_cancel).grid(row=0, column=1, padx=(0, 6), pady=(0, 2))
        

        # key bindings
        win.bind("<Escape>", _on_cancel)
        win.bind("<Return>", _on_continue)

        # show + center + modal
        win.deiconify()
        _center_on_screen(win)
        try:
            win.lift()
            win.attributes("-topmost", True)
            win.update()
            win.after(200, lambda: win.attributes("-topmost", False))
        except Exception:
            pass
        try:
            win.grab_set()
        except Exception:
            pass
        try:
            rb_build.focus_set()
        except Exception:
            pass

        # block until closed
        try:
            win.wait_window()
        except Exception:
            pass

        if created_root:
            try:
                _parent.destroy()
            except Exception:
                pass

        return result["v"]





    cfg = prompt_manifold_action(
        default_mode="build",
        default_num_topics=100,
        default_top_n_topics=3,
        default_top_m_keywords=7,
        default_cos_threshold=0.50,
        default_srcdst_threshold=0.90,
        default_endpoint="visualize",
        default_doc_id=None,
        parent=root,
        verbose=True
    )

    if cfg is None:
        print("I'm not feeling it.")
        raise SystemExit(0)

    num_topics       = cfg["num_topics"]
    top_n_topics     = cfg["top_n_topics"]
    top_m_keywords   = cfg["top_m_keywords"]
    cos_threshold    = cfg["cos_threshold"]
    srcdst_threshold = cfg["srcdst_threshold"]
    doc_id           = cfg["doc_id"]
    fpaths           = cfg["files"]

    lda_int_topics_list = load_topic_labels(fpaths["lda_labels"], expected_size=100)


    if cfg["mode"] == "load":
        print("Loading saved .pkl and the LDA model files")
        with open(fpaths["manifold_pkl"], "rb") as fp:
            document_delta_dict = pickle.load(fp)
        if document_delta_dict is None:
            print("No manifold selected; exiting.")
            raise SystemExit(0)
        with open(fpaths["manifold_segments_pkl"], "rb") as fp:
            segments_by_doc = pickle.load(fp)
        if segments_by_doc is None:
            print("No segments selected; exiting.")
            raise SystemExit(0)
        LdaModel_loaded = LdaModel.load(fpaths["lda_model"])
        if LdaModel_loaded is None:
            print("No LDA model selected; exiting.")
            raise SystemExit(0)
        set_cluster_text_reader_sources(document_delta_dict, segments_by_doc, parent=root)
        
    elif cfg["mode"] == "build":
        log("Building new manifolds from source texts...")

        item_text_dict = load_document_texts_by_prefix(21)

        # Choose a sensible worker count; leaving 2 cores free helps responsiveness
        cpu_workers = max(1, os.cpu_count() - 4)

        document_delta_dict = {}
        segments_by_doc = {}

        # Initialize workers ONCE with the heavy models
        with ProcessPoolExecutor(
            max_workers=cpu_workers,
            initializer=_init_worker,
            initargs=(fpaths["lda_model"], fpaths["lda_dict"], num_topics)  # keep "cpu" in workers
        ) as pool:

            futures = [
                pool.submit(manifoldit, item_id, item_text)
                for item_id, item_text in item_text_dict.items()
            ]

            for fut in as_completed(futures):
                item_id, CDM, segments = fut.result()
                print(f"Completing core process for item_id: {item_id}")
                if CDM is not None:
                    document_delta_dict[item_id] = CDM
                    segments_by_doc[item_id] = segments
                else:
                    log(f"Skipping {item_id} (insufficient clusters)")

        # Persist & wire UI ONLY in main
        save_document_delta_dict(document_delta_dict)
        save_segments_by_doc(segments_by_doc)
        set_cluster_text_reader_sources(document_delta_dict, segments_by_doc, parent=root)

    # ============================
    # append key item pairs from a second pkl file assuming they are built identicaly
    if cfg["mode"] == "compare":
        print("Loading saved .pkl and the LDA model files")
        with open(fpaths["manifold_pkl"], "rb") as fp:
            document_delta_dict = pickle.load(fp)
        if document_delta_dict is None:
            print("No manifold selected; exiting.")
            raise SystemExit(0)
        with open(fpaths["manifold_segments_pkl"], "rb") as fp:
            segments_by_doc = pickle.load(fp)
        if segments_by_doc is None:
            print("No segments selected; exiting.")
            raise SystemExit(0)
        LdaModel_loaded = LdaModel.load(fpaths["lda_model"])
        if LdaModel_loaded is None:
            print("No LDA model selected; exiting.")
            raise SystemExit(0)
        print("Loading saved compare .pkl")
        with open(fpaths["compare_manifold_pkl"], "rb") as fp:
            add_document_delta_dict = pickle.load(fp)
        print("Combining manifold datasets...")
        for item_id, document_manifold_data in add_document_delta_dict.items():
            document_delta_dict[item_id] = document_manifold_data
            
        print("Loading saved compare segments .pkl")
        with open(fpaths["compare_segments_pkl"], "rb") as fp:
            add_segments_by_doc = pickle.load(fp)
        print("Combining manifold datasets...")
        for item_id, segments_by_doc_data in add_segments_by_doc.items():
            segments_by_doc[item_id] = segments_by_doc_data

        set_cluster_text_reader_sources(document_delta_dict, segments_by_doc, parent=root)

    for item_id, array in document_delta_dict.items():
        print(item_id)
        assess_topic_frequency_across_clusters(array[3])


    timer1.stop()
    print(f"Total elapsed time before visualization: {int(timer1.elapsed())//60}:min {int(timer1.elapsed())%60}:sec")

    # Endpoint selection
    if cfg["endpoint"] == "visualize":
        visualize_documents_with_directional_overlap(
            document_cluster_data=document_delta_dict,
            lda_model=LdaModel_loaded,
            top_n_topics=top_n_topics,
            top_m_keywords=top_m_keywords,
            cos_threshold=cos_threshold,
            lda_int_topics_list=lda_int_topics_list,
            srcdst_threshold=srcdst_threshold
        )  # topic_modeling.py signature supports these params. :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

    elif cfg["endpoint"] == "analyze":  # "analyze"
        res = analyze_morphism_match_field(
            document_cluster_data=document_delta_dict,
            delta_thresholds=[i / 100 for i in range(99, -1, -3)],
            pc1_thresholds=[i / 100 for i in range(99, -1, -3)],
            top_k_per_delta=1000,
            pc1_only_threshold=0.01,
            delta_max_for_pc1_only=1.00,
            require_cross_doc=True,
            verbose=True
        )
        """
        csv_path = _choose_csv_save_path(parent_window=root)
        
        output_analysis(
        res=res,
        doc_id=doc_id,
        document_cluster_data=document_delta_dict,
        segments_by_doc=segments_by_doc,
        csv_path=csv_path
        )
        """
        plot_morphism_match_field_3d(
        res=res,
        step=0.01,
        doc_id=doc_id,
        log_colors=True
        )
        
    elif cfg["endpoint"] == "arrange":  # "arrange"
        # 1) Extract morphism shapes (zig-zags)
        rows, X = extract_shapes_from_cdm_dict(
            document_delta_dict,
            max_edges_per_doc=2000,      # cap per doc (optional)
            min_weight=0.0,
            include_length=True,
            include_v_bin=True,          # retain coarse global directionality
            vbin_az=12, vbin_el=6,
            align_signs=True,
            dir_weight_beta=1.0          # incorporate flow-consistency into weights
        )
        print(f"{len(rows)} morphisms -> feature dim = {X.shape[1]}")

        # 2) Cluster shapes into K groups
        shape_labels, centroids = cluster_shapes(X, k=64, method="kmeans", random_state=0)
        print("shape clusters:", int(shape_labels.max())+1)

        # 3) Build document membership (distribution over shape types)
        M, doc_ids = doc_membership(rows, shape_labels, weight_mode="weighted", normalize=True)
        print("doc membership matrix:", M.shape)   # (#docs, K_shape)

        # 4) Hierarchical refinement
        tree = build_hierarchy_all(
            document_delta_dict,
            depth=5,            # number of levels
            shape_k=64,
            doc_k=12,
            max_edges_per_doc=2000,
            include_length=True,
            include_v_bin=True,
            align_signs=True,
            dir_weight_beta=1.0,
            hetero_threshold=0.20,  # tweak: higher => fewer splits (more homogeneous required)
            min_docs_to_split=2,
            random_state=0
        )
        
        # 5) LDA topic flow matrix weighted to nodes
        node_details = annotate_tree_with_topic_flow(
            document_delta_dict,
            tree,
            lda_topic_labels=lda_int_topics_list,  # or None
            dir_weight_beta=1.0,                   # match your morphism weighting
            weight_scale=None,                     # per-doc robust median (default)
            top_topic_k=2,                         # top-k topics per morphism side
            top_pairs=4,                           # how many pairs to show
            max_edges_per_doc=2000                 # align with your extract cap
        )
        
        # Sankey with docs on the left
        
        html_save_path = _choose_html_save_path(parent_window=root)
        
        fig = sankey_docs_left(
            tree,
            title="Documents → Arrangement Across a Collections Morphological Membership Refinement Hierarchy",
            max_docs_left=0,                 # cap left-side nodes; set None for all docs
            save_html=html_save_path,
            open_browser=True,
            height_vh=96,          # try 96–98 for near full-screen
            responsive=True
        )
        #fig.show()
        
        '''
        theme_set, themes_df, membership_df, overlaps_df = build_categorical_arrangement_from_cdm_tuple(
            document_delta_dict,
            adapter_kwargs=None,#{
                #"segment_doc_ids": segment_doc_ids,   # majority vote -> doc_id per cluster
                #"doc_ids_by_label": doc_ids_by_label, # overrides majority vote (optional)
                #"topic_labels": topic_labels,         # populates cluster.meta['top_terms'] (optional)
                #"cluster_id_prefix": "cl_",           # optional
                # "weight_scale": 0.25,               # optional: set your own scale for weights
            #},
            # Arrangement knobs:
            two_hop=True,          # allow composed morphisms
            topk_per_src=3,        # prune fan-out on composition
            weight_min=0.0,        # drop very weak edges if you want (<- adjust if needed)
            doc_agg="topk",        # 'sum'|'max'|'mean'|'topk' for doc membership
            doc_topk=3,
            seed_strategy="community",  # 'community' | 'all' | 'explicit'
            merge_protos=True,
            proto_cos_th=0.97,
            doc_jacc_th=0.8,
            membership_th=0.0,
        )

        # Save or inspect
        themes_df.to_csv("global_themes_themes.csv", index=False)
        membership_df.to_csv("global_themes_membership.csv", index=False)
        overlaps_df.to_csv("global_themes_overlaps.csv", index=False)

        # Quick peek
        print(themes_df.head())
        print(membership_df.head())
        print(overlaps_df.head())
        '''

if __name__ == "__main__":
    mp.freeze_support()                 # Windows-safe
    try:
        mp.set_start_method("spawn")    # explicit is better than implicit
    except RuntimeError:
        pass
    main()