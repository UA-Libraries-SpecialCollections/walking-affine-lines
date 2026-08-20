"""
sentence_cluster_projection_endpoint.py

Sentence-level 2D cluster projection endpoint for the document-delta manifold
pipeline.

python.exe "S:\Digital Projects\Encoding\testing\sentence_cluster_projection_endpoint_updated.py" --document-delta-pkl "doc_del_dict.pkl" --segments-pkl "segments.pkl"

What it does
------------
- Loads/accepts the saved document_delta_dict pickle:
    {item_id: (delta_matrix, cluster_order, labels,
               cluster_topic_distributions, cluster_embeddings, cluster_dirs)}
- Loads/accepts the saved segments_by_doc pickle:
    {item_id: [sentence_segment_text, ...]}
- Re-embeds sentence segments when sentence embeddings were not saved.
- Projects every selected sentence embedding into 2D.
- Colors points by item+cluster.
- Lets the user choose two items and display all clusters from both.
- Lets the user select one or more intra-item cluster morphisms Csrc -> Cdst.
- Highlights morphism arrows and the source/destination sentence clusters.
- Can fade unselected clusters to 10% opacity while selected cluster components remain fully visible.
- Draws per-cluster PC1/principal-component arrows from cluster centroids.
- Click a sentence point to inspect its item, cluster, segment index, and text.

The endpoint intentionally uses Tkinter + Matplotlib so it fits the rest of the
current desktop pipeline and avoids requiring Dash/Flask.

Standalone usage
----------------
python sentence_cluster_projection_endpoint.py \
    --document-delta-pkl path/to/document_delta_dict.pkl \
    --segments-pkl path/to/segments_by_doc.pkl

Pipeline usage
--------------
from sentence_cluster_projection_endpoint import visualize_sentence_cluster_projection_endpoint

visualize_sentence_cluster_projection_endpoint(
    document_delta_dict=document_delta_dict,
    segments_by_doc=segments_by_doc,
    parent=root,
)
"""

from __future__ import annotations

import argparse
import math
import pickle
import textwrap
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Data normalization helpers
# -----------------------------------------------------------------------------

def _as_doc_key_display(k: Any) -> str:
    return str(k)


def _label_to_int(label: Any) -> int:
    """The current pipeline uses integer cluster labels. Keep one coercion point."""
    try:
        return int(label)
    except Exception:
        # Fall back to a stable hash bucket if a future pipeline uses non-int labels.
        # The UI will still display the original string where possible.
        return int(abs(hash(str(label))) % (10**9))


def _as_1d_int_array(labels: Any) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.array([_label_to_int(x) for x in arr], dtype=int)


def _normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < eps] = 1.0
    return X / n


def _remove_top_components(E: np.ndarray, n: int = 2) -> np.ndarray:
    """
    Match the common-component removal used by mk_delta_manifold().

    The source pipeline normalizes SBERT embeddings, removes the top shared
    components, and renormalizes before clustering. Reapplying this here lets
    sentence-level points be comparable to the saved cluster labels/centroids
    when embeddings were not persisted in segments_by_doc.
    """
    E = _normalize_rows(np.asarray(E, dtype=float))
    n = int(n or 0)
    if n <= 0 or E.shape[0] <= 2:
        return E
    n = min(n, E.shape[0] - 1, E.shape[1])
    if n <= 0:
        return E
    X = E - E.mean(axis=0, keepdims=True)
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        P = Vt[:n].T
        X2 = X - X @ P @ P.T
        return _normalize_rows(X2)
    except np.linalg.LinAlgError:
        return E


def _unpack_cdm_tuple(cdm_tuple: tuple) -> Tuple[np.ndarray, List[int], np.ndarray, dict, np.ndarray, Optional[np.ndarray]]:
    """
    Normalize a current or legacy CDM tuple.

    Current 6-tuple:
      delta_matrix, cluster_order, labels, cluster_topic_distributions,
      cluster_embeddings, cluster_dirs
    Legacy 5-tuple omits cluster_dirs.
    """
    if len(cdm_tuple) >= 6:
        delta_matrix, cluster_order, labels, topic_dists, cluster_embeddings, cluster_dirs = cdm_tuple[:6]
    elif len(cdm_tuple) == 5:
        delta_matrix, cluster_order, labels, topic_dists, cluster_embeddings = cdm_tuple
        cluster_dirs = None
    else:
        raise ValueError(f"Expected a 5- or 6-tuple CDM entry; got length {len(cdm_tuple)}")

    order = [_label_to_int(x) for x in list(cluster_order)]
    labels_arr = _as_1d_int_array(labels)
    Delta = np.asarray(delta_matrix, dtype=float)

    if isinstance(cluster_embeddings, dict):
        C = np.vstack([np.asarray(cluster_embeddings[l], dtype=float) for l in cluster_order])
    else:
        C = np.asarray(cluster_embeddings, dtype=float)
    if C.ndim == 1:
        C = C[None, :]

    Dirs = None if cluster_dirs is None else np.asarray(cluster_dirs, dtype=float)
    return Delta, order, labels_arr, topic_dists, C, Dirs


def _extract_segments_bundle(bundle: Any) -> Tuple[List[str], Optional[np.ndarray]]:
    """
    Accept the current pipeline's {doc_id: [segments...]} shape, while also
    supporting future richer shapes like:
      {"segments": [...], "embeddings": np.ndarray}
      ([segments...], embeddings)
    """
    segments: Optional[Sequence[str]] = None
    embeddings: Optional[np.ndarray] = None

    if isinstance(bundle, dict):
        for key in ("segments", "sentences", "texts", "segment_texts"):
            if key in bundle:
                segments = bundle[key]
                break
        for key in ("embeddings", "sentence_embeddings", "segment_embeddings"):
            if key in bundle:
                embeddings = np.asarray(bundle[key], dtype=float)
                break
    elif isinstance(bundle, tuple) and len(bundle) >= 2:
        possible_segments, possible_embeddings = bundle[0], bundle[1]
        if isinstance(possible_segments, (list, tuple)) and all(isinstance(x, str) for x in possible_segments):
            segments = possible_segments
            embeddings = np.asarray(possible_embeddings, dtype=float)
    elif isinstance(bundle, (list, tuple)):
        if all(isinstance(x, str) for x in bundle):
            segments = list(bundle)

    if segments is None:
        raise ValueError(
            "Could not read segments. Expected a list[str], a (segments, embeddings) tuple, "
            "or a dict with a 'segments'/'sentences' key."
        )

    return [str(s) for s in segments], embeddings


def _cluster_centroids_from_points(E: np.ndarray, labels: np.ndarray, cluster_order: Sequence[int]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for lab in cluster_order:
        mask = labels == int(lab)
        if np.any(mask):
            rows.append(E[mask].mean(axis=0))
        else:
            rows.append(np.zeros(E.shape[1], dtype=float))
    return np.vstack(rows) if rows else np.zeros((0, E.shape[1]), dtype=float)


def _fit_2d_projection(X: np.ndarray, method: str = "pca", random_state: int = 42) -> np.ndarray:
    """Fit a fresh 2D projection to the currently selected items."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Projection input must be a 2D matrix")
    if X.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    if X.shape[0] == 1:
        return np.zeros((1, 2), dtype=float)

    method = (method or "pca").lower()
    if method.startswith("umap"):
        try:
            try:
                import umap.umap_ as umap_mod  # type: ignore
            except Exception:
                import umap as umap_mod  # type: ignore
            reducer = umap_mod.UMAP(n_components=2, random_state=random_state)
            return np.asarray(reducer.fit_transform(X), dtype=float)
        except Exception as ex:
            print(f"[cluster2d] UMAP unavailable/failed ({ex}); falling back to PCA.")

    from sklearn.decomposition import PCA

    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2), dtype=float)
    pca = PCA(n_components=2, random_state=random_state)
    return np.asarray(pca.fit_transform(X), dtype=float)


@dataclass
class PreparedDoc:
    doc_id: Any
    display_id: str
    segments: List[str]
    labels: np.ndarray
    delta_matrix: np.ndarray
    cluster_order: List[int]
    cluster_embeddings: np.ndarray
    point_embeddings: np.ndarray
    cluster_dirs: Optional[np.ndarray] = None
    point_xy: Optional[np.ndarray] = None
    centroid_xy: Optional[np.ndarray] = None
    pc1_endpoint_xy: Optional[np.ndarray] = None

    @property
    def label_to_order_index(self) -> Dict[int, int]:
        return {int(lab): i for i, lab in enumerate(self.cluster_order)}


# -----------------------------------------------------------------------------
# Viewer
# -----------------------------------------------------------------------------

class SentenceClusterProjectionViewer:
    def __init__(
        self,
        document_delta_dict: Dict[Any, tuple],
        segments_by_doc: Dict[Any, Any],
        *,
        sentence_model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 512,
        remove_top_n_components: int = 2,
        apply_component_removal_to_loaded_embeddings: bool = False,
        projection_method: str = "PCA",
        parent: Any = None,
        initial_item_a: Any = None,
        initial_item_b: Any = None,
    ):
        self.document_delta_dict = document_delta_dict
        self.segments_by_doc = segments_by_doc
        self.sentence_model_name = sentence_model_name
        self.batch_size = int(batch_size)
        self.remove_top_n_components = int(remove_top_n_components)
        self.apply_component_removal_to_loaded_embeddings = bool(apply_component_removal_to_loaded_embeddings)
        self.projection_method = projection_method
        self.parent = parent

        self._st_model = None
        self._prepared_cache: Dict[Any, PreparedDoc] = {}
        self._current_docs: Dict[Any, PreparedDoc] = {}
        self._morphism_entries: List[Tuple[Any, int, int]] = []
        self._scatter_meta: Dict[Any, Dict[str, Any]] = {}
        self._selected_point_artist = None

        self._init_tk(initial_item_a=initial_item_a, initial_item_b=initial_item_b)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _init_tk(self, initial_item_a: Any = None, initial_item_b: Any = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk

        if self.parent is None:
            self.root = tk.Tk()
            self.win = self.root
            self._owns_root = True
        else:
            self.root = self.parent
            self.win = tk.Toplevel(self.parent)
            self._owns_root = False

        self.win.title("Sentence Cluster Projection Endpoint")
        self.win.geometry("1420x900")
        self.win.rowconfigure(1, weight=1)
        self.win.columnconfigure(0, weight=1)

        doc_displays = sorted((_as_doc_key_display(k) for k in self.document_delta_dict.keys()), key=str)
        self.display_to_key = {_as_doc_key_display(k): k for k in self.document_delta_dict.keys()}

        top = ttk.Frame(self.win, padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        for c in (1, 3):
            top.columnconfigure(c, weight=1)

        ttk.Label(top, text="Item A:").grid(row=0, column=0, sticky="w")
        self.item_a_var = tk.StringVar()
        self.item_a_cb = ttk.Combobox(top, textvariable=self.item_a_var, values=doc_displays, state="readonly", width=42)
        self.item_a_cb.grid(row=0, column=1, sticky="ew", padx=(6, 14))

        ttk.Label(top, text="Item B:").grid(row=0, column=2, sticky="w")
        self.item_b_var = tk.StringVar()
        self.item_b_cb = ttk.Combobox(top, textvariable=self.item_b_var, values=doc_displays, state="readonly", width=42)
        self.item_b_cb.grid(row=0, column=3, sticky="ew", padx=(6, 14))

        ttk.Label(top, text="Projection:").grid(row=0, column=4, sticky="w")
        self.proj_var = tk.StringVar(value=self.projection_method)
        self.proj_cb = ttk.Combobox(top, textvariable=self.proj_var, values=["PCA", "UMAP"], state="readonly", width=8)
        self.proj_cb.grid(row=0, column=5, sticky="w", padx=(6, 10))

        self.render_btn = ttk.Button(top, text="Render / update", command=self.render)
        self.render_btn.grid(row=0, column=6, sticky="w", padx=(0, 8))

        self.clear_btn = ttk.Button(top, text="Clear morphism highlight", command=self.clear_morphism_selection)
        self.clear_btn.grid(row=0, column=7, sticky="w")

        self.fade_unselected_var = tk.BooleanVar(value=False)
        self.fade_unselected_btn = ttk.Checkbutton(
            top,
            text="Fade unselected clusters",
            variable=self.fade_unselected_var,
            command=self.redraw,
        )
        self.fade_unselected_btn.grid(row=1, column=6, columnspan=2, sticky="w", pady=(6, 0))

        # Main body: plot left, controls right
        body = ttk.Panedwindow(self.win, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        plot_frame = ttk.Frame(body)
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        body.add(plot_frame, weight=4)

        side = ttk.Frame(body, padding=(8, 0))
        side.rowconfigure(1, weight=1)
        side.rowconfigure(4, weight=1)
        side.columnconfigure(0, weight=1)
        body.add(side, weight=1)

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(11, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        self.toolbar.grid(row=1, column=0, sticky="ew")
        self.canvas.mpl_connect("pick_event", self._on_pick_sentence)

        ttk.Label(side, text="Morphisms to highlight (multi-select):").grid(row=0, column=0, sticky="w")
        list_frame = ttk.Frame(side)
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.morphism_list = tk.Listbox(list_frame, selectmode="extended", exportselection=False, height=18)
        self.morphism_list.grid(row=0, column=0, sticky="nsew")
        morph_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.morphism_list.yview)
        morph_scroll.grid(row=0, column=1, sticky="ns")
        self.morphism_list.configure(yscrollcommand=morph_scroll.set)
        self.morphism_list.bind("<<ListboxSelect>>", lambda _evt: self.redraw())

        help_text = (
            "Workflow:\n"
            "1. Choose two items and render.\n"
            "2. Select one or more Csrc→Cdst morphisms.\n"
            "3. Yellow arrows highlight projected cluster-to-cluster morphisms.\n"
            "4. Source/destination sentence points are outlined.\n"
            "5. Toggle fading to keep only selected clusters fully visible.\n"
            "6. PC1 arrows show each cluster principal direction.\n"
            "7. Click any sentence point to inspect its text."
        )
        ttk.Label(side, text=help_text, justify="left", wraplength=330).grid(row=2, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(side, text="Selected sentence / status:").grid(row=3, column=0, sticky="w")
        text_frame = ttk.Frame(side)
        text_frame.grid(row=4, column=0, sticky="nsew", pady=(4, 0))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        self.info_text = tk.Text(text_frame, wrap="word", height=15)
        self.info_text.grid(row=0, column=0, sticky="nsew")
        info_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.info_text.yview)
        info_scroll.grid(row=0, column=1, sticky="ns")
        self.info_text.configure(yscrollcommand=info_scroll.set)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.win, textvariable=self.status_var, anchor="w", padding=(10, 4)).grid(row=2, column=0, sticky="ew")

        # Defaults
        if doc_displays:
            a_display = _as_doc_key_display(initial_item_a) if initial_item_a is not None else doc_displays[0]
            b_display = _as_doc_key_display(initial_item_b) if initial_item_b is not None else (doc_displays[1] if len(doc_displays) > 1 else doc_displays[0])
            if a_display not in self.display_to_key:
                a_display = doc_displays[0]
            if b_display not in self.display_to_key:
                b_display = doc_displays[1] if len(doc_displays) > 1 else doc_displays[0]
            self.item_a_var.set(a_display)
            self.item_b_var.set(b_display)

        self._write_info("Ready. Choose two items, then click Render / update.")
        self.render()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)
        try:
            self.win.update_idletasks()
        except Exception:
            pass

    def _write_info(self, msg: str) -> None:
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", msg)
        self.info_text.configure(state="normal")

    def _get_model(self):
        if self._st_model is None:
            self._set_status(f"Loading sentence-transformer model: {self.sentence_model_name}")
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st_model = SentenceTransformer(self.sentence_model_name)
        return self._st_model

    def _encode_segments(self, doc_id: Any, segments: List[str]) -> np.ndarray:
        model = self._get_model()
        self._set_status(f"Embedding {len(segments)} segments for {doc_id} ...")
        E = model.encode(
            segments,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        E = _remove_top_components(E, n=self.remove_top_n_components)
        return E

    def _prepare_doc(self, doc_id: Any) -> PreparedDoc:
        if doc_id in self._prepared_cache:
            return self._prepared_cache[doc_id]

        if doc_id not in self.document_delta_dict:
            raise KeyError(f"{doc_id!r} not found in document_delta_dict")
        if doc_id not in self.segments_by_doc:
            raise KeyError(f"{doc_id!r} not found in segments_by_doc")

        Delta, cluster_order, labels, _topic_dists, cluster_embeddings, cluster_dirs = _unpack_cdm_tuple(
            self.document_delta_dict[doc_id]
        )
        segments, stored_embeddings = _extract_segments_bundle(self.segments_by_doc[doc_id])

        if len(segments) != len(labels):
            raise ValueError(
                f"Segment/label length mismatch for {doc_id}: "
                f"segments={len(segments)} labels={len(labels)}. "
                "The endpoint needs the segments pickle generated with the same document_delta_dict."
            )

        if stored_embeddings is not None and stored_embeddings.shape[0] == len(segments):
            E = _normalize_rows(stored_embeddings)
            if self.apply_component_removal_to_loaded_embeddings:
                E = _remove_top_components(E, n=self.remove_top_n_components)
        else:
            E = self._encode_segments(doc_id, segments)

        # Prefer stored CDM centroids if dimensionality matches the sentence points;
        # otherwise compute centroids from the sentence embeddings and labels.
        if cluster_embeddings.ndim == 2 and cluster_embeddings.shape[1] == E.shape[1] and cluster_embeddings.shape[0] == len(cluster_order):
            C = np.asarray(cluster_embeddings, dtype=float)
        else:
            C = _cluster_centroids_from_points(E, labels, cluster_order)

        # Principal-component directions are saved as cluster_dirs in the current
        # CDM tuple. Normalize them once and keep only shape-compatible arrays.
        Dirs = None
        if cluster_dirs is not None:
            D = np.asarray(cluster_dirs, dtype=float)
            if D.ndim == 2 and D.shape[0] == len(cluster_order) and D.shape[1] == C.shape[1]:
                Dirs = _normalize_rows(D)

        prepared = PreparedDoc(
            doc_id=doc_id,
            display_id=_as_doc_key_display(doc_id),
            segments=segments,
            labels=labels,
            delta_matrix=Delta,
            cluster_order=cluster_order,
            cluster_embeddings=C,
            point_embeddings=E,
            cluster_dirs=Dirs,
        )
        self._prepared_cache[doc_id] = prepared
        return prepared

    def _selected_doc_keys(self) -> List[Any]:
        keys: List[Any] = []
        for display in (self.item_a_var.get(), self.item_b_var.get()):
            key = self.display_to_key.get(display)
            if key is not None and key not in keys:
                keys.append(key)
        return keys

    def _project_docs(self, docs: List[PreparedDoc]) -> None:
        X_parts: List[np.ndarray] = []
        slices: List[Tuple[Any, str, int, int]] = []
        cursor = 0

        # Reset projected coordinates before refitting.
        for doc in docs:
            doc.point_xy = None
            doc.centroid_xy = None
            doc.pc1_endpoint_xy = None

        # Use a high-dimensional PC1 arrow length comparable to the document's
        # intra-cluster displacement scale. This keeps arrows visible without
        # letting synthetic PC1 endpoints dominate the projection fit.
        delta_norms: List[float] = []
        for doc in docs:
            D = np.asarray(doc.delta_matrix, dtype=float)
            if D.ndim == 3 and D.size:
                norms = np.linalg.norm(D.reshape(-1, D.shape[-1]), axis=1)
                delta_norms.extend(float(x) for x in norms if np.isfinite(x) and x > 1e-12)
        pc1_scale = float(np.median(delta_norms)) * 0.55 if delta_norms else 0.20
        if not np.isfinite(pc1_scale) or pc1_scale <= 0:
            pc1_scale = 0.20

        for doc in docs:
            start = cursor
            X_parts.append(doc.point_embeddings)
            cursor += doc.point_embeddings.shape[0]
            slices.append((doc.doc_id, "points", start, cursor))

            start = cursor
            X_parts.append(doc.cluster_embeddings)
            cursor += doc.cluster_embeddings.shape[0]
            slices.append((doc.doc_id, "centroids", start, cursor))

            if (
                doc.cluster_dirs is not None
                and doc.cluster_dirs.shape == doc.cluster_embeddings.shape
                and doc.cluster_embeddings.shape[0] > 0
            ):
                pc1_endpoints = doc.cluster_embeddings + (doc.cluster_dirs * pc1_scale)
                start = cursor
                X_parts.append(pc1_endpoints)
                cursor += pc1_endpoints.shape[0]
                slices.append((doc.doc_id, "pc1_endpoints", start, cursor))

        X = np.vstack(X_parts) if X_parts else np.zeros((0, 2), dtype=float)
        xy = _fit_2d_projection(X, method=self.proj_var.get(), random_state=42)

        by_id = {doc.doc_id: doc for doc in docs}
        for doc_id, kind, start, stop in slices:
            if kind == "points":
                by_id[doc_id].point_xy = xy[start:stop]
            elif kind == "centroids":
                by_id[doc_id].centroid_xy = xy[start:stop]
            elif kind == "pc1_endpoints":
                by_id[doc_id].pc1_endpoint_xy = xy[start:stop]

    # ------------------------------------------------------------------
    # Render/redraw
    # ------------------------------------------------------------------
    def render(self) -> None:
        try:
            doc_keys = self._selected_doc_keys()
            if not doc_keys:
                return
            docs = [self._prepare_doc(k) for k in doc_keys]
            self._set_status(f"Projecting {', '.join(str(d.doc_id) for d in docs)} with {self.proj_var.get()} ...")
            self._project_docs(docs)
            self._current_docs = {d.doc_id: d for d in docs}
            self._populate_morphisms(docs)
            self.redraw()
            total_points = sum(len(d.segments) for d in docs)
            total_clusters = sum(len(d.cluster_order) for d in docs)
            self._set_status(f"Rendered {total_points} sentence points across {total_clusters} clusters.")
        except Exception as ex:
            self._set_status(f"Error: {ex}")
            self._write_info(traceback.format_exc())
            try:
                from tkinter import messagebox
                messagebox.showerror("Cluster 2D endpoint", str(ex), parent=self.win)
            except Exception:
                pass

    def _populate_morphisms(self, docs: List[PreparedDoc]) -> None:
        # Preserve selected labels where possible.
        previous_selected = {self.morphism_list.get(i) for i in self.morphism_list.curselection()}
        self.morphism_list.delete(0, "end")
        self._morphism_entries = []

        for doc in docs:
            order = list(doc.cluster_order)
            for i, src_lab in enumerate(order):
                for j, dst_lab in enumerate(order):
                    if i == j:
                        continue
                    try:
                        delta_len = float(np.linalg.norm(doc.delta_matrix[i, j]))
                    except Exception:
                        delta_len = math.nan
                    label = f"{doc.display_id} | C{src_lab} → C{dst_lab} | |Δ|={delta_len:.3f}"
                    self._morphism_entries.append((doc.doc_id, int(src_lab), int(dst_lab)))
                    self.morphism_list.insert("end", label)
                    if label in previous_selected:
                        self.morphism_list.selection_set("end")

    def clear_morphism_selection(self) -> None:
        self.morphism_list.selection_clear(0, "end")
        self.redraw()

    def _selected_morphisms(self) -> List[Tuple[Any, int, int]]:
        out: List[Tuple[Any, int, int]] = []
        for idx in self.morphism_list.curselection():
            if 0 <= int(idx) < len(self._morphism_entries):
                out.append(self._morphism_entries[int(idx)])
        return out

    def redraw(self) -> None:
        import matplotlib.cm as cm
        from matplotlib.lines import Line2D

        self.ax.clear()
        self._scatter_meta = {}
        self._selected_point_artist = None

        docs = list(self._current_docs.values())
        if not docs:
            self.canvas.draw_idle()
            return

        cluster_keys: List[Tuple[Any, int]] = []
        for doc in docs:
            for lab in doc.cluster_order:
                cluster_keys.append((doc.doc_id, int(lab)))
        cmap = cm.get_cmap("tab20", max(1, len(cluster_keys)))
        cluster_color = {key: cmap(i % 20) for i, key in enumerate(cluster_keys)}
        doc_marker = {doc.doc_id: ("o" if i == 0 else "^" if i == 1 else "s") for i, doc in enumerate(docs)}

        selected_morphisms = self._selected_morphisms()
        highlighted_clusters = set()
        for doc_id, src_lab, dst_lab in selected_morphisms:
            highlighted_clusters.add((doc_id, int(src_lab)))
            highlighted_clusters.add((doc_id, int(dst_lab)))

        fade_selected_only = bool(
            getattr(self, "fade_unselected_var", None) is not None
            and self.fade_unselected_var.get()
            and highlighted_clusters
        )

        def _component_alpha(doc_id: Any, lab: int, default_alpha: float = 0.62) -> float:
            if not fade_selected_only:
                return default_alpha
            return 1.0 if (doc_id, int(lab)) in highlighted_clusters else 0.10

        # Base sentence points, centroids, labels, and per-cluster PC1 arrows.
        for doc in docs:
            if doc.point_xy is None or doc.centroid_xy is None:
                continue
            for lab in doc.cluster_order:
                lab = int(lab)
                mask = doc.labels == lab
                idxs = np.where(mask)[0]
                if idxs.size == 0:
                    continue
                xy = doc.point_xy[idxs]
                color = cluster_color[(doc.doc_id, lab)]
                marker = doc_marker.get(doc.doc_id, "o")
                alpha = _component_alpha(doc.doc_id, lab, default_alpha=0.62)
                zbase = 6 if alpha >= 1.0 else 3
                sc = self.ax.scatter(
                    xy[:, 0], xy[:, 1],
                    s=28 if alpha >= 1.0 else 26,
                    alpha=alpha,
                    marker=marker,
                    color=color,
                    linewidths=0.2,
                    edgecolors="none",
                    picker=True,
                    label=f"{doc.display_id} C{lab} (n={idxs.size})",
                    zorder=zbase,
                )
                self._scatter_meta[sc] = {"doc_id": doc.doc_id, "cluster": lab, "segment_indices": idxs}

                ci = doc.label_to_order_index.get(lab)
                if ci is not None and ci < doc.centroid_xy.shape[0]:
                    cx, cy = doc.centroid_xy[ci]
                    self.ax.scatter(
                        [cx], [cy],
                        s=140 if alpha >= 1.0 else 130,
                        marker="X",
                        color=color,
                        alpha=alpha,
                        edgecolors="black",
                        linewidths=1.1 if alpha >= 1.0 else 1.0,
                        zorder=zbase + 2,
                    )
                    txt = self.ax.text(cx, cy, f" {doc.display_id}\n C{lab}", fontsize=8, zorder=zbase + 3)
                    txt.set_alpha(alpha)

                    if (
                        doc.pc1_endpoint_xy is not None
                        and ci < doc.pc1_endpoint_xy.shape[0]
                    ):
                        ex, ey = doc.pc1_endpoint_xy[ci]
                        pc_alpha = 1.0 if alpha >= 1.0 else (0.78 if not fade_selected_only else 0.10)
                        pc_lw = 2.1 if alpha >= 1.0 else 1.45
                        self.ax.annotate(
                            "",
                            xy=(ex, ey),
                            xytext=(cx, cy),
                            arrowprops=dict(
                                arrowstyle="-|>",
                                color=color,
                                linewidth=pc_lw,
                                shrinkA=7,
                                shrinkB=0,
                                alpha=pc_alpha,
                                mutation_scale=12 if alpha < 1.0 else 14,
                            ),
                            zorder=zbase + 1,
                        )

        # Highlight selected source/destination clusters on top.
        for doc_id, lab in highlighted_clusters:
            doc = self._current_docs.get(doc_id)
            if doc is None or doc.point_xy is None:
                continue
            idxs = np.where(doc.labels == int(lab))[0]
            if idxs.size == 0:
                continue
            xy = doc.point_xy[idxs]
            self.ax.scatter(
                xy[:, 0], xy[:, 1],
                s=98,
                facecolors="none",
                edgecolors="black",
                linewidths=1.45,
                marker=doc_marker.get(doc_id, "o"),
                alpha=1.0,
                zorder=9,
            )

        # Morphism arrows.
        arrow_handles: List[Line2D] = []
        for n, (doc_id, src_lab, dst_lab) in enumerate(selected_morphisms, start=1):
            doc = self._current_docs.get(doc_id)
            if doc is None or doc.centroid_xy is None:
                continue
            idx_map = doc.label_to_order_index
            if src_lab not in idx_map or dst_lab not in idx_map:
                continue
            sidx, didx = idx_map[src_lab], idx_map[dst_lab]
            if sidx >= doc.centroid_xy.shape[0] or didx >= doc.centroid_xy.shape[0]:
                continue
            src_xy = doc.centroid_xy[sidx]
            dst_xy = doc.centroid_xy[didx]
            arrow_color = "yellow"
            self.ax.annotate(
                "",
                xy=(dst_xy[0], dst_xy[1]),
                xytext=(src_xy[0], src_xy[1]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=arrow_color,
                    linewidth=3.0,
                    shrinkA=8,
                    shrinkB=8,
                    alpha=0.98,
                ),
                zorder=12,
            )
            mid = (src_xy + dst_xy) / 2.0
            self.ax.text(
                mid[0], mid[1],
                f"M{n}: {doc.display_id} C{src_lab}→C{dst_lab}",
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="yellow", alpha=0.88, pad=2),
                zorder=13,
            )
            arrow_handles.append(Line2D([0], [0], color=arrow_color, lw=3.0, label=f"M{n}: {doc.display_id} C{src_lab}→C{dst_lab}"))

        self.ax.set_title(
            "Sentence embeddings by item cluster\n"
            "points = sentence segments; X = cluster centroid; colored arrows = cluster PC1; yellow arrows = selected intra-item morphisms"
        )
        self.ax.set_xlabel(f"{self.proj_var.get()} dimension 1")
        self.ax.set_ylabel(f"{self.proj_var.get()} dimension 2")
        self.ax.grid(True, alpha=0.25)

        # Compact legend: many clusters can make this large, so keep it outside.
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            # De-duplicate labels while preserving order.
            seen = set()
            h2, l2 = [], []
            for h, lab in zip(handles, labels):
                if lab not in seen:
                    seen.add(lab)
                    h2.append(h)
                    l2.append(lab)
            self.ax.legend(h2 + arrow_handles, l2 + [h.get_label() for h in arrow_handles], loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def _on_pick_sentence(self, event: Any) -> None:
        artist = getattr(event, "artist", None)
        meta = self._scatter_meta.get(artist)
        if meta is None:
            return
        if not hasattr(event, "ind") or len(event.ind) == 0:
            return

        local_idx = int(event.ind[0])
        segment_indices = meta["segment_indices"]
        if local_idx < 0 or local_idx >= len(segment_indices):
            return
        seg_idx = int(segment_indices[local_idx])
        doc_id = meta["doc_id"]
        cluster = int(meta["cluster"])
        doc = self._current_docs.get(doc_id)
        if doc is None or doc.point_xy is None:
            return
        text = doc.segments[seg_idx]
        preview = textwrap.fill(text, width=92)

        self._write_info(
            f"Item: {doc.display_id}\n"
            f"Cluster: C{cluster}\n"
            f"Segment index: {seg_idx} of {len(doc.segments) - 1}\n\n"
            f"{preview}"
        )

        # Draw/replace a selected-point marker.
        try:
            if self._selected_point_artist is not None:
                self._selected_point_artist.remove()
        except Exception:
            pass
        x, y = doc.point_xy[seg_idx]
        self._selected_point_artist = self.ax.scatter(
            [x], [y],
            s=170,
            marker="*",
            color="yellow",
            edgecolors="black",
            linewidths=1.0,
            zorder=20,
        )
        self.canvas.draw_idle()

    def show(self, block: bool = True) -> None:
        try:
            self.win.lift()
        except Exception:
            pass
        if block:
            if self._owns_root:
                self.root.mainloop()
            else:
                self.win.wait_window()


# -----------------------------------------------------------------------------
# Public endpoint
# -----------------------------------------------------------------------------

def visualize_sentence_cluster_projection_endpoint(
    document_delta_dict: Optional[Dict[Any, tuple]] = None,
    segments_by_doc: Optional[Dict[Any, Any]] = None,
    *,
    document_delta_pkl: Optional[str] = None,
    segments_pkl: Optional[str] = None,
    sentence_model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 512,
    remove_top_n_components: int = 2,
    apply_component_removal_to_loaded_embeddings: bool = False,
    projection_method: str = "PCA",
    parent: Any = None,
    initial_item_a: Any = None,
    initial_item_b: Any = None,
    block: bool = True,
) -> SentenceClusterProjectionViewer:
    """
    Launch the sentence-level cluster projection endpoint.

    Provide in-memory dictionaries OR pickle paths. This is the function to call
    from generate_document_delta_manifold_PP.py once document_delta_dict and
    segments_by_doc have been loaded/built.
    """
    if document_delta_dict is None:
        if not document_delta_pkl:
            raise ValueError("Provide document_delta_dict or document_delta_pkl.")
        with open(document_delta_pkl, "rb") as fp:
            document_delta_dict = pickle.load(fp)
    if segments_by_doc is None:
        if not segments_pkl:
            raise ValueError("Provide segments_by_doc or segments_pkl.")
        with open(segments_pkl, "rb") as fp:
            segments_by_doc = pickle.load(fp)

    if not isinstance(document_delta_dict, dict) or not document_delta_dict:
        raise ValueError("document_delta_dict must be a non-empty dict.")
    if not isinstance(segments_by_doc, dict) or not segments_by_doc:
        raise ValueError("segments_by_doc must be a non-empty dict.")

    viewer = SentenceClusterProjectionViewer(
        document_delta_dict=document_delta_dict,
        segments_by_doc=segments_by_doc,
        sentence_model_name=sentence_model_name,
        batch_size=batch_size,
        remove_top_n_components=remove_top_n_components,
        apply_component_removal_to_loaded_embeddings=apply_component_removal_to_loaded_embeddings,
        projection_method=projection_method,
        parent=parent,
        initial_item_a=initial_item_a,
        initial_item_b=initial_item_b,
    )
    viewer.show(block=block)
    return viewer


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sentence-level 2D cluster projection endpoint")
    ap.add_argument("--document-delta-pkl", required=True, help="Pickle containing document_delta_dict.")
    ap.add_argument("--segments-pkl", required=True, help="Pickle containing segments_by_doc.")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name/path.")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--remove-top-components", type=int, default=2,
                    help="Apply the same common-component removal used by mk_delta_manifold when re-embedding text.")
    ap.add_argument("--projection", choices=["PCA", "UMAP"], default="PCA")
    ap.add_argument("--item-a", default=None)
    ap.add_argument("--item-b", default=None)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    visualize_sentence_cluster_projection_endpoint(
        document_delta_pkl=args.document_delta_pkl,
        segments_pkl=args.segments_pkl,
        sentence_model_name=args.model,
        batch_size=args.batch_size,
        remove_top_n_components=args.remove_top_components,
        projection_method=args.projection,
        initial_item_a=args.item_a,
        initial_item_b=args.item_b,
        block=True,
    )


if __name__ == "__main__":
    main()
