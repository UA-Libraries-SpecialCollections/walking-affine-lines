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
    analyze_morphism_match_field_parallel,
    save_morphism_comparison_pickle,
    enrich_morphism_comparison_diagnostics,
    recommend_morphism_compare_workers,
    output_analysis,
    output_acuity_candidates_csv,
    plot_morphism_match_field_3d,
    analyze_anchor_null_match_field,
    plot_anchor_null_match_field_3d,
    output_anchor_null_match_csvs,
    save_anchor_null_field_pickle,
    cluster_semantic_quality_score
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

_worker_ctx = {"LDA": None, "DICT": None, "NUM_TOPICS": None, "SBERT_M": None, "SBERT_MODEL_NAME": "all-MiniLM-L6-v2", "FLUENCY": {}}

def _init_worker(lda_path, dict_path, num_topics, fluency_params: dict | None = None, device: str | None = None):
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
    _worker_ctx["SBERT_MODEL_NAME"] = "all-MiniLM-L6-v2"
    _worker_ctx["SBERT_M"] = SentenceTransformer(_worker_ctx["SBERT_MODEL_NAME"], device=device)
    _worker_ctx["FLUENCY"] = dict(fluency_params or {})

def manifoldit(item_id, item_text):
    """Per-document build step (runs in workers)."""
    from topic_modeling import mk_delta_manifold, build_cluster_delta_matrix

    # Reuse the worker's cached model for this document.  Request raw SBERT
    # embeddings in addition to the document-centered manifold embeddings so the
    # CDM can store both global raw-document and local residual-document baselines.
    _dm = mk_delta_manifold(
        item_id,
        item_text,
        _worker_ctx["SBERT_M"],
        return_raw_embeddings=True,
    )
    if len(_dm) >= 7:
        delta_matrix, cluster_order, labels, segments, embeddings, k, raw_sbert_embeddings = _dm[:7]
    else:
        delta_matrix, cluster_order, labels, segments, embeddings, k = _dm
        raw_sbert_embeddings = None

    if k < 3 or len(set(labels)) < 3:
        # Not enough structure to form deltas; return a marker
        return (item_id, None, None)

    flu = _worker_ctx.get("FLUENCY", {}) or {}
    CDM = build_cluster_delta_matrix(
        segments, embeddings, labels,
        _worker_ctx["LDA"], _worker_ctx["DICT"], _worker_ctx["NUM_TOPICS"],
        n_clusters=k,
        semantic_fluency_enabled=bool(flu.get("enabled", True)),
        semantic_fluency_model_name=flu.get("model_name", "distilgpt2"),
        semantic_fluency_device=flu.get("device", "cpu"),
        semantic_fluency_batch_size=int(flu.get("batch_size", 8)),
        semantic_fluency_max_length=int(flu.get("max_length", 128)),
        semantic_fluency_min_tokens=int(flu.get("min_tokens", 3)),
        semantic_fluency_calibration=flu.get("calibration", "hybrid"),
        semantic_fluency_absolute_center=float(flu.get("absolute_center", 5.5)),
        semantic_fluency_absolute_scale=float(flu.get("absolute_scale", 1.15)),
        raw_sbert_embeddings=raw_sbert_embeddings,
        raw_sbert_model_name=_worker_ctx.get("SBERT_MODEL_NAME", "all-MiniLM-L6-v2"),
    )
    return (item_id, CDM, segments)

def main():
    print("Initializing WaAffL Iron")
    # ==== Cluster Text Reader Helpers ============================================
    # A tiny tools window with a "Cluster text Reader" button (disabled by default).
    # Call `set_cluster_text_reader_sources(document_delta_dict, segments_by_doc, parent=root)`
    # after your data are loaded/built to enable the button.

    # Globals that survive your prompt window lifecycle
    _CTR_TOOLS = {"win": None, "btn": None, "lex_btn": None, "readers": [], "lex_readers": []}
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

        def _launch_lexical_reader():
            _open_lexical_overlap_reader(parent=win)

        btn = ttk.Button(frm, text="Open new cluster text reader",
                         command=_launch_reader, state="disabled")
        btn.grid(row=0, column=0, padx=6, pady=(6, 3), sticky="ew")

        lex_btn = ttk.Button(frm, text="Open lexical pair overlap reader",
                             command=_launch_lexical_reader, state="disabled")
        lex_btn.grid(row=1, column=0, padx=6, pady=(3, 6), sticky="ew")

        _CTR_TOOLS["win"] = win
        _CTR_TOOLS["btn"] = btn
        _CTR_TOOLS["lex_btn"] = lex_btn
        return win

    def set_cluster_text_reader_sources(document_delta_dict, segments_by_doc, parent=None):
        """
        Provide the data the reader needs and enable the button.
        - document_delta_dict[item_id] = (delta_matrix, cluster_order, labels,
                                          cluster_topic_distributions, cluster_embeddings,
                                          cluster_dirs, optional cluster_semantic_quality)
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
            if _CTR_TOOLS.get("lex_btn") is not None:
                _CTR_TOOLS["lex_btn"].configure(state="normal")
        except Exception as ex:
            print(f"[CTR] Unable to enable Cluster text Reader: {ex}")

    # -------------------------------------------------------------------------
    # Paired lexical overlap reader helpers
    # -------------------------------------------------------------------------
    def _ctr_labels_for_item(iid):
        """Return labels list for an item, or [] if unavailable."""
        try:
            bundle = _CTR_SOURCES["doc_delta"].get(iid)
            labels = bundle[2]
            return labels.tolist() if hasattr(labels, "tolist") else list(labels)
        except Exception:
            return []

    def _ctr_segments_for_item(iid):
        try:
            return list((_CTR_SOURCES["segments"] or {}).get(iid, []))
        except Exception:
            return []

    def _ctr_cluster_texts(iid, cluster_label):
        """Return all segment strings assigned to one cluster label."""
        labels = _ctr_labels_for_item(iid)
        segments = _ctr_segments_for_item(iid)
        out = []
        try:
            target = int(cluster_label)
        except Exception:
            target = cluster_label
        for i, lab in enumerate(labels):
            if i >= len(segments):
                continue
            try:
                same = int(lab) == int(target)
            except Exception:
                same = lab == target
            if same:
                out.append(str(segments[i]))
        return out

    def _ctr_cluster_quality_record(iid, cluster_label):
        """Return the full stored cluster semantic quality record when available."""
        def _text_only_record():
            try:
                texts = _ctr_cluster_texts(iid, cluster_label)
                if texts:
                    rec = cluster_semantic_quality_score(
                        texts,
                        embeddings=None,
                        return_components=True,
                        spread_score_override=0.65,
                    )
                    if isinstance(rec, dict):
                        rec = dict(rec)
                        rec.setdefault("quality_source", "text_only_repair")
                        return rec
            except Exception:
                pass
            return {"quality": 1.0, "quality_source": "default_missing"}

        try:
            bundle = _CTR_SOURCES["doc_delta"].get(iid)
            if not isinstance(bundle, (tuple, list)) or len(bundle) < 7:
                return _text_only_record()
            q_payload = bundle[6]
            rec = None
            if isinstance(q_payload, dict):
                candidates = [cluster_label]
                try:
                    candidates.append(int(cluster_label))
                except Exception:
                    pass
                candidates.append(str(cluster_label))
                for key in candidates:
                    if key in q_payload:
                        rec = q_payload[key]
                        break
            else:
                try:
                    arr = list(q_payload)
                    order = list(bundle[1])
                    idx = order.index(cluster_label) if cluster_label in order else order.index(int(cluster_label))
                    rec = arr[idx]
                except Exception:
                    rec = None
            if isinstance(rec, dict):
                out = dict(rec)
            elif rec is not None:
                out = {"quality": float(rec)}
            else:
                return _text_only_record()
            try:
                q = float(out.get("quality", 1.0))
                if q < 1e-4:
                    repaired = _text_only_record()
                    repaired.setdefault("stored_near_zero_quality", q)
                    return repaired
            except Exception:
                pass
            out.setdefault("quality_source", "stored_payload")
            return out
        except Exception:
            return _text_only_record()

    def _ctr_quality_value(iid, cluster_label):
        rec = _ctr_cluster_quality_record(iid, cluster_label)
        try:
            return max(0.0, min(1.0, float(rec.get("quality", 1.0))))
        except Exception:
            return 1.0

    def _ctr_format_quality_record(rec, title=""):
        """Human-readable semantic-quality component summary."""
        if not isinstance(rec, dict):
            return f"{title}\n  Q=—\n"
        primary = [
            ("quality", "Q"),
            ("median_segment_quality", "median segment Q"),
            ("p25_segment_quality", "p25 segment Q"),
            ("medoid_segment_quality", "medoid segment Q"),
            ("semantic_core_quality", "semantic core Q"),
            ("semantic_center_support", "semantic center support"),
            ("usable_segment_ratio", "usable segment ratio"),
            ("fragment_burden", "fragment burden"),
            ("contamination_penalty", "contamination penalty"),
            ("non_template_score", "non-template score"),
            ("spread_score", "spread score"),
            ("spread_median_cosine_distance", "spread median cos distance"),
            ("lm_fluency_score_median", "LM fluency median"),
            ("lm_fluency_score_p25", "LM fluency p25"),
            ("lm_fluency_score_mean", "LM fluency mean"),
            ("lm_fluency_nll_median", "LM NLL median"),
            ("lm_fluency_nll_p75", "LM NLL p75"),
            ("n_segments", "n segments"),
        ]
        meta = [
            ("quality_model", "quality model"),
            ("quality_source", "quality source"),
            ("lm_fluency_model", "LM fluency model"),
            ("lm_fluency_available", "LM fluency available"),
            ("lm_fluency_calibration", "LM fluency calibration"),
        ]
        lines = []
        if title:
            lines.append(str(title))
        for key, lab in primary:
            val = rec.get(key, "")
            if val == "" or val is None:
                continue
            try:
                if isinstance(val, bool):
                    lines.append(f"  {lab}: {val}")
                elif key == "n_segments":
                    lines.append(f"  {lab}: {int(float(val))}")
                else:
                    lines.append(f"  {lab}: {float(val):.4f}")
            except Exception:
                lines.append(f"  {lab}: {val}")
        for key, lab in meta:
            val = rec.get(key, "")
            if val == "" or val is None:
                continue
            lines.append(f"  {lab}: {val}")
        return "\n".join(lines) + "\n"

    def _ctr_cluster_pairs_for_item(iid):
        labels = _ctr_labels_for_item(iid)
        clusters = sorted({int(c) for c in labels}) if labels else []
        display_to_tuple = {}
        displays = []
        for s_lab in clusters:
            for d_lab in clusters:
                if s_lab == d_lab:
                    continue
                q_s = _ctr_quality_value(iid, s_lab)
                q_d = _ctr_quality_value(iid, d_lab)
                edge_q = min(q_s, q_d)
                disp = f"C{s_lab} (Q={q_s:.3f}) → C{d_lab} (Q={q_d:.3f}) | edgeQ={edge_q:.3f}"
                display_to_tuple[disp] = (s_lab, d_lab)
                displays.append(disp)
        return displays, display_to_tuple

    def _open_lexical_overlap_reader(parent=None):
        """Open a paired edge reader for lexical overlap and Q-component inspection."""
        import math
        import tkinter as tk
        from tkinter import ttk, messagebox
        from collections import Counter

        doc_delta = _CTR_SOURCES["doc_delta"]
        segs = _CTR_SOURCES["segments"]
        if not isinstance(doc_delta, dict) or not isinstance(segs, dict) or not doc_delta:
            messagebox.showwarning(
                "Lexical pair overlap reader",
                "Data not loaded yet. Load/build your manifold and segments first.",
                parent=parent,
            )
            return

        stopwords = {
            "the", "and", "for", "that", "with", "from", "this", "these", "those", "were", "was", "are", "is", "be",
            "been", "being", "have", "has", "had", "not", "but", "or", "nor", "than", "then", "there", "their", "they",
            "them", "its", "into", "onto", "over", "under", "between", "within", "without", "after", "before", "during",
            "while", "where", "when", "which", "what", "who", "whom", "whose", "can", "could", "may", "might", "must",
            "shall", "should", "will", "would", "also", "such", "only", "more", "most", "other", "some", "any", "all",
            "each", "both", "one", "two", "three", "his", "her", "him", "she", "you", "your", "our", "ours", "out",
            "about", "upon", "per", "via", "fig", "table", "vol", "no", "pp", "page", "pages", "et", "al",
        }

        def _tokens(texts, min_len=3, remove_stop=True):
            if isinstance(texts, str):
                text = texts
            else:
                text = "\n".join(str(t) for t in (texts or []))
            raw = re.findall(r"[A-Za-z][A-Za-z'\-]*", text.lower())
            toks = []
            for tok in raw:
                tok = tok.strip("'-")
                if len(tok) < min_len:
                    continue
                if remove_stop and tok in stopwords:
                    continue
                toks.append(tok)
            return toks

        def _counter(texts):
            return Counter(_tokens(texts))

        def _metrics(ca, cb):
            sa, sb = set(ca), set(cb)
            inter = sa & sb
            union = sa | sb
            total_a = sum(ca.values())
            total_b = sum(cb.values())
            if not union:
                return {
                    "tokens_a": total_a, "tokens_b": total_b,
                    "unique_a": len(sa), "unique_b": len(sb),
                    "shared_unique": 0, "jaccard": 0.0, "dice": 0.0,
                    "overlap_coefficient": 0.0, "containment_a_in_b": 0.0,
                    "containment_b_in_a": 0.0, "weighted_jaccard": 0.0,
                    "count_cosine": 0.0, "shared_token_mass_a": 0.0, "shared_token_mass_b": 0.0,
                }
            min_sum = sum(min(ca.get(t, 0), cb.get(t, 0)) for t in union)
            max_sum = sum(max(ca.get(t, 0), cb.get(t, 0)) for t in union)
            dot = sum(ca.get(t, 0) * cb.get(t, 0) for t in union)
            norm_a = math.sqrt(sum(v * v for v in ca.values()))
            norm_b = math.sqrt(sum(v * v for v in cb.values()))
            mass_a = sum(ca[t] for t in inter)
            mass_b = sum(cb[t] for t in inter)
            return {
                "tokens_a": total_a,
                "tokens_b": total_b,
                "unique_a": len(sa),
                "unique_b": len(sb),
                "shared_unique": len(inter),
                "jaccard": len(inter) / len(union) if union else 0.0,
                "dice": (2 * len(inter) / (len(sa) + len(sb))) if (sa or sb) else 0.0,
                "overlap_coefficient": len(inter) / max(1, min(len(sa), len(sb))),
                "containment_a_in_b": len(inter) / max(1, len(sa)),
                "containment_b_in_a": len(inter) / max(1, len(sb)),
                "weighted_jaccard": min_sum / max(1, max_sum),
                "count_cosine": dot / max(1e-12, norm_a * norm_b),
                "shared_token_mass_a": mass_a / max(1, total_a),
                "shared_token_mass_b": mass_b / max(1, total_b),
            }

        def _fmt_metrics(label, ca, cb):
            m = _metrics(ca, cb)
            return (
                f"{label}\n"
                f"  tokens: A={m['tokens_a']}  B={m['tokens_b']}\n"
                f"  unique terms: A={m['unique_a']}  B={m['unique_b']}  shared={m['shared_unique']}\n"
                f"  Jaccard(unique): {m['jaccard']:.4f}\n"
                f"  Dice(unique): {m['dice']:.4f}\n"
                f"  Overlap coefficient: {m['overlap_coefficient']:.4f}\n"
                f"  Containment A⊂B: {m['containment_a_in_b']:.4f}   B⊂A: {m['containment_b_in_a']:.4f}\n"
                f"  Weighted Jaccard(counts): {m['weighted_jaccard']:.4f}\n"
                f"  Count cosine: {m['count_cosine']:.4f}\n"
                f"  Shared token mass: A={m['shared_token_mass_a']:.4f}  B={m['shared_token_mass_b']:.4f}\n"
            )

        def _norm_freq(c):
            total = sum(c.values())
            if total <= 0:
                return {}
            return {t: v / total for t, v in c.items()}

        def _delta_terms(src_counter, dst_counter):
            fs = _norm_freq(src_counter)
            fd = _norm_freq(dst_counter)
            keys = set(fs) | set(fd)
            return {t: fd.get(t, 0.0) - fs.get(t, 0.0) for t in keys}

        def _top_delta(delta, n=30, positive=True):
            vals = [(t, v) for t, v in delta.items() if (v > 0 if positive else v < 0)]
            vals.sort(key=lambda x: abs(x[1]), reverse=True)
            return vals[:n]

        def _fmt_terms(title, rows, max_rows=35):
            lines = [title]
            if not rows:
                lines.append("  —")
                return "\n".join(lines)
            for row in rows[:max_rows]:
                if len(row) == 3:
                    t, a, b = row
                    lines.append(f"  {t:<24} A={a:<5} B={b:<5}")
                elif len(row) == 2:
                    t, v = row
                    lines.append(f"  {t:<24} {v:+.5f}")
                else:
                    lines.append("  " + str(row))
            return "\n".join(lines)

        def _shared_terms(ca, cb, n=40):
            rows = [(t, ca[t], cb[t]) for t in (set(ca) & set(cb))]
            rows.sort(key=lambda r: (min(r[1], r[2]), r[1] + r[2], r[0]), reverse=True)
            return rows[:n]

        def _unique_terms(ca, cb, n=40):
            fa = _norm_freq(ca)
            fb = _norm_freq(cb)
            rows = []
            for t in set(fa) | set(fb):
                diff = fa.get(t, 0.0) - fb.get(t, 0.0)
                if diff != 0:
                    rows.append((t, diff))
            rows.sort(key=lambda x: abs(x[1]), reverse=True)
            return rows[:n]

        def _termset(rows):
            return {t for t, _ in rows}

        def _set_jaccard(a, b):
            if not a and not b:
                return 0.0
            return len(a & b) / max(1, len(a | b))

        def _set_overlap(a, b):
            if not a or not b:
                return 0.0
            return len(a & b) / max(1, min(len(a), len(b)))

        idx = len(_CTR_TOOLS.get("lex_readers", [])) + 1
        win = tk.Toplevel(parent)
        win.title(f"Lexical pair overlap reader #{idx}")
        win.geometry("1320x820")
        win.rowconfigure(1, weight=1)
        win.columnconfigure(0, weight=1)
        _CTR_TOOLS.setdefault("lex_readers", []).append(win)

        def _on_destroy(event=None, w=win):
            try:
                _CTR_TOOLS["lex_readers"].remove(w)
            except Exception:
                pass
        win.bind("<Destroy>", _on_destroy)

        top = ttk.Frame(win, padding=(8, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        item_ids = sorted(str(k) for k in doc_delta.keys())
        display_to_key = {str(k): k for k in doc_delta.keys()}
        side_state = {}
        pair_maps = {"A": {}, "B": {}}

        def _make_side(parent_frame, side, title):
            lf = ttk.LabelFrame(parent_frame, text=title)
            lf.grid(row=0, column=(0 if side == "A" else 1), sticky="ew", padx=(0, 6) if side == "A" else (6, 0))
            lf.columnconfigure(1, weight=1)
            item_var = tk.StringVar()
            pair_var = tk.StringVar()
            ttk.Label(lf, text="Item ID:").grid(row=0, column=0, sticky="w", padx=6, pady=3)
            item_cb = ttk.Combobox(lf, textvariable=item_var, values=item_ids, state="readonly", width=44)
            item_cb.grid(row=0, column=1, sticky="ew", padx=6, pady=3)
            ttk.Label(lf, text="Cluster pair:").grid(row=1, column=0, sticky="w", padx=6, pady=3)
            pair_cb = ttk.Combobox(lf, textvariable=pair_var, values=[], state="disabled", width=76)
            pair_cb.grid(row=1, column=1, sticky="ew", padx=6, pady=3)
            side_state[side] = {"item_var": item_var, "pair_var": pair_var, "item_cb": item_cb, "pair_cb": pair_cb}

        _make_side(top, "A", "Edge A")
        _make_side(top, "B", "Edge B")

        action_row = ttk.Frame(win, padding=(8, 0))
        action_row.grid(row=2, column=0, sticky="ew")
        action_row.columnconfigure(0, weight=1)
        status_var = tk.StringVar(value="Select two item IDs and cluster pairs to compare lexical overlap.")
        ttk.Label(action_row, textvariable=status_var).grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Button(action_row, text="Recalculate lexical overlap", command=lambda: _recalculate()).grid(row=0, column=1, sticky="e", padx=4, pady=4)

        notebook = ttk.Notebook(win)
        notebook.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 6))

        def _text_tab(name):
            frm = ttk.Frame(notebook)
            frm.rowconfigure(0, weight=1)
            frm.columnconfigure(0, weight=1)
            txt = tk.Text(frm, wrap="word")
            y = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
            txt.configure(yscrollcommand=y.set)
            txt.grid(row=0, column=0, sticky="nsew")
            y.grid(row=0, column=1, sticky="ns")
            notebook.add(frm, text=name)
            return txt

        metrics_txt = _text_tab("Overlap metrics")
        quality_txt = _text_tab("Semantic Q components")
        terms_txt = _text_tab("Shared / distinctive terms")
        texts_txt = _text_tab("Cluster texts")

        def _set_text(txt, content):
            txt.configure(state="normal")
            txt.delete("1.0", "end")
            txt.insert("1.0", content)
            txt.configure(state="normal")

        def _update_pairs(side, *_):
            item_display = side_state[side]["item_var"].get()
            iid = display_to_key.get(item_display)
            pair_cb = side_state[side]["pair_cb"]
            pair_var = side_state[side]["pair_var"]
            pair_maps[side].clear()
            if iid is None:
                pair_cb.configure(values=[], state="disabled")
                return
            displays, mapping = _ctr_cluster_pairs_for_item(iid)
            pair_maps[side].update(mapping)
            pair_cb.configure(values=displays, state=("readonly" if displays else "disabled"))
            if displays:
                pair_var.set(displays[0])
            _recalculate()

        def _selected(side):
            item_display = side_state[side]["item_var"].get()
            pair_display = side_state[side]["pair_var"].get()
            iid = display_to_key.get(item_display)
            pair = pair_maps[side].get(pair_display)
            if iid is None or pair is None:
                return None
            return iid, pair[0], pair[1]

        def _edge_bundle(side):
            sel = _selected(side)
            if sel is None:
                return None
            iid, src_lab, dst_lab = sel
            src_texts = _ctr_cluster_texts(iid, src_lab)
            dst_texts = _ctr_cluster_texts(iid, dst_lab)
            src_q = _ctr_cluster_quality_record(iid, src_lab)
            dst_q = _ctr_cluster_quality_record(iid, dst_lab)
            return {
                "side": side,
                "iid": iid,
                "src": src_lab,
                "dst": dst_lab,
                "src_texts": src_texts,
                "dst_texts": dst_texts,
                "edge_texts": src_texts + dst_texts,
                "src_q": src_q,
                "dst_q": dst_q,
                "src_counter": _counter(src_texts),
                "dst_counter": _counter(dst_texts),
                "edge_counter": _counter(src_texts + dst_texts),
            }

        def _recalculate():
            A = _edge_bundle("A")
            B = _edge_bundle("B")
            if A is None or B is None:
                status_var.set("Select two item IDs and cluster pairs to compare lexical overlap.")
                return
            lines = []
            lines.append("Lexical-overlap metrics use lowercased alphabetic content tokens, length ≥ 3, with a compact stopword list removed.")
            lines.append("These metrics are lexical diagnostics only; they do not replace Δ/PC1/Q morphism alignment scores.\n")
            lines.append(f"Edge A: {A['iid']}: C{A['src']} → C{A['dst']}  (src n={len(A['src_texts'])}, dst n={len(A['dst_texts'])})")
            lines.append(f"Edge B: {B['iid']}: C{B['src']} → C{B['dst']}  (src n={len(B['src_texts'])}, dst n={len(B['dst_texts'])})\n")
            lines.append(_fmt_metrics("A source cluster vs B source cluster", A["src_counter"], B["src_counter"]))
            lines.append(_fmt_metrics("A destination cluster vs B destination cluster", A["dst_counter"], B["dst_counter"]))
            lines.append(_fmt_metrics("A combined edge text vs B combined edge text", A["edge_counter"], B["edge_counter"]))
            lines.append(_fmt_metrics("Within Edge A: source cluster vs destination cluster", A["src_counter"], A["dst_counter"]))
            lines.append(_fmt_metrics("Within Edge B: source cluster vs destination cluster", B["src_counter"], B["dst_counter"]))

            delta_A = _delta_terms(A["src_counter"], A["dst_counter"])
            delta_B = _delta_terms(B["src_counter"], B["dst_counter"])
            gain_A = _top_delta(delta_A, n=30, positive=True)
            gain_B = _top_delta(delta_B, n=30, positive=True)
            loss_A = _top_delta(delta_A, n=30, positive=False)
            loss_B = _top_delta(delta_B, n=30, positive=False)
            gain_set_A = _termset(gain_A); gain_set_B = _termset(gain_B)
            loss_set_A = _termset(loss_A); loss_set_B = _termset(loss_B)
            lines.append("Directional lexical-shift overlap, using top 30 normalized-frequency gains/losses from source → destination:")
            lines.append(f"  gained-term Jaccard: {_set_jaccard(gain_set_A, gain_set_B):.4f}")
            lines.append(f"  gained-term overlap coefficient: {_set_overlap(gain_set_A, gain_set_B):.4f}")
            lines.append(f"  lost-term Jaccard: {_set_jaccard(loss_set_A, loss_set_B):.4f}")
            lines.append(f"  lost-term overlap coefficient: {_set_overlap(loss_set_A, loss_set_B):.4f}")
            lines.append(f"  A gained vs B lost Jaccard: {_set_jaccard(gain_set_A, loss_set_B):.4f}")
            lines.append(f"  A lost vs B gained Jaccard: {_set_jaccard(loss_set_A, gain_set_B):.4f}")
            _set_text(metrics_txt, "\n".join(lines))

            q_lines = []
            q_lines.append(_ctr_format_quality_record(A["src_q"], f"Edge A source — {A['iid']} C{A['src']}"))
            q_lines.append(_ctr_format_quality_record(A["dst_q"], f"Edge A destination — {A['iid']} C{A['dst']}"))
            q_lines.append(_ctr_format_quality_record(B["src_q"], f"Edge B source — {B['iid']} C{B['src']}"))
            q_lines.append(_ctr_format_quality_record(B["dst_q"], f"Edge B destination — {B['iid']} C{B['dst']}"))
            _set_text(quality_txt, "\n".join(q_lines))

            t_lines = []
            t_lines.append(_fmt_terms("Shared terms: A source vs B source", _shared_terms(A["src_counter"], B["src_counter"])))
            t_lines.append("")
            t_lines.append(_fmt_terms("Shared terms: A destination vs B destination", _shared_terms(A["dst_counter"], B["dst_counter"])))
            t_lines.append("")
            t_lines.append(_fmt_terms("Shared terms: A combined edge vs B combined edge", _shared_terms(A["edge_counter"], B["edge_counter"])))
            t_lines.append("")
            t_lines.append(_fmt_terms("Most distinctive normalized-frequency terms for A combined edge relative to B", _unique_terms(A["edge_counter"], B["edge_counter"])))
            t_lines.append("")
            t_lines.append(_fmt_terms("Top Edge A source→destination gained terms", gain_A))
            t_lines.append("")
            t_lines.append(_fmt_terms("Top Edge B source→destination gained terms", gain_B))
            t_lines.append("")
            t_lines.append(_fmt_terms("Top Edge A source→destination lost terms", loss_A))
            t_lines.append("")
            t_lines.append(_fmt_terms("Top Edge B source→destination lost terms", loss_B))
            _set_text(terms_txt, "\n".join(t_lines))

            text_lines = []
            for edge in (A, B):
                text_lines.append(f"================ {edge['side']}  {edge['iid']}: C{edge['src']} → C{edge['dst']} ================")
                text_lines.append(f"\n--- {edge['side']} SOURCE C{edge['src']} | n={len(edge['src_texts'])} ---\n")
                text_lines.append("\n\n".join(edge["src_texts"]) if edge["src_texts"] else "(no segments)")
                text_lines.append(f"\n\n--- {edge['side']} DESTINATION C{edge['dst']} | n={len(edge['dst_texts'])} ---\n")
                text_lines.append("\n\n".join(edge["dst_texts"]) if edge["dst_texts"] else "(no segments)")
                text_lines.append("\n")
            _set_text(texts_txt, "\n".join(text_lines))
            status_var.set(
                f"Compared {A['iid']} C{A['src']}→C{A['dst']} with {B['iid']} C{B['src']}→C{B['dst']}"
            )

        for side in ("A", "B"):
            side_state[side]["item_cb"].bind("<<ComboboxSelected>>", lambda evt, s=side: _update_pairs(s))
            side_state[side]["pair_cb"].bind("<<ComboboxSelected>>", lambda evt: _recalculate())

        if item_ids:
            side_state["A"]["item_var"].set(item_ids[0])
            side_state["B"]["item_var"].set(item_ids[1] if len(item_ids) > 1 else item_ids[0])
            _update_pairs("A")
            _update_pairs("B")
            _recalculate()

        try:
            win.lift()
        except Exception:
            pass

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
        pair_cb = ttk.Combobox(top, textvariable=pair_var, state="disabled", width=72)
        pair_cb.grid(row=0, column=3, sticky="w", padx=(6, 0))

        # --- Reading area: two panes --------------------------------------------
        paned = ttk.Panedwindow(win, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        def _mk_reader(parent_frame, title):
            frm = ttk.Frame(parent_frame)
            lbl = ttk.Label(frm, text=title)
            lbl.pack(anchor="w")
            txt = tk.Text(frm, wrap="word", height=28)
            yscroll = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
            txt.configure(yscrollcommand=yscroll.set)
            txt.pack(side="left", fill="both", expand=True)
            yscroll.pack(side="right", fill="y")
            return frm, lbl, txt

        left_frm, left_lbl, left_txt = _mk_reader(paned, "SRC cluster segments")
        right_frm, right_lbl, right_txt = _mk_reader(paned, "DST cluster segments")
        paned.add(left_frm, weight=1)
        paned.add(right_frm, weight=1)

        # --- Helpers -------------------------------------------------------------
        def _cluster_quality_for(iid, cluster_label):
            """Read quality from the optional 7th CDM tuple; repair near-zero legacy Q from text when possible."""
            def _text_only_quality():
                try:
                    bundle = doc_delta[iid]
                    labels = bundle[2]
                    if hasattr(labels, "tolist"):
                        labels = labels.tolist()
                    segments = segs.get(iid, [])
                    texts = [segments[i] for i, lab in enumerate(labels) if int(lab) == int(cluster_label) and i < len(segments)]
                    if texts:
                        return float(cluster_semantic_quality_score(texts, embeddings=None, spread_score_override=0.65))
                except Exception:
                    pass
                return 1.0

            try:
                bundle = doc_delta[iid]
                if not isinstance(bundle, (tuple, list)) or len(bundle) < 7:
                    return _text_only_quality()
                q_payload = bundle[6]
                if isinstance(q_payload, dict):
                    val = q_payload.get(cluster_label, q_payload.get(int(cluster_label), 1.0))
                    if isinstance(val, dict):
                        val = val.get("quality", 1.0)
                    val = max(0.0, min(1.0, float(val)))
                    return _text_only_quality() if val < 1e-4 else val
                arr = list(q_payload)
                cluster_order = list(bundle[1])
                idx = cluster_order.index(cluster_label) if cluster_label in cluster_order else cluster_order.index(int(cluster_label))
                val = max(0.0, min(1.0, float(arr[idx])))
                return _text_only_quality() if val < 1e-4 else val
            except Exception:
                return _text_only_quality()

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
                    q_s = _cluster_quality_for(iid, s)
                    q_d = _cluster_quality_for(iid, d)
                    edge_q = min(q_s, q_d)
                    # Include Q directly in the selector so quality is visible before opening panes.
                    disp = f"C{s} (Q={q_s:.3f}) \u2192 C{d} (Q={q_d:.3f}) | edgeQ={edge_q:.3f}"
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

            # Gather full segment text for each cluster side.
            src_texts = [segments[i] for i, lab in enumerate(labels) if i < len(segments) and int(lab) == int(s)]
            dst_texts = [segments[i] for i, lab in enumerate(labels) if i < len(segments) and int(lab) == int(d)]
            q_s = _cluster_quality_for(iid, s)
            q_d = _cluster_quality_for(iid, d)
            edge_q = min(q_s, q_d)

            # Render Q in pane labels as well as in the pair selector.
            try:
                left_lbl.configure(text=f"SRC C{s} | Q={q_s:.3f} | n={len(src_texts)}")
                right_lbl.configure(text=f"DST C{d} | Q={q_d:.3f} | n={len(dst_texts)} | edgeQ={edge_q:.3f}")
            except Exception:
                pass
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


    def _choose_pkl_save_path(default_name: str = "anchor_null_field.pkl", parent_window=None, title: str = "Save PKL") -> str | None:
        """
        Show a topmost Save-As dialog and return the chosen PKL path (or None if cancelled).
        """
        try:
            parent = parent_window if parent_window is not None else (globals().get("root") or tk._get_default_root())
        except Exception:
            parent = None
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
            title=title,
            defaultextension=".pkl",
            initialfile=default_name,
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
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
        default_quality_threshold=0.0,
        default_analyze_csv_mode="none",
        default_analyze_match_scope="auto",
        default_analyze_engine="parallel",
        default_analyze_parallel_workers="auto",
        default_analyze_source_chunk_size=384,
        default_analyze_target_block_size=8192,
        default_save_morphism_comparison_pkl=True,
        default_analyze_build_legacy_result=False,
        default_analyze_print_summaries=False,
        default_analyze_compact_diagnostics="plot_cache",
        default_analyze_compact_top_candidates=5000,
        default_analyze_plot_cache_step=0.01,
        default_analyze_top_k_per_delta=100,
        default_analyze_pc1_only_threshold=0.60,
        default_analyze_delta_max_for_pc1_only=0.60,
        default_analyze_pc1_only_quality_threshold=0.0,
        default_compute_acuity_for="aligned_only",
        default_acuity_csv_mode="none",
        default_acuity_top_n=500,
        default_visual_projection="global_pca",
        default_visual_surface_mode="quality_ellipsoid_mst",
        default_visual_mst_quality_lambda=2.0,
        default_visual_q_node_radius_scale=10.0,
        default_visual_q_edge_width_scale=20.0,
        default_semantic_fluency_enabled=True,
        default_semantic_fluency_model_name="distilgpt2",
        default_semantic_fluency_device="cpu",
        default_semantic_fluency_batch_size=8,
        default_semantic_fluency_max_length=128,
        default_semantic_fluency_min_tokens=3,
        default_semantic_fluency_calibration="hybrid",
        default_semantic_fluency_worker_cap=4,
        default_null_replicates=50,
        default_null_random_seed=0,
        default_null_step=0.01,
        default_null_strategy="association_shuffle",
        default_null_max_field_csv_rows=250000,
        default_endpoint="visualize",  # "visualize" | "analyze" | "null_compare" | "arrange"
        default_doc_id=None,           # optional analysis focus doc_id
        parent=None,
        verbose=False
    ):
        """
        Modal configuration dialog for manifold workflow.

        Returns:
            dict with keys:
              - mode: "build" | "load" | "compare"
              - num_topics, top_n_topics, top_m_keywords, cos_threshold, srcdst_threshold, quality_threshold, analyze_csv_mode, analyze_match_scope
              - analyze_engine, analyze_parallel_workers, analyze_source_chunk_size, analyze_target_block_size, save_morphism_comparison_pkl
              - analyze_top_k_per_delta, analyze_pc1_only_threshold, analyze_delta_max_for_pc1_only, analyze_pc1_only_quality_threshold, compute_acuity_for
              - visual_projection, visual_surface_mode
              - visual_mst_quality_lambda, visual_q_node_radius_scale, visual_q_edge_width_scale
              - semantic_fluency_* options for optional LM-derived segment fluency
              - endpoint: "visualize" | "analyze" | "null_compare" | "arrange"
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

        def _parse_float_nonnegative(name, s):
            try:
                v = float(str(s).strip())
                if v < 0.0:
                    raise ValueError
                return v, None
            except Exception:
                return None, f"“{name}” must be a non-negative number."

        def _parse_boolish(name, s):
            raw = str(s).strip().lower()
            if raw in ("1", "true", "yes", "y", "on", "enable", "enabled"):
                return True, None
            if raw in ("0", "false", "no", "n", "off", "disable", "disabled"):
                return False, None
            return None, f"“{name}” must be true/false, yes/no, or 1/0."

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
                "quality_threshold": default_quality_threshold,
                "analyze_csv_mode": default_analyze_csv_mode,
                "analyze_match_scope": default_analyze_match_scope,
                "analyze_engine": default_analyze_engine,
                "analyze_parallel_workers": default_analyze_parallel_workers,
                "analyze_source_chunk_size": int(default_analyze_source_chunk_size),
                "analyze_target_block_size": int(default_analyze_target_block_size),
                "save_morphism_comparison_pkl": bool(default_save_morphism_comparison_pkl),
                "analyze_build_legacy_result": bool(default_analyze_build_legacy_result),
                "analyze_print_summaries": bool(default_analyze_print_summaries),
                "analyze_compact_diagnostics": str(default_analyze_compact_diagnostics),
                "analyze_compact_top_candidates": int(default_analyze_compact_top_candidates),
                "analyze_plot_cache_step": float(default_analyze_plot_cache_step),
                "analyze_top_k_per_delta": int(default_analyze_top_k_per_delta),
                "analyze_pc1_only_threshold": float(default_analyze_pc1_only_threshold),
                "analyze_delta_max_for_pc1_only": float(default_analyze_delta_max_for_pc1_only),
                "analyze_pc1_only_quality_threshold": float(default_analyze_pc1_only_quality_threshold),
                "compute_acuity_for": default_compute_acuity_for,
                "acuity_csv_mode": default_acuity_csv_mode,
                "acuity_top_n": int(default_acuity_top_n),
                "visual_projection": default_visual_projection,
                "visual_surface_mode": default_visual_surface_mode,
                "visual_mst_quality_lambda": default_visual_mst_quality_lambda,
                "visual_q_node_radius_scale": default_visual_q_node_radius_scale,
                "visual_q_edge_width_scale": default_visual_q_edge_width_scale,
                "semantic_fluency_enabled": bool(default_semantic_fluency_enabled),
                "semantic_fluency_model_name": default_semantic_fluency_model_name,
                "semantic_fluency_device": default_semantic_fluency_device,
                "semantic_fluency_batch_size": default_semantic_fluency_batch_size,
                "semantic_fluency_max_length": default_semantic_fluency_max_length,
                "semantic_fluency_min_tokens": default_semantic_fluency_min_tokens,
                "semantic_fluency_calibration": default_semantic_fluency_calibration,
                "semantic_fluency_worker_cap": default_semantic_fluency_worker_cap,
                "null_replicates": int(default_null_replicates),
                "null_random_seed": int(default_null_random_seed),
                "null_step": float(default_null_step),
                "null_strategy": default_null_strategy,
                "null_max_field_csv_rows": int(default_null_max_field_csv_rows),
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
        win.resizable(True, True)
        try:
            win.transient(_parent)
        except Exception:
            pass

        outer = ttk.Frame(win, padding=(14, 12))
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        # Two-column layout keeps the bottom action buttons visible on smaller screens.
        # Left column: mode + primary numeric/text parameters.
        # Right column: endpoint selector + file pickers.
        left_col = ttk.Frame(outer)
        right_col = ttk.Frame(outer)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right_col.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        left_col.columnconfigure(0, weight=1)
        right_col.columnconfigure(0, weight=1)

        left_row = 0
        right_row = 0
        
        # --- Load ---
        load_frame = ttk.LabelFrame(left_col, text="Load")
        load_frame.grid(row=left_row, column=0, sticky="ew", pady=(0, 8))
        left_row += 1
        
        ttk.Button(load_frame, text="Load plot", command=_plot_pkl_picker).grid(row=0, column=0, pady=(0, 2))
        
        # --- Mode ---
        mode_frame = ttk.LabelFrame(left_col, text="Mode")
        mode_frame.grid(row=left_row, column=0, sticky="ew", pady=(0, 8))
        left_row += 1

        mode_var = tk.StringVar(value=default_mode)
        rb_build   = ttk.Radiobutton(mode_frame, text="Build new manifold", value="build",   variable=mode_var)
        rb_load    = ttk.Radiobutton(mode_frame, text="Load saved document manifolds", value="load",    variable=mode_var)
        rb_compare = ttk.Radiobutton(mode_frame, text="Compare with another manifold (.pkl)", value="compare", variable=mode_var)

        rb_build.grid(row=0, column=0, sticky="w", padx=8, pady=4)
        rb_load.grid(row=1, column=0, sticky="w", padx=8, pady=4)
        rb_compare.grid(row=2, column=0, sticky="w", padx=8, pady=4)

        # --- Parameters ---
        params = ttk.LabelFrame(left_col, text="Parameters")
        params.grid(row=left_row, column=0, sticky="ew", pady=(0, 8))
        left_row += 1
        params.columnconfigure(1, weight=1)

        labels = [
            ("num_topics (int ≥ 1)", str(default_num_topics)),
            ("top_n_topics (int ≥ 1)", str(default_top_n_topics)),
            ("top_m_keywords (int ≥ 1)", str(default_top_m_keywords)),
            ("cos_threshold (0–1)", f"{default_cos_threshold}"),
            ("srcdst_threshold (0–1)", f"{default_srcdst_threshold}"),
            ("semantic_quality_threshold (0–1)", f"{default_quality_threshold}"),
            ("analyze_csv_mode (full/selected/none)", f"{default_analyze_csv_mode}"),
            ("analyze_match_scope (auto/anchor/full)", f"{default_analyze_match_scope}"),
            ("analyze_engine (parallel/serial)", f"{default_analyze_engine}"),
            ("analyze_parallel_workers (auto/int ≥1)", f"{default_analyze_parallel_workers}"),
            ("analyze_source_chunk_size (int ≥1)", f"{default_analyze_source_chunk_size}"),
            ("analyze_target_block_size (int ≥1)", f"{default_analyze_target_block_size}"),
            ("save_morphism_comparison_pkl (true/false)", f"{str(bool(default_save_morphism_comparison_pkl)).lower()}"),
            ("analyze_build_legacy_result (true/false)", f"{str(bool(default_analyze_build_legacy_result)).lower()}"),
            ("analyze_print_summaries (true/false)", f"{str(bool(default_analyze_print_summaries)).lower()}"),
            ("analyze_compact_diagnostics (none/plot_cache/top_candidates/all_matches)", f"{default_analyze_compact_diagnostics}"),
            ("analyze_compact_top_candidates (int ≥0)", f"{default_analyze_compact_top_candidates}"),
            ("analyze_plot_cache_step (0.001–0.25)", f"{default_analyze_plot_cache_step}"),
            ("acuity_csv_mode (full/selected/none)", f"{default_acuity_csv_mode}"),
            ("acuity_top_n (int ≥1)", f"{default_acuity_top_n}"),
            ("visual_projection (global_pca/raw_first3)", f"{default_visual_projection}"),
            ("visual_surface_mode (quality_ellipsoid_mst/hull/quality_ellipsoid/skeleton/none)", f"{default_visual_surface_mode}"),
            ("visual_mst_quality_lambda (≥0)", f"{default_visual_mst_quality_lambda}"),
            ("visual_q_node_radius_scale (≥0)", f"{default_visual_q_node_radius_scale}"),
            ("visual_q_edge_width_scale (≥0)", f"{default_visual_q_edge_width_scale}"),
            ("semantic_fluency_enabled (true/false)", f"{str(bool(default_semantic_fluency_enabled)).lower()}"),
            ("semantic_fluency_model_name", f"{default_semantic_fluency_model_name}"),
            ("semantic_fluency_device (cpu/cuda/auto)", f"{default_semantic_fluency_device}"),
            ("semantic_fluency_batch_size (int ≥1)", f"{default_semantic_fluency_batch_size}"),
            ("semantic_fluency_max_length (int ≥8)", f"{default_semantic_fluency_max_length}"),
            ("semantic_fluency_min_tokens (int ≥1)", f"{default_semantic_fluency_min_tokens}"),
            ("semantic_fluency_calibration (hybrid/absolute/relative)", f"{default_semantic_fluency_calibration}"),
            ("semantic_fluency_worker_cap (int ≥1)", f"{default_semantic_fluency_worker_cap}"),
            ("null_replicates (int ≥1)", f"{default_null_replicates}"),
            ("null_random_seed (int)", f"{default_null_random_seed}"),
            ("null_step (0.001–0.25)", f"{default_null_step}"),
            ("null_strategy (association_shuffle)", f"{default_null_strategy}"),
            ("null_max_field_csv_rows (int ≥100)", f"{default_null_max_field_csv_rows}"),
        ]
        entries = {}
        for r, (lab, val) in enumerate(labels):
            ttk.Label(params, text=lab).grid(row=r, column=0, sticky="w", padx=8, pady=2)
            e = ttk.Entry(params, width=22)
            e.insert(0, val)
            e.grid(row=r, column=1, sticky="ew", padx=(6, 8), pady=2)
            entries[lab] = e

        # --- Endpoint + doc_id (optional) ---
        end_frame = ttk.LabelFrame(right_col, text="Endpoint")
        end_frame.grid(row=right_row, column=0, sticky="ew", pady=(0, 8))
        right_row += 1
        end_frame.columnconfigure(1, weight=1)

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

        ttk.Radiobutton(
            end_frame,
            text="Null compare (anchor observed vs edge-shuffled null field)",
            variable=endpoint_var,
            value="null_compare"
        ).grid(row=2, column=0, sticky="w", padx=8, pady=4)

        # NEW endpoint: arrange
        ttk.Radiobutton(
            end_frame,
            text="Arrange (build_categorical_arrangement_from_cdm_tuple)",
            variable=endpoint_var,
            value="arrange"
        ).grid(row=3, column=0, sticky="w", padx=8, pady=4)

        # doc_id applies to Analyze and Null compare; leave disabled for Visualize/Arrange
        doc_id_var = tk.StringVar(value="" if default_doc_id is None else str(default_doc_id))
        ttk.Label(end_frame, text="doc_id / anchor item id (for analyze or null compare)").grid(row=4, column=0, sticky="w", padx=8, pady=(0, 6))
        ent_doc_id = ttk.Entry(end_frame, textvariable=doc_id_var, width=40)
        ent_doc_id.grid(row=4, column=1, sticky="ew", padx=(6, 8), pady=(0, 6))

        # --- Analyze tuning ---------------------------------------------------
        # These controls primarily affect high-k runs, where k×(k-1) source
        # morphisms and permissive PC1-only settings can create very large
        # retained match sets.
        analyze_tune_frame = ttk.LabelFrame(right_col, text="Analyze tuning")
        analyze_tune_frame.grid(row=right_row, column=0, sticky="ew", pady=(0, 8))
        right_row += 1
        analyze_tune_frame.columnconfigure(1, weight=1)

        analyze_tune_labels = [
            ("analyze_top_k_per_delta (int ≥1)", f"{default_analyze_top_k_per_delta}"),
            ("analyze_pc1_only_threshold (0–1)", f"{default_analyze_pc1_only_threshold}"),
            ("analyze_delta_max_for_pc1_only (0–1)", f"{default_analyze_delta_max_for_pc1_only}"),
            ("analyze_pc1_only_quality_threshold (0–1)", f"{default_analyze_pc1_only_quality_threshold}"),
            ("compute_acuity_for (aligned_only/aligned_plus_pc1_only/pc1_only/none)", f"{default_compute_acuity_for}"),
        ]
        for r, (lab, val) in enumerate(analyze_tune_labels):
            ttk.Label(analyze_tune_frame, text=lab).grid(row=r, column=0, sticky="w", padx=8, pady=2)
            e = ttk.Entry(analyze_tune_frame, width=22)
            e.insert(0, val)
            e.grid(row=r, column=1, sticky="ew", padx=(6, 8), pady=2)
            entries[lab] = e

        # --- Files / pickers ---
        files_frame = ttk.LabelFrame(right_col, text="Files")
        files_frame.grid(row=right_row, column=0, sticky="ew", pady=(0, 8))
        right_row += 1
        files_frame.columnconfigure(1, weight=1)

        def add_picker(row_idx, label_text, var, browse_cmd):
            ttk.Label(files_frame, text=label_text).grid(row=row_idx, column=0, sticky="w", padx=8, pady=3)
            ent = ttk.Entry(files_frame, textvariable=var, width=48)
            ent.grid(row=row_idx, column=1, sticky="ew", padx=(6, 6), pady=3)
            btn = ttk.Button(files_frame, text="Browse…", command=browse_cmd)
            btn.grid(row=row_idx, column=2, sticky="w", padx=(0, 8), pady=3)
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
            enable = (endpoint_var.get() in ("analyze", "null_compare"))
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
        btns.grid(row=1, column=0, columnspan=2, sticky="e", pady=(8, 0))

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
            v_quality_threshold, err = _parse_float01("semantic_quality_threshold", entries["semantic_quality_threshold (0–1)"].get())
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_analyze_csv_mode = str(entries["analyze_csv_mode (full/selected/none)"].get()).strip().lower().replace("-", "_").replace(" ", "_") or "none"
            if v_analyze_csv_mode in ("all", "everything", "complete"):
                v_analyze_csv_mode = "full"
            elif v_analyze_csv_mode in ("selected_doc", "selected_document", "doc", "doc_id", "anchor", "involving", "focus"):
                v_analyze_csv_mode = "selected"
            elif v_analyze_csv_mode in ("no", "no_csv", "off", "false", "0", "skip"):
                v_analyze_csv_mode = "none"
            if v_analyze_csv_mode not in ("full", "selected", "none"):
                return messagebox.showerror(
                    "Invalid input",
                    "analyze_csv_mode must be full, selected, or none.",
                    parent=win
                )

            v_analyze_match_scope = str(entries["analyze_match_scope (auto/anchor/full)"].get()).strip().lower().replace("-", "_").replace(" ", "_") or "auto"
            if v_analyze_match_scope in ("selected", "selected_doc", "selected_document", "doc", "doc_id", "focus", "source", "source_only", "anchor_source"):
                v_analyze_match_scope = "anchor"
            elif v_analyze_match_scope in ("all", "global", "complete", "everything", "collection"):
                v_analyze_match_scope = "full"
            if v_analyze_match_scope not in ("auto", "anchor", "full"):
                return messagebox.showerror(
                    "Invalid input",
                    "analyze_match_scope must be auto, anchor, or full.",
                    parent=win
                )

            v_analyze_engine = str(entries["analyze_engine (parallel/serial)"].get()).strip().lower().replace("-", "_").replace(" ", "_") or "parallel"
            if v_analyze_engine in ("mp", "multiprocessing", "parallel_chunked", "chunked_parallel"):
                v_analyze_engine = "parallel"
            elif v_analyze_engine in ("single", "classic", "legacy", "reference"):
                v_analyze_engine = "serial"
            if v_analyze_engine not in ("parallel", "serial"):
                return messagebox.showerror(
                    "Invalid input",
                    "analyze_engine must be parallel or serial.",
                    parent=win
                )

            raw_workers = str(entries["analyze_parallel_workers (auto/int ≥1)"].get()).strip().lower() or "auto"
            if raw_workers in ("auto", "default", "recommended", ""):
                v_analyze_parallel_workers = "auto"
            else:
                v_analyze_parallel_workers, err = _parse_int("analyze_parallel_workers", raw_workers, 1, None)
                if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_analyze_source_chunk_size, err = _parse_int(
                "analyze_source_chunk_size",
                entries["analyze_source_chunk_size (int ≥1)"].get(),
                1, None
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_analyze_target_block_size, err = _parse_int(
                "analyze_target_block_size",
                entries["analyze_target_block_size (int ≥1)"].get(),
                1, None
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_save_morphism_comparison_pkl, err = _parse_boolish(
                "save_morphism_comparison_pkl",
                entries["save_morphism_comparison_pkl (true/false)"].get()
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)
            v_analyze_build_legacy_result, err = _parse_boolish(
                "analyze_build_legacy_result",
                entries["analyze_build_legacy_result (true/false)"].get()
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)
            v_analyze_print_summaries, err = _parse_boolish(
                "analyze_print_summaries",
                entries["analyze_print_summaries (true/false)"].get()
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_analyze_compact_diagnostics = str(entries["analyze_compact_diagnostics (none/plot_cache/top_candidates/all_matches)"].get()).strip().lower().replace("-", "_").replace(" ", "_") or "plot_cache"
            if v_analyze_compact_diagnostics in ("default", "cache", "plot", "plotcache", "graph", "graph_cache"):
                v_analyze_compact_diagnostics = "plot_cache"
            elif v_analyze_compact_diagnostics in ("top", "top_candidate", "candidate", "candidates"):
                v_analyze_compact_diagnostics = "top_candidates"
            elif v_analyze_compact_diagnostics in ("all", "full", "full_matches"):
                v_analyze_compact_diagnostics = "all_matches"
            elif v_analyze_compact_diagnostics in ("no", "off", "false", "0", "skip"):
                v_analyze_compact_diagnostics = "none"
            if v_analyze_compact_diagnostics not in ("none", "plot_cache", "top_candidates", "all_matches"):
                return messagebox.showerror(
                    "Invalid input",
                    "analyze_compact_diagnostics must be none, plot_cache, top_candidates, or all_matches.",
                    parent=win
                )

            v_analyze_compact_top_candidates, err = _parse_int(
                "analyze_compact_top_candidates",
                entries["analyze_compact_top_candidates (int ≥0)"].get(),
                0, None
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            try:
                v_analyze_plot_cache_step = float(str(entries["analyze_plot_cache_step (0.001–0.25)"].get()).strip())
                if not (0.001 <= v_analyze_plot_cache_step <= 0.25):
                    raise ValueError
            except Exception:
                return messagebox.showerror("Invalid input", "analyze_plot_cache_step must be a number between 0.001 and 0.25.", parent=win)

            v_analyze_top_k_per_delta, err = _parse_int(
                "analyze_top_k_per_delta",
                entries["analyze_top_k_per_delta (int ≥1)"].get(),
                1, None
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_analyze_pc1_only_threshold, err = _parse_float01(
                "analyze_pc1_only_threshold",
                entries["analyze_pc1_only_threshold (0–1)"].get()
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_analyze_delta_max_for_pc1_only, err = _parse_float01(
                "analyze_delta_max_for_pc1_only",
                entries["analyze_delta_max_for_pc1_only (0–1)"].get()
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_analyze_pc1_only_quality_threshold, err = _parse_float01(
                "analyze_pc1_only_quality_threshold",
                entries["analyze_pc1_only_quality_threshold (0–1)"].get()
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_compute_acuity_for = str(entries["compute_acuity_for (aligned_only/aligned_plus_pc1_only/pc1_only/none)"].get()).strip().lower().replace("-", "_").replace(" ", "_") or "aligned_only"
            if v_compute_acuity_for in ("aligned", "strict", "default"):
                v_compute_acuity_for = "aligned_only"
            elif v_compute_acuity_for in ("all", "both", "aligned_plus_pc1", "aligned_and_pc1_only"):
                v_compute_acuity_for = "aligned_plus_pc1_only"
            elif v_compute_acuity_for in ("pc1",):
                v_compute_acuity_for = "pc1_only"
            elif v_compute_acuity_for in ("no", "off", "false", "0", "skip"):
                v_compute_acuity_for = "none"
            if v_compute_acuity_for not in ("aligned_only", "aligned_plus_pc1_only", "pc1_only", "none"):
                return messagebox.showerror(
                    "Invalid input",
                    "compute_acuity_for must be aligned_only, aligned_plus_pc1_only, pc1_only, or none.",
                    parent=win
                )

            v_acuity_csv_mode = str(entries["acuity_csv_mode (full/selected/none)"].get()).strip().lower().replace("-", "_").replace(" ", "_") or "none"
            if v_acuity_csv_mode in ("all", "everything", "complete"):
                v_acuity_csv_mode = "full"
            elif v_acuity_csv_mode in ("selected_doc", "selected_document", "doc", "doc_id", "anchor", "involving", "focus"):
                v_acuity_csv_mode = "selected"
            elif v_acuity_csv_mode in ("no", "no_csv", "off", "false", "0", "skip"):
                v_acuity_csv_mode = "none"
            if v_acuity_csv_mode not in ("full", "selected", "none"):
                return messagebox.showerror(
                    "Invalid input",
                    "acuity_csv_mode must be full, selected, or none.",
                    parent=win
                )

            v_acuity_top_n, err = _parse_int(
                "acuity_top_n",
                entries["acuity_top_n (int ≥1)"].get(),
                1, None
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_visual_projection = str(entries["visual_projection (global_pca/raw_first3)"].get()).strip().lower().replace("-", "_")
            if v_visual_projection in ("pca", "globalpca", "global_pca", "pca3", "pca_3d"):
                v_visual_projection = "global_pca"
            elif v_visual_projection in ("raw", "first3", "first_3", "raw3", "raw_first3"):
                v_visual_projection = "raw_first3"
            else:
                return messagebox.showerror(
                    "Invalid input",
                    "visual_projection must be either global_pca or raw_first3.",
                    parent=win
                )

            v_visual_surface_mode = str(entries["visual_surface_mode (quality_ellipsoid_mst/hull/quality_ellipsoid/skeleton/none)"].get()).strip().lower()
            v_visual_surface_mode = v_visual_surface_mode.replace("-", "_").replace(" ", "_").replace("+", "_").replace("/", "_")
            v_visual_surface_mode = re.sub(r"_+", "_", v_visual_surface_mode).strip("_")
            if v_visual_surface_mode in ("", "default", "both", "all", "q_ellipsoid_mst", "ellipsoid_mst",
                                         "quality_ellipsoid_mst", "quality_weighted_ellipsoid_mst",
                                         "quality_ellipsoid_skeleton", "quality_weighted_ellipsoid_skeleton",
                                         "ellipsoid_skeleton", "ellipsoid_and_mst", "ellipsoid_with_mst",
                                         "qellipsoid_mst"):
                v_visual_surface_mode = "quality_ellipsoid_mst"
            elif v_visual_surface_mode in ("none", "off", "no", "nosurface", "no_surface", "surface_off", "hide"):
                v_visual_surface_mode = "none"
            elif v_visual_surface_mode in ("hull", "convex_hull", "convexhull", "poly_hull", "polygon_hull"):
                v_visual_surface_mode = "hull"
            elif v_visual_surface_mode in ("ellipsoid", "quality_ellipsoid", "q_ellipsoid", "qellipsoid",
                                           "quality_weighted_ellipsoid", "weighted_ellipsoid", "covariance_ellipsoid",
                                           "q_weighted_ellipsoid"):
                v_visual_surface_mode = "quality_ellipsoid"
            elif v_visual_surface_mode in ("skeleton", "mst", "mst_skeleton", "tree", "minimum_spanning_tree"):
                v_visual_surface_mode = "skeleton"
            else:
                return messagebox.showerror(
                    "Invalid input",
                    "visual_surface_mode must be one of: quality_ellipsoid_mst, hull, quality_ellipsoid, skeleton, none.",
                    parent=win
                )

            v_visual_mst_quality_lambda, err = _parse_float_nonnegative(
                "visual_mst_quality_lambda",
                entries["visual_mst_quality_lambda (≥0)"].get()
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_visual_q_node_radius_scale, err = _parse_float_nonnegative(
                "visual_q_node_radius_scale",
                entries["visual_q_node_radius_scale (≥0)"].get()
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_visual_q_edge_width_scale, err = _parse_float_nonnegative(
                "visual_q_edge_width_scale",
                entries["visual_q_edge_width_scale (≥0)"].get()
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_semantic_fluency_enabled, err = _parse_boolish(
                "semantic_fluency_enabled",
                entries["semantic_fluency_enabled (true/false)"].get()
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_semantic_fluency_model_name = str(entries["semantic_fluency_model_name"].get()).strip()
            v_semantic_fluency_device = str(entries["semantic_fluency_device (cpu/cuda/auto)"].get()).strip() or "cpu"

            v_semantic_fluency_batch_size, err = _parse_int(
                "semantic_fluency_batch_size",
                entries["semantic_fluency_batch_size (int ≥1)"].get(),
                1, None
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_semantic_fluency_max_length, err = _parse_int(
                "semantic_fluency_max_length",
                entries["semantic_fluency_max_length (int ≥8)"].get(),
                8, None
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_semantic_fluency_min_tokens, err = _parse_int(
                "semantic_fluency_min_tokens",
                entries["semantic_fluency_min_tokens (int ≥1)"].get(),
                1, None
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_semantic_fluency_calibration = str(entries["semantic_fluency_calibration (hybrid/absolute/relative)"].get()).strip().lower().replace("-", "_")
            if v_semantic_fluency_calibration in ("", "default"):
                v_semantic_fluency_calibration = "hybrid"
            if v_semantic_fluency_calibration not in ("hybrid", "absolute", "relative"):
                return messagebox.showerror(
                    "Invalid input",
                    "semantic_fluency_calibration must be hybrid, absolute, or relative.",
                    parent=win
                )

            v_semantic_fluency_worker_cap, err = _parse_int(
                "semantic_fluency_worker_cap",
                entries["semantic_fluency_worker_cap (int ≥1)"].get(),
                1, None
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            v_null_replicates, err = _parse_int(
                "null_replicates",
                entries["null_replicates (int ≥1)"].get(),
                1, None
            )
            if err: return messagebox.showerror("Invalid input", err, parent=win)

            try:
                v_null_random_seed = int(str(entries["null_random_seed (int)"].get()).strip())
            except Exception:
                return messagebox.showerror("Invalid input", "null_random_seed must be an integer.", parent=win)

            try:
                v_null_step = float(str(entries["null_step (0.001–0.25)"].get()).strip())
                if not (0.001 <= v_null_step <= 0.25):
                    raise ValueError
            except Exception:
                return messagebox.showerror("Invalid input", "null_step must be between 0.001 and 0.25.", parent=win)

            v_null_strategy = str(entries["null_strategy (association_shuffle)"].get()).strip().lower().replace("-", "_").replace(" ", "_") or "association_shuffle"
            if v_null_strategy not in ("association_shuffle", "feature_shuffle", "shuffle"):
                return messagebox.showerror("Invalid input", "null_strategy currently supports association_shuffle only.", parent=win)
            if v_null_strategy in ("feature_shuffle", "shuffle"):
                v_null_strategy = "association_shuffle"

            v_null_max_field_csv_rows, err = _parse_int(
                "null_max_field_csv_rows",
                entries["null_max_field_csv_rows (int ≥100)"].get(),
                100, None
            )
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
            if endpoint_var.get() == "null_compare" and not raw_doc_id:
                return messagebox.showerror(
                    "Missing anchor item id",
                    "Null compare requires doc_id / anchor item id.",
                    parent=win
                )
            if endpoint_var.get() == "analyze" and v_analyze_csv_mode == "selected" and not raw_doc_id:
                return messagebox.showerror(
                    "Missing selected document id",
                    "Analyze CSV mode 'selected' requires doc_id / anchor item id. Choose full or none to run without doc_id.",
                    parent=win
                )
            if endpoint_var.get() == "analyze" and v_acuity_csv_mode == "selected" and not raw_doc_id:
                return messagebox.showerror(
                    "Missing selected document id",
                    "Acuity CSV mode 'selected' requires doc_id / anchor item id. Choose full or none to run without doc_id.",
                    parent=win
                )
            if endpoint_var.get() == "analyze" and v_analyze_match_scope == "anchor" and not raw_doc_id:
                return messagebox.showerror(
                    "Missing anchor item id",
                    "Analyze match scope 'anchor' requires doc_id / anchor item id.",
                    parent=win
                )
            result["v"] = {
                "mode": mode,
                "num_topics": v_num_topics,
                "top_n_topics": v_top_n_topics,
                "top_m_keywords": v_top_m_keywords,
                "cos_threshold": v_cos_threshold,
                "srcdst_threshold": v_srcdst_threshold,
                "quality_threshold": v_quality_threshold,
                "analyze_csv_mode": v_analyze_csv_mode,
                "analyze_match_scope": v_analyze_match_scope,
                "analyze_engine": v_analyze_engine,
                "analyze_parallel_workers": v_analyze_parallel_workers,
                "analyze_source_chunk_size": int(v_analyze_source_chunk_size),
                "analyze_target_block_size": int(v_analyze_target_block_size),
                "save_morphism_comparison_pkl": bool(v_save_morphism_comparison_pkl),
                "analyze_build_legacy_result": bool(v_analyze_build_legacy_result),
                "analyze_print_summaries": bool(v_analyze_print_summaries),
                "analyze_compact_diagnostics": v_analyze_compact_diagnostics,
                "analyze_compact_top_candidates": int(v_analyze_compact_top_candidates),
                "analyze_plot_cache_step": float(v_analyze_plot_cache_step),
                "analyze_top_k_per_delta": int(v_analyze_top_k_per_delta),
                "analyze_pc1_only_threshold": float(v_analyze_pc1_only_threshold),
                "analyze_delta_max_for_pc1_only": float(v_analyze_delta_max_for_pc1_only),
                "analyze_pc1_only_quality_threshold": float(v_analyze_pc1_only_quality_threshold),
                "compute_acuity_for": v_compute_acuity_for,
                "acuity_csv_mode": v_acuity_csv_mode,
                "acuity_top_n": int(v_acuity_top_n),
                "visual_projection": v_visual_projection,
                "visual_surface_mode": v_visual_surface_mode,
                "visual_mst_quality_lambda": v_visual_mst_quality_lambda,
                "visual_q_node_radius_scale": v_visual_q_node_radius_scale,
                "visual_q_edge_width_scale": v_visual_q_edge_width_scale,
                "semantic_fluency_enabled": bool(v_semantic_fluency_enabled),
                "semantic_fluency_model_name": v_semantic_fluency_model_name,
                "semantic_fluency_device": v_semantic_fluency_device,
                "semantic_fluency_batch_size": int(v_semantic_fluency_batch_size),
                "semantic_fluency_max_length": int(v_semantic_fluency_max_length),
                "semantic_fluency_min_tokens": int(v_semantic_fluency_min_tokens),
                "semantic_fluency_calibration": v_semantic_fluency_calibration,
                "semantic_fluency_worker_cap": int(v_semantic_fluency_worker_cap),
                "null_replicates": int(v_null_replicates),
                "null_random_seed": int(v_null_random_seed),
                "null_step": float(v_null_step),
                "null_strategy": v_null_strategy,
                "null_max_field_csv_rows": int(v_null_max_field_csv_rows),
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
        default_quality_threshold=0.0,
        default_analyze_csv_mode="none",
        default_analyze_match_scope="auto",
        default_analyze_engine="parallel",
        default_analyze_parallel_workers="auto",
        default_analyze_source_chunk_size=384,
        default_analyze_target_block_size=8192,
        default_save_morphism_comparison_pkl=True,
        default_analyze_build_legacy_result=False,
        default_analyze_print_summaries=False,
        default_analyze_compact_diagnostics="plot_cache",
        default_analyze_compact_top_candidates=5000,
        default_analyze_plot_cache_step=0.01,
        default_analyze_top_k_per_delta=100,
        default_analyze_pc1_only_threshold=0.60,
        default_analyze_delta_max_for_pc1_only=0.60,
        default_analyze_pc1_only_quality_threshold=0.0,
        default_compute_acuity_for="aligned_only",
        default_acuity_csv_mode="none",
        default_acuity_top_n=500,
        default_visual_projection="global_pca",
        default_visual_surface_mode="quality_ellipsoid_mst",
        default_visual_mst_quality_lambda=2.0,
        default_visual_q_node_radius_scale=10.0,
        default_visual_q_edge_width_scale=20.0,
        default_semantic_fluency_enabled=True,
        default_semantic_fluency_model_name="distilgpt2",
        default_semantic_fluency_device="cpu",
        default_semantic_fluency_batch_size=8,
        default_semantic_fluency_max_length=128,
        default_semantic_fluency_min_tokens=3,
        default_semantic_fluency_calibration="hybrid",
        default_semantic_fluency_worker_cap=4,
        default_null_replicates=50,
        default_null_random_seed=0,
        default_null_step=0.01,
        default_null_strategy="association_shuffle",
        default_null_max_field_csv_rows=250000,
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
    quality_threshold = cfg.get("quality_threshold", 0.0)
    analyze_csv_mode = str(cfg.get("analyze_csv_mode", "none")).strip().lower().replace("-", "_").replace(" ", "_")
    if analyze_csv_mode in ("all", "everything", "complete"):
        analyze_csv_mode = "full"
    elif analyze_csv_mode in ("selected_doc", "selected_document", "doc", "doc_id", "anchor", "involving", "focus"):
        analyze_csv_mode = "selected"
    elif analyze_csv_mode in ("no", "no_csv", "off", "false", "0", "skip"):
        analyze_csv_mode = "none"
    if analyze_csv_mode not in ("full", "selected", "none"):
        analyze_csv_mode = "none"
    analyze_match_scope = str(cfg.get("analyze_match_scope", "auto")).strip().lower().replace("-", "_").replace(" ", "_")
    if analyze_match_scope in ("selected", "selected_doc", "selected_document", "doc", "doc_id", "focus", "source", "source_only", "anchor_source"):
        analyze_match_scope = "anchor"
    elif analyze_match_scope in ("all", "global", "complete", "everything", "collection"):
        analyze_match_scope = "full"
    if analyze_match_scope not in ("auto", "anchor", "full"):
        analyze_match_scope = "auto"

    analyze_engine = str(cfg.get("analyze_engine", "parallel")).strip().lower().replace("-", "_").replace(" ", "_")
    if analyze_engine in ("mp", "multiprocessing", "parallel_chunked", "chunked_parallel"):
        analyze_engine = "parallel"
    elif analyze_engine in ("single", "classic", "legacy", "reference"):
        analyze_engine = "serial"
    if analyze_engine not in ("parallel", "serial"):
        analyze_engine = "parallel"

    raw_parallel_workers = cfg.get("analyze_parallel_workers", "auto")
    if raw_parallel_workers is None or str(raw_parallel_workers).strip().lower() in ("", "auto", "default", "recommended"):
        analyze_parallel_workers = None
    else:
        analyze_parallel_workers = max(1, int(raw_parallel_workers))

    analyze_source_chunk_size = max(1, int(cfg.get("analyze_source_chunk_size", 384)))
    analyze_target_block_size = max(1, int(cfg.get("analyze_target_block_size", 8192)))
    save_morphism_comparison_pkl = bool(cfg.get("save_morphism_comparison_pkl", True))
    analyze_build_legacy_result = bool(cfg.get("analyze_build_legacy_result", False))
    analyze_print_summaries = bool(cfg.get("analyze_print_summaries", False))
    analyze_compact_diagnostics = str(cfg.get("analyze_compact_diagnostics", "plot_cache")).strip().lower().replace("-", "_").replace(" ", "_")
    if analyze_compact_diagnostics in ("default", "cache", "plot", "plotcache", "graph", "graph_cache"):
        analyze_compact_diagnostics = "plot_cache"
    elif analyze_compact_diagnostics in ("top", "top_candidate", "candidate", "candidates"):
        analyze_compact_diagnostics = "top_candidates"
    elif analyze_compact_diagnostics in ("all", "full", "full_matches"):
        analyze_compact_diagnostics = "all_matches"
    elif analyze_compact_diagnostics in ("no", "off", "false", "0", "skip"):
        analyze_compact_diagnostics = "none"
    if analyze_compact_diagnostics not in ("none", "plot_cache", "top_candidates", "all_matches"):
        analyze_compact_diagnostics = "plot_cache"
    analyze_compact_top_candidates = max(0, int(cfg.get("analyze_compact_top_candidates", 5000)))
    analyze_plot_cache_step = float(cfg.get("analyze_plot_cache_step", 0.01))
    analyze_plot_cache_step = max(0.001, min(0.25, analyze_plot_cache_step))

    analyze_top_k_per_delta = int(cfg.get("analyze_top_k_per_delta", 100))
    analyze_top_k_per_delta = max(1, analyze_top_k_per_delta)
    analyze_pc1_only_threshold = float(cfg.get("analyze_pc1_only_threshold", 0.60))
    analyze_pc1_only_threshold = max(0.0, min(1.0, analyze_pc1_only_threshold))
    analyze_delta_max_for_pc1_only = float(cfg.get("analyze_delta_max_for_pc1_only", 0.60))
    analyze_delta_max_for_pc1_only = max(0.0, min(1.0, analyze_delta_max_for_pc1_only))
    analyze_pc1_only_quality_threshold = float(cfg.get("analyze_pc1_only_quality_threshold", 0.0))
    analyze_pc1_only_quality_threshold = max(0.0, min(1.0, analyze_pc1_only_quality_threshold))

    compute_acuity_for = str(cfg.get("compute_acuity_for", "aligned_only")).strip().lower().replace("-", "_").replace(" ", "_")
    if compute_acuity_for in ("aligned", "strict", "default"):
        compute_acuity_for = "aligned_only"
    elif compute_acuity_for in ("all", "both", "aligned_plus_pc1", "aligned_and_pc1_only"):
        compute_acuity_for = "aligned_plus_pc1_only"
    elif compute_acuity_for in ("pc1",):
        compute_acuity_for = "pc1_only"
    elif compute_acuity_for in ("no", "off", "false", "0", "skip"):
        compute_acuity_for = "none"
    if compute_acuity_for not in ("aligned_only", "aligned_plus_pc1_only", "pc1_only", "none"):
        compute_acuity_for = "aligned_only"

    acuity_csv_mode = str(cfg.get("acuity_csv_mode", "none")).strip().lower().replace("-", "_").replace(" ", "_")
    if acuity_csv_mode in ("all", "everything", "complete"):
        acuity_csv_mode = "full"
    elif acuity_csv_mode in ("selected_doc", "selected_document", "doc", "doc_id", "anchor", "involving", "focus"):
        acuity_csv_mode = "selected"
    elif acuity_csv_mode in ("no", "no_csv", "off", "false", "0", "skip"):
        acuity_csv_mode = "none"
    if acuity_csv_mode not in ("full", "selected", "none"):
        acuity_csv_mode = "none"
    acuity_top_n = int(cfg.get("acuity_top_n", 500))
    visual_projection = cfg.get("visual_projection", "global_pca")
    visual_surface_mode = cfg.get("visual_surface_mode", "quality_ellipsoid_mst")
    visual_mst_quality_lambda = cfg.get("visual_mst_quality_lambda", 2.0)
    visual_q_node_radius_scale = cfg.get("visual_q_node_radius_scale", 10.0)
    visual_q_edge_width_scale = cfg.get("visual_q_edge_width_scale", 20.0)
    semantic_fluency_enabled = bool(cfg.get("semantic_fluency_enabled", True))
    semantic_fluency_model_name = cfg.get("semantic_fluency_model_name", "distilgpt2")
    semantic_fluency_device = cfg.get("semantic_fluency_device", "cpu")
    semantic_fluency_batch_size = int(cfg.get("semantic_fluency_batch_size", 8))
    semantic_fluency_max_length = int(cfg.get("semantic_fluency_max_length", 128))
    semantic_fluency_min_tokens = int(cfg.get("semantic_fluency_min_tokens", 3))
    semantic_fluency_calibration = cfg.get("semantic_fluency_calibration", "hybrid")
    semantic_fluency_worker_cap = int(cfg.get("semantic_fluency_worker_cap", 4))
    null_replicates = int(cfg.get("null_replicates", 50))
    null_random_seed = int(cfg.get("null_random_seed", 0))
    null_step = float(cfg.get("null_step", 0.01))
    null_strategy = cfg.get("null_strategy", "association_shuffle")
    null_max_field_csv_rows = int(cfg.get("null_max_field_csv_rows", 250000))
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

        # Choose a sensible worker count.  LM fluency scoring loads a causal
        # language model per worker, so cap workers by default to avoid multiplying
        # RAM use and model-startup cost.
        cpu_workers = max(1, 10)
        if semantic_fluency_enabled:
            cpu_workers = max(1, min(cpu_workers, int(semantic_fluency_worker_cap)))
            print(
                f"[fluency] enabled: model={semantic_fluency_model_name!r}, "
                f"device={semantic_fluency_device!r}, calibration={semantic_fluency_calibration!r}, "
                f"worker_cap={cpu_workers}",
                flush=True
            )

        fluency_params = {
            "enabled": bool(semantic_fluency_enabled),
            "model_name": semantic_fluency_model_name,
            "device": semantic_fluency_device,
            "batch_size": int(semantic_fluency_batch_size),
            "max_length": int(semantic_fluency_max_length),
            "min_tokens": int(semantic_fluency_min_tokens),
            "calibration": semantic_fluency_calibration,
            "absolute_center": 5.5,
            "absolute_scale": 1.15,
        }

        document_delta_dict = {}
        segments_by_doc = {}

        # Initialize workers ONCE with the heavy models
        with ProcessPoolExecutor(
            max_workers=cpu_workers,
            initializer=_init_worker,
            initargs=(fpaths["lda_model"], fpaths["lda_dict"], num_topics, fluency_params)  # keep "cpu" in workers
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


    # Ensure the main process has an LDA model for visualization/analysis labels.
    if "LdaModel_loaded" not in locals() or LdaModel_loaded is None:
        LdaModel_loaded = LdaModel.load(fpaths["lda_model"])

    timer1.stop()
    print(f"Total elapsed time before visualization: {int(timer1.elapsed())//60}:min {int(timer1.elapsed())%60}:sec")


    # Legacy Analyze output controls.  The compact parallel backend can save and
    # plot directly from res["morphism_comparison"].  The old nested match maps
    # are only built when the serial engine is used or when CSV/console legacy
    # output is explicitly requested.
    if analyze_engine == "serial":
        if not analyze_build_legacy_result:
            print("[main] Serial Analyze requires the legacy result object; enabling analyze_build_legacy_result=True.")
        analyze_build_legacy_result = True
    if (analyze_csv_mode != "none" or acuity_csv_mode != "none" or analyze_print_summaries) and not analyze_build_legacy_result:
        print("[main] Legacy Analyze CSV/summary output requested; enabling analyze_build_legacy_result=True.")
        analyze_build_legacy_result = True

    # Endpoint selection
    if cfg["endpoint"] == "visualize":
        visualize_documents_with_directional_overlap(
            document_cluster_data=document_delta_dict,
            lda_model=LdaModel_loaded,
            top_n_topics=top_n_topics,
            top_m_keywords=top_m_keywords,
            cos_threshold=cos_threshold,
            lda_int_topics_list=lda_int_topics_list,
            srcdst_threshold=srcdst_threshold,
            quality_threshold=quality_threshold,
            segments_by_doc=segments_by_doc,
            visual_projection=visual_projection,
            visual_surface_mode=visual_surface_mode,
            visual_mst_quality_lambda=visual_mst_quality_lambda,
            visual_q_node_radius_scale=visual_q_node_radius_scale,
            visual_q_edge_width_scale=visual_q_edge_width_scale
        )

    elif cfg["endpoint"] == "analyze":  # "analyze"
        # Resolve Analyze computation scope.  In auto mode, selected CSV output
        # uses anchor-scoped matching so the analyzer only searches the selected
        # document's source morphisms against the collection.
        analyze_scope_effective = analyze_match_scope
        if analyze_scope_effective == "auto":
            analyze_scope_effective = "anchor" if (analyze_csv_mode == "selected" and doc_id) else "full"
        source_doc_filter = doc_id if analyze_scope_effective == "anchor" else None
        print(f"[main] Analyze match scope: {analyze_scope_effective}" + (f"; anchor={source_doc_filter}" if source_doc_filter else ""))
        recommended_workers = recommend_morphism_compare_workers()
        effective_parallel_workers = analyze_parallel_workers if analyze_parallel_workers is not None else recommended_workers
        print(
            f"[main] Analyze tuning: engine={analyze_engine}; top_k_per_delta={analyze_top_k_per_delta}; "
            f"pc1_only_threshold={analyze_pc1_only_threshold:.3f}; "
            f"delta_max_for_pc1_only={analyze_delta_max_for_pc1_only:.3f}; "
            f"pc1_only_quality_threshold={analyze_pc1_only_quality_threshold:.3f}; "
            f"compute_acuity_for={compute_acuity_for}"
        )
        print(
            f"[main] Analyze legacy outputs: build_legacy={analyze_build_legacy_result}; "
            f"print_summaries={analyze_print_summaries}; analyze_csv_mode={analyze_csv_mode}; "
            f"acuity_csv_mode={acuity_csv_mode}"
        )
        print(
            f"[main] Compact diagnostics: mode={analyze_compact_diagnostics}; "
            f"top_candidates={analyze_compact_top_candidates}; plot_cache_step={analyze_plot_cache_step:.3f}"
        )
        if analyze_engine == "parallel":
            print(
                f"[main] Parallel Analyze settings: workers={effective_parallel_workers} "
                f"(recommended={recommended_workers}); source_chunk={analyze_source_chunk_size}; "
                f"target_block={analyze_target_block_size}; BLAS_threads_per_worker=1"
            )
            res = analyze_morphism_match_field_parallel(
                document_cluster_data=document_delta_dict,
                delta_thresholds=[i / 100 for i in range(99, -1, -3)],
                pc1_thresholds=[i / 100 for i in range(99, -1, -3)],
                quality_thresholds=[i / 100 for i in range(99, -1, -1)],
                segments_by_doc=segments_by_doc,
                top_k_per_delta=analyze_top_k_per_delta,
                pc1_only_threshold=analyze_pc1_only_threshold,
                delta_max_for_pc1_only=analyze_delta_max_for_pc1_only,
                pc1_only_quality_threshold=analyze_pc1_only_quality_threshold,
                compute_acuity_for=compute_acuity_for,
                pc1_match_axis="dst",
                source_doc_filter=source_doc_filter,
                analyze_scope=analyze_scope_effective,
                require_cross_doc=True,
                max_workers=effective_parallel_workers,
                source_chunk_size=analyze_source_chunk_size,
                target_block_size=analyze_target_block_size,
                blas_threads=1,
                include_edge_vectors_in_result=True,
                include_delta_full_in_vectors=False,
                compact_only=(not analyze_build_legacy_result),
                verbose=True
            )
        else:
            print("[main] Serial Analyze selected; using reference analyzer.")
            res = analyze_morphism_match_field(
                document_cluster_data=document_delta_dict,
                delta_thresholds=[i / 100 for i in range(99, -1, -3)],
                pc1_thresholds=[i / 100 for i in range(99, -1, -3)],
                quality_thresholds=[i / 100 for i in range(99, -1, -1)],
                segments_by_doc=segments_by_doc,
                top_k_per_delta=analyze_top_k_per_delta,
                pc1_only_threshold=analyze_pc1_only_threshold,
                delta_max_for_pc1_only=analyze_delta_max_for_pc1_only,
                pc1_only_quality_threshold=analyze_pc1_only_quality_threshold,
                compute_acuity_for=compute_acuity_for,
                pc1_match_axis="dst",
                source_doc_filter=source_doc_filter,
                analyze_scope=analyze_scope_effective,
                require_cross_doc=True,
                verbose=True
            )

        if analyze_engine == "parallel" and analyze_compact_diagnostics != "none" and isinstance(res, dict) and res.get("morphism_comparison"):
            res = enrich_morphism_comparison_diagnostics(
                res,
                document_cluster_data=document_delta_dict,
                segments_by_doc=segments_by_doc,
                diagnostics_mode=analyze_compact_diagnostics,
                step=analyze_plot_cache_step,
                top_candidates=analyze_compact_top_candidates,
                verbose=True,
            )
        elif analyze_compact_diagnostics == "none":
            print("[main] Compact diagnostic enrichment skipped (mode=none).")

        if save_morphism_comparison_pkl:
            if doc_id:
                safe_doc = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(doc_id))[:120]
                default_cmp_name = f"morphism_comparison_selected_{safe_doc}.pkl"
            elif analyze_scope_effective == "full":
                default_cmp_name = "morphism_comparison_full.pkl"
            else:
                default_cmp_name = "morphism_comparison.pkl"
            cmp_pkl_path = _choose_pkl_save_path(default_name=default_cmp_name, parent_window=root, title="Save morphism comparison PKL")
            if cmp_pkl_path:
                save_morphism_comparison_pickle(res, cmp_pkl_path, include_legacy_result=False)
            else:
                print("[main] Morphism comparison PKL save cancelled; continuing with in-memory result.")
        else:
            print("[main] Morphism comparison PKL output skipped.")
        
        if analyze_build_legacy_result:
            csv_path = None
            if analyze_csv_mode != "none":
                if analyze_csv_mode == "selected" and doc_id:
                    safe_doc = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(doc_id))[:120]
                    default_csv_name = f"morphism_matches_selected_{safe_doc}.txt"
                elif analyze_csv_mode == "full":
                    default_csv_name = "morphism_matches_full.txt"
                else:
                    default_csv_name = "morphism_matches.txt"
                csv_path = _choose_csv_save_path(default_name=default_csv_name, parent_window=root)
                if not csv_path:
                    print("[main] Analyze CSV save cancelled; CSV output will be skipped.")
                    analyze_csv_mode = "none"
            else:
                print("[main] Analyze CSV output skipped because analyze_csv_mode='none'.")

            if analyze_print_summaries or analyze_csv_mode != "none":
                output_analysis(
                    res=res,
                    doc_id=doc_id,
                    document_cluster_data=document_delta_dict,
                    segments_by_doc=segments_by_doc,
                    csv_path=csv_path,
                    csv_mode=analyze_csv_mode,
                    print_summary=analyze_print_summaries,
                )
            else:
                print("[main] Legacy human-readable Analyze summaries skipped.")

            acuity_csv_path = None
            if acuity_csv_mode != "none":
                if acuity_csv_mode == "selected" and doc_id:
                    safe_doc = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(doc_id))[:120]
                    default_acuity_name = f"top_acuity_candidates_selected_{safe_doc}.txt"
                elif acuity_csv_mode == "full":
                    default_acuity_name = "top_acuity_candidates_full.txt"
                else:
                    default_acuity_name = "top_acuity_candidates.txt"
                acuity_csv_path = _choose_csv_save_path(default_name=default_acuity_name, parent_window=root)
                if not acuity_csv_path:
                    print("[main] Top acuity candidates CSV save cancelled; acuity CSV output will be skipped.")
                    acuity_csv_mode = "none"
            else:
                print("[main] Top acuity candidates CSV output skipped because acuity_csv_mode='none'.")

            if acuity_csv_mode != "none":
                output_acuity_candidates_csv(
                    res=res,
                    csv_path=acuity_csv_path,
                    doc_id=doc_id,
                    csv_mode=acuity_csv_mode,
                    top_n=acuity_top_n,
                    step=0.01,
                    peak_per_bin=True
                )
            else:
                print("[main] Legacy top-acuity CSV exporter skipped.")
        else:
            print("[main] Legacy Analyze result expansion/output skipped; using compact morphism_comparison for PKL and plot.")

        plot_morphism_match_field_3d(
        res=res,
        step=0.01,
        doc_id=doc_id,
        log_colors=True,
        initial_quality_floor=quality_threshold,
        cumulative=False,
        include_cumulative=False,
        pc1_axis_mode="dst"
        )
        
    elif cfg["endpoint"] == "null_compare":
        if not doc_id:
            messagebox.showerror(
                "Missing anchor item id",
                "The null_compare endpoint requires doc_id / anchor item id.",
                parent=root
            )
            raise SystemExit(1)
        if doc_id not in document_delta_dict:
            messagebox.showerror(
                "Anchor item not found",
                f"Anchor item id not found in loaded document delta dict:\n{doc_id}",
                parent=root
            )
            raise SystemExit(1)

        null_res = analyze_anchor_null_match_field(
            document_cluster_data=document_delta_dict,
            anchor_doc_id=doc_id,
            segments_by_doc=segments_by_doc,
            n_null_replicates=null_replicates,
            null_strategy=null_strategy,
            random_seed=null_random_seed,
            step=null_step,
            pc1_match_axis="dst",
            edge_support_delta_threshold=cos_threshold,
            edge_support_pc1_threshold=srcdst_threshold,
            edge_support_quality_threshold=quality_threshold,
            max_match_csv_rows=null_max_field_csv_rows,
            verbose=True
        )

        # Save full null-field result as a PKL for reuse/audit.
        pkl_path = _choose_pkl_save_path(
            default_name=f"{doc_id}_anchor_null_field.pkl",
            parent_window=root,
            title="Save anchor null field PKL"
        )
        if pkl_path:
            save_anchor_null_field_pickle(null_res, pkl_path)
        else:
            print("[null] null field PKL save skipped.")

        field_csv_path = _choose_csv_save_path(
            default_name=f"{doc_id}_anchor_null_field_voxels.txt",
            parent_window=root
        )
        edge_csv_path = _choose_csv_save_path(
            default_name=f"{doc_id}_anchor_edge_contributions.txt",
            parent_window=root
        )
        match_csv_path = _choose_csv_save_path(
            default_name=f"{doc_id}_anchor_null_pair_matches.txt",
            parent_window=root
        )
        output_anchor_null_match_csvs(
            null_res,
            field_csv_path=field_csv_path,
            edge_csv_path=edge_csv_path,
            match_csv_path=match_csv_path,
            max_field_rows=null_max_field_csv_rows
        )

        plot_anchor_null_match_field_3d(
            null_res,
            max_plot_points=150000,
            initial_view="positive_residual"
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
            max_docs_left=None,                 # cap left-side nodes; set None for all docs
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