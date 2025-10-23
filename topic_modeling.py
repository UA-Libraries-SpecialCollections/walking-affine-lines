#!/usr/bin/python
# OpenAI api uses Python 3.10 :  C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\topic_modeling.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu


# This python resource file contains function definitions supporting visualizations and 
# data analysis outputs for the generate_document_delta_manifold.py pipeline
# 

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.

from backrooms import Timer
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

import re, csv, os

import numpy as np

from typing import List, Tuple, Optional, Dict, Any

import nltk

from tkinter import filedialog

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

import umap #umap-learn

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Cursor
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D
import mplcursors

from sentence_transformers import SentenceTransformer
from collections import defaultdict
from collections import Counter

from scipy.spatial import ConvexHull, cKDTree

from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_punctuation,
    strip_numeric,
    remove_stopwords,
    strip_short
)

###### ONE TIME INSTALL #######
#nltk.download('punkt_tab')

# -----------------------------------------------------------------------------
# GLOBALS
_MODEL = None
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Function: clean_text
# Summary:
#   Lowercases raw OCR text, removes non‑alphabetic characters, and collapses
#   whitespace to single spaces to yield a simple, stable token stream.
# Effect:
#   Reduces OCR noise so downstream vectorizers (TF‑IDF, LDA) see cleaner
#   statistics, making later topic models and morphism interpretation steadier.
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Simple OCR cleanup: lowercase, strip punctuation/numbers, normalize whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # remove non-alpha chars
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    return text.strip()

# -----------------------------------------------------------------------------
# Function: extract_topics_from_ocr
# Summary:
#   Cleans input strings with `clean_text`, builds a TF‑IDF matrix
#   (min_df/max_df guarded for tiny corpora), fits an NMF topic model, and
#   returns the top words per topic (optionally also returns model, features,
#   TF‑IDF matrix, and vectorizer for reuse).
# Effect:
#   Produces light‑weight topical scaffolding over noisy OCR that can seed
#   human labels and provide interpretable context for later delta directions.
# -----------------------------------------------------------------------------
def extract_topics_from_ocr(
    documents: List[str],
    n_topics: int = 10,
    top_n_words: int = 10,
    min_df: int = 3,
    max_df: float = 0.95,
    stop_words: Optional[str] = 'english',
    return_model: bool = False
) -> Tuple[List[List[str]], Optional[TransformerMixin], Optional[List[str]]]:

    
    if len(documents) <= min_df:
        min_df = int(len(documents) * 0.01) + 1
        print(f"min_df adjusted due to small document corpus")
        
    if len(documents) == 1:
        min_df = 1
        max_df = 1.0
        
    print("")
    print(f"n_topics: {n_topics}")
    print(f"top_n_words: {top_n_words}")
    print(f"min_df: {min_df}")
    print(f"max_df: {max_df}")
    
    """
    Clean OCR text, perform TF-IDF vectorization, and extract topics using NMF.

    Parameters:
        documents: List of OCR strings (dirty input texts).
        n_topics: Number of topics to extract.
        top_n_words: Number of top words to return per topic.
        min_df: Minimum doc freq for terms in TF-IDF vectorizer. (ignore words that appear in fewer than n documents)
        max_df: Maximum doc freq (as fraction) for terms in TF-IDF vectorizer. (ignore words that appear in more than % of documents)
        stop_words: Stopword list; 'english' uses built-in, or None disables.
        return_model: If True, return the fitted model and feature names as well.

    Returns:
        topics: List of topic word lists.
        (Optional) model: Fitted NMF model.
        (Optional) feature_names: TF-IDF vocabulary terms.
    """
    # Step 1: Preprocess OCR text
    cleaned = [clean_text(doc) for doc in documents]

    # Step 2: TF-IDF Vectorization
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(cleaned)
    feature_names = vectorizer.get_feature_names_out()

    # Step 3: Fit NMF
    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(tfidf_matrix)  # document-topic weights
    H = nmf.components_                 # topic-word weights

    # Step 4: Get top words per topic
    topics = []
    for topic_idx, topic_weights in enumerate(H):
        top_indices = topic_weights.argsort()[::-1][:top_n_words]
        topic_words = [feature_names[i] for i in top_indices]
        topics.append(topic_words)

    if return_model:
        return topics, nmf, feature_names.tolist(), tfidf_matrix, vectorizer
    else:
        return topics, None, None

# -----------------------------------------------------------------------------
# Function: summarize_document_topic
# Summary:
#   Projects one document through a trained NMF model to get topic weights,
#   selects its top‑N topics, and reports each topic’s weight and top words.
# Effect:
#   Offers compact, document‑level summaries that tie later morphisms (cluster
#   deltas) back to human‑legible topical cues for interpretation.
# -----------------------------------------------------------------------------
def summarize_document_topic(
    doc_index: int,
    nmf_model: NMF,
    tfidf_matrix: np.ndarray,
    tfidf_vectorizer: TfidfVectorizer,
    top_n_topics: int = 3,
    top_n_words: int = 10,
) -> Dict[str, Any]:
    """
    Returns the most relevant topic info for a single document based on a trained NMF model.

    Parameters:
        doc_index: Index of the document to summarize.
        nmf_model: Trained NMF model.
        tfidf_matrix: TF-IDF matrix of the documents.
        tfidf_vectorizer: Fitted TfidfVectorizer instance.
        top_n_topics: Number of top topics to return.
        top_n_words: Number of top topic words to include.

    Returns:
        Dictionary containing:
            - document_index: Index of the document
            - topics: List of dicts with:
                - topic_index: Topic with highest weight for the doc
                - topic_weight: Topic weight score
                - topic_words: Top N words in the topic
    """

    # Get topic weights for the document
    doc_topic_matrix = nmf_model.transform(tfidf_matrix)
    topic_weights = doc_topic_matrix[doc_index]

    # Get indices of the top N topics
    top_topic_indices = topic_weights.argsort()[::-1][:top_n_topics]

    # Get vocabulary
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Build topic summaries
    topics_summary = []
    for topic_idx in top_topic_indices:
        topic_weight = float(topic_weights[topic_idx])
        topic_word_weights = nmf_model.components_[topic_idx]
        top_word_indices = topic_word_weights.argsort()[::-1][:top_n_words]
        topic_words = [feature_names[i] for i in top_word_indices]
        topics_summary.append({
            "topic_index": int(topic_idx),
            "topic_weight": topic_weight,
            "topic_words": topic_words
        })

    return {
        "document_index": doc_index,
        "doc_topic_matrix": doc_topic_matrix,
        "topics": topics_summary
    }
    

# -----------------------------------------------------------------------------
# Function: plot_documents_by_topic
# Summary:
#   Reduces document‑topic weights to 2D via PCA and colors each document by
#   its dominant topic to visualize coarse topical neighborhoods.
# Effect:
#   Provides an overview “map” so that later morphism‑based comparisons can be
#   understood against broad topical structure in the collection.
# -----------------------------------------------------------------------------
def plot_documents_by_topic(doc_topic_matrix, top_n_topics: int = 10):
    """
    Projects NMF document-topic weights to 2D using PCA and plots them,
    coloring by the dominant topic.

    Parameters:
        doc_topic_matrix: result of nmf_model.transform(tfidf_matrix)
        top_n_topics: number of topic clusters to color (others marked as 'Other')
    """
    # Dimensionality reduction
    pca = PCA(n_components=2)
    doc_2d = pca.fit_transform(doc_topic_matrix)

    # Assign each doc to its most influential topic
    topic_labels = np.argmax(doc_topic_matrix, axis=1)

    # Optionally remap topics to only top N most common
    top_topics = np.bincount(topic_labels).argsort()[::-1][:top_n_topics]
    remap = {topic: i for i, topic in enumerate(top_topics)}
    colored_labels = [remap[t] if t in remap else top_n_topics for t in topic_labels]

    # Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(doc_2d[:, 0], doc_2d[:, 1],
                          c=colored_labels, cmap='tab10', alpha=0.7)
    plt.title("Document Clusters by Dominant Topic")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, ticks=range(top_n_topics+1), label="Topic Index")
    plt.tight_layout()
    plt.show()
    
# -----------------------------------------------------------------------------
# Function: plot_documents_with_umap_annotations
# Summary:
#   Embeds doc‑topic weights to 2D using UMAP, colors by common topics, draws
#   per‑topic keyword labels, and adds interactive hover previews of documents.
# Effect:
#   Gives curators an intuitive, interactive lens on topical clusters, which
#   complements the morphism view (flows between clusters) by showing context.
# -----------------------------------------------------------------------------
def plot_documents_with_umap_annotations(
    doc_topic_matrix,
    documents,
    nmf_model,
    tfidf_vectorizer,
    top_n_words=5,
    top_n_topics=5,
    preview_chars=200
):
    # Reduce dimensionality to 2D with UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    doc_2d = reducer.fit_transform(doc_topic_matrix)

    # Assign each document to its most influential topic
    topic_labels = np.argmax(doc_topic_matrix, axis=1)

    # Select the most common top_n_topics
    top_topics = np.bincount(topic_labels).argsort()[::-1][:top_n_topics]
    remap = {topic: i for i, topic in enumerate(top_topics)}
    colored_labels = [remap[t] if t in remap else top_n_topics for t in topic_labels]

    # Get top keywords per topic
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topic_keywords = []
    for topic_idx, topic_weights in enumerate(nmf_model.components_):
        top_word_indices = topic_weights.argsort()[::-1][:top_n_words]
        keywords = ", ".join(feature_names[i] for i in top_word_indices)
        topic_keywords.append(keywords)

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(doc_2d[:, 0], doc_2d[:, 1], c=colored_labels, cmap='tab10', alpha=0.7, s=1)

    # Add topic keyword annotations for top N topics
    for topic, idx in remap.items():
        topic_docs = np.where(topic_labels == topic)[0]
        if len(topic_docs) == 0:
            continue
        mean_x = np.mean(doc_2d[topic_docs, 0])
        mean_y = np.mean(doc_2d[topic_docs, 1])
        plt.text(mean_x, mean_y, topic_keywords[topic], fontsize=10, weight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    # Enable interactive hover to preview document text
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        doc_index = sel.index
        sel.annotation.set_text(documents[doc_index][:preview_chars] + "...")
        sel.annotation.get_bbox_patch().set(alpha=0.8)

    plt.title("Document Clusters by Dominant Topic (UMAP)")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.colorbar(scatter, ticks=range(top_n_topics+1), label="Topic Index")
    plt.tight_layout()
    plt.show()
    

# -----------------------------------------------------------------------------
# Function: mk_delta_manifold helper functions for speed improvements
def get_sentence_model(device: str | None = None):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return _MODEL

def batched_encode(segments: list[str], batch_size: int = 2048, device: str | None = None):
    m = get_sentence_model(device=device)
    return m.encode(
        segments,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


# -----------------------------------------------------------------------------
# Function: mk_delta_manifold
# Summary:
#   Segments a single document into sentences, embeds them with a Sentence
#   Transformer, chooses a cluster count k (silhouette‑guided), clusters the
#   segments, computes per‑cluster centroids, and builds a pairwise delta
#   tensor Δ[i,j] = centroid_j − centroid_i. Returns (Δ, cluster_order, raw
#   segment labels, segment texts, segment embeddings, k).
# Effect:
#   Converts a document into a *directional manifold* of semantic shifts
#   between its clusters—the core object used to compare documents by the
#   shapes and alignments of these morphisms.
# -----------------------------------------------------------------------------
def mk_delta_manifold(
    item_id,
    item_text,
    st_model,
    cluster_method: str = "agglomerative",     # "agglomerative" | "spectral"
    spectral_params: dict | None = None        # e.g., {"n_neighbors": 10, "assign_labels": "kmeans", "self_weight": 1.0}
):

    # Load sentence transformer model
    #st_model = get_sentence_model()  # cached, single GPU model
    
    # Summary: segment_document — sentence‐tokenize the document (NLTK).
    # Effect: stable granularity for embedding and clustering.
    def segment_document(text: str):
        """Segment a document into sentences using NLTK."""
        return nltk.sent_tokenize(text)

    # Summary: embed_segments — encode segments with a normalized sentence‑BERT.
    # Effect: places text in a metric space where vector deltas are meaningful.
    def embed_segments(segments):
        """Generate embeddings for each segment using a sentence transformer."""
        return st_model.encode(segments, normalize_embeddings=True)
    
    # Project out the top-n principal directions (common components) from L2-normalized embeddings,
    # then renormalize. Works well with n in {1,2,3}.    
    def remove_top_components(E: np.ndarray, n: int = 2) -> np.ndarray:
        X = E - E.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        P = Vt[:n].T                     # (d, n)
        X2 = X - X @ P @ P.T             # remove the shared subspace
        X2 /= (np.linalg.norm(X2, axis=1, keepdims=True) + 1e-12)
        return X2    
        
    # Summary: choose_k_via_silhouette — select a robust cluster count using
    #          agglomerative clustering and cosine‐silhouette (with safe fallbacks).
    # Effect: yields enough structure for Δ without over/under‑segmenting small docs.
    def choose_k_via_silhouette(embeddings, k_min=3, k_max=10):
        """
        Pick k robustly for small-n documents.
        - If n <= 1: k=1
        - If n == 2: k=2 (silhouette not meaningful)
        - Else: search 2..min(k_max, n-1); fall back to k=2 if none works.
        """
        n = len(embeddings)
        if n <= 1:
            return 1
        if n == 2:
            return 2

        k_low  = 2                       # silhouette needs at least 2 clusters
        k_high = min(k_max, n - 1)       # and strictly less than n
        start  = max(k_low, min(k_min, k_high))

        best_k, best_score = None, -1.0
        for k in range(start, k_high + 1):
            try:
                model = AgglomerativeClustering(n_clusters=k, linkage='complete', metric='cosine')
            except TypeError:
                model = AgglomerativeClustering(n_clusters=k, linkage='complete', affinity='cosine')  # older sklearn
            labels = model.fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            try:
                score = silhouette_score(embeddings, labels, metric='cosine')
            except Exception:
                continue
            if score > best_score:
                best_k, best_score = k, score

        return best_k if best_k is not None else 2  # safe fallback

    def choose_k_size_aware(
        embeddings,
        k_min=3,
        k_max=10,
        alpha=0.25,            # reward balance (normalized entropy)
        min_cluster_size=5,    # penalize tiny clusters
        method: str = "agglomerative",
        spectral_params: dict | None = None
    ):
        n = len(embeddings)
        if n <= 1: return 1
        if n == 2: return 2
        k_hi = min(int(k_max), n - 1)
        k_lo = max(2, int(k_min))

        best_k, best_score = None, -1e9
        for k in range(k_lo, k_hi + 1):
            try:
                if method == "spectral":
                    labels = _spectral_labels(embeddings, k, spectral_params)
                else:
                    try:
                        model = AgglomerativeClustering(n_clusters=k, linkage='average', metric='cosine')
                    except TypeError:  # older sklearn
                        model = AgglomerativeClustering(n_clusters=k, linkage='average', affinity='cosine')
                    labels = model.fit_predict(embeddings)
            except Exception:
                continue

            # Need at least 2 non-empty clusters for silhouette
            if len(set(labels)) < 2:
                continue

            try:
                s = silhouette_score(embeddings, labels, metric='cosine')
            except Exception:
                s = -1.0

            H, f_max, counts = _cluster_balance_metrics(labels, k)
            penalty = 0.10 * int((counts < min_cluster_size).sum())
            score = s + alpha * H - penalty

            if score > best_score:
                best_k, best_score = k, score
        return best_k if best_k is not None else 2

    # Summary: cluster_segments — agglomerative clustering with cosine distance
    #          (clamps degenerate cases, handles k==1 fast path).
    # Effect: provides coherent cluster assignments that anchor Δ endpoints.
    def cluster_segments(embeddings, k, method: str = "agglomerative", spectral_params: dict | None = None):
        """
        Cluster segments. For k==1 return a single label; otherwise
        use Agglomerative (cosine) or Spectral (cosine-affinity) per 'method'.
        """
        n = len(embeddings)
        if n == 0:
            return np.array([], dtype=int)

        k = int(max(1, min(k, n)))  # clamp

        if k == 1:
            return np.zeros(n, dtype=int)

        if method == "spectral":
            return _spectral_labels(embeddings, k, spectral_params)

        # default: agglomerative
        try:
            clustering = AgglomerativeClustering(n_clusters=k, linkage='complete', metric='cosine')
        except TypeError:
            clustering = AgglomerativeClustering(n_clusters=k, linkage='complete', affinity='cosine')
        return clustering.fit_predict(embeddings)

    def _cluster_balance_metrics(labels, k=None, eps=1e-12):
        labs = np.asarray(labels, dtype=int)
        K = int(k if k is not None else (labs.max() + 1))
        counts = np.bincount(labs, minlength=K).astype(float)
        p = counts / (counts.sum() + eps)
        # normalized entropy in [0,1]; 1 == perfectly even
        H = -np.sum(p * np.log(p + eps)) / np.log(max(2, K))
        f_max = (counts.max() / max(1.0, counts.sum()))
        return float(H), float(f_max), counts.astype(int)

    # ---- Spectral clustering helpers (cosine affinity, optional k-NN pruning) ----
    def _build_cosine_affinity(E: np.ndarray, n_neighbors: int | None = 10, self_weight: float = 1.0) -> np.ndarray:
        """
        Build a nonnegative, symmetric cosine affinity matrix from L2-normalized embeddings.
        If n_neighbors is set, keeps a symmetric k-NN graph to sharpen the spectrum.
        """
        # Cosine sim == dot for unit vectors
        S = E @ E.T
        # Clamp to [-1,1], then keep only nonnegative weights (common in spectral graph use)
        S = np.clip(S, -1.0, 1.0)
        S = np.maximum(S, 0.0)

        n = S.shape[0]
        if n_neighbors is not None and 0 < n_neighbors < max(1, n - 1):
            A = np.zeros_like(S)
            # keep top-k neighbors per row (exclude self)
            order = np.argsort(S, axis=1)[:, ::-1]
            for i in range(n):
                tops = [j for j in order[i] if j != i][:n_neighbors]
                if tops:
                    A[i, tops] = S[i, tops]
            # symmetrize (max to preserve any one-sided strong tie)
            S = np.maximum(A, A.T)

        # Diagonal strength
        np.fill_diagonal(S, float(self_weight))
        return S

    def _spectral_labels(embeddings: np.ndarray, k: int, params: dict | None = None) -> np.ndarray:
        """Run SpectralClustering with a precomputed cosine affinity; returns labels."""
        n = len(embeddings)
        if n == 0:
            return np.array([], dtype=int)
        if k <= 1:
            return np.zeros(n, dtype=int)

        p = params or {}
        nn  = int(p.get("n_neighbors", 10))
        lab = str(p.get("assign_labels", "kmeans"))
        sw  = float(p.get("self_weight", 1.0))

        A = _build_cosine_affinity(embeddings, n_neighbors=nn, self_weight=sw)
        try:
            model = SpectralClustering(
                n_clusters=int(k),
                affinity="precomputed",
                assign_labels=lab,
                random_state=0
            )
        except TypeError:
            # older sklearn without random_state on SpectralClustering
            model = SpectralClustering(
                n_clusters=int(k),
                affinity="precomputed",
                assign_labels=lab
            )
        labels = model.fit_predict(A)
        return labels.astype(int)

    def bisecting_kmeans_spherical(X, k, min_gain=0.01, random_state=0):
        # start with all points in one cluster; repeatedly split the largest
        from sklearn.cluster import KMeans
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        labels = np.zeros(Xn.shape[0], dtype=int)
        K = 1
        rng = np.random.default_rng(random_state)
        while K < k:
            # pick the largest cluster to split
            sizes = np.bincount(labels, minlength=K)
            c = int(np.argmax(sizes))
            idx = np.where(labels == c)[0]
            if len(idx) < 4:     # too small to split stably
                break
            km = KMeans(n_clusters=2, n_init=10, random_state=rng.integers(1e9))
            sub = km.fit_predict(Xn[idx])
            # simple gain proxy: variance explained increase
            old_cent = Xn[idx].mean(axis=0)
            new_cent = np.vstack([Xn[idx][sub==0].mean(axis=0), Xn[idx][sub==1].mean(axis=0)])
            gain = float(np.linalg.norm(new_cent[0]-old_cent) + np.linalg.norm(new_cent[1]-old_cent))
            if gain < min_gain:
                break
            # relabel: one child keeps 'c', the other becomes new cluster id K
            child = (sub.max() if sizes[c] >= 2 else 0)
            labels[idx[sub == child]] = c
            labels[idx[sub != child]] = K
            K += 1
        return labels

    # Summary: compute_cluster_embeddings — mean vector per cluster.
    # Effect: makes cluster centroids to serve as consistent Δ endpoints.
    def compute_cluster_embeddings(embeddings, labels):
        """Compute mean embedding for each cluster."""
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(embeddings[idx])
        cluster_embeddings = {label: np.mean(vectors, axis=0) for label, vectors in clusters.items()}
        return cluster_embeddings

    # Summary: compute_delta_matrix — build Δ[i,j] = μ_j − μ_i and ordered labels.
    # Effect: encodes directed transformations (morphisms) between all cluster pairs.
    def compute_delta_matrix(cluster_embeddings):
        """Compute displacement vectors between each pair of cluster embeddings."""
        labels = sorted(cluster_embeddings.keys())
        n = len(labels)
        delta_matrix = np.zeros((n, n, len(next(iter(cluster_embeddings.values())))))
        E = np.vstack([cluster_embeddings[l] for l in labels])
        delta_matrix = E[None, :, :] - E[:, None, :]

        #for i in range(n):
        #    for j in range(n):
        #        delta_matrix[i][j] = cluster_embeddings[labels[j]] - cluster_embeddings[labels[i]]
        
        return delta_matrix, labels



    segments = segment_document(item_text)
    embeddings = embed_segments(segments)
    #embeddings = batched_encode(segments, batch_size=2048)
    #embeddings = remove_top_components(embeddings, n=1)   # try n=1..3
    #k = choose_k_via_silhouette(embeddings)
    k = choose_k_size_aware(
        embeddings,
        k_min=3, k_max=10,
        alpha=0.25,
        min_cluster_size=5,
        method=cluster_method,                 
        spectral_params=spectral_params        
    )
    labels = cluster_segments(
        embeddings, k,
        method=cluster_method,                 
        spectral_params=spectral_params        
    )
    #H, f_max, counts = _cluster_balance_metrics(labels, k)
    #if H < 0.55 or f_max > 0.85:
    #    labels = bisecting_kmeans_spherical(embeddings, k, min_gain=0.01, random_state=0)
    cluster_embs = compute_cluster_embeddings(embeddings, labels)
    #cluster_order = sorted(cluster_embs.keys())
    delta_matrix, cluster_order = compute_delta_matrix(cluster_embs)

    return delta_matrix, cluster_order, labels, segments, embeddings, k


# -----------------------------------------------------------------------------
# Function: build_cluster_delta_matrix
# Summary:
#   Clusters segments (agglomerative), aggregates cluster texts & mean
#   embeddings, computes per‑cluster LDA topic distributions, estimates a
#   principal direction (PC1) for each cluster (SVD on member embeddings,
#   oriented away from the doc mean), and then builds the Δ tensor aligned to
#   the cluster order. Returns the canonical 6‑tuple:
#   (Δ, cluster_order, raw_labels, topic_dists, cluster_embeddings[n×d], cluster_dirs[n×d]).
# Effect:
#   Produces the standardized per‑document artifact the rest of the pipeline
#   consumes, unifying *geometry* (Δ, centroids, PC1) with *semantics* (LDA
#   distributions) so morphisms can be matched, labeled, and analyzed.
# -----------------------------------------------------------------------------
def build_cluster_delta_matrix(segments, embeddings, labels, lda_model, lda_dictionary, num_topics=10, n_clusters: int = 5):
    # Step 1: Cluster segments using embeddings
    #clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    #labels = clustering.fit_predict(embeddings)

    # Step 2: Organize segments and embeddings by cluster
    cluster_texts = defaultdict(list)
    cluster_embeddings_raw = defaultdict(list)

    for i, label in enumerate(labels):
        cluster_texts[label].append(segments[i])
        cluster_embeddings_raw[label].append(embeddings[i])

    # Step 3: Compute mean embeddings and concatenated texts per cluster
    cluster_embeddings_dict = {}
    cluster_topic_distributions = {}

    for label in sorted(cluster_texts.keys()):
        text_blob = " ".join(cluster_texts[label])
        mean_embedding = np.mean(cluster_embeddings_raw[label], axis=0)
        cluster_embeddings_dict[label] = mean_embedding

        # Topic distribution using LDA
        bow = lda_dictionary.doc2bow(text_blob.lower().split())
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
        dist_vector = np.zeros(num_topics)
        for topic_id, prob in topic_dist:
            dist_vector[topic_id] = prob
        cluster_topic_distributions[label] = dist_vector
        
    # --- per-cluster principal directions (PC1) ---
    doc_mean = np.mean(embeddings, axis=0)
    cluster_principal_dirs = {}

    for label in sorted(cluster_texts.keys()):
        X = np.vstack(cluster_embeddings_raw[label])  # (n_i, d)
        Xc = X - X.mean(axis=0, keepdims=True)
        if Xc.shape[0] >= 2:
            # SVD: first right singular vector = principal direction
            # 
            try:
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                pc1 = Vt[0]
            except np.linalg.LinAlgError:
                pc1 = cluster_embeddings_dict[label] - doc_mean
        else:
            # Fallback when only 1 vector: use (cluster_mean - doc_mean)
            pc1 = cluster_embeddings_dict[label] - doc_mean

        # Orient consistently (point away from doc mean toward cluster mean)
        ref = cluster_embeddings_dict[label] - doc_mean
        if np.dot(pc1, ref) < 0:
            pc1 = -pc1

        # Normalize
        nrm = np.linalg.norm(pc1)
        pc1 = pc1 / nrm if nrm > 0 else pc1
        cluster_principal_dirs[label] = pc1
        
    # Convert to ordered array for visualization
    cluster_order = sorted(cluster_embeddings_dict.keys())
    cluster_embeddings = np.array([cluster_embeddings_dict[label] for label in cluster_order])
    
    # --- NEW: directions aligned to cluster_order ---
    cluster_dirs = np.array([cluster_principal_dirs[label] for label in cluster_order])

    # Step 4: Build delta matrix (pairwise deltas)
    n = len(cluster_order)
    d = embeddings[0].shape[0]
    delta_matrix = np.zeros((n, n, d))
    delta_matrix = cluster_embeddings[None, :, :] - cluster_embeddings[:, None, :]

    #for i in range(n):
    #    for j in range(n):
    #        delta_matrix[i, j, :] = cluster_embeddings[j] - cluster_embeddings[i]

    return (
        delta_matrix,                  # [n x n x d]
        cluster_order,                 # list of label integers
        labels,                        # original labels per segment
        cluster_topic_distributions,   # dict[label] → np.array[num_topics]
        cluster_embeddings,            # np.array[n x d]
        cluster_dirs                   # --- np.array[n x d], PC1 per cluster
    )


# -----------------------------------------------------------------------------
# Function: preprocess_text
# Summary:
#   Tokenizes and normalizes text with Gensim’s filters (strip punctuation,
#   numbers, stop‑words; drop very short tokens).
# Effect:
#   Ensures consistent lexical inputs for topic models and keyword displays used
#   when interpreting morphisms.
# -----------------------------------------------------------------------------
def preprocess_text(text: str) -> List[str]:
    # Define consistent preprocessing filters
    CUSTOM_FILTERS = [
        strip_punctuation,
        strip_numeric,
        remove_stopwords,
        strip_short
    ]
    """Preprocess a single LCSH string with custom filters."""
    return preprocess_string(text, CUSTOM_FILTERS)


# -----------------------------------------------------------------------------
# Function: assess_topic_frequency_across_clusters
# Summary:
#   Tallies the most probable topic (argmax) per cluster and prints a frequency
#   count across clusters for a quick sanity check.
# Effect:
#   Fast diagnostic of topical redundancy/diversity before deeper morphism work.
# -----------------------------------------------------------------------------
def assess_topic_frequency_across_clusters(cluster_topic_distributions):
    topic_ids = []
    for dist_vector in cluster_topic_distributions.values():
        topic_ids.append(int(np.argmax(dist_vector)))

    print("Top topic frequency across clusters:", Counter(topic_ids))


# -----------------------------------------------------------------------------
# Function: build_topic_embeddings_for_doc
# Summary:
#   Builds topic‑anchored vectors by weighting each segment’s embedding by its
#   LDA topic probabilities and averaging per topic; returns [num_topics × d].
# Effect:
#   Bridges symbolic topics and the embedding space, letting Δ directions be
#   interpreted as movements toward/away from topic vectors.
# -----------------------------------------------------------------------------
def build_topic_embeddings_for_doc(lda_model, dictionary, segments, embeddings, num_topics):
    """
    Builds [k x d] topic embeddings for a single document.
    """
    d = embeddings[0].shape[0]
    topic_vectors = np.zeros((num_topics, d))
    topic_weights = np.zeros(num_topics)
    
    for seg_text, seg_emb in zip(segments, embeddings):
        bow = dictionary.doc2bow(seg_text.lower().split())
        topic_probs = lda_model.get_document_topics(bow, minimum_probability=0.0)
        
        for topic_id, prob in topic_probs:
            topic_vectors[topic_id] += prob * seg_emb
            topic_weights[topic_id] += prob
    
    # Normalize by total weight per topic
    for t in range(num_topics):
        if topic_weights[t] > 0:
            topic_vectors[t] /= topic_weights[t]
    
    return topic_vectors


# -----------------------------------------------------------------------------
# Function: interpret_direction
# Summary:
#   Interprets a full‑dimensional direction vector d by: (1) contrasting dst
#   vs src topic probabilities to list topics that increase/decrease; and
#   (2) projecting candidate words onto d (via embeddings) to list
#   words aligned with +d and −d.
# Effect:
#   Turns opaque geometry into human‑readable “topic flow” and lexical cues,
#   enabling principled labeling of morphisms and nodes in higher‑level graphs.
# -----------------------------------------------------------------------------
def interpret_direction(
    dir_full: np.ndarray,
    src_topic_probs: np.ndarray,
    dst_topic_probs: np.ndarray,
    lda_model,
    lda_labels: list[str] | None,
    embed_fn,                      # e.g., lambda texts: model.encode(texts, normalize_embeddings=True)
    candidate_words: list[str],
    top_n_topics: int = 3,
    top_m_keywords: int = 5,
    top_k_words: int = 12
) -> dict:
    """
    Produce an interpretable summary of a semantic direction vector.
    Returns a dict with topic-flow, lexical projections, and (optionally) topic labels.
    """
    # 1) Topic-flow
    d_tau = dst_topic_probs - src_topic_probs
    up_ids = np.argsort(d_tau)[::-1][:top_n_topics]
    down_ids = np.argsort(d_tau)[:top_n_topics]
    def topic_block(ids):
        rows = []
        for tid in ids:
            label = f"T{tid}" if lda_labels is None or tid >= len(lda_labels) else f"T{tid}: {lda_labels[tid]}"
            try:
                twords = [w for (w, _) in lda_model.show_topic(int(tid), topn=top_m_keywords)]
            except Exception:
                twords = []
            rows.append((int(tid), float(d_tau[tid]), label, twords))
        return rows

    topics_up   = topic_block(up_ids)
    topics_down = topic_block(down_ids)

    # 2) Lexical projection (positive/negative)
    cw = list(dict.fromkeys([w for w in candidate_words if w and w.isalpha()]))  # dedupe, simple filter
    W = embed_fn(cw)                          # shape: [V, d], normalized
    d = dir_full / (np.linalg.norm(dir_full) + 1e-12)
    scores = W @ d
    pos_idx = np.argsort(scores)[::-1][:top_k_words]
    neg_idx = np.argsort(scores)[:top_k_words]
    words_pos = [(cw[i], float(scores[i])) for i in pos_idx]
    words_neg = [(cw[i], float(scores[i])) for i in neg_idx]

    return {
        "topic_flow_up": topics_up,           # (topic_id, delta, label, top_words)
        "topic_flow_down": topics_down,
        "words_pos": words_pos,               # words aligned with +d
        "words_neg": words_neg,               # words aligned with -d
    }


# -----------------------------------------------------------------------------
# Function: visualize_documents_with_directional_overlap
# Summary:
#   Builds a 3D, interactive matplotlib scene of all Δ vectors from every
#   document: shows document convex hulls and centroids (with LDA hover text),
#   draws Δ quivers (red if they match another doc by cosine), overlays per‑
#   cluster PC1 arrows when available, lets users click a Δ to highlight all
#   cross‑document matches, and opens a details table that also includes
#   `interpret_direction` semantics. Sliders let you re‑threshold Δ and PC1.
# Effect:
#   Provides an exploratory interface to *morphism alignment*—revealing shared
#   semantic transitions across documents, surfacing candidate thematic links,
#   and grounding arrangement/labeling decisions.
# -----------------------------------------------------------------------------
def visualize_documents_with_directional_overlap(
    document_cluster_data: dict,
    lda_model=None,
    top_n_topics: int = 3,
    top_m_keywords: int = 3,
    cos_threshold: float = 0.9,
    lda_int_topics_list=None,
    srcdst_threshold: float = None   # NEW: defaults to cos_threshold if None
):
    if srcdst_threshold is None:
        srcdst_threshold = cos_threshold
    
    # Keep current thresholds in one place
    current_thr = {"delta": float(cos_threshold), "pc1": float(srcdst_threshold)}
    
    """
    3D visualization of directional overlap between document delta manifolds, with:
      - Colored convex hulls per document (toggleable)
      - Cluster points with LDA hover labels
      - Delta vectors: red if directionally similar (cosine ≥ cos_threshold) to any delta in another doc, else olivedrab
      - Click-to-select on deltas: highlights the selected delta + all directionally similar deltas
      - 'Restore Styles' button to reset delta styles survey
      - Centers view on data centroid
      - NEW: Opens/updates a separate window with a table of descriptive fields for the selected group
    """
    
    timer2 = Timer()
    timer2.start()
    def _unpack_data(data):
        """
        Accept 5- or 6-tuple document data.
        Returns: delta_matrix, cluster_order, labels, cluster_topic_distributions, cluster_embeddings_arr, cluster_dirs_or_None
        Also normalizes cluster_embeddings to a 2D np.ndarray aligned to cluster_order.
        """
        # Summary: _unpack_data — normalize 5‑ or 6‑tuples into aligned arrays.
        # Effect: makes viz robust to legacy tuple shapes.
        if len(data) >= 6:
            delta_matrix, cluster_order, labels, cluster_topic_distributions, cluster_embeddings, cluster_dirs = data[:6]
        else:
            delta_matrix, cluster_order, labels, cluster_topic_distributions, cluster_embeddings = data[:5]
            cluster_dirs = None

        # Normalize cluster_embeddings to ndarray aligned with cluster_order
        if isinstance(cluster_embeddings, np.ndarray):
            emb_arr = cluster_embeddings
        elif isinstance(cluster_embeddings, dict):
            # Older path: dict keyed by label
            emb_arr = np.vstack([cluster_embeddings[label] for label in cluster_order])
        else:
            # List-like fallback
            emb_arr = np.asarray(cluster_embeddings)

        # Ensure 2D
        if emb_arr.ndim == 1:
            emb_arr = emb_arr[None, :]

        return delta_matrix, cluster_order, labels, cluster_topic_distributions, emb_arr, cluster_dirs
        
    # Palette per document
    doc_ids = list(document_cluster_data.keys())
    colormap = cm.get_cmap('tab20', len(doc_ids))
    doc_colors = {doc: colormap(i) for i, doc in enumerate(doc_ids)}

# 1) Collect all deltas across docs (+ attach cluster principal directions)
    all_entries = []  # dicts: {doc, start3, delta3, dir_full, src_dir_full, dst_dir_full, i_idx, j_idx, i_lab, j_lab}
    for doc_id, data in document_cluster_data.items():

        delta_matrix, cluster_order, _, _, cluster_embeddings, cluster_dirs = _unpack_data(data)

        # Use correct 3D points (aligned with cluster_order)
        emb3d = cluster_embeddings[:, :3]  # rows already aligned to cluster_order

        n, d = cluster_embeddings.shape
        doc_has_dirs = (isinstance(cluster_dirs, np.ndarray) and cluster_dirs.shape == (n, d))

        # Pre-normalize cluster principal dirs if present
        if doc_has_dirs:
            # guard against zero norms
            norms = np.linalg.norm(cluster_dirs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            cluster_dirs_unit = cluster_dirs / norms
        else:
            cluster_dirs_unit = None

        # Build entries for all pairwise deltas
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                start3 = emb3d[i]
                delta3 = delta_matrix[i, j, :3]

                # FULL-dim direction for robust cosine matching
                dfull = delta_matrix[i, j, :]
                nrm = np.linalg.norm(dfull)
                if nrm <= 0:
                    continue
                dir_full = dfull / nrm

                # Attach source/dest cluster principal directions (full dim) if available
                if doc_has_dirs:
                    src_dir_full = cluster_dirs_unit[i]
                    dst_dir_full = cluster_dirs_unit[j]
                else:
                    src_dir_full = None
                    dst_dir_full = None

                all_entries.append({
                    "doc": doc_id,
                    "start3": start3,
                    "delta3": delta3,
                    "dir_full": dir_full,
                    "delta_full": dfull,
                    "src_dir_full": src_dir_full,
                    "dst_dir_full": dst_dir_full,
                    "i_idx": i,
                    "j_idx": j,
                    "i_lab": cluster_order[i],
                    "j_lab": cluster_order[j],
                })

    if not all_entries:
        print("No delta entries to visualize.")
        return

    docs = np.array([e["doc"] for e in all_entries], dtype=object)

    dirs_full = np.vstack([e["dir_full"] for e in all_entries])  # (M, d)
    delta_sims = dirs_full @ dirs_full.T  # unit → dot == cosine

    # If any document lacks cluster dirs, fall back to delta-only criterion
    have_srcdst = all(e["src_dir_full"] is not None and e["dst_dir_full"] is not None for e in all_entries)
    if have_srcdst:
        src_dirs_full = np.vstack([e["src_dir_full"] for e in all_entries])  # (M, d)
        dst_dirs_full = np.vstack([e["dst_dir_full"] for e in all_entries])  # (M, d)
        src_sims = np.abs(src_dirs_full @ src_dirs_full.T)
        dst_sims = np.abs(dst_dirs_full @ dst_dirs_full.T)

    neighbors = []
    overlap_mask = np.zeros(len(all_entries), dtype=bool)
    for i in range(len(all_entries)):
        cross_doc = (docs != docs[i])
        if have_srcdst:
            mask = (delta_sims[i] >= current_thr["delta"]) & (src_sims[i] >= current_thr["pc1"]) & (dst_sims[i] >= current_thr["pc1"]) & cross_doc
        else:
            mask = (delta_sims[i] >= current_thr["delta"]) & cross_doc
        nbrs = np.where(mask)[0].tolist()
        neighbors.append(nbrs)
        overlap_mask[i] = len(nbrs) > 0

    # --- diagnostics: where are the cosines across docs? ---
    print("")
    print("--- Document Concordance Survey ---")
    cross = (docs[:, None] != docs[None, :])
    vals_delta = delta_sims[cross]
    q = np.quantile(vals_delta, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    print("[Δ cos] min,25,50,75,90,95,99,max:", np.array2string(q, precision=3))

    if have_srcdst:
        vals_src = src_sims[cross];  vals_dst = dst_sims[cross]
        q_src = np.quantile(vals_src, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        q_dst = np.quantile(vals_dst, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        print("[PC1 cos] SRC  quantiles:", np.array2string(q_src, precision=3))
        print("[PC1 cos] DEST quantiles:", np.array2string(q_dst, precision=3))
    print("")
    
    # 2) Plot hulls + cluster points
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    selected_text = ax.text2D(
        0.02, 0.98, "", transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(fc="lavender", alpha=0.95),
        visible=False
    )
    hull_artists = []        # store hull collections for toggling
    all_scatter_pts = []     # cluster point artists for hover
    hover_texts_pts = []     # LDA topic hover text
    all_coords = []          # for centering
    
    # NEW: eigenvector artists (per-cluster PC1 arrows)
    eigen_artists = []          # flattened list of all eigenvector quivers (across docs)
    eigen_base_colors = []      # base face colors for restore
    eigen_base_alphas = []      # base alphas for restore
    eigen_base_lws = []         # base linewidths for restore
    eig_artist_map = {}         # key: (doc_id, ordered_cluster_index) -> quiver artist

    for doc_id, data in document_cluster_data.items():
        delta_matrix, cluster_order, _, cluster_topic_distributions, cluster_embeddings, cluster_dirs = _unpack_data(data)
        color = doc_colors[doc_id]
        emb3d = cluster_embeddings[:, :3]

        # Scale for eigenvector arrow length (10–12% of doc extent)
        doc_extent = (emb3d.max(axis=0) - emb3d.min(axis=0)).max()
        eig_scale = 0.12 * doc_extent if doc_extent > 0 else 1.0

        # Convex hull (one collection per doc)
        try:
            hull = ConvexHull(emb3d)
            verts = [emb3d[s] for s in hull.simplices]
            poly = Poly3DCollection(verts, alpha=0.2, facecolor=color)
            poly.set_edgecolor('k')
            ax.add_collection3d(poly)
            hull_artists.append(poly)
        except Exception:
            # too few/co-planar points
            pass

        # Cluster centroids + LDA hover labels
        for idx, label in enumerate(cluster_order):
            x, y, z = emb3d[idx]
            all_coords.append((x, y, z))
            sc = ax.scatter(x, y, z, color=color, s=30)
            all_scatter_pts.append(sc)

            if lda_model is not None:
                dist = cluster_topic_distributions.get(label)
                if dist is not None:
                    top_ids = np.argsort(dist)[::-1][:top_n_topics]
                    blocks = []
                    for tid in top_ids:
                        w = dist[tid]
                        kws = [w_ for (w_, _) in lda_model.show_topic(tid, topn=top_m_keywords)]
                        blocks.append(f"T{tid}: ({lda_int_topics_list[tid]}) ({w:.3f}): " + ", ".join(kws))
                    txt = "\n".join(blocks)
                else:
                    txt = f"{doc_id}_C{label} (no topic dist)"
            else:
                txt = f"{doc_id}_C{label}"

            # Append PC1 (3D) to the hover label if available
            if (cluster_dirs is not None and
                isinstance(cluster_dirs, np.ndarray) and
                cluster_dirs.shape[0] == cluster_embeddings.shape[0]):

                pc1_3d = cluster_dirs[idx][:3].astype(float)
                nrm3 = np.linalg.norm(pc1_3d)
                if nrm3 > 0:
                    pc1_3d = pc1_3d / nrm3
                txt = txt + f"\nPC1≈({pc1_3d[0]:.2f}, {pc1_3d[1]:.2f}, {pc1_3d[2]:.2f})"

            hover_texts_pts.append(txt)
            ax.text(x, y, z, f"{doc_id}_C{label}", fontsize=8, color='black')
            # --- NEW: plot eigenvector (PC1) at this cluster centroid, if available ---
            if (cluster_dirs is not None and
                isinstance(cluster_dirs, np.ndarray) and
                cluster_dirs.shape[0] == cluster_embeddings.shape[0]):

                d3 = cluster_dirs[idx][:3].astype(float)
                nrm = np.linalg.norm(d3)
                if nrm > 0:
                    d3 = (d3 / nrm) * eig_scale
                    ev = ax.quiver(x, y, z, d3[0], d3[1], d3[2],
                                   color=color, alpha=0.8, linewidth=1.5)
                    eigen_artists.append(ev)
                    eigen_base_colors.append(color)
                    eigen_base_alphas.append(0.8)
                    eigen_base_lws.append(1.5)
                    # Map by ordered index (idx), not raw label
                    eig_artist_map[(doc_id, idx)] = ev

    # 3) Plot deltas; store artists & base styles for manual restore
    delta_artists = []
    base_colors = []
    base_alphas = []
    base_lws = []

    for idx, e in enumerate(all_entries):
        x, y, z = e["start3"]
        dx, dy, dz = e["delta3"]
        base_color = 'red' if overlap_mask[idx] else 'olivedrab'
        art = ax.quiver(x, y, z, dx, dy, dz,
                        color=base_color, alpha=0.6, linewidth=1, picker=True)
        delta_artists.append(art)
        base_colors.append(base_color)
        base_alphas.append(0.6)
        base_lws.append(1.0)

    # 4) Center view on data centroid
    coords = np.array(all_coords)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    # 5) Legend
    patches = [Patch(facecolor=doc_colors[doc], edgecolor='k', label=doc) for doc in doc_ids]
    ax.legend(handles=patches, loc='best')

    # ==================== Helpers functions ====================
    
    # Summary: _set_linewidth — set line width for different artist types.
    # Effect: consistent highlighting and reset behavior.
    def _set_linewidth(artist, lw):
        if hasattr(artist, "set_linewidths"):
            artist.set_linewidths(lw)
        else:
            try:
                artist.set_linewidth(lw)
            except Exception:
                pass

    # Summary: _restore_base_styles — restore original colors/alphas/line widths.
    # Effect: quick un‑select and clean reset for the scene.
    def _restore_base_styles():
        # deltas
        for a, c, al, lw in zip(delta_artists, base_colors, base_alphas, base_lws):
            a.set_color(c)
            a.set_alpha(al)
            _set_linewidth(a, lw)
        # eigenvectors
        for a, c, al, lw in zip(eigen_artists, eigen_base_colors, eigen_base_alphas, eigen_base_lws):
            try:
                a.set_color(c)
            except Exception:
                pass
            a.set_alpha(al)
            _set_linewidth(a, lw)
        fig.canvas.draw_idle()

    # --- DEPRECIATED:  ---
    # Summary: _apply_srcdst_threshold — (deprecated) older path to recolor by PC1.
    # Effect: retained for compatibility; superseded by _recompute_matches.
    def _apply_srcdst_threshold(new_thr):
        # If any doc lacks cluster PC1 vectors, PC1 criterion is not active
        if not have_srcdst:
            return
        t = float(new_thr)

        # Recompute neighbor lists and overlap mask using precomputed cos matrices
        for i in range(len(all_entries)):
            cross_doc = (docs != docs[i])
            mask = (delta_sims[i] >= current_thr["delta"]) & (src_sims[i] >= t) & (dst_sims[i] >= t) & cross_doc
            nbrs = np.where(mask)[0].tolist()
            neighbors[i] = nbrs
            overlap_mask[i] = (len(nbrs) > 0)

        # Update base colors + visible colors: red if has matches, else olivedrab
        for k, a in enumerate(delta_artists):
            c = 'red' if overlap_mask[k] else 'olivedrab'
            a.set_color(c)
            a.set_alpha(0.6)
            _set_linewidth(a, 1.0)
            base_colors[k] = c  # so "Restore Styles" reflects the new base coloring

        # Clear any selection overlay and close the details table if open
        try:
            selected_text.set_visible(False)    # present in v5; guard just in case
        except Exception:
            pass
        if details_state["fig"] is not None and plt.fignum_exists(details_state["fig"].number):
            try:
                plt.close(details_state["fig"])
            except Exception:
                pass
            details_state["fig"] = None
            details_state["ax"]  = None

        fig.canvas.draw_idle()

    # --- recompute matches & recolor deltas when PC1 src/dst threshold changes ---
    # Summary: _recompute_matches — recompute neighbor sets using current thresholds
    #          (Δ and PC1 at src/dst independently) and recolor all Δ accordingly.
    # Effect: interactive control over the notion of “directional overlap.”
    def _recompute_matches(delta_thr=None, pc1_thr=None):
        """Recompute neighbor sets and recolor deltas based on thresholds."""
        if delta_thr is not None:
            current_thr["delta"] = float(delta_thr)
        if have_srcdst and (pc1_thr is not None):
            current_thr["pc1"] = float(pc1_thr)

        for i in range(len(all_entries)):
            cross_doc = (docs != docs[i])
            if have_srcdst:
                mask = (delta_sims[i] >= current_thr["delta"]) \
                       & (src_sims[i] >= current_thr["pc1"]) \
                       & (dst_sims[i] >= current_thr["pc1"]) \
                       & cross_doc
            else:
                mask = (delta_sims[i] >= current_thr["delta"]) & cross_doc
            neighbors[i]    = np.where(mask)[0].tolist()
            overlap_mask[i] = len(neighbors[i]) > 0

        # recolor lines (red=has matches, olivedrab=no matches)
        for k, a in enumerate(delta_artists):
            c = 'red' if overlap_mask[k] else 'olivedrab'
            a.set_color(c); a.set_alpha(0.6)
            if hasattr(a, "set_linewidths"): a.set_linewidths(1.0)
            else: a.set_linewidth(1.0)
            base_colors[k] = c

        # hide selection overlay & close details window if open
        try:
            selected_text.set_visible(False)
        except Exception:
            pass
        if details_state.get("fig") is not None and plt.fignum_exists(details_state["fig"].number):
            plt.close(details_state["fig"])
            details_state["fig"] = None; details_state["ax"] = None

        fig.canvas.draw_idle()

    # ---  embed helper for interpret_direction() ---
    # Determine the embedding dimensionality from the first delta direction
    _dim_d = len(all_entries[0]["dir_full"])

    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Summary: _embed_texts — embed candidate words (Sentence‑BERT) for d·w scoring.
        # Effect: enables lexical projection in interpretive details.
        def _embed_texts(texts: list[str]) -> np.ndarray:
            # shape: [len(texts), d], normalized
            return _st_model.encode(texts, normalize_embeddings=True)
    except Exception:
        # Safe fallback: keep dimensions consistent; scores will be 0 → semantics block won’t crash
        def _embed_texts(texts: list[str]) -> np.ndarray:
            return np.zeros((len(texts), _dim_d), dtype=float)
    # --- end embed helper ---
    
    # ---  collect candidate words for a specific delta entry ---
    # Summary: _candidate_words_for_entry — gather top words from src/dst topics.
    # Effect: constrains lexical interpretation to local topical neighborhoods.
    def _candidate_words_for_entry(e, n_topics: int, per_topic_words: int) -> list[str]:
        """
        Build a compact candidate vocabulary for lexical projection by unioning
        top terms from the top-N topics of the src and dst clusters.
        """
        if lda_model is None:
            return []
        try:
            dist_src = document_cluster_data[e["doc"]][3].get(e["i_lab"])
            dist_dst = document_cluster_data[e["doc"]][3].get(e["j_lab"])
        except Exception:
            return []

        if dist_src is None or dist_dst is None:
            return []

        # top topic ids for src/dst
        src_top = np.argsort(dist_src)[::-1][:n_topics]
        dst_top = np.argsort(dist_dst)[::-1][:n_topics]

        words = []
        for tid in list(dict.fromkeys(list(src_top) + list(dst_top))):
            try:
                kws = [w for (w, _) in lda_model.show_topic(int(tid), topn=per_topic_words)]
                words.extend(kws)
            except Exception:
                pass

        # de-dup while preserving order
        return list(dict.fromkeys(words))
    # --- end candidate words helper ---    
    
    

    # =============== details window (table) ===============
    details_state = {"fig": None, "ax": None}
    # Summary: _topic_block_for — format top‑N topics/weights/keywords for one cluster.
    # Effect: compact, human‑legible hover/details content.
    def _topic_block_for(doc_id: str, cluster_label, n_topics=3, n_words=3) -> str:
        """
        Return a multi-line string of the top-N topics (with weights) for the given cluster:
          T<id> (<weight>): kw1, kw2, kw3
          ...
        Falls back to "" if lda_model or distribution is unavailable.
        """
        if lda_model is None:
            return ""
        try:
            dist = document_cluster_data[doc_id][3].get(cluster_label)
        except Exception:
            return ""
        if dist is None:
            return ""

        # Top-N topics by probability
        top_ids = np.argsort(dist)[::-1][:n_topics]
        lines = []
        for tid in top_ids:
            w = float(dist[int(tid)])
            try:
                words = [w_ for (w_, _) in lda_model.show_topic(int(tid), topn=n_words)]
            except Exception:
                words = []
            lines.append(f"T{int(tid)}: ({lda_int_topics_list[tid]}) ({w:.3f}): " + ", ".join(words))
        return "\n".join(lines)

    # --- DROP-IN: details table with Δ-semantics via interpret_direction() ---
    # Summary: _format_interpretation_summary — render interpret_direction output.
    # Effect: readable ↑/↓ topic and ±word summaries per Δ.
    def _format_interpretation_summary(ir: dict, max_topics: int = 2, max_words: int = 4) -> str:
        """
        Turn interpret_direction() output into a compact multiline string.
        """
        def _fmt_topic_rows(rows):
            out = []
            for (tid, delta_w, label, twords) in rows[:max_topics]:
                # label already contains "T<id>: <name>" if you passed lda_labels
                twords_str = ", ".join(twords[:max_words]) if twords else ""
                out.append(f"{label} Δ={delta_w:+.3f} [{twords_str}]")
            return "\n".join(out)

        up   = _fmt_topic_rows(ir.get("topic_flow_up", []))
        down = _fmt_topic_rows(ir.get("topic_flow_down", []))

        wpos = ", ".join([f"{w}({s:+.2f})" for (w, s) in ir.get("words_pos", [])[:max_words]])
        wneg = ", ".join([f"{w}({s:+.2f})" for (w, s) in ir.get("words_neg", [])[:max_words]])

        blocks = []
        if up:   blocks.append("↑ topics:\n" + up)
        if down: blocks.append("↓ topics:\n" + down)
        if wpos: blocks.append("+words: " + wpos)
        if wneg: blocks.append("-words: " + wneg)
        return "\n".join(blocks) if blocks else "—"
        
    # Summary: _show_details_table — open/update a side window listing the selected
    #          Δ and all its matches with Δ/PC1 scores and interpretive text.
    # Effect: durable, export‑like view that pairs numbers with semantics.
    def _show_details_table(selected_idx: int, group_indices: list[int]):
        """
        Create or update a separate window with a table for the selected delta and
        all directionally matching deltas. Includes Δ-semantics derived from interpret_direction().
        """
        # Build rows
        rows = []
        headers = [
            "★", "doc", "from", "to", "|Δ|", "cos→sel",
            "src_PC1(3D)", "dst_PC1(3D)",
            "src_topics: Number, Inferred category, Weight over cluster, Top topic terms",
            "dst_topics: Number, Inferred category, Weight over cluster, Top topic terms",
            "Δ semantics (topics↑/↓; words±)"
        ]

        sel_dir = all_entries[selected_idx]["dir_full"]

        for g in group_indices:
            e = all_entries[g]
            length = float(np.linalg.norm(e["delta3"]))
            cos_to_sel = float(np.dot(sel_dir, e["dir_full"]))

            # PC1 pretty print (if present)
            def _pc1_str(vec):
                if vec is None:
                    return "—"
                v3 = np.asarray(vec[:3], dtype=float)
                n  = np.linalg.norm(v3)
                if n > 0:
                    v3 = v3 / n
                return f"({v3[0]:.2f}, {v3[1]:.2f}, {v3[2]:.2f})"

            src_pc1 = _pc1_str(e.get("src_dir_full"))
            dst_pc1 = _pc1_str(e.get("dst_dir_full"))

            # topic blocks (you already had these helpers)
            src_block = _topic_block_for(e["doc"], e["i_lab"], n_topics=top_n_topics, n_words=top_m_keywords)
            dst_block = _topic_block_for(e["doc"], e["j_lab"], n_topics=top_n_topics, n_words=top_m_keywords)

            # ---------- NEW: Δ-semantics via interpret_direction ----------
            try:
                # 1) fetch probability vectors for src/dst clusters
                dist_src = document_cluster_data[e["doc"]][3].get(e["i_lab"])
                dist_dst = document_cluster_data[e["doc"]][3].get(e["j_lab"])

                # 2) pick candidate words from the src/dst topic neighborhoods
                cand_words = _candidate_words_for_entry(e, n_topics=top_n_topics, per_topic_words=max(8, top_m_keywords))

                # 3) call your interpret_direction()
                ir = interpret_direction(
                    dir_full       = e["dir_full"],
                    src_topic_probs= dist_src,
                    dst_topic_probs= dist_dst,
                    lda_model      = lda_model,
                    lda_labels     = lda_int_topics_list,     # optional labels list you pass to the viz function
                    embed_fn       = _embed_texts,            # Snippet A
                    candidate_words= cand_words,
                    top_n_topics   = top_n_topics,
                    top_m_keywords = top_m_keywords,
                    top_k_words    = 12
                )
                delta_sem_txt = _format_interpretation_summary(ir, max_topics=2, max_words=4)
            except Exception as ex:
                delta_sem_txt = f"(interpretation error: {ex})"
            # -------------------------------------------------------------

            rows.append([
                "★" if g == selected_idx else "",
                e["doc"],
                f"C{e['i_lab']}",
                f"C{e['j_lab']}",
                f"{length:.3f}",
                f"{cos_to_sel:.3f}",
                src_pc1,
                dst_pc1,
                src_block,
                dst_block,
                delta_sem_txt
            ])

        # Create or refresh figure
        if details_state["fig"] is None or not plt.fignum_exists(details_state["fig"].number):
            details_state["fig"] = plt.figure(figsize=(12, 0.7 + 0.50 * max(3, len(rows))))
            details_state["ax"]  = details_state["fig"].add_subplot(111)
        else:
            details_state["fig"].clf()
            details_state["ax"] = details_state["fig"].add_subplot(111)

        details_state["ax"].axis('off')
        table = details_state["ax"].table(
            cellText=rows,
            colLabels=headers,
            loc='center',
            cellLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col=list(range(len(headers))))
        details_state["fig"].suptitle(
            "Cluster Pairs With Similar Transform Directionality\n"
            "and PC1 Concordance Across SRC→SRC and DST→DST Clusters",
            fontsize=12
        )
        details_state["fig"].tight_layout()
        details_state["fig"].canvas.draw_idle()
        plt.show(block=False)
    # --- end details table with Δ-semantics ---

    # ---------------- Hover: cluster points ----------------
    cursor_pts = mplcursors.cursor(all_scatter_pts, hover=True, annotation_kwargs=dict(arrowprops=None))

    @cursor_pts.connect("add")
    # Summary: on_add_point — hover handler for cluster points; sets annotated text.
    # Effect: discoverability of cluster context while browsing.
    def on_add_point(sel):
        artist = sel.artist
        idx = all_scatter_pts.index(artist)
        ann = getattr(sel, "annotation", None)
        if ann is not None:
            ann.set_text(hover_texts_pts[idx])
            bbox = ann.get_bbox_patch()
            if bbox is not None:
                bbox.set(fc="lightyellow", alpha=0.9)

    # ---------------- Click-select: delta vectors (persistent highlight) ----------------
    cursor_deltas = mplcursors.cursor(delta_artists, hover=False, multiple=False, annotation_kwargs=dict(arrowprops=None))

    @cursor_deltas.connect("add")
    # Summary: on_select_delta — click handler; highlights one Δ and its matches,
    #          dims others, shows PC1 overlays, and populates the details table.
    # Effect: focuses attention on a single morphism “family” across documents.
    def on_select_delta(sel):
        # Which delta?
        artist = sel.artist
        try:
            idx = delta_artists.index(artist)
        except ValueError:
            return

        # Group: selected + all directional matches (other docs) above threshold
        group = [idx] + neighbors[idx]

        # Reset first, then dim non-group and highlight group
        _restore_base_styles()
        for k, a in enumerate(delta_artists):
            if k not in group:
                a.set_alpha(0.05)
                _set_linewidth(a, 0.25)
        for g in group:
            a = delta_artists[g]
            a.set_alpha(1.0)
            _set_linewidth(a, 2.5)
            a.set_color('yellow')
        # --- NEW: dim all eigenvectors, then highlight those belonging to group deltas ---
        for ev in eigen_artists:
            ev.set_alpha(0.05)
            _set_linewidth(ev, 0.25)

        for g in group:
            eg = all_entries[g]
            # Each delta has a source and destination cluster index within its doc
            key_src = (eg["doc"], eg["i_idx"])
            key_dst = (eg["doc"], eg["j_idx"])
            for key in (key_src, key_dst):
                ev = eig_artist_map.get(key)
                if ev is not None:
                    ev.set_alpha(1.0)
                    # slightly thicker for the selected delta's own endpoints
                    _set_linewidth(ev, 3.0 if g == idx else 2.0)
                    # keep doc color; you could set a highlight color if desired:
                    # ev.set_color('orange' if g != idx else 'red')

        # Annotation for selected delta + list of matching deltas
        e = all_entries[idx]
        length = float(np.linalg.norm(e["delta3"]))
        matched = neighbors[idx]
        max_lines = 8
        if matched:
            lines = [
                f"• {all_entries[m]['doc']}: C{all_entries[m]['i_lab']}→C{all_entries[m]['j_lab']}"
                for m in matched[:max_lines]
            ]
            more = len(matched) - max_lines
            if more > 0:
                lines.append(f"  (+{more} more)")
            match_block = "\n" + "\n".join(lines)
        else:
            match_block = "\n(no directional matches)"
            
        # Show the text in our 2D overlay (robust in 3D)
        # Append PC1 lines if available
        src_pc1 = e.get("src_dir_full")
        dst_pc1 = e.get("dst_dir_full")
        pc1_lines = ""
        if src_pc1 is not None and dst_pc1 is not None:
            sp = np.asarray(src_pc1[:3], dtype=float)
            dp = np.asarray(dst_pc1[:3], dtype=float)
            if np.linalg.norm(sp) > 0: sp = sp / np.linalg.norm(sp)
            if np.linalg.norm(dp) > 0: dp = dp / np.linalg.norm(dp)
            pc1_lines = (
                f"\nsrc_PC1≈({sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f})  "
                f"dst_PC1≈({dp[0]:.2f}, {dp[1]:.2f}, {dp[2]:.2f})"
            )

        selected_text.set_text(
            f"{e['doc']}: C{e['i_lab']}→C{e['j_lab']}\n"
            f"‖Δ‖={length:.3f}, matches={len(matched)}{pc1_lines}{match_block}"
        )

        selected_text.set_visible(True)
        fig.canvas.draw_idle()
        
        try:
            _show_details_table(idx, group)
        except Exception as ex:
            print(f"[details] error: {ex}")

    # ---------------- Buttons: Toggle hulls + Restore styles ----------------
    hulls_visible = [True]  # mutable in closure

    ax_btn_hulls = fig.add_axes([0.02, 0.92, 0.10, 0.03])   # [left, bottom, width, height]
    btn_hulls = Button(ax_btn_hulls, 'Toggle Hulls')
    
    # Summary: on_toggle_hulls — show/hide document convex hull surfaces.
    # Effect: declutter or contextualize geometry as needed.
    def on_toggle_hulls(event):
        hulls_visible[0] = not hulls_visible[0]
        for poly in hull_artists:
            poly.set_visible(hulls_visible[0])
        fig.canvas.draw_idle()

    btn_hulls.on_clicked(on_toggle_hulls)

    ax_btn_restore = fig.add_axes([0.12, 0.92, 0.12, 0.03])
    btn_restore = Button(ax_btn_restore, 'Restore Styles')

    # Summary: on_restore_styles — clear selection and restore default styling.
    # Effect: quick return to baseline view.
    def on_restore_styles(event):
        _restore_base_styles()

    btn_restore.on_clicked(on_restore_styles)

    # ------------------------------------------------------------------------
    ax.set_title("Visualizing Inter-Document Feature Manifold Similarity,\nThrough Intra-Document Transform Morphisms")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    timer2.stop()
    print(f"Total elapsed time to render visualization: {int(timer2.elapsed())//60}:min {int(timer2.elapsed())%60}:sec")
    # Current view (for reset)
    init_elev, init_azim = ax.elev, ax.azim

    # Sliders for azimuth and elevation
    ax_slider_az = fig.add_axes([0.78, 0.92, 0.20, 0.03])  # [left, bottom, width, height]
    ax_slider_el = fig.add_axes([0.78, 0.88, 0.20, 0.03])
    ax_slider_az.set_in_layout(False)
    ax_slider_el.set_in_layout(False)

    slider_az = Slider(ax_slider_az, "Azim", 0, 360, valinit=ax.azim, valstep=1)
    slider_el = Slider(ax_slider_el, "Elev", -90, 90, valinit=ax.elev, valstep=1)
    
    # Summary: _update_view — adjust camera by sliders.
    # Effect: ergonomic navigation of 3D scene.
    def _update_view(_):
        ax.view_init(elev=slider_el.val, azim=slider_az.val)
        fig.canvas.draw_idle()

    slider_az.on_changed(_update_view)
    slider_el.on_changed(_update_view)

    # Auto-rotate toggle
    ax_btn_auto = fig.add_axes([0.78, 0.83, 0.08, 0.04]); ax_btn_auto.set_in_layout(False)
    btn_auto = Button(ax_btn_auto, "Auto")

    rotating = [False]
    rot_step = 0.5  # degrees per tick
    rot_timer = fig.canvas.new_timer(interval=20)  # ~25 FPS

    def _tick(_event=None):
        # Advance azimuth; using set_val triggers _update_view
        slider_az.set_val((slider_az.val + rot_step) % 360)

    rot_timer.add_callback(_tick)

    # Summary: _toggle_auto / _tick — auto‑rotate camera until toggled off.
    # Effect: slow survey of the scene to reveal hidden alignments.
    def _toggle_auto(_event):
        rotating[0] = not rotating[0]
        if rotating[0]:
            btn_auto.label.set_text("Stop")
            rot_timer.start()
        else:
            btn_auto.label.set_text("Auto")
            rot_timer.stop()
        fig.canvas.draw_idle()

    btn_auto.on_clicked(_toggle_auto)

    # Reset view button
    ax_btn_reset = fig.add_axes([0.88, 0.83, 0.10, 0.04]); ax_btn_reset.set_in_layout(False)
    btn_reset = Button(ax_btn_reset, "Reset View")
    
    # Summary: _reset_view — jump back to initial camera.
    # Effect: navigational safety‑net while exploring.
    def _reset_view(_event):
        slider_el.set_val(init_elev)
        slider_az.set_val(init_azim)

    btn_reset.on_clicked(_reset_view)
    # --- end axis rotation controls ---
    
    # --- NEW: Slider for PC1 src/dst cosine threshold ---
    # Only meaningful if every document supplied PC1 (cluster_dirs)
    if have_srcdst:
    
        # Slider for Δ-direction cosine
        ax_slider_delta = fig.add_axes([0.78, 0.79, 0.20, 0.03]); ax_slider_delta.set_in_layout(False)
        slider_delta = Slider(ax_slider_delta, "Δ dir cos", 0.0, 1.0, valinit=current_thr["delta"], valstep=0.01)

        # Summary: _on_delta_change / _on_srcdst_change — slider callbacks to recompute
        #          matches and recolor Δ with new thresholds.
        # Effect: fast “what‑if” analysis of alignment criteria.
        def _on_delta_change(val):
            # pass current PC1 slider value if present; else None
            pc1_val = slider_srcdst.val if have_srcdst else None
            _recompute_matches(delta_thr=val, pc1_thr=pc1_val)

        slider_delta.on_changed(_on_delta_change)
        
        # Slider for PC1 cosine 
        ax_slider_srcdst = fig.add_axes([0.78, 0.74, 0.20, 0.03])  # left, bottom, width, height
        ax_slider_srcdst.set_in_layout(False)
        slider_srcdst = Slider(ax_slider_srcdst, "PC1 match thshld", 0.0, 1.0,
                               valinit=current_thr["pc1"], valstep=0.01)

        def _on_srcdst_change(val):
            _recompute_matches(delta_thr=slider_delta.val, pc1_thr=val)

        slider_srcdst.on_changed(_on_srcdst_change)
    else:
        # Reserve space + indicate inactive (some docs lacked PC1)
        ax_lbl_srcdst = fig.add_axes([0.78, 0.74, 0.20, 0.03])
        ax_lbl_srcdst.set_in_layout(False)
        ax_lbl_srcdst.axis('off')
        ax_lbl_srcdst.text(0.0, 0.5, "PC1 src/dst: n/a", transform=ax_lbl_srcdst.transAxes, va='center')
    
    attach_matplotlib_save_button(fig, default_name="visualization.pkl", parent=globals().get("root"))
    plt.show()
    

# -----------------------------------------------------------------------------
# Function: analyze_morphism_match_field
# Summary:
#   Computes a *morphism match field*: for each Δ, searches other docs for
#   matches that satisfy independent thresholds on (Δ cosine) AND (|PC1_src|)
#   AND (|PC1_dst|), trying strict→looser levels until matches appear. Also
#   records “PC1‑only” matches (strong PC1 at both ends, weak Δ). Returns
#   detailed maps, per‑docpair counters, and parameter metadata.
# Effect:
#   Quantifies how robust cross‑document morphism alignments are across
#   threshold grids—evidence you can use to pick defensible cutoffs and to
#   generate exports for downstream arrangement/labeling.
# -----------------------------------------------------------------------------
def analyze_morphism_match_field(
    document_cluster_data: dict,
    # Try strict → looser thresholds; first level that yields ≥1 match per Δ is used
    delta_thresholds = (0.98, 0.95, 0.92, 0.90, 0.88, 0.85),
    pc1_thresholds   = (0.98, 0.95, 0.92, 0.90, 0.88, 0.85),
    top_k_per_delta: int = 5,         # max matches to keep per Δ at the winning level
    pc1_only_threshold: float = 0.90, # strong PC1 match
    delta_max_for_pc1_only: float = 0.60,  # Δ must be below this to qualify as "PC1-only"
    require_cross_doc: bool = True,   # only match against other docs
    verbose: bool = True
) -> dict:
    """
    Analyze 'morphism match field' across documents.

    Returns a dictionary with:
      - 'aligned_matches':   {i -> [match, ...]}  # best matches per Δ entry
      - 'pc1_only_matches':  {i -> [match, ...]}  # strong PC1 (src & dst independently) but weak Δ
      - 'index':             {i -> entry_meta}    # quick lookup of Δ entries
      - 'summary':           { 'aligned_per_docpair': Counter, 'pc1_only_per_docpair': Counter }
      - 'shapes':            { 'num_entries': int, 'dim': int }
      - 'params':            { thresholds..., flags... }

    Semantics:
      * Δ-direction cosine uses signed dot (like the viz). PC1 concordance uses absolute cosines (|cos|).
      * Gating for aligned matches requires ALL THREE axes to pass independently:
            Δ >= delta_threshold AND |src_pc1| >= pc1_threshold AND |dst_pc1| >= pc1_threshold
      * We retain a 'pc1_composite' value (min(|src_pc1|, |dst_pc1|)) for backward compatibility,
        but it is NOT used for gating; only for ranking and legacy CSV consumers.
      * If any document lacks per-cluster PC1 (older 5-tuple), PC1-based parts degrade gracefully.
    """
    # Local imports (robust if module-level imports are absent)
    from collections import Counter
    import numpy as np

    # Summary: _first_scalar — defensive float extraction from scalars/arrays.
    # Effect: guards against odd shapes in numeric inputs.
    def _first_scalar(x):
        """Return a float from a scalar/arraylike, robust to shapes."""
        try:
            a = np.asarray(x, dtype=float)
            return float(a.flat[0])
        except Exception:
            try:
                return float(x)
            except Exception:
                return 0.0

    # --- Local helper (mirrors the visualizer) --------------------------------
    # Summary: _unpack_data — normalize 5‑ or 6‑tuples into aligned arrays.
    # Effect: consistent basis for Δ/PC1 computation.
    def _unpack_data(data):
        """
        Accept 5- or 6-tuple document data.
        Returns: delta_matrix, cluster_order, labels, cluster_topic_distributions, emb_arr, cluster_dirs
        """
        if len(data) >= 6:
            delta_matrix, cluster_order, labels, cluster_topic_distributions, cluster_embeddings, cluster_dirs = data[:6]
        else:
            delta_matrix, cluster_order, labels, cluster_topic_distributions, cluster_embeddings = data[:5]
            cluster_dirs = None

        # Normalize cluster_embeddings to ndarray aligned with cluster_order
        if isinstance(cluster_embeddings, np.ndarray):
            emb_arr = cluster_embeddings
        elif isinstance(cluster_embeddings, dict):
            emb_arr = np.vstack([cluster_embeddings[label] for label in cluster_order])
        else:
            emb_arr = np.asarray(cluster_embeddings)

        if emb_arr.ndim == 1:
            emb_arr = emb_arr[None, :]

        return delta_matrix, cluster_order, labels, cluster_topic_distributions, emb_arr, cluster_dirs
    # --------------------------------------------------------------------------

    # 1) Collect all Δ entries across docs (same fields/shape as the visualizer)
    all_entries = []  # each: {doc, start3, delta3, dir_full, delta_full, src_dir_full, dst_dir_full, i_idx, j_idx, i_lab, j_lab}
    doc_ids = list(document_cluster_data.keys())

    for doc_id, data in document_cluster_data.items():
        delta_matrix, cluster_order, _, _, cluster_embeddings, cluster_dirs = _unpack_data(data)
        emb3d = cluster_embeddings[:, :3]
        n, d = cluster_embeddings.shape

        doc_has_dirs = (isinstance(cluster_dirs, np.ndarray) and cluster_dirs.shape == (n, d))
        if doc_has_dirs:
            norms = np.linalg.norm(cluster_dirs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            cluster_dirs_unit = cluster_dirs / norms
        else:
            cluster_dirs_unit = None

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                start3 = emb3d[i]
                delta3 = delta_matrix[i, j, :3]

                dfull = delta_matrix[i, j, :]
                nrm = np.linalg.norm(dfull)
                if nrm <= 0:
                    continue
                dir_full = dfull / nrm

                if cluster_dirs_unit is not None:
                    src_dir_full = cluster_dirs_unit[i]
                    dst_dir_full = cluster_dirs_unit[j]
                else:
                    src_dir_full = None
                    dst_dir_full = None

                all_entries.append({
                    "doc": doc_id,
                    "start3": start3,
                    "delta3": delta3,
                    "dir_full": dir_full,
                    "delta_full": dfull,
                    "src_dir_full": src_dir_full,
                    "dst_dir_full": dst_dir_full,
                    "i_idx": i,
                    "j_idx": j,
                    "i_lab": cluster_order[i],
                    "j_lab": cluster_order[j],
                })

    M = len(all_entries)
    if M == 0:
        if verbose:
            print("[analyze] No delta entries found.")
        return {
            "aligned_matches": {}, "pc1_only_matches": {}, "index": {}, "summary": {},
            "shapes": {"num_entries": 0, "dim": 0},
            "params": {
                "delta_thresholds": list(delta_thresholds),
                "pc1_thresholds": list(pc1_thresholds),
                "top_k_per_delta": int(top_k_per_delta),
                "pc1_only_threshold": float(pc1_only_threshold),
                "delta_max_for_pc1_only": float(delta_max_for_pc1_only),
                "require_cross_doc": bool(require_cross_doc),
                "gating": "delta & src_pc1 & dst_pc1 (independent axes)"
            }
        }

    # 2) Cosine matrices (Δ signed; PC1 absolute), mirroring viz logic
    docs_arr = np.array([e["doc"] for e in all_entries], dtype=object)

    dirs_full = np.vstack([e["dir_full"] for e in all_entries])              # (M, d)
    delta_sims = dirs_full @ dirs_full.T                                     # signed cosines

    have_srcdst = all(e["src_dir_full"] is not None and e["dst_dir_full"] is not None for e in all_entries)
    if have_srcdst:
        src_dirs_full = np.vstack([e["src_dir_full"] for e in all_entries])  # (M, d)
        dst_dirs_full = np.vstack([e["dst_dir_full"] for e in all_entries])  # (M, d)
        src_sims = np.abs(src_dirs_full @ src_dirs_full.T)                   # absolute cosines
        dst_sims = np.abs(dst_dirs_full @ dst_dirs_full.T)
    else:
        src_sims = None
        dst_sims = None
        if verbose:
            print("[analyze] PC1 directions missing for at least one document; will perform Δ-only matching where applicable.")

    # 3) Matching helpers -------------------------------------------------------
    # Summary: _joint_min — conservative ranking score = min(Δ, |src|, |dst|).
    # Effect: favors matches strong along every axis (used only for ranking).
    def _joint_min(delta_vec, src_vec=None, dst_vec=None):
        """
        Conservative joint score for ranking only (NOT gating):
        - with PC1s: min(delta, src, dst)
        - without PC1s: delta only
        """
        if src_vec is None or dst_vec is None:
            return delta_vec
        return np.minimum.reduce([delta_vec, src_vec, dst_vec])

    # Summary: _cross_doc_mask — mask restricting matches to other documents.
    # Effect: ensures cross‑document alignment when required.
    def _cross_doc_mask(i):
        return (docs_arr != docs_arr[i]) if require_cross_doc else np.ones(M, dtype=bool)

    # 4) For each Δ entry, find best-case matches by threshold levels ----------
    aligned_matches = {}
    for i in range(M):
        found = []
        cross_doc = _cross_doc_mask(i)

        if have_srcdst:
            # Try strict→looser levels; stop at first (Δ, PC1) level that yields ≥1 match
            done = False
            for dt in delta_thresholds:
                if done:
                    break
                for pt in pc1_thresholds:
                    dt = float(dt); pt = float(pt)
                    # Independent-axis gating
                    mask = (delta_sims[i] >= dt) & cross_doc
                    mask &= (src_sims[i] >= pt) & (dst_sims[i] >= pt)
                    mask[i] = False  # exclude self

                    idxs = np.where(mask)[0]
                    if idxs.size > 0:
                        # Rank by conservative joint score (min of the three); gating already independent.
                        joint = _joint_min(delta_sims[i, idxs], src_sims[i, idxs], dst_sims[i, idxs])
                        order = np.argsort(-joint)
                        keep = idxs[order][:top_k_per_delta]
                        for j in keep:
                            # Include independent axis values & legacy composite for downstream CSVs
                            pc1_comp = float(min(src_sims[i, j], dst_sims[i, j]))
                            found.append({
                                "j": int(j),
                                "doc": all_entries[j]["doc"],
                                "from": int(all_entries[j]["i_lab"]),
                                "to":   int(all_entries[j]["j_lab"]),
                                "scores": {
                                    "delta_cos": float(delta_sims[i, j]),
                                    "src_pc1":   float(src_sims[i, j]),
                                    "dst_pc1":   float(dst_sims[i, j]),
                                    "pc1_composite": pc1_comp,  # legacy field; not used for gating
                                    "joint_min": float(min(delta_sims[i, j], src_sims[i, j], dst_sims[i, j]))
                                },
                                "flags": {
                                    "delta_ok": bool(delta_sims[i, j] >= dt),
                                    "src_pc1_ok": bool(src_sims[i, j] >= pt),
                                    "dst_pc1_ok": bool(dst_sims[i, j] >= pt)
                                },
                                "level": {"delta": dt, "pc1": pt}
                            })
                        done = True
                        break
        else:
            # Δ-only fallback (use loosest Δ thr as baseline)
            dt = float(delta_thresholds[-1])
            mask = (delta_sims[i] >= dt) & cross_doc
            mask[i] = False
            idxs = np.where(mask)[0]
            if idxs.size > 0:
                joint = delta_sims[i, idxs]
                order = np.argsort(-joint)
                keep = idxs[order][:top_k_per_delta]
                for j in keep:
                    found.append({
                        "j": int(j),
                        "doc": all_entries[j]["doc"],
                        "from": int(all_entries[j]["i_lab"]),
                        "to":   int(all_entries[j]["j_lab"]),
                        "scores": {"delta_cos": float(delta_sims[i, j])},
                        "flags": {"delta_ok": bool(delta_sims[i, j] >= dt)},
                        "level": {"delta": dt}
                    })

        aligned_matches[i] = found

    # 5) PC1-only matches: strong PC1 at both ends but weak Δ -------------------
    pc1_only_matches = {}
    if have_srcdst:
        for i in range(M):
            cross_doc = _cross_doc_mask(i)
            dt_max = float(delta_max_for_pc1_only)
            pt_only = float(pc1_only_threshold)

            # Independent axis gating: weak Δ, strong PC1 on both ends
            mask = (delta_sims[i] < dt_max) & cross_doc
            mask &= (src_sims[i] >= pt_only) & (dst_sims[i] >= pt_only)
            mask[i] = False

            idxs = np.where(mask)[0]
            found = []
            if idxs.size > 0:
                # Rank by min(|src_pc1|, |dst_pc1|)
                pc1_comp_vec = np.minimum(src_sims[i, idxs], dst_sims[i, idxs])
                order = np.argsort(-pc1_comp_vec)
                keep = idxs[order][:top_k_per_delta]
                for j in keep:
                    found.append({
                        "j": int(j),
                        "doc": all_entries[j]["doc"],
                        "from": int(all_entries[j]["i_lab"]),
                        "to":   int(all_entries[j]["j_lab"]),
                        "scores": {
                            "delta_cos": float(delta_sims[i, j]),
                            "src_pc1":   float(src_sims[i, j]),
                            "dst_pc1":   float(dst_sims[i, j]),
                            "pc1_composite": float(min(src_sims[i, j], dst_sims[i, j]))  # legacy/ordering
                        },
                        "criteria": {
                            "pc1_only_threshold": pt_only,
                            "delta_max": dt_max
                        }
                    })
            pc1_only_matches[i] = found
    else:
        if verbose:
            print("[analyze] Skipping PC1-only search (no PC1 directions).")

    # 6) Summaries by doc pair --------------------------------------------------
    # Summary: _docpair_key — canonicalize doc→doc pair keys.
    # Effect: stable counters and CSV ordering.
    def _docpair_key(a_doc, b_doc):
        return (str(a_doc), str(b_doc))

    aligned_counter = Counter()
    pc1_only_counter = Counter()
    for i, ms in aligned_matches.items():
        src_doc = all_entries[i]["doc"]
        for m in ms:
            aligned_counter[_docpair_key(src_doc, m["doc"])] += 1
    for i, ms in pc1_only_matches.items():
        src_doc = all_entries[i]["doc"]
        for m in ms:
            pc1_only_counter[_docpair_key(src_doc, m["doc"])] += 1

    if verbose:
        print("\n[analyze] Aligned (Δ & src_pc1 & dst_pc1) match counts by doc→doc:")
        for k, v in aligned_counter.most_common():
            print("  ", k, ":", v)
        if have_srcdst:
            print("\n[analyze] PC1-only (weak Δ) match counts by doc→doc:")
            for k, v in pc1_only_counter.most_common():
                print("  ", k, ":", v)
        print("")

    # 7) Build index for quick lookup of each Δ entry’s identity ----------------
    index = {
        i: {
            "doc": e["doc"],
            "from": int(e["i_lab"]),
            "to":   int(e["j_lab"])
        } for i, e in enumerate(all_entries)
    }

    return {
        "aligned_matches": aligned_matches,
        "pc1_only_matches": pc1_only_matches,
        "index": index,
        "summary": {
            "aligned_per_docpair": aligned_counter,
            "pc1_only_per_docpair": pc1_only_counter
        },
        "shapes": {"num_entries": M, "dim": int(len(all_entries[0]["dir_full"]))},
        "params": {
            "delta_thresholds": list(delta_thresholds),
            "pc1_thresholds": list(pc1_thresholds),
            "top_k_per_delta": int(top_k_per_delta),
            "pc1_only_threshold": float(pc1_only_threshold),
            "delta_max_for_pc1_only": float(delta_max_for_pc1_only),
            "require_cross_doc": bool(require_cross_doc),
            "gating": "delta & src_pc1 & dst_pc1 (independent axes); ranking uses joint_min"
        }
    }
    

# -----------------------------------------------------------------------------
# Function: output_analysis
# Summary:
#   Prints three human‑readable sections (per‑document alignments; PC1‑only
#   alignments; and dst‑PC1 highlights) and writes a tab‑delimited CSV with
#   one row per unique match, including independent axes (Δ, src, dst), a
#   conservative ranking score, detected thresholds, and optional full text.
# Effect:
#   Produces durable artifacts (console + CSV) that can be archived, audited,
#   or joined to other systems for curation and arrangement work.
# -----------------------------------------------------------------------------
def output_analysis(
    res,
    doc_id=None,
    document_cluster_data=None,
    segments_by_doc=None,
    csv_path=None
):
    """
    Print analysis (Sections A/B/C) and also write a CSV with one row per unique match.

    Args:
        res: dict returned by analyze_morphism_match_field(...)
        doc_id: optional doc id string for Section A focus
        document_cluster_data: optional dict {doc_id: tuple(...)} as used by viz/analyzer.
        segments_by_doc: optional dict {doc_id: list_of_segment_strings}
        csv_path: optional filesystem path to write the CSV. If None, a Save-As dialog
            will prompt for a location.

    Notes:
        - Display and CSV now show independent axes:
              delta_cos, src_pc1, dst_pc1
          plus a conservative ranking score 'joint_min' (min of the three) when available.
        - 'pc1_composite' (min of src/dst) is retained for legacy reads and PC1-only rows.
        - Gating/selection semantics come from analyze_morphism_match_field (Δ ∧ src ∧ dst).
    """
    import csv
    from tkinter import filedialog
    import tkinter as tk

    # ---- small helpers ----
    # Summary: _fmt — print‑friendly number formatting with “—” for missing.
    # Effect: tidy console and CSV output.
    def _fmt(x):
        return "—" if x is None or x == "" else f"{float(x):.3f}"

    # Summary: _get_param — pull a parameter from results metadata (with fallbacks).
    # Effect: preserves provenance (what thresholds were in effect).
    def _get_param(name, default_val):
        # Read from top-level params if present; else scan criteria in pc1_only rows; else default
        try:
            params = res.get("params", {})
            if name in params:
                return float(params[name])
        except Exception:
            pass
        if name == "pc1_only_threshold":
            try:
                for _i, lst in res.get("pc1_only_matches", {}).items():
                    for m in lst or []:
                        crit = m.get("criteria") or {}
                        if "pc1_only_threshold" in crit:
                            return float(crit["pc1_only_threshold"])
            except Exception:
                pass
        if name == "delta_max_for_pc1_only":
            try:
                for _i, lst in res.get("pc1_only_matches", {}).items():
                    for m in lst or []:
                        crit = m.get("criteria") or {}
                        if "delta_max" in crit:
                            return float(crit["delta_max"])
            except Exception:
                pass
        return float(default_val)

    # Summary: _choose_csv_save_path — topmost Save‑As dialog for CSV path.
    # Effect: avoids silent cancel and keeps the workflow smooth.
    def _choose_csv_save_path(default_name: str = "morphism_matches.csv", parent_window=None):
        """Top-most Save-As dialog parented to a live root to avoid 'silent cancel' issues."""
        parent = None
        try:
            parent = parent_window if parent_window is not None else (globals().get("root") or tk._get_default_root())
        except Exception:
            parent = None
        try:
            if parent is not None:
                parent.update_idletasks()
                try:
                    parent.lift()
                    parent.attributes("-topmost", True)
                    parent.after(200, lambda: parent.attributes("-topmost", False))
                except Exception:
                    pass
        except Exception:
            pass

        try:
            return filedialog.asksaveasfilename(
                parent=parent,
                title="Save morphism matches CSV",
                defaultextension=".csv",
                initialfile=default_name,
                filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
            ) or None
        except Exception:
            return None

    # Summary: _get_labels_array / _get_segments_list — recover labels/segments
    #          for looking up cluster texts when exporting.
    # Effect: optional richer CSVs with cluster excerpts.
    # locate labels array for a doc
    def _get_labels_array(doc):
        if not document_cluster_data or doc not in document_cluster_data:
            return None
        try:
            data = document_cluster_data[doc]
            # standard 6-tuple: (..., cluster_order, labels, cluster_topic_distributions, cluster_embeddings, cluster_dirs)
            return data[2]
        except Exception:
            return None

    # try to get segments list for a doc
    def _get_segments_list(doc):
        # prefer explicitly provided segments_by_doc
        if segments_by_doc and doc in segments_by_doc:
            return segments_by_doc[doc]
        # fall back: if your tuple happens to include segments (non-standard), try to detect them
        if document_cluster_data and doc in document_cluster_data:
            data = document_cluster_data[doc]
            # Heuristic: if there's a list[str] somewhere after labels, accept it
            try:
                for item in list(data)[3:]:
                    if isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
                        return list(item)
            except Exception:
                pass
        return None  # segments unavailable

    # Summary: _cluster_text — concatenate the raw segments for a cluster.
    # Effect: human context for matches (when enabled).
    # assemble full text for a single cluster
    def _cluster_text(doc, cluster_label):
        labels_arr = _get_labels_array(doc)
        segs = _get_segments_list(doc)
        if labels_arr is None or segs is None:
            return "N/A"  # we don't have what we need
        try:
            idxs = [i for i, lab in enumerate(labels_arr) if int(lab) == int(cluster_label)]
            # join with double newline to preserve boundaries (CSV will quote)
            return "\n\n".join(str(segs[i]) for i in idxs)
        except Exception:
            return "N/A"

    # ---- guard ----
    if not res or "index" not in res or not res["index"]:
        print("[output_analysis] No entries in result index.")
        return

    # ==================== Section A: Per-selected document ====================
    first_idx, first_meta = next(iter(res["index"].items()))
    display_doc = first_meta["doc"] if doc_id is None else doc_id

    entries = [(i, meta) for i, meta in res["index"].items() if meta["doc"] == display_doc]
    entries.sort(key=lambda t: (t[1]["from"], t[1]["to"]))

    print(f"\n=== Matches for document: {display_doc} ===\n")

    for i, meta in entries:
        src_lab = meta["from"]
        dst_lab = meta["to"]
        print(f"C{src_lab} \u2192 C{dst_lab}")  # →

        aligned = res.get("aligned_matches", {}).get(i, [])
        pc1_only = res.get("pc1_only_matches", {}).get(i, [])

        # Δ & PC1 aligned (or Δ-only if PC1s were unavailable)
        if aligned:
            print("  Δ & PC1 aligned:")
            for m in aligned:
                sc = m.get("scores", {}) or {}
                flags = m.get("flags", {}) or {}
                print(
                    "    → {doc}: C{f}→C{t} | Δ={d} src={s} dst={p} joint={j}".format(
                        doc=m["doc"],
                        f=m["from"], t=m["to"],
                        d=_fmt(sc.get("delta_cos")),
                        s=_fmt(sc.get("src_pc1")),
                        p=_fmt(sc.get("dst_pc1")),
                        j=_fmt(sc.get("joint_min")),
                    )
                )
        else:
            print("  Δ & PC1 aligned: (none)")

        # PC1-only (weak Δ + strong src/dst PC1 independently)
        if pc1_only:
            print("  PC1-only:")
            for m in pc1_only:
                sc = m.get("scores", {}) or {}
                print(
                    "    → {doc}: C{f}→C{t} | Δ={d} src={s} dst={p} pc1_comp={pc}".format(
                        doc=m["doc"],
                        f=m["from"], t=m["to"],
                        d=_fmt(sc.get("delta_cos")),
                        s=_fmt(sc.get("src_pc1")),
                        p=_fmt(sc.get("dst_pc1")),
                        pc=_fmt(sc.get("pc1_composite")),
                    )
                )
        else:
            print("  PC1-only: (none)")

        print()  # blank line

    # ==================== Section B: PC1-only across ALL documents ====================
    print("\n=== PC1-only matches across ALL documents ===\n")

    any_pc1_only = False
    for i, meta in sorted(res["index"].items(), key=lambda t: (t[1]["doc"], t[1]["from"], t[1]["to"])):
        pc1_only_list = res.get("pc1_only_matches", {}).get(i, [])
        if not pc1_only_list:
            continue

        any_pc1_only = True
        src_doc = meta["doc"]; src_lab = meta["from"]; dst_lab = meta["to"]
        print(f"{src_doc}: C{src_lab} \u2192 C{dst_lab}")
        for m in pc1_only_list:
            sc = m.get("scores", {}) or {}
            print(
                "  → {doc}: C{f}→C{t} | Δ={d} src={s} dst={p} pc1_comp={pc}".format(
                    doc=m["doc"],
                    f=m["from"], t=m["to"],
                    d=_fmt(sc.get("delta_cos")),
                    s=_fmt(sc.get("src_pc1")),
                    p=_fmt(sc.get("dst_pc1")),
                    pc=_fmt(sc.get("pc1_composite")),
                )
            )
        print()

    if not any_pc1_only:
        print("  (none found)\n")

    # ==================== Section C: Dst PC1-only across ALL documents ====================
    dst_thr = _get_param("pc1_only_threshold", 0.90)
    print(f"\n=== Dst PC1-only across ALL documents (dst_pc1 ≥ {dst_thr:.2f}) ===\n")

    any_dst_only = False

    aligned_map   = res.get("aligned_matches", {}) or {}
    pc1_only_map  = res.get("pc1_only_matches", {}) or {}
    index_map     = res.get("index", {}) or {}

    # Summary: _src_key — sort key for deterministic console layout.
    # Effect: stable, readable output across runs.
    def _src_key(item):
        i, meta = item
        return (meta["doc"], meta["from"], meta["to"])

    for i, meta in sorted(index_map.items(), key=_src_key):
        # Combine candidates from both aligned and pc1_only for this source edge
        cand = []
        cand.extend(aligned_map.get(i, []))
        cand.extend(pc1_only_map.get(i, []))

        # Filter to dst-only criterion
        seen_j = set()
        dst_only_hits = []
        for m in cand:
            sc = (m.get("scores", {}) or {})
            dst_val = sc.get("dst_pc1", None)
            j_idx   = m.get("j", None)
            try:
                if dst_val is None:
                    continue
                if float(dst_val) >= dst_thr:
                    if j_idx is not None and j_idx in seen_j:
                        continue
                    seen_j.add(j_idx)
                    dst_only_hits.append(m)
            except Exception:
                continue

        if not dst_only_hits:
            continue

        any_dst_only = True
        src_doc = meta["doc"]; src_lab = meta["from"]; dst_lab = meta["to"]
        print(f"{src_doc}: C{src_lab} \u2192 C{dst_lab}")
        for m in dst_only_hits:
            sc = m.get("scores", {}) or {}
            print(
                "  → {doc}: C{f}→C{t} | dst={dst}  Δ={d} src={s}".format(
                    doc=m["doc"],
                    f=m["from"], t=m["to"],
                    dst=_fmt(sc.get("dst_pc1")),
                    d=_fmt(sc.get("delta_cos")),
                    s=_fmt(sc.get("src_pc1")),
                )
            )
        print()

    if not any_dst_only:
        print("  (none found)\n")

    # ==================== CSV: one row per unique match ====================
    # Columns (independent axes + legacy composites kept explicitly named)
    header = [
        "match_type",                 # "aligned" | "pc1_only"
        "src_doc", "src_from", "src_to",
        "tgt_doc", "tgt_from", "tgt_to",
        "delta_cos", "src_pc1", "dst_pc1",
        "pc1_composite",             # min(src_pc1, dst_pc1) — legacy/ranking
        "joint_min",                 # min(delta_cos, src_pc1, dst_pc1) — ranking only
        "delta_ok", "src_pc1_ok", "dst_pc1_ok",
        "detected_delta_thr", "detected_pc1_thr",
        "pc1_only_thr", "delta_max_for_pc1_only",
        # Optional full-text columns can be re-enabled later:
        # "src_doc_src_cluster_text", "src_doc_dst_cluster_text",
        # "tgt_doc_src_cluster_text", "tgt_doc_dst_cluster_text",
    ]

    # Where to save
    if csv_path is None:
        csv_path = _choose_csv_save_path(parent_window=globals().get("root"))

    if not csv_path:
        print("[output_analysis] CSV save cancelled (no file chosen).")
        return

    emitted = set()  # dedupe (i_src, j_tgt) across both maps

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        # Tab-delimited (keeps compatibility with prior text outputs)
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
        writer.writerow(header)

        # Summary: _emit_rows — write deduplicated rows to the CSV for aligned and
        #          PC1‑only matches with all required fields.
        # Effect: single source of truth for file output.
        def _emit_rows(kind: str, mmap: dict):
            for i_src, lst in mmap.items():
                i_src = int(i_src)
                src_meta = res["index"].get(i_src, {})
                src_doc = src_meta.get("doc")
                src_from = src_meta.get("from")
                src_to   = src_meta.get("to")
                for m in lst or []:
                    j_tgt = m.get("j")
                    if j_tgt is None:
                        continue
                    key = (i_src, int(j_tgt))
                    if key in emitted:
                        continue  # already written from the other map
                    emitted.add(key)

                    tgt_doc = m.get("doc")
                    tgt_from = m.get("from")
                    tgt_to   = m.get("to")

                    sc = m.get("scores", {}) or {}
                    flags = m.get("flags", {}) or {}
                    level = m.get("level", {}) or {}
                    crit  = m.get("criteria", {}) or {}

                    # compute joint_min if not present (e.g., pc1-only rows might omit it)
                    try:
                        d = sc.get("delta_cos", None)
                        s = sc.get("src_pc1", None)
                        p = sc.get("dst_pc1", None)
                        joint_min = min([v for v in (d, s, p) if v is not None])
                    except Exception:
                        joint_min = ""

                    # Gather full text for each of the four clusters (optional)
                    # src_src_text = _cluster_text(src_doc, src_from).replace('\n', '').replace('\r', '')
                    # src_dst_text = _cluster_text(src_doc, src_to).replace('\n', '').replace('\r', '')
                    # tgt_src_text = _cluster_text(tgt_doc, tgt_from).replace('\n', '').replace('\r', '')
                    # tgt_dst_text = _cluster_text(tgt_doc, tgt_to).replace('\n', '').replace('\r', '')

                    row = [
                        kind,
                        src_doc, src_from, src_to,
                        tgt_doc, tgt_from, tgt_to,
                        sc.get("delta_cos", ""),
                        sc.get("src_pc1", ""),
                        sc.get("dst_pc1", ""),
                        sc.get("pc1_composite", ""),   # legacy/ranking
                        joint_min,
                        flags.get("delta_ok", ""),
                        flags.get("src_pc1_ok", ""),
                        flags.get("dst_pc1_ok", ""),
                        # thresholds where the aligned match was detected
                        level.get("delta", ""),
                        level.get("pc1", ""),
                        # pc1-only criteria (present only for pc1_only rows)
                        crit.get("pc1_only_threshold", ""),
                        crit.get("delta_max", ""),
                        # src_src_text, src_dst_text, tgt_src_text, tgt_dst_text,
                    ]
                    writer.writerow(row)

        _emit_rows("aligned",   res.get("aligned_matches", {}) or {})
        _emit_rows("pc1_only",  res.get("pc1_only_matches", {}) or {})

    print(f"[output_analysis] CSV written to: {csv_path}")


# -----------------------------------------------------------------------------
# Function: plot_morphism_match_field_3d
# Summary:
#   Builds an interactive 3D view of the *incremental* match counts as a
#   function of thresholds (X=Δ cosine, Y=PC1 concordance). Supports three
#   modes: AND (Δ & src & dst), Δ&dst only, and src&dst only (Δ ignored and
#   drawn at X=1.00). Uses a persistent colorbar, an optional green overlay for
#   matches involving a chosen doc_id, and a convex‑hull edge curve on the XY
#   projection to summarize the frontier of non‑zero cells.
# Effect:
#   Visual “phase diagram” for threshold selection—makes it easy to spot
#   stable regimes where many new matches appear (or don’t) as criteria relax.
# -----------------------------------------------------------------------------
def plot_morphism_match_field_3d(
    res: dict,
    step: float = 0.01,
    figsize=(12, 9),
    cmap_name: str = "YlOrRd",
    doc_id: str | None = None,
    log_colors=True
) -> None:
    """
    3D scatter with mode toggles (independent-axis semantics):
      - Δ & src & dst : require delta_cos ≥ X AND |src_pc1| ≥ Y AND |dst_pc1| ≥ Y
                         (Y applies to both ends via min(|src_pc1|, |dst_pc1|))
      - Δ & dst       : require delta_cos ≥ X AND |dst_pc1| ≥ Y
      - src & dst     : require |src_pc1| ≥ Y AND |dst_pc1| ≥ Y (Δ ignored; results drawn at X=1.00)

    Z counts ONLY the *additional* matches that first qualify at that (X, Y) cell.
    Grey for Z=0; non-zero colored yellow→red scaled to the largest Z in the current view.
    Z-axis is fixed to [0 .. number_of_pairwise_cluster_deltas].

    If `doc_id` is provided, any cells that include matches involving that document
    (as source OR target) are overlaid as bright-green points.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # ----------------- validation -----------------
    if not res or "index" not in res or not res["index"]:
        print("[plot_morphism_match_field_3d] Empty or invalid 'res'.")
        return

    # ----------------- grid & caps -----------------
    n_bins = int(round(1.0 / step)) + 1
    thr = np.round(np.linspace(1.0, 0.0, n_bins), 2)  # 1.00, 0.99, ..., 0.00
    X, Y = np.meshgrid(thr, thr, indexing="ij")

    # total # pairwise cluster deltas (cap for Z)
    max_pairs = (
        int(res.get("shapes", {}).get("num_entries", 0))
        or int(len(res.get("index", {})))
    )

    matches_maps = {
        "aligned":  res.get("aligned_matches", {}) or {},
        "pc1only":  res.get("pc1_only_matches", {}) or {},
    }
    index_map = res.get("index", {}) or {}

    # ----------------- helpers -----------------
    # Summary: _bin_idx — map a value in [0,1] to the grid index along an axis.
    # Effect: consistent placement of matches on the threshold lattice.
    def _bin_idx(v: float) -> int:
        v = float(np.clip(v, 0.0, 1.0))
        v = float(np.round(v + 1e-12, 2))
        idx = int(np.round((1.0 - v) / step))
        return int(np.clip(idx, 0, n_bins - 1))

    # Summary: _build_grid — aggregate unique (i_src, j_tgt) matches into two
    #          incremental grids (all matches; matches involving a specific doc).
    # Effect: supplies the counts that drive the 3D scatter and highlights.
    def _build_grid(mode: str):
        """
        Build two incremental (point-source) grids for a given mode:
          arr[i0, j0] += 1  for all matches
          hl [i0, j0] += 1  for matches involving `doc_id` (if provided)
        """
        arr = np.zeros((n_bins, n_bins), dtype=np.int32)
        hl  = np.zeros((n_bins, n_bins), dtype=np.int32)
        seen = set()  # dedupe (source_entry_idx, target_entry_idx)

        def _ingest(mmap: dict):
            nonlocal arr, hl
            for i_src, lst in mmap.items():
                for m in (lst or []):
                    j_tgt = m.get("j")
                    if j_tgt is None:
                        continue
                    key = (int(i_src), int(j_tgt))
                    if key in seen:
                        continue  # avoid double count across aligned/pc1-only

                    sc = m.get("scores", {}) or {}
                    d  = sc.get("delta_cos", None)
                    sp = sc.get("src_pc1",  None)
                    dp = sc.get("dst_pc1",  None)

                    # Determine minimal (i0, j0) where this match first qualifies
                    if mode == "AND":
                        # require all three axes; Y encodes both ends via min(|src|, |dst|)
                        if d is None or sp is None or dp is None:
                            continue
                        pc_min = float(min(abs(float(sp)), abs(float(dp))))
                        i0 = _bin_idx(float(d))
                        j0 = _bin_idx(pc_min)

                    elif mode == "DELDST":
                        # Δ & dst only
                        if d is None or dp is None:
                            continue
                        i0 = _bin_idx(float(d))
                        j0 = _bin_idx(abs(float(dp)))

                    elif mode == "SRC_DST":
                        # src & dst only (Δ ignored); draw at X=1.00 (i=0)
                        if sp is None or dp is None:
                            continue
                        pc_min = float(min(abs(float(sp)), abs(float(dp))))
                        i0 = 0
                        j0 = _bin_idx(pc_min)
                    else:
                        continue

                    # Increment main grid
                    arr[i0, j0] += 1

                    # Highlight grid: if doc_id is involved as source or target
                    if doc_id:
                        try:
                            src_doc = index_map[int(i_src)]["doc"]
                        except Exception:
                            src_doc = None
                        tgt_doc = m.get("doc")
                        if (src_doc == doc_id) or (tgt_doc == doc_id):
                            hl[i0, j0] += 1

                    seen.add(key)

        _ingest(matches_maps["aligned"])
        _ingest(matches_maps["pc1only"])
        return arr, hl

    # Summary: _draw_upper_hull — compute/draw the positive‑Y edge of the XY
    #          convex hull of non‑zero cells (with z=max per (x,y)).
    # Effect: summarizes the “frontier” where matches first appear as thresholds relax.
    def _draw_upper_hull(ax, Zgrid):
        """Draw green curve on the positive-Y edge of the XY convex hull where Z>0."""
        try:
            from scipy.spatial import ConvexHull
            mask_nz = (Zgrid > 0)
            if np.count_nonzero(mask_nz) < 2:
                return None

            x_nz = np.round(X[mask_nz], 2)
            y_nz = np.round(Y[mask_nz], 2)
            z_nz = Zgrid[mask_nz]

            xy_to_z = {}
            for x, y, z in zip(x_nz, y_nz, z_nz):
                key = (float(x), float(y))
                if key not in xy_to_z or float(z) > xy_to_z[key]:
                    xy_to_z[key] = float(z)

            P = np.array(list(xy_to_z.keys()), dtype=float)
            if P.shape[0] == 2:
                P2 = P[np.argsort(P[:, 0])]
                z2 = np.array([xy_to_z[(float(np.round(x, 2)), float(np.round(y, 2)))] for x, y in P2], dtype=float)
                return ax.plot3D(P2[:, 0], P2[:, 1], z2, color='green', linewidth=2.5, alpha=0.95, label='hull (upper edge)')[0]

            if P.shape[0] >= 3:
                hull = ConvexHull(P)
                H = P[hull.vertices]
                li = int(np.argmin(H[:, 0]))
                ri = int(np.argmax(H[:, 0]))
                N = H.shape[0]
                idxs_fwd = [(li + k) % N for k in range((ri - li) % N + 1)]
                idxs_bwd = [(li - k) % N for k in range((li - ri) % N + 1)]
                chain_fwd = H[idxs_fwd]
                chain_bwd = H[idxs_bwd][::-1]
                chain_xy = chain_fwd if np.nanmean(chain_fwd[:, 1]) >= np.nanmean(chain_bwd[:, 1]) else chain_bwd
                order = np.argsort(chain_xy[:, 0])
                chain_xy = chain_xy[order]
                z_chain = np.array([xy_to_z[(float(np.round(x, 2)), float(np.round(y, 2)))] for x, y in chain_xy], dtype=float)
                return ax.plot3D(chain_xy[:, 0], chain_xy[:, 1], z_chain, color='green', linewidth=2.5, alpha=0.95, label='hull (upper edge)')[0]
            return None
        except Exception as ex:
            print(f"[plot] hull positive-edge curve error: {ex}")
            return None

    # ----------------- precompute grids per mode -----------------
    ARR_AND,    HL_AND    = _build_grid("AND")
    ARR_DELDST, HL_DELDST = _build_grid("DELDST")
    ARR_SRCDST, HL_SRCDST = _build_grid("SRC_DST")
    ARR_BY_MODE = {"AND": ARR_AND, "DELDST": ARR_DELDST, "SRC_DST": ARR_SRCDST}
    HL_BY_MODE  = {"AND": HL_AND,  "DELDST": HL_DELDST,  "SRC_DST": HL_SRCDST}

    # ----------------- figure & persistent colorbar -----------------
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Create ONE colorbar and update it on each refresh to avoid shrinking.
    cmap = cm.get_cmap(cmap_name)
    base_norm = mcolors.Normalize(vmin=1.0, vmax=1.0)
    base_sm = cm.ScalarMappable(norm=base_norm, cmap=cmap)
    base_sm.set_array([])
    cbar_obj = plt.colorbar(base_sm, ax=ax, fraction=0.03, pad=0.08)
    # If tight_layout/constrained_layout is active in your env, this prevents re-layout effects:
    # cbar_obj.ax.set_in_layout(False)

    current_mode = {"v": "AND"}  # mutable holder
    hull_line = {"obj": None}
    legend_obj = {"obj": None}

    # Summary: _colors_for — map counts to RGBA with grey zeros.
    # Effect: makes “no new matches” visually distinct from low but positive counts.
    def _colors_for(Zgrid, norm):
        """Map Zgrid to RGBA, keeping zeros grey."""
        Zc = np.maximum(Zgrid.astype(np.float32), 0.0)
        flat = Zc.ravel()
        colors = np.empty((flat.size, 4), dtype=np.float32)
        grey = np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32)
        mask0 = (flat <= 0.0)
        colors[mask0] = grey
        nz = ~mask0
        if np.any(nz) and norm is not None:
            colors[nz] = cmap(norm(flat[nz]))
        else:
            colors[nz] = grey
        return colors

    # Summary: _refresh — (re)draw the scene for a selected mode and update the
    #          persistent colorbar, hull, highlight overlay, labels, and legend.
    # Effect: keeps interactions responsive without shrinking the figure.
    def _refresh(mode_key: str):
        Zgrid = ARR_BY_MODE[mode_key].astype(np.float32)  # with Patch A
        zmax_cell = float(Zgrid.max()) if Zgrid.size else 0.0
        if log_colors:
            # Ignore zeros; colorbar will reflect orders of magnitude
            norm = mcolors.LogNorm(vmin=1.0, vmax=max(1.0, zmax_cell))
        else:
            norm = mcolors.Normalize(vmin=1.0, vmax=max(1.0, zmax_cell))
        colors = _colors_for(Zgrid, norm)
        cbar_obj.update_normal(cm.ScalarMappable(norm=norm, cmap=cmap))

        # main scatter
        if not hasattr(_refresh, "scat"):
            _refresh.scat = ax.scatter(
                X.ravel(), Y.ravel(), Zgrid.ravel(),
                c=colors, s=8, depthshade=True
            )
        else:
            _refresh.scat._offsets3d = (X.ravel(), Y.ravel(), Zgrid.ravel())
            _refresh.scat.set_color(colors)

        # doc highlight overlay (bright green)
        HLgrid = HL_BY_MODE[mode_key]
        mask_hl = (HLgrid > 0)
        if getattr(_refresh, "scat_hl", None) is not None:
            try:
                _refresh.scat_hl.remove()
            except Exception:
                pass
            _refresh.scat_hl = None

        if doc_id and np.count_nonzero(mask_hl) > 0:
            hx = X[mask_hl]; hy = Y[mask_hl]; hz = Zgrid[mask_hl]
            _refresh.scat_hl = ax.scatter(
                hx, hy, hz,
                s=28, c="#00ff55", edgecolors="k", linewidths=0.7, alpha=0.95, depthshade=False
            )

        # labels & title per mode
        if mode_key == "AND":
            ax.set_ylabel("PC1 concordance threshold (≥ Y)\n(require |src_pc1| ≥ Y AND |dst_pc1| ≥ Y)")
            ax.set_xlabel("Δ direction cosine threshold (≥ X)")
            ax.set_title("Morphism Match Field (Incremental): Δ & src & dst")
        elif mode_key == "DELDST":
            ax.set_ylabel("Dst PC1 concordance threshold (≥ Y)")
            ax.set_xlabel("Δ direction cosine threshold (≥ X)")
            ax.set_title("Morphism Match Field (Incremental): Δ & dst")
        else:  # SRC_DST
            ax.set_ylabel("PC1 concordance threshold (≥ Y)\n(require |src_pc1| ≥ Y AND |dst_pc1| ≥ Y)")
            ax.set_xlabel("Δ direction cosine threshold (ignored in this view)")
            ax.set_title("Morphism Match Field (Incremental): src & dst")

        ax.set_xlim(1.0, 0.0)
        ax.set_ylim(1.0, 0.0)
        ax.set_zlim(0.0, float(Zgrid.max() if Zgrid.size else 1.0))
        ax.set_zlabel(f"Additional matches at (X, Y)  [0 .. {max_pairs}]")

        # hull update
        if hull_line["obj"] is not None:
            try:
                hull_line["obj"].remove()
            except Exception:
                pass
            hull_line["obj"] = None
        # hull_line["obj"] = _draw_upper_hull(ax, Zgrid) # toggle this commented line to show the green hull

        # legend: replace, do not pile up
        if legend_obj["obj"] is not None:
            try:
                legend_obj["obj"].remove()
            except Exception:
                pass
            legend_obj["obj"] = None

        handles = [Patch(facecolor=(0.7, 0.7, 0.7, 1.0), edgecolor='k', label="0 additional matches")]
        if hull_line["obj"] is not None:
            handles.append(hull_line["obj"])
        if doc_id and np.count_nonzero(mask_hl) > 0:
            handles.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor="#00ff55", markeredgecolor='k',
                                  markersize=8, linewidth=0,
                                  label=f"matches involving '{doc_id}'"))
        legend_obj["obj"] = ax.legend(handles=handles, loc="upper right")

        fig.canvas.draw_idle()

    # initial draw
    _refresh("AND")

    # ----------------- UI: toggle buttons -----------------
    ax_btn_and    = fig.add_axes([0.80, 0.92, 0.18, 0.04]); ax_btn_and.set_in_layout(False)
    ax_btn_deldst = fig.add_axes([0.80, 0.87, 0.18, 0.04]); ax_btn_deldst.set_in_layout(False)
    ax_btn_srcdst = fig.add_axes([0.80, 0.82, 0.18, 0.04]); ax_btn_srcdst.set_in_layout(False)

    from matplotlib.widgets import Button as _Btn
    btn_and    = _Btn(ax_btn_and,    "Δ & src & dst")
    btn_deldst = _Btn(ax_btn_deldst, "Δ & dst")
    btn_srcdst = _Btn(ax_btn_srcdst, "src & dst")

    # Summary: _on_and / _on_deldst / _on_srcdst — button callbacks to switch modes.
    # Effect: quick toggling between the three match semantics.
    def _on_and(_evt):    _refresh("AND")
    def _on_deldst(_evt): _refresh("DELDST")
    def _on_srcdst(_evt): _refresh("SRC_DST")

    btn_and.on_clicked(_on_and)
    btn_deldst.on_clicked(_on_deldst)
    btn_srcdst.on_clicked(_on_srcdst)

    plt.tight_layout()
    attach_matplotlib_save_button(fig, default_name="morphism_match_plot.pkl", parent=globals().get("root"))
    plt.show()
