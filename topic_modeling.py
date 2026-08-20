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

import re, csv, os, math, zlib

import numpy as np

from typing import List, Tuple, Optional, Dict, Any

import nltk

from tkinter import filedialog

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin
from sklearn.cluster import AgglomerativeClustering
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
def mk_delta_manifold(item_id, item_text, st_model, return_raw_embeddings: bool = False):

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
        alpha=0.25,            # how much to reward balance (normalized entropy)
        min_cluster_size=5     # penalize k that produces tiny clusters
    ):
        n = len(embeddings)
        if n <= 1: return 1
        if n == 2: return 2
        k_hi = min(int(k_max), n - 1)
        k_lo = max(2, int(k_min))

        best_k, best_score = None, -1e9
        for k in range(k_lo, k_hi + 1):
            try:
                try:
                    model = AgglomerativeClustering(n_clusters=k, linkage='average', metric='cosine')
                except TypeError:
                    model = AgglomerativeClustering(n_clusters=k, linkage='average', affinity='cosine')
                labels = model.fit_predict(embeddings)
            except Exception:
                continue

            if len(set(labels)) < 2:
                continue

            try:
                s = silhouette_score(embeddings, labels, metric='cosine')
            except Exception:
                s = -1.0

            H, f_max, counts = _cluster_balance_metrics(labels, k)
            # small penalty for tiny clusters
            penalty = 0.10 * int((counts < min_cluster_size).sum())
            score = s + alpha * H - penalty

            if score > best_score:
                best_k, best_score = k, score
        return best_k if best_k is not None else 2

    # Summary: cluster_segments — agglomerative clustering with cosine distance
    #          (clamps degenerate cases, handles k==1 fast path).
    # Effect: provides coherent cluster assignments that anchor Δ endpoints.
    def cluster_segments(embeddings, k):
        """
        Cluster with Agglomerative, but clamp k into [1, n] and avoid sklearn if k==1.
        """
        n = len(embeddings)
        if n == 0:
            return np.array([], dtype=int)

        k = int(max(1, min(k, n)))  # clamp

        if k == 1:
            # single cluster label for all segments
            return np.zeros(n, dtype=int)

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
    # Raw, normalized SBERT segment embeddings.  These are kept separate from
    # the document-centered manifold embeddings so downstream analysis can
    # distinguish global SBERT document proximity from local residual morphism
    # geometry.
    raw_sbert_embeddings = embed_segments(segments)
    #embeddings = batched_encode(segments, batch_size=2048)
    embeddings = remove_top_components(raw_sbert_embeddings, n=0)   # try n=1..3
    #k = choose_k_via_silhouette(embeddings)
    k = choose_k_size_aware(embeddings, k_min=8, k_max=12, alpha=0.25, min_cluster_size=8)
    labels = cluster_segments(embeddings, k)
    H, f_max, counts = _cluster_balance_metrics(labels, k)
    if H < 0.55 or f_max > 0.85:
        labels = bisecting_kmeans_spherical(embeddings, k, min_gain=0.01, random_state=0)
    cluster_embs = compute_cluster_embeddings(embeddings, labels)
    #cluster_order = sorted(cluster_embs.keys())
    delta_matrix, cluster_order = compute_delta_matrix(cluster_embs)

    if return_raw_embeddings:
        return delta_matrix, cluster_order, labels, segments, embeddings, k, raw_sbert_embeddings
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

# -----------------------------------------------------------------------------
# Semantic / discursive quality scoring helpers
# -----------------------------------------------------------------------------
_QUALITY_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_QUALITY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+|[^\w\s]")
_QUALITY_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "because", "although",
    "while", "of", "for", "to", "from", "in", "on", "at", "by", "with", "as",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "this", "that", "these", "those", "which", "who", "whom",
    "whose", "not", "no", "it", "its", "they", "their", "we", "our", "you", "your"
}
_QUALITY_VERB_CUES = {
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "do", "does", "did", "can", "could",
    "may", "might", "must", "shall", "should", "will", "would"
}


def _quality_clamp01(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _quality_hmean(vals, eps: float = 1e-6, component_floor: float | None = None) -> float:
    """
    Harmonic mean for quality components.

    A pure harmonic mean is intentionally conservative, but one miscalibrated
    diagnostic component can collapse every cluster toward zero.  `component_floor`
    lets non-core diagnostic components be softly floored before aggregation.
    True zeros from empty text still remain zero unless a floor is explicitly used
    by the caller.
    """
    vals = [_quality_clamp01(v) for v in vals if v is not None]
    if not vals:
        return 0.0
    if component_floor is not None:
        floor = _quality_clamp01(component_floor)
        vals = [floor + (1.0 - floor) * v for v in vals]
    return _quality_clamp01(len(vals) / sum(1.0 / max(v, eps) for v in vals))


def _quality_soft_floor(x, floor: float = 0.10) -> float:
    """Soft floor for diagnostic components so they modulate Q without annihilating it."""
    floor = _quality_clamp01(floor)
    return _quality_clamp01(floor + (1.0 - floor) * _quality_clamp01(x))


def _quality_quantiles(vals) -> dict:
    arr = np.asarray([float(v) for v in vals if np.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    qs = np.quantile(arr, [0.0, 0.25, 0.50, 0.75, 1.0])
    return {"min": float(qs[0]), "p25": float(qs[1]), "median": float(qs[2]), "p75": float(qs[3]), "max": float(qs[4])}




# -----------------------------------------------------------------------------
# Optional language-model fluency scoring for segment semantic quality
# -----------------------------------------------------------------------------
_LM_FLUENCY_CACHE: dict[tuple, dict] = {}
_LM_FLUENCY_WARNED: set[str] = set()


def _quality_sigmoid(x: float) -> float:
    """Numerically guarded logistic function."""
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x >= 40.0:
        return 1.0
    if x <= -40.0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-x)))


def _normalize_lm_nll_to_fluency_scores(
    nll_values,
    calibration: str = "hybrid",
    absolute_center: float = 5.5,
    absolute_scale: float = 1.15,
) -> list[float | None]:
    """
    Convert per-token negative log-likelihood values into [0,1] fluency scores.

    Lower NLL is more fluent.  The default `hybrid` calibration combines:
      - an absolute GPT-style anchor, so globally improbable OCR/layout strings
        remain low even if a document is mostly poor text; and
      - a document-local relative rank, so historically unusual but internally
        coherent prose is not over-penalized simply because the LM finds the
        whole document somewhat surprising.
    """
    raw = []
    for v in nll_values or []:
        try:
            fv = float(v)
            raw.append(fv if np.isfinite(fv) else float("nan"))
        except Exception:
            raw.append(float("nan"))

    finite = np.asarray([v for v in raw if np.isfinite(v)], dtype=float)
    mode = str(calibration or "hybrid").strip().lower().replace("-", "_")
    if mode in {"doc", "document", "relative", "rank", "local"}:
        mode = "relative"
    elif mode in {"abs", "absolute", "global"}:
        mode = "absolute"
    else:
        mode = "hybrid"

    # Relative calibration: values near the document's low-NLL quartile are good;
    # values near the high-NLL tail are poor.  A floor prevents total collapse.
    if finite.size >= 4:
        good = float(np.quantile(finite, 0.20))
        bad = float(np.quantile(finite, 0.90))
        if bad <= good + 1e-6:
            bad = good + 1.0
    elif finite.size > 0:
        good = float(np.min(finite))
        bad = float(np.max(finite))
        if bad <= good + 1e-6:
            bad = good + 1.0
    else:
        good, bad = 3.5, 8.0

    out: list[float | None] = []
    abs_scale = max(1e-6, float(absolute_scale))
    for v in raw:
        if not np.isfinite(v):
            out.append(None)
            continue

        abs_score = _quality_sigmoid((float(absolute_center) - v) / abs_scale)
        rel = 1.0 - _quality_clamp01((v - good) / max(1e-6, bad - good))
        rel_score = 0.05 + 0.95 * rel

        if mode == "absolute":
            score = abs_score
        elif mode == "relative":
            score = rel_score
        else:
            # Hybrid keeps the LM useful for noisy cultural heritage data: the
            # relative component protects unusual but coherent prose, while the
            # absolute component stops all-garbage documents from self-normalizing
            # to high fluency.
            score = 0.55 * rel_score + 0.45 * abs_score
        out.append(_quality_clamp01(score))
    return out


def _get_lm_fluency_model(model_name: str, device: str | None = None):
    """
    Lazy-load and cache a causal language model for fluency scoring.

    This intentionally imports transformers/torch inside the function so the
    pipeline can still run in environments where the optional fluency model is
    disabled or unavailable.
    """
    name = str(model_name or "").strip()
    if not name:
        raise ValueError("empty LM fluency model name")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as ex:
        raise RuntimeError(f"transformers/torch unavailable for LM fluency scoring: {ex}") from ex

    dev = str(device or "").strip().lower()
    if not dev or dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    key = (name, dev)
    if key in _LM_FLUENCY_CACHE:
        return _LM_FLUENCY_CACHE[key]

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(name)
    model.to(dev)
    model.eval()

    payload = {"tokenizer": tokenizer, "model": model, "device": dev, "model_name": name}
    _LM_FLUENCY_CACHE[key] = payload
    return payload


def compute_lm_fluency_scores(
    texts,
    model_name: str | None = "distilgpt2",
    enabled: bool = True,
    device: str | None = "cpu",
    batch_size: int = 8,
    max_length: int = 128,
    min_tokens: int = 3,
    calibration: str = "hybrid",
    absolute_center: float = 5.5,
    absolute_scale: float = 1.15,
    verbose: bool = False,
) -> dict:
    """
    Score segment fluency with a causal language model.

    Returns a dict with:
      - scores: list[float|None], one per input segment, in [0,1]
      - nll: list[float|None], average token negative log-likelihood
      - token_counts: list[int]
      - available: bool
      - error: str

    If disabled or unavailable, scores are None.  The existing segment quality
    function treats None as the historical neutral placeholder, so old behavior
    remains available as a fallback.
    """
    segs = [" ".join(str(t or "").split()) for t in (texts or [])]
    n = len(segs)
    disabled_names = {"", "none", "off", "false", "0", "disable", "disabled", "no"}
    name = str(model_name or "").strip()
    if (not enabled) or name.lower() in disabled_names:
        return {
            "scores": [None] * n,
            "nll": [None] * n,
            "token_counts": [0] * n,
            "available": False,
            "enabled": bool(enabled),
            "model_name": name,
            "device": device,
            "calibration": calibration,
            "error": "LM fluency scoring disabled",
        }

    try:
        import torch
        import torch.nn.functional as F
        lm = _get_lm_fluency_model(name, device=device)
        tokenizer = lm["tokenizer"]
        model = lm["model"]
        dev = lm["device"]

        bs = max(1, int(batch_size or 1))
        mx = max(8, int(max_length or 128))
        min_tok = max(1, int(min_tokens or 1))

        nlls: list[float | None] = []
        token_counts: list[int] = []

        with torch.no_grad():
            for start in range(0, n, bs):
                batch_texts = segs[start:start + bs]
                enc = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=mx,
                )
                input_ids = enc["input_ids"].to(dev)
                attention_mask = enc.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                attention_mask = attention_mask.to(dev)

                # Need at least one shifted target token to compute causal NLL.
                counts = attention_mask.sum(dim=1).detach().cpu().numpy().astype(int).tolist()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1, :]
                targets = input_ids[:, 1:]
                mask = attention_mask[:, 1:].float()

                log_probs = F.log_softmax(logits, dim=-1)
                tok_nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
                denom = mask.sum(dim=1)
                seq_nll = (tok_nll * mask).sum(dim=1) / torch.clamp(denom, min=1.0)
                vals = seq_nll.detach().cpu().numpy().astype(float).tolist()

                for cnt, val in zip(counts, vals):
                    token_counts.append(int(cnt))
                    if int(cnt) < min_tok:
                        nlls.append(None)
                    elif np.isfinite(float(val)):
                        nlls.append(float(val))
                    else:
                        nlls.append(None)

        scores = _normalize_lm_nll_to_fluency_scores(
            nlls,
            calibration=calibration,
            absolute_center=absolute_center,
            absolute_scale=absolute_scale,
        )

        return {
            "scores": scores,
            "nll": nlls,
            "token_counts": token_counts,
            "available": True,
            "enabled": True,
            "model_name": name,
            "device": dev,
            "calibration": str(calibration or "hybrid"),
            "absolute_center": float(absolute_center),
            "absolute_scale": float(absolute_scale),
            "error": "",
        }

    except Exception as ex:
        msg = str(ex)
        warn_key = f"{name}|{msg[:120]}"
        if verbose or warn_key not in _LM_FLUENCY_WARNED:
            print(f"[fluency] LM fluency scoring unavailable; using neutral fallback. model={name!r}; error={msg}", flush=True)
            _LM_FLUENCY_WARNED.add(warn_key)
        return {
            "scores": [None] * n,
            "nll": [None] * n,
            "token_counts": [0] * n,
            "available": False,
            "enabled": True,
            "model_name": name,
            "device": device,
            "calibration": calibration,
            "error": msg,
        }


def _quality_count_score(n: int, lo: int = 5, hi: int = 80) -> float:
    """Soft length score. Very short fragments are weak; very long blobs decay slowly."""
    n = int(max(0, n))
    if n <= 0:
        return 0.0
    if n < lo:
        return _quality_clamp01(n / max(1.0, float(lo)))
    if n <= hi:
        return 1.0
    # Do not destroy long coherent sentences, but make overlong OCR/layout blobs less certain.
    return _quality_clamp01(1.0 - min(0.65, (n - hi) / max(1.0, 2.5 * hi)))


def segment_textuality_score(text: str) -> float:
    """
    Type-agnostic text-likeness score in [0,1].
    It measures alphabetic semantic density, usable word length, lexical diversity,
    and lack of digit/punctuation dominance without assigning content categories.
    """
    s = " ".join(str(text or "").split())
    if not s:
        return 0.0

    words = _QUALITY_WORD_RE.findall(s)
    content_words = [w for w in words if len(w) >= 3]
    n_content = len(content_words)
    n_chars = max(1, len(s))
    alpha_ratio = sum(ch.isalpha() for ch in s) / n_chars
    digit_ratio = sum(ch.isdigit() for ch in s) / n_chars
    punct_ratio = sum((not ch.isalnum()) and (not ch.isspace()) for ch in s) / n_chars

    length_score = _quality_count_score(n_content, lo=5, hi=90)
    alpha_score = _quality_clamp01((alpha_ratio - 0.30) / 0.55)
    digit_score = math.exp(-5.0 * digit_ratio)
    punct_score = math.exp(-4.0 * punct_ratio)

    if n_content:
        unique_ratio = len({w.lower() for w in content_words}) / max(1, n_content)
        lexical_score = _quality_clamp01((unique_ratio - 0.20) / 0.70)
    else:
        lexical_score = 0.0

    fn_count = sum(1 for w in words if w.lower() in _QUALITY_FUNCTION_WORDS)
    # Captions/headings can be meaningful without many function words, so keep this soft.
    function_word_score = 0.45 + 0.55 * _quality_clamp01(fn_count / 3.0)

    return _quality_hmean([
        length_score,
        alpha_score,
        digit_score,
        punct_score,
        lexical_score,
        function_word_score,
    ])


def rough_predication_score(text: str) -> float:
    """
    Lightweight complete-thought likelihood. This is not a document-type tagger;
    it only estimates whether the segment has enough predicate/relational structure
    to support discursive interpretation.
    """
    words = [w.lower() for w in _QUALITY_WORD_RE.findall(str(text or ""))]
    if len(words) < 3:
        return 0.08

    has_aux = any(w in _QUALITY_VERB_CUES for w in words)
    has_verbish = any(
        w.endswith(("ed", "ing", "ize", "izes", "ized", "ise", "ises", "ised", "ate", "ates", "ated", "fy", "fies", "fied"))
        for w in words
    )
    has_relation = any(w in {"because", "although", "while", "when", "where", "which", "that", "therefore", "however", "after", "before", "during"} for w in words)

    if has_aux and (has_verbish or len(words) >= 6):
        return 1.0
    if has_aux or has_verbish:
        return 0.78
    if has_relation and len(words) >= 7:
        return 0.70
    if len(words) >= 12:
        return 0.55
    if len(words) >= 6:
        return 0.35
    return 0.18


def segment_semantic_quality_score(text: str, fluency_score: float | None = None) -> float:
    """
    Segment-level semantic/discursive quality in [0,1].

    When `fluency_score` is supplied, it should be an LM-derived [0,1] score
    where higher means more fluent/predictable as language.  When it is None,
    the historical neutral placeholder (0.65) is used as a fallback so older
    workflows and environments without transformers still run.
    """
    textuality = segment_textuality_score(text)
    predication = rough_predication_score(text)
    fluency = 0.65 if fluency_score is None else _quality_clamp01(fluency_score)
    return _quality_hmean([textuality, predication, fluency])


def _quality_skeletonize(text: str) -> str:
    """Surface-form skeleton used to penalize repeated layout/form templates."""
    s = " ".join(str(text or "").split())
    s = re.sub(r"[A-Z][a-z]+", "Aaaa", s)
    s = re.sub(r"[A-Z]", "A", s)
    s = re.sub(r"[a-z]", "a", s)
    s = re.sub(r"\d", "0", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def cluster_non_template_score(texts: list[str]) -> float:
    """
    Cluster-level anti-template score in [0,1].
    Tight structural artifacts often repeat punctuation/number/name skeletons and
    compress strongly; coherent prose clusters should be less surface-repetitive.
    """
    clean_texts = [" ".join(str(t or "").split()) for t in texts if str(t or "").strip()]
    if not clean_texts:
        return 0.0

    joined = "\n".join(clean_texts)
    raw = joined.encode("utf-8", errors="ignore")
    if len(raw) < 40:
        compression_score = 0.35
    else:
        cr = len(zlib.compress(raw)) / max(1, len(raw))
        compression_score = _quality_clamp01((cr - 0.22) / 0.50)

    skeletons = [_quality_skeletonize(t) for t in clean_texts]
    unique_skeleton_ratio = len(set(skeletons)) / max(1, len(skeletons))
    unique_text_ratio = len(set(clean_texts)) / max(1, len(clean_texts))

    # Penalize clusters whose members are mostly very short repeated forms.
    short_ratio = sum(1 for t in clean_texts if len(_QUALITY_WORD_RE.findall(t)) < 5) / max(1, len(clean_texts))
    short_penalty_score = 1.0 - 0.65 * short_ratio

    return _quality_hmean([
        compression_score,
        unique_skeleton_ratio,
        unique_text_ratio,
        short_penalty_score,
    ])



# -----------------------------------------------------------------------------
# Lexical-overlap helpers for morphism-match acuity diagnostics
# -----------------------------------------------------------------------------
_LEXICAL_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_LEXICAL_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "that", "this", "from", "into",
    "they", "their", "there", "then", "than", "them", "these", "those", "was", "were", "been", "being",
    "have", "has", "had", "does", "did", "can", "could", "would", "should", "shall", "will", "may",
    "might", "must", "its", "his", "her", "she", "him", "our", "ours", "out", "about", "after", "before",
    "over", "under", "between", "during", "through", "where", "which", "while", "when", "what", "who",
    "whom", "whose", "because", "although", "also", "such", "only", "more", "most", "some", "any", "all",
    "each", "other", "same", "very", "using", "used", "use", "one", "two", "three", "within", "without",
    "upon", "onto", "per", "via", "fig", "figure", "table", "vol", "no", "pp", "et", "al"
}


def lexical_content_tokens(texts, min_len: int = 3, remove_stopwords: bool = True) -> list[str]:
    """Return lowercased alphabetic content tokens for lexical-overlap diagnostics."""
    if isinstance(texts, str):
        raw_text = texts
    else:
        try:
            raw_text = "\n".join(str(t or "") for t in texts)
        except Exception:
            raw_text = str(texts or "")
    toks = []
    for tok in _LEXICAL_TOKEN_RE.findall(raw_text.lower()):
        tok = tok.strip("'-")
        if len(tok) < int(min_len):
            continue
        if remove_stopwords and tok in _LEXICAL_STOPWORDS:
            continue
        toks.append(tok)
    return toks


def lexical_counter_from_texts(texts, min_len: int = 3, remove_stopwords: bool = True):
    """Counter of content tokens for one cluster or edge text bundle."""
    return Counter(lexical_content_tokens(texts, min_len=min_len, remove_stopwords=remove_stopwords))


def lexical_overlap_metrics_from_counters(ca, cb) -> dict:
    """
    Compute lexical-overlap metrics between two token-count Counters.

    The overlap_coefficient is shared_unique / min(unique_A, unique_B), which is
    the default lexical-overlap score used for semantic/lexical acuity.  It is
    more tolerant of unequal cluster sizes than Jaccard.
    """
    ca = Counter(ca or {})
    cb = Counter(cb or {})
    sa, sb = set(ca), set(cb)
    inter = sa & sb
    union = sa | sb
    total_a = int(sum(ca.values()))
    total_b = int(sum(cb.values()))
    out = {
        "tokens_a": total_a,
        "tokens_b": total_b,
        "unique_a": int(len(sa)),
        "unique_b": int(len(sb)),
        "shared_unique": int(len(inter)),
        "jaccard": 0.0,
        "dice": 0.0,
        "overlap_coefficient": 0.0,
        "containment_a_in_b": 0.0,
        "containment_b_in_a": 0.0,
        "weighted_jaccard": 0.0,
        "count_cosine": 0.0,
        "shared_token_mass_a": 0.0,
        "shared_token_mass_b": 0.0,
        "lexical_available": bool(total_a > 0 and total_b > 0),
    }
    if not union:
        return out
    min_sum = sum(min(ca.get(t, 0), cb.get(t, 0)) for t in union)
    max_sum = sum(max(ca.get(t, 0), cb.get(t, 0)) for t in union)
    dot = sum(ca.get(t, 0) * cb.get(t, 0) for t in union)
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    mass_a = sum(ca[t] for t in inter)
    mass_b = sum(cb[t] for t in inter)
    out.update({
        "jaccard": float(len(inter) / len(union)) if union else 0.0,
        "dice": float((2 * len(inter) / (len(sa) + len(sb))) if (sa or sb) else 0.0),
        "overlap_coefficient": float(len(inter) / max(1, min(len(sa), len(sb)))),
        "containment_a_in_b": float(len(inter) / max(1, len(sa))),
        "containment_b_in_a": float(len(inter) / max(1, len(sb))),
        "weighted_jaccard": float(min_sum / max(1, max_sum)),
        "count_cosine": float(dot / max(1e-12, norm_a * norm_b)),
        "shared_token_mass_a": float(mass_a / max(1, total_a)),
        "shared_token_mass_b": float(mass_b / max(1, total_b)),
    })
    return out


def lexical_overlap_metrics_from_texts(texts_a, texts_b) -> dict:
    """Convenience wrapper around lexical_counter_from_texts + overlap metrics."""
    return lexical_overlap_metrics_from_counters(
        lexical_counter_from_texts(texts_a),
        lexical_counter_from_texts(texts_b),
    )


def _counter_add(ca, cb):
    """Counter addition that tolerates None and ordinary dict-like inputs."""
    out = Counter(ca or {})
    out.update(Counter(cb or {}))
    return out

def cluster_spread_median_distance(cluster_embeddings) -> float:
    """Return median within-cluster cosine distance, or NaN when unavailable."""
    if cluster_embeddings is None:
        return float("nan")
    X = np.asarray(cluster_embeddings, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    n = X.shape[0]
    if n < 3:
        return float("nan")

    # Guard very large clusters by deterministic subsampling.
    if n > 500:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=500, replace=False)
        X = X[idx]
        n = X.shape[0]

    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sims = np.clip(X @ X.T, -1.0, 1.0)
    iu = np.triu_indices(n, k=1)
    if len(iu[0]) == 0:
        return float("nan")
    dists = 1.0 - sims[iu]
    return float(np.median(dists))


def calibrate_cluster_spread_scores(spread_dist_by_label: dict, floor: float = 0.20) -> dict:
    """
    Convert raw within-cluster spread distances into schema-free Goldilocks scores.

    The previous implementation used a fixed cosine-distance target.  That made
    Q collapse to ~0 when a collection/model/preprocessing regime had naturally
    wider or narrower clusters.  This calibration is local to the document: the
    median spread is treated as the center, and clusters in the extreme tight/loose
    tails are softly penalized.  Scores are floored so spread modulates semantic
    quality rather than annihilating it.
    """
    labels = list(spread_dist_by_label.keys())
    finite = np.asarray([v for v in spread_dist_by_label.values() if np.isfinite(v)], dtype=float)
    floor = _quality_clamp01(floor)
    if finite.size == 0:
        return {lab: 0.65 for lab in labels}

    center = float(np.median(finite))
    q10, q25, q75, q90 = np.quantile(finite, [0.10, 0.25, 0.75, 0.90])
    # Robust scale; guard degenerate documents where all clusters have similar spread.
    scale_candidates = [
        float((q75 - q25) / 1.349) if q75 > q25 else 0.0,
        float((q90 - q10) / 2.563) if q90 > q10 else 0.0,
        0.08,
    ]
    scale = max(scale_candidates)

    out = {}
    for lab, dist in spread_dist_by_label.items():
        if not np.isfinite(dist):
            score = 0.65
        else:
            raw = math.exp(-((float(dist) - center) ** 2) / (2.0 * scale ** 2))
            score = floor + (1.0 - floor) * _quality_clamp01(raw)
        out[lab] = _quality_clamp01(score)
    return out


def cluster_spread_quality_score(cluster_embeddings, target: float | None = None, sigma: float | None = None) -> float:
    """
    Goldilocks score over within-cluster cosine distance.

    If target/sigma are supplied, use the explicit curve; otherwise return a
    broad, non-annihilating local diagnostic.  Document-level calibration is
    preferred via `calibrate_cluster_spread_scores()`.
    """
    med_dist = cluster_spread_median_distance(cluster_embeddings)
    if not np.isfinite(med_dist):
        return 0.60
    if target is None:
        # Broad fallback: penalize only pathological extremes.
        if med_dist < 0.02:
            return 0.35
        if med_dist > 1.20:
            return 0.45
        return 0.75
    if sigma is None or sigma <= 0:
        sigma = 0.25
    return _quality_clamp01(math.exp(-((med_dist - float(target)) ** 2) / (2.0 * float(sigma) ** 2)))


def cluster_semantic_quality_score(
    texts,
    embeddings=None,
    return_components: bool = False,
    spread_score_override: float | None = None,
    spread_distance: float | None = None,
    fluency_scores=None,
    fluency_nlls=None,
    fluency_model_name: str | None = None,
    fluency_available: bool | None = None,
    fluency_calibration: str | None = None,
):
    """
    Cluster-level semantic/discursive quality in [0,1].

    v4 adjustment: p25_segment_quality is retained as a diagnostic, but it no
    longer participates as a raw harmonic-mean component.  In mixed clusters,
    a low lower-quartile score often means "fragment contamination" rather than
    "no semantic core."  The final Q therefore estimates a usable semantic core
    and then applies contamination/support penalties derived partly from p25.

    Components stored in the quality payload:
      - median_segment_quality: central segment-level discursive quality
      - p25_segment_quality: lower-quartile diagnostic, not a hard veto
      - medoid_segment_quality: quality of the centroid-nearest segment
      - semantic_core_quality: median score among usable segments
      - usable_segment_ratio: fraction of segments above usable_threshold
      - fragment_burden: fraction of near-zero fragments
      - p25_penalty_score: soft penalty derived from p25, floored at 0.35
      - contamination_penalty: combined p25/support/fragment penalty
      - non_template_score: anti-template/repetition diagnostic
      - spread_score: calibrated embedding-spread diagnostic
    """
    texts = list(texts or [])
    if not texts:
        return {"quality": 0.0} if return_components else 0.0

    # Segment scores use LM fluency when supplied.  Missing fluency values fall
    # back to the neutral placeholder inside segment_semantic_quality_score().
    if fluency_scores is None:
        fluency_list = [None] * len(texts)
    else:
        fluency_list = list(fluency_scores)
        if len(fluency_list) < len(texts):
            fluency_list = fluency_list + [None] * (len(texts) - len(fluency_list))

    seg_scores = np.asarray([
        segment_semantic_quality_score(t, fluency_score=fluency_list[i] if i < len(fluency_list) else None)
        for i, t in enumerate(texts)
    ], dtype=float)
    seg_scores = np.asarray([_quality_clamp01(v) for v in seg_scores], dtype=float)

    fluency_arr = np.asarray([
        np.nan if v is None else _quality_clamp01(v)
        for v in fluency_list[:len(texts)]
    ], dtype=float)
    nll_arr = np.asarray([
        np.nan if v is None else float(v)
        for v in (list(fluency_nlls)[:len(texts)] if fluency_nlls is not None else [None] * len(texts))
    ], dtype=float)
    finite_fluency = fluency_arr[np.isfinite(fluency_arr)]
    finite_nll = nll_arr[np.isfinite(nll_arr)]

    if embeddings is None:
        X = np.zeros((0, 0), dtype=float)
    else:
        X = np.asarray(embeddings, dtype=float)
        if X.ndim == 1:
            X = X[None, :]

    if X.shape[0] == len(texts) and X.shape[0] > 0:
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        centroid = Xn.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        medoid_idx = int(np.argmax(Xn @ centroid))
        medoid_q = float(seg_scores[medoid_idx])
    else:
        medoid_q = float(np.median(seg_scores))

    median_q = float(np.median(seg_scores))
    q25 = float(np.quantile(seg_scores, 0.25))
    non_template = float(cluster_non_template_score(texts))
    spread = float(cluster_spread_quality_score(X) if spread_score_override is None else _quality_clamp01(spread_score_override))

    # Thresholds are intentionally soft diagnostics, not content-type categories.
    usable_threshold = 0.20
    fragment_threshold = 0.05

    usable_mask = seg_scores >= usable_threshold
    fragment_mask = seg_scores < fragment_threshold
    usable_ratio = float(np.mean(usable_mask)) if seg_scores.size else 0.0
    fragment_burden = float(np.mean(fragment_mask)) if seg_scores.size else 1.0

    if np.any(usable_mask):
        usable_scores = seg_scores[usable_mask]
        semantic_core_quality = float(np.median(usable_scores))
        core_p25_quality = float(np.quantile(usable_scores, 0.25))
    else:
        # If nothing clears the usable threshold, let the center of the distribution
        # speak weakly instead of inventing a high core.
        semantic_core_quality = 0.50 * median_q
        core_p25_quality = 0.50 * q25

    # p25 now lowers confidence as a soft penalty.  It cannot by itself push Q
    # to ~0; pure fragment clusters still score near 0 because their median/core
    # values are also near 0.
    p25_penalty_score = 0.35 + 0.65 * _quality_clamp01(q25 / usable_threshold)
    usable_support_score = 0.35 + 0.65 * _quality_clamp01(usable_ratio)
    fragment_penalty_score = 1.0 - 0.55 * _quality_clamp01(fragment_burden)

    contamination_penalty = _quality_hmean([
        p25_penalty_score,
        usable_support_score,
        fragment_penalty_score,
    ])

    # The semantic core score is no longer harmonically tied to the raw lower
    # quartile.  Median/medoid still contribute, but as a soft center-support
    # penalty.  This lets a mixed cluster with a real discursive/title/prose core
    # land in a low-to-medium Q range instead of being rounded to zero merely
    # because its bottom quartile is made of page-range or locator fragments.
    center_reference = max(median_q, medoid_q)
    semantic_center_support = 0.45 + 0.55 * _quality_clamp01(center_reference / usable_threshold)
    semantic_core_bundle = _quality_clamp01(semantic_core_quality * semantic_center_support)

    structural_penalty = _quality_hmean([
        _quality_soft_floor(non_template, 0.08),
        _quality_soft_floor(spread, 0.08),
    ])

    quality = _quality_clamp01(
        semantic_core_bundle
        * contamination_penalty
        * structural_penalty
    )

    if finite_fluency.size:
        lm_fluency_median = float(np.median(finite_fluency))
        lm_fluency_p25 = float(np.quantile(finite_fluency, 0.25))
        lm_fluency_mean = float(np.mean(finite_fluency))
    else:
        lm_fluency_median = ""
        lm_fluency_p25 = ""
        lm_fluency_mean = ""

    if finite_nll.size:
        lm_nll_median = float(np.median(finite_nll))
        lm_nll_p75 = float(np.quantile(finite_nll, 0.75))
    else:
        lm_nll_median = ""
        lm_nll_p75 = ""

    if return_components:
        return {
            "quality": float(quality),
            "median_segment_quality": median_q,
            "p25_segment_quality": q25,
            "medoid_segment_quality": medoid_q,
            "semantic_core_quality": float(semantic_core_quality),
            "semantic_center_support": float(semantic_center_support),
            "core_p25_quality": float(core_p25_quality),
            "usable_segment_ratio": float(usable_ratio),
            "fragment_burden": float(fragment_burden),
            "p25_penalty_score": float(p25_penalty_score),
            "usable_support_score": float(usable_support_score),
            "fragment_penalty_score": float(fragment_penalty_score),
            "contamination_penalty": float(contamination_penalty),
            "non_template_score": non_template,
            "spread_score": spread,
            "spread_median_cosine_distance": (float(spread_distance) if spread_distance is not None and np.isfinite(spread_distance) else ""),
            "lm_fluency_score_median": lm_fluency_median,
            "lm_fluency_score_p25": lm_fluency_p25,
            "lm_fluency_score_mean": lm_fluency_mean,
            "lm_fluency_nll_median": lm_nll_median,
            "lm_fluency_nll_p75": lm_nll_p75,
            "lm_fluency_model": str(fluency_model_name or ""),
            "lm_fluency_available": bool(fluency_available) if fluency_available is not None else bool(finite_fluency.size),
            "lm_fluency_calibration": str(fluency_calibration or ""),
            "n_segments": int(len(texts)),
            "quality_model": "v11_lm_fluency_core_quality_x_contamination_penalty",
            "usable_threshold": float(usable_threshold),
            "fragment_threshold": float(fragment_threshold),
        }
    return float(quality)


def _cluster_quality_map_from_segments(labels, segments) -> dict:
    """Compute {cluster_label: quality} from text/labels only, no type schema required."""
    if labels is None or segments is None:
        return {}
    try:
        labs = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        segs = list(segments)
    except Exception:
        return {}
    by_lab = defaultdict(list)
    for i, lab in enumerate(labs):
        if i < len(segs):
            by_lab[int(lab)].append(str(segs[i]))
    return {
        int(lab): float(cluster_semantic_quality_score(texts, embeddings=None, spread_score_override=0.65))
        for lab, texts in by_lab.items()
    }


def _cluster_quality_record_from_payload(q_payload, label) -> dict:
    """Return the full quality payload record for a cluster label, if available."""
    if not isinstance(q_payload, dict):
        return {}
    candidates = [label]
    try:
        candidates.append(int(label))
    except Exception:
        pass
    candidates.append(str(label))
    for key in candidates:
        if key in q_payload:
            rec = q_payload[key]
            if isinstance(rec, dict):
                return rec
            return {"quality": rec}
    return {}


def _quality_record_float(rec: dict, key: str, default=""):
    try:
        if not isinstance(rec, dict) or key not in rec or rec.get(key) == "":
            return default
        return float(rec.get(key))
    except Exception:
        return default


def _cluster_quality_map_from_cdm(data, default: float = 1.0, segments=None, repair_near_zero: bool = True) -> dict:
    """
    Return {cluster_label: quality}.

    Legacy 5/6-tuples default to `default`, unless `segments` are supplied, in
    which case a text-only Q estimate is computed.  If a 7-tuple quality payload
    is present but all values are near zero, this function treats it as a likely
    pre-calibration payload and repairs it from `segments` when available.
    """
    try:
        cluster_order = list(data[1])
        labels = data[2]
    except Exception:
        return {}

    def _from_payload(q_payload):
        if isinstance(q_payload, dict):
            return {int(k): _quality_clamp01(v.get("quality", v) if isinstance(v, dict) else v) for k, v in q_payload.items()}
        if q_payload is not None:
            try:
                arr = list(q_payload)
                return {int(lab): _quality_clamp01(arr[i]) for i, lab in enumerate(cluster_order) if i < len(arr)}
            except Exception:
                return {}
        return {}

    q_payload = data[6] if isinstance(data, (tuple, list)) and len(data) >= 7 else None
    q_map = _from_payload(q_payload)

    if q_map:
        vals = np.asarray(list(q_map.values()), dtype=float)
        if repair_near_zero and segments is not None and vals.size and float(np.nanmax(vals)) < 1e-4:
            repaired = _cluster_quality_map_from_segments(labels, segments)
            if repaired:
                return repaired
        return q_map

    if segments is not None:
        estimated = _cluster_quality_map_from_segments(labels, segments)
        if estimated:
            return estimated

    return {int(lab): float(default) for lab in cluster_order}


# -----------------------------------------------------------------------------
# Document-level embedding helpers for baseline comparison
# -----------------------------------------------------------------------------
def _unit_normalize_vector(x, eps: float = 1e-12):
    """Return a 1D unit vector, or None if the input is missing/degenerate."""
    try:
        v = np.asarray(x, dtype=float).reshape(-1)
        if v.size == 0 or not np.isfinite(v).all():
            return None
        n = float(np.linalg.norm(v))
        if n <= eps:
            return None
        return v / n
    except Exception:
        return None


def compute_document_embedding_payload(
    embeddings,
    cluster_embeddings=None,
    cluster_order=None,
    cluster_semantic_quality=None,
    raw_sbert_embeddings=None,
    sbert_model_name: str | None = "all-MiniLM-L6-v2",
    normalize_embeddings: bool = True,
) -> dict:
    """
    Build item-level embedding baselines stored inside each CDM tuple.

    Two document vectors are now stored explicitly:
      - ``manifold_residual_document_embedding``: normalized mean of the
        document-centered / row-renormalized embeddings used to build the local
        CDM morphism geometry.
      - ``raw_sbert_document_embedding``: normalized mean of the raw normalized
        SBERT segment embeddings before document centering.

    The older ``document_embedding`` key is retained as a deprecated alias for
    ``manifold_residual_document_embedding`` so legacy readers can still open
    existing outputs, but new comparison diagnostics should use the explicit
    raw/residual field names.
    """
    payload = {
        "manifold_residual_document_embedding": None,
        "manifold_residual_document_embedding_method": "mean_document_centered_segment_embeddings",
        "manifold_residual_document_embedding_source": "manifold_residual_segment_embeddings_after_document_centering_and_row_renorm",
        "manifold_residual_document_embedding_available": False,
        "manifold_residual_document_embedding_dim": 0,
        "manifold_residual_document_embedding_norm_before_unit": "",

        "raw_sbert_document_embedding": None,
        "raw_sbert_document_embedding_method": "mean_raw_sbert_segment_embeddings",
        "raw_sbert_document_embedding_source": "raw_sbert_segment_embeddings_before_document_centering",
        "raw_sbert_document_embedding_available": False,
        "raw_sbert_document_embedding_dim": 0,
        "raw_sbert_document_embedding_norm_before_unit": "",
        "raw_sbert_model_name": str(sbert_model_name or ""),
        "raw_sbert_normalize_embeddings": bool(normalize_embeddings),

        # Deprecated alias retained for backwards compatibility only.
        "document_embedding": None,
        "document_embedding_method": "deprecated_alias_for_manifold_residual_document_embedding",
        "document_embedding_source": "deprecated_alias_for_manifold_residual_document_embedding",
        "document_embedding_available": False,
        "document_embedding_dim": 0,
        "document_embedding_norm_before_unit": "",

        "document_embedding_cluster_mean": None,
        "document_embedding_cluster_quality_weighted": None,
    }

    def _store_doc_vector(prefix: str, arr_like) -> None:
        try:
            E = np.asarray(arr_like, dtype=float)
            if E.ndim == 1:
                E = E[None, :]
            if E.size and E.shape[0] > 0:
                raw = np.nanmean(E, axis=0)
                norm_before = float(np.linalg.norm(raw))
                unit = _unit_normalize_vector(raw)
                if unit is not None:
                    payload[f"{prefix}_document_embedding"] = unit
                    payload[f"{prefix}_document_embedding_available"] = True
                    payload[f"{prefix}_document_embedding_dim"] = int(unit.shape[0])
                    payload[f"{prefix}_document_embedding_norm_before_unit"] = norm_before
        except Exception as ex:
            payload[f"{prefix}_document_embedding_error"] = str(ex)

    _store_doc_vector("manifold_residual", embeddings)
    if raw_sbert_embeddings is not None:
        _store_doc_vector("raw_sbert", raw_sbert_embeddings)

    # Backward-compatible alias: old code that requests document_embedding gets
    # the residual/manifold baseline, but new outputs use explicit cosine names.
    if payload.get("manifold_residual_document_embedding") is not None:
        payload["document_embedding"] = payload.get("manifold_residual_document_embedding")
        payload["document_embedding_available"] = bool(payload.get("manifold_residual_document_embedding_available"))
        payload["document_embedding_dim"] = int(payload.get("manifold_residual_document_embedding_dim") or 0)
        payload["document_embedding_norm_before_unit"] = payload.get("manifold_residual_document_embedding_norm_before_unit", "")

    # Optional cluster-mean baseline.
    try:
        C = np.asarray(cluster_embeddings, dtype=float)
        if C.ndim == 1:
            C = C[None, :]
        if C.size and C.shape[0] > 0:
            cm = _unit_normalize_vector(np.nanmean(C, axis=0))
            if cm is not None:
                payload["document_embedding_cluster_mean"] = cm
    except Exception:
        pass

    # Optional Q-weighted cluster-centroid baseline.  Kept separate so the
    # default baseline remains an ordinary pooled document embedding.
    try:
        C = np.asarray(cluster_embeddings, dtype=float)
        if C.ndim == 1:
            C = C[None, :]
        if C.size and C.shape[0] > 0:
            weights = []
            for idx, lab in enumerate(list(cluster_order or range(C.shape[0]))):
                q = 1.0
                if isinstance(cluster_semantic_quality, dict):
                    rec = _cluster_quality_record_from_payload(cluster_semantic_quality, lab)
                    if isinstance(rec, dict) and rec:
                        q = _quality_clamp01(rec.get("quality", 1.0))
                    elif lab in cluster_semantic_quality:
                        q = _quality_clamp01(cluster_semantic_quality.get(lab, 1.0))
                weights.append(max(1e-6, float(q)))
            w = np.asarray(weights[: C.shape[0]], dtype=float)
            if w.size == C.shape[0] and np.isfinite(w).all() and float(w.sum()) > 0:
                qmean = np.average(C, axis=0, weights=w)
                qvec = _unit_normalize_vector(qmean)
                if qvec is not None:
                    payload["document_embedding_cluster_quality_weighted"] = qvec
    except Exception:
        pass

    return payload


def _document_embedding_from_cdm(data, preferred_key: str = "manifold_residual_document_embedding") -> tuple:
    """
    Return (unit_vector_or_None, metadata_dict) for a CDM tuple.

    Preferred explicit keys:
      - ``manifold_residual_document_embedding`` for the document-centered CDM
        residual baseline.
      - ``raw_sbert_document_embedding`` for the raw normalized SBERT baseline
        before document centering.

    Legacy tuples are repaired only for the residual/manifold baseline, using the
    old ``document_embedding`` alias or a normalized mean of cluster centroids.
    Raw SBERT vectors cannot be recovered from old CDMs without re-embedding.
    """
    preferred_key = str(preferred_key or "manifold_residual_document_embedding")
    meta = {
        "available": False,
        "source": "missing",
        "method": "",
        "dim": 0,
        "key": preferred_key,
    }

    payload = None
    if isinstance(data, (tuple, list)) and len(data) >= 8:
        payload = data[7]

    if preferred_key in {"raw", "raw_sbert", "raw_document_embedding"}:
        preferred_key = "raw_sbert_document_embedding"
    elif preferred_key in {"residual", "manifold", "manifold_residual"}:
        preferred_key = "manifold_residual_document_embedding"

    if preferred_key == "raw_sbert_document_embedding":
        candidate_keys = ("raw_sbert_document_embedding", "raw_document_embedding")
        allow_legacy_fallback = False
    elif preferred_key == "manifold_residual_document_embedding":
        candidate_keys = (
            "manifold_residual_document_embedding",
            "document_embedding",  # deprecated alias for residual/manifold baseline
            "embedding",
            "document_vector",
            "document_embedding_cluster_mean",
        )
        allow_legacy_fallback = True
    else:
        candidate_keys = (preferred_key, "document_embedding", "embedding", "document_vector", "document_embedding_cluster_mean")
        allow_legacy_fallback = True

    if isinstance(payload, dict):
        for key in candidate_keys:
            if key in payload:
                v = _unit_normalize_vector(payload.get(key))
                if v is not None:
                    method_key = f"{key}_method"
                    source_key = f"{key}_source"
                    dim_key = f"{key}_dim"
                    meta.update({
                        "available": True,
                        "source": str(payload.get(source_key, payload.get("document_embedding_source", "payload"))),
                        "method": str(payload.get(method_key, payload.get("document_embedding_method", key))),
                        "dim": int(payload.get(dim_key, int(v.shape[0])) or int(v.shape[0])),
                        "key": key,
                    })
                    if key == "raw_sbert_document_embedding":
                        meta["model_name"] = str(payload.get("raw_sbert_model_name", ""))
                        meta["normalize_embeddings"] = bool(payload.get("raw_sbert_normalize_embeddings", True))
                    return v, meta
    elif payload is not None and allow_legacy_fallback:
        v = _unit_normalize_vector(payload)
        if v is not None:
            meta.update({"available": True, "source": "payload_array", "method": "payload_array", "dim": int(v.shape[0]), "key": "payload_array"})
            return v, meta

    if not allow_legacy_fallback:
        meta.update({"available": False, "source": "missing_raw_sbert_document_embedding", "method": "requires_new_build_or_reembedding"})
        return None, meta

    # Legacy fallback for residual/manifold baseline: mean cluster centroids. This
    # is not identical to a true segment-pooled residual embedding, but it keeps
    # old saved CDM dictionaries usable for baseline comparison.
    try:
        cluster_order = data[1]
        cluster_embeddings = data[4]
        if isinstance(cluster_embeddings, np.ndarray):
            C = np.asarray(cluster_embeddings, dtype=float)
        elif isinstance(cluster_embeddings, dict):
            C = np.vstack([cluster_embeddings[label] for label in cluster_order])
        else:
            C = np.asarray(cluster_embeddings, dtype=float)
        if C.ndim == 1:
            C = C[None, :]
        v = _unit_normalize_vector(np.nanmean(C, axis=0))
        if v is not None:
            meta.update({"available": True, "source": "legacy_cluster_centroid_mean", "method": "mean_cluster_centroids", "dim": int(v.shape[0]), "key": "legacy_cluster_centroid_mean"})
            return v, meta
    except Exception as ex:
        meta["error"] = str(ex)

    return None, meta

def _document_embedding_cosine_from_maps(src_doc, tgt_doc, embedding_map: dict) -> tuple:
    """Return (cosine_or_blank, available_bool) for two document ids."""
    try:
        a = embedding_map.get(src_doc)
        b = embedding_map.get(tgt_doc)
        if a is None:
            a = embedding_map.get(str(src_doc))
        if b is None:
            b = embedding_map.get(str(tgt_doc))
        if a is None or b is None:
            return "", False
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        if a.shape != b.shape or a.size == 0:
            return "", False
        return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12))), True
    except Exception:
        return "", False


# -----------------------------------------------------------------------------
# Function: build_cluster_delta_matrix
# Summary:
#   Clusters segments, aggregates cluster texts & embeddings, computes LDA topic
#   distributions, PC1 directions, pairwise deltas, and a schema-free harmonic
#   mean semantic/discursive quality score per cluster, and an item-level
#   document embedding baseline. Returns an 8-tuple:
#   (Δ, cluster_order, raw_labels, topic_dists, cluster_embeddings, cluster_dirs,
#    cluster_semantic_quality, document_embedding_payload). Downstream unpackers
#   accept legacy 6/7-tuples and the new 8-tuples.
# -----------------------------------------------------------------------------
def build_cluster_delta_matrix(
    segments,
    embeddings,
    labels,
    lda_model,
    lda_dictionary,
    num_topics=10,
    n_clusters: int = 5,
    semantic_fluency_enabled: bool = True,
    semantic_fluency_model_name: str | None = "distilgpt2",
    semantic_fluency_device: str | None = "cpu",
    semantic_fluency_batch_size: int = 8,
    semantic_fluency_max_length: int = 128,
    semantic_fluency_min_tokens: int = 3,
    semantic_fluency_calibration: str = "hybrid",
    semantic_fluency_absolute_center: float = 5.5,
    semantic_fluency_absolute_scale: float = 1.15,
    raw_sbert_embeddings=None,
    raw_sbert_model_name: str | None = "all-MiniLM-L6-v2",
):
    # Step 1: Cluster segments using embeddings
    #clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    #labels = clustering.fit_predict(embeddings)

    # Step 2: Organize segments and embeddings by cluster
    cluster_texts = defaultdict(list)
    cluster_embeddings_raw = defaultdict(list)

    for i, label in enumerate(labels):
        cluster_texts[label].append(segments[i])
        cluster_embeddings_raw[label].append(embeddings[i])

    # Step 3: Compute mean embeddings, concatenated texts, LDA topic distributions,
    # and cluster semantic quality.
    cluster_embeddings_dict = {}
    cluster_topic_distributions = {}
    cluster_semantic_quality = {}

    # Optional real LM fluency scoring.  This is computed once per document and
    # then sliced by cluster so the same per-segment scores are reused for every
    # cluster-level quality calculation.  If the model is unavailable, the
    # segment scorer falls back to its neutral historical placeholder.
    fluency_payload = compute_lm_fluency_scores(
        segments,
        model_name=semantic_fluency_model_name,
        enabled=bool(semantic_fluency_enabled),
        device=semantic_fluency_device,
        batch_size=int(semantic_fluency_batch_size),
        max_length=int(semantic_fluency_max_length),
        min_tokens=int(semantic_fluency_min_tokens),
        calibration=semantic_fluency_calibration,
        absolute_center=float(semantic_fluency_absolute_center),
        absolute_scale=float(semantic_fluency_absolute_scale),
        verbose=False,
    )
    segment_fluency_scores = list(fluency_payload.get("scores", [None] * len(segments)))
    segment_fluency_nlls = list(fluency_payload.get("nll", [None] * len(segments)))

    # Calibrate embedding-spread quality within the document rather than against
    # a fixed global target.  This prevents the spread diagnostic from flattening
    # all Q values when a collection has naturally wide/narrow SBERT clusters.
    cluster_spread_distances = {
        label: cluster_spread_median_distance(cluster_embeddings_raw[label])
        for label in sorted(cluster_texts.keys())
    }
    cluster_spread_scores = calibrate_cluster_spread_scores(cluster_spread_distances)

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

        # NEW: schema-free semantic/discursive quality score with optional
        # LM fluency values aligned to the original segment list.
        cluster_segment_indices = [i for i, lab in enumerate(labels) if int(lab) == int(label)]
        cluster_fluency_scores = [
            segment_fluency_scores[i] if i < len(segment_fluency_scores) else None
            for i in cluster_segment_indices
        ]
        cluster_fluency_nlls = [
            segment_fluency_nlls[i] if i < len(segment_fluency_nlls) else None
            for i in cluster_segment_indices
        ]
        cluster_semantic_quality[label] = cluster_semantic_quality_score(
            cluster_texts[label],
            cluster_embeddings_raw[label],
            return_components=True,
            spread_score_override=cluster_spread_scores.get(label, 0.65),
            spread_distance=cluster_spread_distances.get(label),
            fluency_scores=cluster_fluency_scores,
            fluency_nlls=cluster_fluency_nlls,
            fluency_model_name=fluency_payload.get("model_name", semantic_fluency_model_name),
            fluency_available=bool(fluency_payload.get("available", False)),
            fluency_calibration=fluency_payload.get("calibration", semantic_fluency_calibration),
        )
        cluster_semantic_quality[label]["quality"] = float(cluster_semantic_quality[label]["quality"])
        cluster_semantic_quality[label]["label"] = int(label)
        cluster_semantic_quality[label]["lm_fluency_enabled"] = bool(semantic_fluency_enabled)
        cluster_semantic_quality[label]["lm_fluency_error"] = str(fluency_payload.get("error", ""))
        cluster_semantic_quality[label]["lm_fluency_token_max_length"] = int(semantic_fluency_max_length)

    # --- per-cluster principal directions (PC1) ---
    doc_mean = np.mean(embeddings, axis=0)
    cluster_principal_dirs = {}

    for label in sorted(cluster_texts.keys()):
        X = np.vstack(cluster_embeddings_raw[label])  # (n_i, d)
        Xc = X - X.mean(axis=0, keepdims=True)
        if Xc.shape[0] >= 2:
            # SVD: first right singular vector = principal direction
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
    cluster_dirs = np.array([cluster_principal_dirs[label] for label in cluster_order])

    # Step 4: Build delta matrix (pairwise deltas)
    n = len(cluster_order)
    d = embeddings[0].shape[0]
    delta_matrix = np.zeros((n, n, d))
    delta_matrix = cluster_embeddings[None, :, :] - cluster_embeddings[:, None, :]

    # Step 5: Document-level embedding baseline.  This is stored at the item
    # level of the CDM tuple so top-acuity morphism matches can report an
    # ordinary document-embedding cosine alongside edge-quality scores.
    document_embedding_payload = compute_document_embedding_payload(
        embeddings=embeddings,
        cluster_embeddings=cluster_embeddings,
        cluster_order=cluster_order,
        cluster_semantic_quality=cluster_semantic_quality,
        raw_sbert_embeddings=raw_sbert_embeddings,
        sbert_model_name=raw_sbert_model_name,
        normalize_embeddings=True,
    )

    return (
        delta_matrix,                  # [n x n x d]
        cluster_order,                 # list of label integers
        labels,                        # original labels per segment
        cluster_topic_distributions,   # dict[label] → np.array[num_topics]
        cluster_embeddings,            # np.array[n x d]
        cluster_dirs,                  # np.array[n x d], PC1 per cluster
        cluster_semantic_quality,      # dict[label] → {quality, components...}
        document_embedding_payload     # dict with item-level embedding baseline
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
    srcdst_threshold: float = None,
    quality_threshold: float = 0.0,
    segments_by_doc: dict | None = None,
    visual_projection: str = "global_pca",
    visual_surface_mode: str = "quality_ellipsoid_mst",
    visual_mst_quality_lambda: float = 2.0,
    visual_q_node_radius_scale: float = 10.0,
    visual_q_edge_width_scale: float = 20.0,
):
    """
    3D visualization of directional overlap between document delta manifolds.

    v6 semantic-quality update:
      - Accepts legacy 5/6-tuples and new 7-tuples whose final payload is
        cluster_semantic_quality.
      - Carries per-cluster Q into cluster hover text, point size/alpha, delta
        entries, match masks, selected-delta annotation, and the details table.
      - Adds a semantic-quality floor slider.  A cross-document match must pass:
            Δ >= delta_threshold
            src_PC1 >= pc1_threshold and dst_PC1 >= pc1_threshold, when PC1 exists
            Q_pair >= semantic_quality_threshold
        where Q_pair is the harmonic mean of the two edge qualities.
      - Lazy-loads the SBERT lexical helper only when the details table needs it.

    v7 projection update:
      - `visual_projection="global_pca"` fits one PCA over all document cluster
        centroids and uses that shared coordinate system for centroid points,
        document hulls, delta arrows, and displayed PC1 arrows.
      - `visual_projection="raw_first3"` restores the previous behavior of
        plotting embedding dimensions 0, 1, and 2 directly.
      - Full-dimensional Δ/PC1/Q matching is unchanged; projection only affects
        where geometry is drawn in the 3D scene.

    v8 surface update:
      - `visual_surface_mode="quality_ellipsoid_mst"` is the default. It draws a
        semantic-Q-weighted covariance ellipsoid plus an MST skeleton per document.
      - Other supported surface modes: `hull`, `quality_ellipsoid`, `skeleton`,
        and `none`. The in-figure Surface button cycles among those modes.

    v9 Q-aware skeleton display update:
      - The MST now minimizes the Q-penalized edge cost:
            cost(i,j) = distance(i,j) * exp(lambda * (1 - average_Q(i,j)))
        so geometrically plausible high-Q links are favored over low-Q links.
      - Cluster point radius is directly proportional to Q:
            radius_points = visual_q_node_radius_scale * Q
        and marker area is derived from that radius.
      - Skeleton line width is directly proportional to the average endpoint Q,
        using `visual_q_edge_width_scale` as the width at Q=1.0.
      - Skeleton segments use a brighter, more saturated version of each item color.
    """
    if srcdst_threshold is None:
        srcdst_threshold = cos_threshold

    current_thr = {
        "delta": float(cos_threshold),
        "pc1": float(srcdst_threshold),
        "quality": float(quality_threshold),
    }

    timer2 = Timer()
    timer2.start()

    def _safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    def _fmt_q(x):
        return f"{_safe_float(x):.3f}"

    def _q_hmean(a, b, eps=1e-12):
        a = _quality_clamp01(a)
        b = _quality_clamp01(b)
        den = a + b
        if den <= eps:
            return 0.0
        return float(2.0 * a * b / den)

    def _q_record_from_payload(q_payload, label):
        """Return a per-cluster quality record when the 7th tuple item is dict-like."""
        if not isinstance(q_payload, dict):
            return None
        candidates = [label]
        try:
            candidates.append(int(label))
        except Exception:
            pass
        candidates.append(str(label))
        for key in candidates:
            if key in q_payload:
                rec = q_payload[key]
                if isinstance(rec, dict):
                    return rec
                return {"quality": rec}
        return None

    def _unpack_data(data, doc_id=None):
        """
        Accept 5-, 6-, or 7-tuples.
        Returns:
          delta_matrix, cluster_order, labels, topic_dists, emb_arr, cluster_dirs, q_map, q_payload
        """
        if len(data) >= 6:
            delta_matrix, cluster_order, labels, cluster_topic_distributions, cluster_embeddings, cluster_dirs = data[:6]
        else:
            delta_matrix, cluster_order, labels, cluster_topic_distributions, cluster_embeddings = data[:5]
            cluster_dirs = None

        # Normalize cluster_embeddings to ndarray aligned with cluster_order.
        if isinstance(cluster_embeddings, np.ndarray):
            emb_arr = cluster_embeddings
        elif isinstance(cluster_embeddings, dict):
            emb_arr = np.vstack([cluster_embeddings[label] for label in cluster_order])
        else:
            emb_arr = np.asarray(cluster_embeddings)
        if emb_arr.ndim == 1:
            emb_arr = emb_arr[None, :]

        segs = None
        if isinstance(segments_by_doc, dict) and doc_id in segments_by_doc:
            segs = segments_by_doc.get(doc_id)

        # Uses the v4/v5 helper.  Legacy tuples default to Q=1.0 unless segment
        # text is supplied, and near-zero pre-calibration payloads can be repaired.
        q_map = _cluster_quality_map_from_cdm(data, default=1.0, segments=segs, repair_near_zero=True)
        q_payload = data[6] if isinstance(data, (tuple, list)) and len(data) >= 7 else None

        return delta_matrix, cluster_order, labels, cluster_topic_distributions, emb_arr, cluster_dirs, q_map, q_payload

    # Palette per document.
    doc_ids = list(document_cluster_data.keys())
    colormap = cm.get_cmap('tab10', len(doc_ids))
    doc_colors = {doc: colormap(i) for i, doc in enumerate(doc_ids)}

    # ---------------- Visual projection -------------------------------------
    # The match calculations below always use the full SBERT-dimensional
    # vectors.  This projection object controls only the 3D coordinates drawn in
    # Matplotlib.  Global PCA makes spatial relationships in the visualizer more
    # meaningful than the previous raw-first-3-dim slice, while still remaining
    # a lossy 3D display of high-dimensional geometry.
    projection_mode = str(visual_projection or "global_pca").strip().lower().replace("-", "_")
    if projection_mode in {"pca", "globalpca", "global_pca", "pca3", "pca_3d"}:
        projection_mode = "global_pca"
    elif projection_mode in {"raw", "first3", "first_3", "raw3", "raw_first3"}:
        projection_mode = "raw_first3"
    else:
        print(f"[viz] unknown visual_projection={visual_projection!r}; using global_pca", flush=True)
        projection_mode = "global_pca"

    def _pad_to_3(A):
        A = np.asarray(A, dtype=float)
        was_1d = (A.ndim == 1)
        if was_1d:
            A = A[None, :]
        if A.shape[1] >= 3:
            out = A[:, :3]
        else:
            out = np.zeros((A.shape[0], 3), dtype=float)
            if A.shape[1] > 0:
                out[:, :A.shape[1]] = A
        return out[0] if was_1d else out

    def _build_visual_projection():
        centroid_rows = []
        centroid_keys = []

        for _doc_id, _data in document_cluster_data.items():
            try:
                _, _cluster_order, _, _, _E, _, _, _ = _unpack_data(_data, doc_id=_doc_id)
                _E = np.asarray(_E, dtype=float)
                if _E.ndim == 1:
                    _E = _E[None, :]
                for _idx, _label in enumerate(_cluster_order):
                    if _idx < _E.shape[0]:
                        centroid_rows.append(_E[_idx])
                        centroid_keys.append((_doc_id, int(_idx)))
            except Exception as _ex:
                print(f"[viz] projection skip for {_doc_id}: {_ex}", flush=True)

        if not centroid_rows:
            def _zero_project_vec(_v):
                return np.zeros(3, dtype=float)
            return {
                "mode": "empty",
                "label": "Empty projection",
                "coord_by_key": {},
                "project_vec": _zero_project_vec,
                "axis_labels": ("X", "Y", "Z"),
                "explained_variance_ratio": np.zeros(3, dtype=float),
            }

        X_cent = np.vstack(centroid_rows).astype(float)
        X_cent = np.nan_to_num(X_cent, nan=0.0, posinf=0.0, neginf=0.0)

        if projection_mode == "global_pca" and X_cent.shape[0] >= 2 and X_cent.shape[1] >= 2:
            n_comp = int(min(3, X_cent.shape[0], X_cent.shape[1]))
            try:
                pca = PCA(n_components=n_comp)
                coords = pca.fit_transform(X_cent)
                coords3 = _pad_to_3(coords)
                components = np.asarray(pca.components_, dtype=float)
                evr = _pad_to_3(np.asarray(pca.explained_variance_ratio_, dtype=float))
                coord_by_key = {key: coords3[i] for i, key in enumerate(centroid_keys)}

                def _pca_project_vec(v):
                    vv = np.asarray(v, dtype=float).reshape(-1)
                    if vv.shape[0] != components.shape[1]:
                        # Defensive fallback for malformed legacy data.
                        return _pad_to_3(vv)
                    return _pad_to_3(components @ vv)

                evr_txt = ", ".join(f"PC{k+1}={evr[k]:.1%}" for k in range(3))
                return {
                    "mode": "global_pca",
                    "label": f"Global PCA-3D over {len(centroid_keys)} cluster centroids ({evr_txt})",
                    "coord_by_key": coord_by_key,
                    "project_vec": _pca_project_vec,
                    "axis_labels": ("PC1", "PC2", "PC3"),
                    "explained_variance_ratio": evr,
                    "pca_components": components,
                }
            except Exception as _ex:
                print(f"[viz] global PCA projection failed ({_ex}); falling back to raw_first3", flush=True)

        # Raw fallback / legacy mode.
        coords3 = _pad_to_3(X_cent)
        coord_by_key = {key: coords3[i] for i, key in enumerate(centroid_keys)}

        def _raw_project_vec(v):
            return _pad_to_3(np.asarray(v, dtype=float).reshape(-1))

        return {
            "mode": "raw_first3",
            "label": "Raw embedding dimensions 0, 1, 2",
            "coord_by_key": coord_by_key,
            "project_vec": _raw_project_vec,
            "axis_labels": ("dim0", "dim1", "dim2"),
            "explained_variance_ratio": np.zeros(3, dtype=float),
        }

    projection_state = _build_visual_projection()
    coord_by_key = projection_state["coord_by_key"]
    project_vec3 = projection_state["project_vec"]
    print(f"[viz] visual projection: {projection_state['label']}", flush=True)

    # ---------------- Surface / volume mode ---------------------------------
    # Surface drawing is intentionally display-only: it summarizes the projected
    # cluster-centroid geometry and does not affect full-dimensional Δ/PC1/Q
    # matching.  The default emphasizes high-Q semantic mass rather than raw
    # outer extrema by using a Q-weighted covariance ellipsoid plus an MST
    # skeleton.
    def _normalize_surface_mode(mode) -> str:
        m = str(mode or "quality_ellipsoid_mst").strip().lower()
        m = m.replace("-", "_").replace(" ", "_").replace("+", "_").replace("/", "_")
        m = re.sub(r"_+", "_", m).strip("_")

        if m in {"", "default", "both", "all", "q_ellipsoid_mst", "ellipsoid_mst",
                 "quality_ellipsoid_mst", "quality_weighted_ellipsoid_mst",
                 "quality_ellipsoid_skeleton", "quality_weighted_ellipsoid_skeleton",
                 "ellipsoid_skeleton", "ellipsoid_and_mst", "ellipsoid_with_mst",
                 "qellipsoid_mst"}:
            return "quality_ellipsoid_mst"
        if m in {"none", "off", "no", "nosurface", "no_surface", "surface_off", "hide"}:
            return "none"
        if m in {"hull", "convex_hull", "convexhull", "poly_hull", "polygon_hull"}:
            return "hull"
        if m in {"ellipsoid", "quality_ellipsoid", "q_ellipsoid", "qellipsoid",
                 "quality_weighted_ellipsoid", "weighted_ellipsoid", "covariance_ellipsoid",
                 "q_weighted_ellipsoid"}:
            return "quality_ellipsoid"
        if m in {"skeleton", "mst", "mst_skeleton", "tree", "minimum_spanning_tree"}:
            return "skeleton"

        print(f"[viz] unknown visual_surface_mode={mode!r}; using quality_ellipsoid_mst", flush=True)
        return "quality_ellipsoid_mst"

    surface_modes = ["quality_ellipsoid_mst", "hull", "quality_ellipsoid", "skeleton", "none"]
    surface_labels = {
        "quality_ellipsoid_mst": "Q ellipsoid + MST",
        "hull": "Hull",
        "quality_ellipsoid": "Q ellipsoid",
        "skeleton": "MST skeleton",
        "none": "None",
    }
    surface_state = {"mode": _normalize_surface_mode(visual_surface_mode)}
    print(f"[viz] surface mode: {surface_labels[surface_state['mode']]}", flush=True)

    # 1) Collect all deltas across docs (+ PC1 and semantic-quality metadata).
    print("[viz] collecting directed cluster morphisms", flush=True)
    all_entries = []
    q_payload_by_doc = {}
    q_map_by_doc = {}

    for doc_id, data in document_cluster_data.items():
        delta_matrix, cluster_order, _, _, cluster_embeddings, cluster_dirs, q_map, q_payload = _unpack_data(data, doc_id=doc_id)
        q_payload_by_doc[doc_id] = q_payload
        q_map_by_doc[doc_id] = q_map

        emb3d = np.vstack([
            coord_by_key.get((doc_id, int(_idx)), _pad_to_3(cluster_embeddings[_idx]))
            for _idx in range(cluster_embeddings.shape[0])
        ])
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

                dfull = delta_matrix[i, j, :]
                nrm = np.linalg.norm(dfull)
                if nrm <= 0:
                    continue

                i_lab = cluster_order[i]
                j_lab = cluster_order[j]
                q_src = _quality_clamp01(q_map.get(int(i_lab), q_map.get(str(i_lab), 1.0)))
                q_dst = _quality_clamp01(q_map.get(int(j_lab), q_map.get(str(j_lab), 1.0)))
                q_edge_min = float(min(q_src, q_dst))
                q_edge_hmean = _q_hmean(q_src, q_dst)

                all_entries.append({
                    "doc": doc_id,
                    "start3": emb3d[i],
                    "delta3": emb3d[j] - emb3d[i],
                    "dir_full": dfull / nrm,
                    "delta_full": dfull,
                    "src_dir_full": cluster_dirs_unit[i] if cluster_dirs_unit is not None else None,
                    "dst_dir_full": cluster_dirs_unit[j] if cluster_dirs_unit is not None else None,
                    "i_idx": i,
                    "j_idx": j,
                    "i_lab": i_lab,
                    "j_lab": j_lab,
                    "src_quality": float(q_src),
                    "dst_quality": float(q_dst),
                    "edge_quality_min": float(q_edge_min),
                    "edge_quality_hmean": float(q_edge_hmean),
                })

    if not all_entries:
        print("No delta entries to visualize.")
        return

    print(f"[viz] collected {len(all_entries):,} directed morphism entries", flush=True)

    docs = np.array([e["doc"] for e in all_entries], dtype=object)
    dirs_full = np.vstack([e["dir_full"] for e in all_entries])
    delta_sims = dirs_full @ dirs_full.T

    have_srcdst = all(e["src_dir_full"] is not None and e["dst_dir_full"] is not None for e in all_entries)
    if have_srcdst:
        src_dirs_full = np.vstack([e["src_dir_full"] for e in all_entries])
        dst_dirs_full = np.vstack([e["dst_dir_full"] for e in all_entries])
        src_sims = np.abs(src_dirs_full @ src_dirs_full.T)
        dst_sims = np.abs(dst_dirs_full @ dst_dirs_full.T)
    else:
        src_sims = None
        dst_sims = None

    # Pairwise semantic-quality matrix.  This uses the same smooth harmonic logic
    # as the analyze endpoint, preserving a conservative minimum as metadata.
    edge_q_hmean = np.asarray([e["edge_quality_hmean"] for e in all_entries], dtype=np.float32)
    edge_q_min = np.asarray([e["edge_quality_min"] for e in all_entries], dtype=np.float32)
    q_den = edge_q_hmean[:, None] + edge_q_hmean[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        quality_pairs_hmean = np.where(
            q_den > 1e-12,
            (2.0 * edge_q_hmean[:, None] * edge_q_hmean[None, :]) / q_den,
            0.0,
        ).astype(np.float32)
    quality_pairs_min = np.minimum(edge_q_min[:, None], edge_q_min[None, :]).astype(np.float32)

    def _compute_neighbors_and_mask():
        neighbors_new = []
        overlap_new = np.zeros(len(all_entries), dtype=bool)
        for i in range(len(all_entries)):
            cross_doc = (docs != docs[i])
            if have_srcdst:
                mask = (delta_sims[i] >= current_thr["delta"]) \
                       & (src_sims[i] >= current_thr["pc1"]) \
                       & (dst_sims[i] >= current_thr["pc1"]) \
                       & (quality_pairs_hmean[i] >= current_thr["quality"]) \
                       & cross_doc
            else:
                mask = (delta_sims[i] >= current_thr["delta"]) \
                       & (quality_pairs_hmean[i] >= current_thr["quality"]) \
                       & cross_doc
            mask[i] = False
            nbrs = np.where(mask)[0].tolist()
            neighbors_new.append(nbrs)
            overlap_new[i] = len(nbrs) > 0
        return neighbors_new, overlap_new

    neighbors, overlap_mask = _compute_neighbors_and_mask()

    # --- diagnostics ---------------------------------------------------------
    print("")
    print("--- Document Concordance Survey ---")
    cross = (docs[:, None] != docs[None, :])
    vals_delta = delta_sims[cross]
    q = np.quantile(vals_delta, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    print("[Δ cos] min,25,50,75,90,95,99,max:", np.array2string(q, precision=3))

    if have_srcdst:
        vals_src = src_sims[cross]
        vals_dst = dst_sims[cross]
        q_src = np.quantile(vals_src, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        q_dst = np.quantile(vals_dst, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        print("[PC1 cos] SRC  quantiles:", np.array2string(q_src, precision=3))
        print("[PC1 cos] DEST quantiles:", np.array2string(q_dst, precision=3))

    q_edge_quant = np.quantile(edge_q_hmean, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    q_pair_vals = quality_pairs_hmean[cross]
    q_pair_quant = np.quantile(q_pair_vals, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]) if q_pair_vals.size else q_edge_quant
    print("[Q edge] min,25,50,75,90,95,99,max:", np.array2string(q_edge_quant, precision=3))
    print("[Q pair] min,25,50,75,90,95,99,max:", np.array2string(q_pair_quant, precision=3))
    print(f"[thresholds] Δ≥{current_thr['delta']:.2f}; PC1≥{current_thr['pc1']:.2f}; Q≥{current_thr['quality']:.2f}")
    print("")

    # 2) Plot surface summaries + cluster points.
    print("[viz] building Matplotlib 3D figure", flush=True)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    selected_text = ax.text2D(
        0.02, 0.98, "", transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(fc="lavender", alpha=0.95),
        visible=False,
    )
    all_scatter_pts = []
    hover_texts_pts = []
    all_coords = []

    eigen_artists = []
    eigen_base_colors = []
    eigen_base_alphas = []
    eigen_base_lws = []
    eig_artist_map = {}

    def _cluster_q(doc_id, label):
        qmap = q_map_by_doc.get(doc_id, {})
        try:
            return _quality_clamp01(qmap.get(int(label), qmap.get(str(label), 1.0)))
        except Exception:
            return 1.0

    def _cluster_q_text(doc_id, label):
        q_payload = q_payload_by_doc.get(doc_id)
        rec = _q_record_from_payload(q_payload, label)
        qv = _cluster_q(doc_id, label)
        if not rec:
            return f"Q={qv:.3f}"
        parts = [f"Q={qv:.3f}"]
        for key, lab in [
            ("median_segment_quality", "med"),
            ("p25_segment_quality", "p25"),
            ("semantic_core_quality", "core"),
            ("lm_fluency_score_median", "LM-flu"),
            ("lm_fluency_nll_median", "LM-nll"),
            ("usable_segment_ratio", "usable"),
            ("fragment_burden", "frag"),
            ("non_template_score", "non-template"),
            ("spread_score", "spread"),
        ]:
            if key in rec and rec[key] != "":
                try:
                    parts.append(f"{lab}={float(rec[key]):.3f}")
                except Exception:
                    pass
        return "; ".join(parts)

    surface_artists_by_component = {"hull": [], "quality_ellipsoid": [], "skeleton": []}

    def _artist_set_visible(artist, visible: bool):
        try:
            artist.set_visible(bool(visible))
        except Exception:
            pass

    def _surface_components_for_mode(mode: str) -> tuple[str, ...]:
        mode = _normalize_surface_mode(mode)
        if mode == "quality_ellipsoid_mst":
            return ("quality_ellipsoid", "skeleton")
        if mode == "quality_ellipsoid":
            return ("quality_ellipsoid",)
        if mode == "skeleton":
            return ("skeleton",)
        if mode == "hull":
            return ("hull",)
        return tuple()

    def _draw_quality_weighted_ellipsoid(
        pts3,
        color,
        weights=None,
        n_std: float = 1.55,
        alpha: float = 0.12,
        resolution: int = 18,
        ridge: float = 1e-6,
    ):
        """Draw a Q-weighted covariance ellipsoid for projected cluster centroids."""
        pts = np.asarray(pts3, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 3:
            return None
        try:
            if weights is None:
                w = np.ones(pts.shape[0], dtype=float)
            else:
                w = np.asarray(weights, dtype=float).reshape(-1)
                if w.shape[0] != pts.shape[0]:
                    w = np.ones(pts.shape[0], dtype=float)
                w = np.clip(np.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                # Soft floor keeps low-Q structural clusters visible as context without
                # letting them dominate the ellipsoid.  Squaring emphasizes high-Q mass.
                w = np.maximum(w, 0.05) ** 1.5
            if float(w.sum()) <= 1e-12:
                w = np.ones(pts.shape[0], dtype=float)
            w = w / (float(w.sum()) + 1e-12)

            center = np.sum(pts * w[:, None], axis=0)
            Xc = pts - center
            C = (Xc * w[:, None]).T @ Xc
            C = C + float(ridge) * np.eye(3)

            vals, vecs = np.linalg.eigh(C)
            vals = np.maximum(vals, float(ridge))
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            radii = float(n_std) * np.sqrt(vals)

            u = np.linspace(0, 2 * np.pi, int(resolution))
            v = np.linspace(0, np.pi, int(resolution))
            xs = np.outer(np.cos(u), np.sin(v))
            ys = np.outer(np.sin(u), np.sin(v))
            zs = np.outer(np.ones_like(u), np.cos(v))
            sphere = np.stack([xs, ys, zs], axis=-1)
            ell = sphere @ np.diag(radii) @ vecs.T + center

            surf = ax.plot_surface(
                ell[:, :, 0], ell[:, :, 1], ell[:, :, 2],
                color=color, alpha=float(alpha), linewidth=0, shade=False, antialiased=True
            )
            try:
                surf.set_edgecolor("none")
            except Exception:
                pass
            return surf
        except Exception as ex:
            print(f"[viz] ellipsoid surface error: {ex}", flush=True)
            return None

    # Q → display geometry scaling.  Matplotlib scatter uses marker area in
    # points², so we convert the requested Q-proportional visual radius into area.
    node_radius_scale_pt = max(0.0, _safe_float(visual_q_node_radius_scale, 10.0))
    edge_width_scale_pt = max(0.0, _safe_float(visual_q_edge_width_scale, 20.0))
    mst_quality_lambda = max(0.0, _safe_float(visual_mst_quality_lambda, 2.0))

    def _q_to_node_radius_pt(qv: float) -> float:
        # Direct relationship requested: Q=0.5 -> radius=5 when scale=10.
        return float(node_radius_scale_pt * _quality_clamp01(qv))

    def _q_to_marker_area(qv: float) -> float:
        r = _q_to_node_radius_pt(qv)
        return float(np.pi * r * r)

    def _q_to_edge_linewidth(edge_q: float) -> float:
        # Direct relationship requested: linewidth = scale * edge_Q.
        # Default scale=20 makes the line roughly diameter-equivalent to a node
        # whose radius scale is 10.
        return float(edge_width_scale_pt * _quality_clamp01(edge_q))

    def _bright_saturated_color(color, saturation_boost: float = 1.65, value_boost: float = 1.25):
        """Return a brighter, more saturated RGBA derived from a Matplotlib color."""
        try:
            import colorsys
            r, g, b, a = mcolors.to_rgba(color)
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            s = min(1.0, max(0.0, s * float(saturation_boost) + 0.08))
            v = min(1.0, max(0.68, v * float(value_boost)))
            rr, gg, bb = colorsys.hsv_to_rgb(h, s, v)
            return (rr, gg, bb, a)
        except Exception:
            return color

    def _draw_mst_skeleton(pts3, color, qualities=None):
        """
        Draw a Q-aware MST over projected centroids.

        Edge selection uses the requested cost function:
            cost(i,j) = distance(i,j) * exp(lambda * (1 - average_Q(i,j)))

        Higher-Q links therefore have lower cost for the same projected distance.
        The tree is still an MST: it connects all centroids with n-1 edges and no
        cycles, but it now prefers high-quality routes when geometry allows.
        """
        pts = np.asarray(pts3, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
            return []
        try:
            from scipy.sparse.csgraph import minimum_spanning_tree

            n_pts = pts.shape[0]
            diff = pts[:, None, :] - pts[None, :, :]
            D = np.sqrt(np.sum(diff * diff, axis=2))

            if qualities is None:
                q = np.ones(n_pts, dtype=float)
            else:
                q = np.asarray(qualities, dtype=float).reshape(-1)
                if q.shape[0] != n_pts:
                    q = np.ones(n_pts, dtype=float)
                q = np.clip(np.nan_to_num(q, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

            avg_q = 0.5 * (q[:, None] + q[None, :])
            cost = D * np.exp(mst_quality_lambda * (1.0 - avg_q))

            # SciPy sparse MST treats structural zeros as absent edges.  If two
            # projected centroids coincide, give that off-diagonal connection a
            # tiny positive cost so the graph remains complete.
            offdiag = ~np.eye(n_pts, dtype=bool)
            positive = cost[offdiag & (cost > 0)]
            eps_cost = float(np.min(positive) * 1e-6) if positive.size else 1e-9
            cost[offdiag & (cost <= 0)] = eps_cost
            np.fill_diagonal(cost, 0.0)

            T = minimum_spanning_tree(cost).tocoo()
            skeleton_color = _bright_saturated_color(color)

            artists = []
            for i, j in zip(T.row, T.col):
                i = int(i); j = int(j)
                p0 = pts[i]; p1 = pts[j]
                edge_q = float(0.5 * (q[i] + q[j]))
                edge_alpha = 0.20 + 0.80 * edge_q
                lw = _q_to_edge_linewidth(edge_q)
                if lw <= 0:
                    # Keep zero-Q attachments barely inspectable rather than fully invisible.
                    lw = 0.15
                art = ax.plot(
                    [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=skeleton_color, alpha=edge_alpha, linewidth=lw, linestyle='-', solid_capstyle='round'
                )[0]
                artists.append(art)
            return artists
        except Exception as ex:
            print(f"[viz] MST skeleton error: {ex}", flush=True)
            return []

    def _apply_surface_mode(mode: str, draw: bool = True):
        mode = _normalize_surface_mode(mode)
        surface_state["mode"] = mode
        active_components = set(_surface_components_for_mode(mode))
        for comp, artists in surface_artists_by_component.items():
            is_active = comp in active_components
            for art in artists:
                _artist_set_visible(art, is_active)
        try:
            btn_surface.label.set_text("Surface: " + surface_labels.get(mode, mode))
        except Exception:
            pass
        try:
            ax_note_q_text.set_text(
                f"Q gates matches; projection={projection_state['mode']}; surface={surface_labels.get(mode, mode)}"
            )
        except Exception:
            pass
        if draw:
            fig.canvas.draw_idle()

    for doc_id, data in document_cluster_data.items():
        delta_matrix, cluster_order, _, cluster_topic_distributions, cluster_embeddings, cluster_dirs, q_map, q_payload = _unpack_data(data, doc_id=doc_id)
        color = doc_colors[doc_id]
        emb3d = np.vstack([
            coord_by_key.get((doc_id, int(_idx)), _pad_to_3(cluster_embeddings[_idx]))
            for _idx in range(cluster_embeddings.shape[0])
        ])

        doc_extent = (emb3d.max(axis=0) - emb3d.min(axis=0)).max()
        eig_scale = 0.12 * doc_extent if doc_extent > 0 else 1.0

        q_arr = np.asarray([_cluster_q(doc_id, label) for label in cluster_order], dtype=float)

        # Surface component 1: legacy convex hull, kept as an optional mode.
        try:
            if emb3d.shape[0] >= 4:
                hull = ConvexHull(emb3d)
                verts = [emb3d[s] for s in hull.simplices]
                poly = Poly3DCollection(verts, alpha=0.14, facecolor=color)
                poly.set_edgecolor('k')
                ax.add_collection3d(poly)
                poly.set_visible(False)
                surface_artists_by_component["hull"].append(poly)
        except Exception:
            pass

        # Surface component 2: semantic-Q-weighted covariance ellipsoid.
        ell = _draw_quality_weighted_ellipsoid(emb3d, color=color, weights=q_arr)
        if ell is not None:
            ell.set_visible(False)
            surface_artists_by_component["quality_ellipsoid"].append(ell)

        # Surface component 3: quality-aware MST skeleton over displayed centroids.
        for art in _draw_mst_skeleton(emb3d, color=color, qualities=q_arr):
            art.set_visible(False)
            surface_artists_by_component["skeleton"].append(art)

        for idx, label in enumerate(cluster_order):
            x, y, z = emb3d[idx]
            all_coords.append((x, y, z))
            qv = _cluster_q(doc_id, label)
            marker_area = _q_to_marker_area(qv)
            # Keep Q=0 nodes faintly visible for hover/debugging while preserving
            # the direct Q→radius rule for every positive Q value.
            if marker_area <= 0.0:
                marker_area = 0.6
            sc = ax.scatter(x, y, z, color=color, s=marker_area, alpha=0.18 + 0.82 * qv)
            all_scatter_pts.append(sc)

            if lda_model is not None:
                dist = cluster_topic_distributions.get(label)
                if dist is not None:
                    top_ids = np.argsort(dist)[::-1][:top_n_topics]
                    blocks = []
                    for tid in top_ids:
                        w = float(dist[tid])
                        try:
                            topic_name = lda_int_topics_list[tid]
                        except Exception:
                            topic_name = ""
                        try:
                            kws = [w_ for (w_, _) in lda_model.show_topic(int(tid), topn=top_m_keywords)]
                        except Exception:
                            kws = []
                        blocks.append(f"T{int(tid)}: ({topic_name}) ({w:.3f}): " + ", ".join(kws))
                    txt = "\n".join(blocks)
                else:
                    txt = f"{doc_id}_C{label} (no topic dist)"
            else:
                txt = f"{doc_id}_C{label}"

            txt = txt + "\n" + _cluster_q_text(doc_id, label)

            if (cluster_dirs is not None and isinstance(cluster_dirs, np.ndarray)
                    and cluster_dirs.shape[0] == cluster_embeddings.shape[0]):
                pc1_3d = project_vec3(cluster_dirs[idx])
                nrm3 = np.linalg.norm(pc1_3d)
                if nrm3 > 0:
                    pc1_3d = pc1_3d / nrm3
                txt = txt + f"\nPC1({projection_state['axis_labels'][0:3]})≈({pc1_3d[0]:.2f}, {pc1_3d[1]:.2f}, {pc1_3d[2]:.2f})"

            hover_texts_pts.append(txt)
            ax.text(x, y, z, f"{doc_id}_C{label}\nQ={qv:.2f}", fontsize=7, color='black')

            if (cluster_dirs is not None and isinstance(cluster_dirs, np.ndarray)
                    and cluster_dirs.shape[0] == cluster_embeddings.shape[0]):
                d3 = project_vec3(cluster_dirs[idx])
                nrm = np.linalg.norm(d3)
                if nrm > 0:
                    d3 = (d3 / nrm) * eig_scale
                    ev = ax.quiver(x, y, z, d3[0], d3[1], d3[2],
                                   color=color, alpha=0.45 + 0.45 * qv, linewidth=1.5)
                    eigen_artists.append(ev)
                    eigen_base_colors.append(color)
                    eigen_base_alphas.append(0.45 + 0.45 * qv)
                    eigen_base_lws.append(1.5)
                    eig_artist_map[(doc_id, idx)] = ev

    # 3) Plot deltas; line alpha reflects edge Q.
    delta_artists = []
    base_colors = []
    base_alphas = []
    base_lws = []

    for idx, e in enumerate(all_entries):
        x, y, z = e["start3"]
        dx, dy, dz = e["delta3"]
        base_color = 'red' if overlap_mask[idx] else 'olivedrab'
        base_alpha = 0.12 + 0.62 * _quality_clamp01(e.get("edge_quality_hmean", 1.0))
        art = ax.quiver(x, y, z, dx, dy, dz,
                        color=base_color, alpha=base_alpha, linewidth=1, picker=True)
        delta_artists.append(art)
        base_colors.append(base_color)
        base_alphas.append(base_alpha)
        base_lws.append(1.0)

    coords = np.asarray(all_coords, dtype=float)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 2
    if max_range <= 0:
        max_range = 1.0
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    patches = [Patch(facecolor=doc_colors[doc], edgecolor='k', label=doc) for doc in doc_ids]
    patches.append(Patch(facecolor='red', edgecolor='k', label='has cross-doc match at current Δ/PC1/Q'))
    patches.append(Patch(facecolor='olivedrab', edgecolor='k', label='no match at current thresholds'))
    ax.legend(handles=patches, loc='best')

    # ==================== Helper functions ====================
    def _set_linewidth(artist, lw):
        if hasattr(artist, "set_linewidths"):
            artist.set_linewidths(lw)
        else:
            try:
                artist.set_linewidth(lw)
            except Exception:
                pass

    def _restore_base_styles():
        for a, c, al, lw in zip(delta_artists, base_colors, base_alphas, base_lws):
            a.set_color(c)
            a.set_alpha(al)
            _set_linewidth(a, lw)
        for a, c, al, lw in zip(eigen_artists, eigen_base_colors, eigen_base_alphas, eigen_base_lws):
            try:
                a.set_color(c)
            except Exception:
                pass
            a.set_alpha(al)
            _set_linewidth(a, lw)
        fig.canvas.draw_idle()

    def _recompute_matches(delta_thr=None, pc1_thr=None, quality_thr=None):
        """Recompute neighbor sets and recolor deltas using Δ, PC1, and Q floors."""
        nonlocal neighbors, overlap_mask
        if delta_thr is not None:
            current_thr["delta"] = float(delta_thr)
        if have_srcdst and (pc1_thr is not None):
            current_thr["pc1"] = float(pc1_thr)
        if quality_thr is not None:
            current_thr["quality"] = float(quality_thr)

        neighbors, overlap_mask = _compute_neighbors_and_mask()

        for k, a in enumerate(delta_artists):
            c = 'red' if overlap_mask[k] else 'olivedrab'
            a.set_color(c)
            a.set_alpha(base_alphas[k])
            _set_linewidth(a, 1.0)
            base_colors[k] = c

        try:
            selected_text.set_visible(False)
        except Exception:
            pass
        if details_state.get("fig") is not None and plt.fignum_exists(details_state["fig"].number):
            plt.close(details_state["fig"])
            details_state["fig"] = None
            details_state["ax"] = None
        fig.canvas.draw_idle()

    # --- Lazy embed helper for interpret_direction() -------------------------
    _dim_d = len(all_entries[0]["dir_full"])
    _st_model_holder = {"model": None}

    def _embed_texts(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, _dim_d), dtype=float)
        try:
            if _st_model_holder["model"] is None:
                print("[viz] loading SBERT lexical helper on first details-table use", flush=True)
                from sentence_transformers import SentenceTransformer
                _st_model_holder["model"] = SentenceTransformer('all-MiniLM-L6-v2')
            return _st_model_holder["model"].encode(texts, normalize_embeddings=True)
        except Exception as ex:
            print(f"[viz] lexical helper unavailable: {ex}", flush=True)
            return np.zeros((len(texts), _dim_d), dtype=float)

    def _candidate_words_for_entry(e, n_topics: int, per_topic_words: int) -> list[str]:
        if lda_model is None:
            return []
        try:
            dist_src = document_cluster_data[e["doc"]][3].get(e["i_lab"])
            dist_dst = document_cluster_data[e["doc"]][3].get(e["j_lab"])
        except Exception:
            return []
        if dist_src is None or dist_dst is None:
            return []
        src_top = np.argsort(dist_src)[::-1][:n_topics]
        dst_top = np.argsort(dist_dst)[::-1][:n_topics]
        words = []
        for tid in list(dict.fromkeys(list(src_top) + list(dst_top))):
            try:
                words.extend([w for (w, _) in lda_model.show_topic(int(tid), topn=per_topic_words)])
            except Exception:
                pass
        return list(dict.fromkeys(words))

    # =============== details window (table) ===============
    details_state = {"fig": None, "ax": None}

    def _topic_block_for(doc_id: str, cluster_label, n_topics=3, n_words=3) -> str:
        if lda_model is None:
            return ""
        try:
            dist = document_cluster_data[doc_id][3].get(cluster_label)
        except Exception:
            return ""
        if dist is None:
            return ""
        top_ids = np.argsort(dist)[::-1][:n_topics]
        lines = []
        for tid in top_ids:
            w = float(dist[int(tid)])
            try:
                topic_name = lda_int_topics_list[tid]
            except Exception:
                topic_name = ""
            try:
                words = [w_ for (w_, _) in lda_model.show_topic(int(tid), topn=n_words)]
            except Exception:
                words = []
            lines.append(f"T{int(tid)}: ({topic_name}) ({w:.3f}): " + ", ".join(words))
        return "\n".join(lines)

    def _format_interpretation_summary(ir: dict, max_topics: int = 2, max_words: int = 4) -> str:
        def _fmt_topic_rows(rows):
            out = []
            for (tid, delta_w, label, twords) in rows[:max_topics]:
                twords_str = ", ".join(twords[:max_words]) if twords else ""
                out.append(f"{label} Δ={delta_w:+.3f} [{twords_str}]")
            return "\n".join(out)
        up = _fmt_topic_rows(ir.get("topic_flow_up", []))
        down = _fmt_topic_rows(ir.get("topic_flow_down", []))
        wpos = ", ".join([f"{w}({s:+.2f})" for (w, s) in ir.get("words_pos", [])[:max_words]])
        wneg = ", ".join([f"{w}({s:+.2f})" for (w, s) in ir.get("words_neg", [])[:max_words]])
        blocks = []
        if up:
            blocks.append("↑ topics:\n" + up)
        if down:
            blocks.append("↓ topics:\n" + down)
        if wpos:
            blocks.append("+words: " + wpos)
        if wneg:
            blocks.append("-words: " + wneg)
        return "\n".join(blocks) if blocks else "—"

    def _show_details_table(selected_idx: int, group_indices: list[int]):
        rows = []
        headers = [
            "★", "doc", "from", "to", "|Δ|", "cos→sel",
            "src_Q", "dst_Q", "edge_Q", "Q→sel",
            "src_PC1(3D)", "dst_PC1(3D)",
            "src_topics: Number, Inferred category, Weight over cluster, Top topic terms",
            "dst_topics: Number, Inferred category, Weight over cluster, Top topic terms",
            "Δ semantics (topics↑/↓; words±)",
        ]

        sel = all_entries[selected_idx]
        sel_dir = sel["dir_full"]
        sel_edge_q = sel.get("edge_quality_hmean", 1.0)

        for g in group_indices:
            e = all_entries[g]
            length = float(np.linalg.norm(e.get("delta_full", e["delta3"])))
            cos_to_sel = float(np.dot(sel_dir, e["dir_full"]))
            q_to_sel = _q_hmean(sel_edge_q, e.get("edge_quality_hmean", 1.0))

            def _pc1_str(vec):
                if vec is None:
                    return "—"
                v3 = project_vec3(vec)
                n = np.linalg.norm(v3)
                if n > 0:
                    v3 = v3 / n
                return f"({v3[0]:.2f}, {v3[1]:.2f}, {v3[2]:.2f})"

            src_pc1 = _pc1_str(e.get("src_dir_full"))
            dst_pc1 = _pc1_str(e.get("dst_dir_full"))
            src_block = _topic_block_for(e["doc"], e["i_lab"], n_topics=top_n_topics, n_words=top_m_keywords)
            dst_block = _topic_block_for(e["doc"], e["j_lab"], n_topics=top_n_topics, n_words=top_m_keywords)

            try:
                dist_src = document_cluster_data[e["doc"]][3].get(e["i_lab"])
                dist_dst = document_cluster_data[e["doc"]][3].get(e["j_lab"])
                cand_words = _candidate_words_for_entry(e, n_topics=top_n_topics, per_topic_words=max(8, top_m_keywords))
                ir = interpret_direction(
                    dir_full=e["dir_full"],
                    src_topic_probs=dist_src,
                    dst_topic_probs=dist_dst,
                    lda_model=lda_model,
                    lda_labels=lda_int_topics_list,
                    embed_fn=_embed_texts,
                    candidate_words=cand_words,
                    top_n_topics=top_n_topics,
                    top_m_keywords=top_m_keywords,
                    top_k_words=12,
                )
                delta_sem_txt = _format_interpretation_summary(ir, max_topics=2, max_words=4)
            except Exception as ex:
                delta_sem_txt = f"(interpretation error: {ex})"

            rows.append([
                "★" if g == selected_idx else "",
                e["doc"],
                f"C{e['i_lab']}",
                f"C{e['j_lab']}",
                f"{length:.3f}",
                f"{cos_to_sel:.3f}",
                _fmt_q(e.get("src_quality")),
                _fmt_q(e.get("dst_quality")),
                _fmt_q(e.get("edge_quality_hmean")),
                _fmt_q(q_to_sel),
                src_pc1,
                dst_pc1,
                src_block,
                dst_block,
                delta_sem_txt,
            ])

        if details_state["fig"] is None or not plt.fignum_exists(details_state["fig"].number):
            details_state["fig"] = plt.figure(figsize=(14, 0.8 + 0.50 * max(3, len(rows))))
            details_state["ax"] = details_state["fig"].add_subplot(111)
        else:
            details_state["fig"].clf()
            details_state["ax"] = details_state["fig"].add_subplot(111)

        details_state["ax"].axis('off')
        table = details_state["ax"].table(cellText=rows, colLabels=headers, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col=list(range(len(headers))))
        details_state["fig"].suptitle(
            "Cluster Pairs With Similar Transform Directionality, PC1 Concordance, and Semantic Quality",
            fontsize=12,
        )
        details_state["fig"].tight_layout()
        details_state["fig"].canvas.draw_idle()
        plt.show(block=False)

    # ---------------- Hover: cluster points ----------------
    cursor_pts = mplcursors.cursor(all_scatter_pts, hover=True, annotation_kwargs=dict(arrowprops=None))

    @cursor_pts.connect("add")
    def on_add_point(sel):
        artist = sel.artist
        idx = all_scatter_pts.index(artist)
        ann = getattr(sel, "annotation", None)
        if ann is not None:
            ann.set_text(hover_texts_pts[idx])
            bbox = ann.get_bbox_patch()
            if bbox is not None:
                bbox.set(fc="lightyellow", alpha=0.9)

    # ---------------- Click-select: delta vectors ----------------
    cursor_deltas = mplcursors.cursor(delta_artists, hover=False, multiple=False, annotation_kwargs=dict(arrowprops=None))

    @cursor_deltas.connect("add")
    def on_select_delta(sel):
        artist = sel.artist
        try:
            idx = delta_artists.index(artist)
        except ValueError:
            return

        group = [idx] + neighbors[idx]
        _restore_base_styles()

        for k, a in enumerate(delta_artists):
            if k not in group:
                a.set_alpha(0.04)
                _set_linewidth(a, 0.25)
        for g in group:
            a = delta_artists[g]
            a.set_alpha(1.0)
            _set_linewidth(a, 2.5)
            a.set_color('yellow')

        for ev in eigen_artists:
            ev.set_alpha(0.04)
            _set_linewidth(ev, 0.25)
        for g in group:
            eg = all_entries[g]
            for key in ((eg["doc"], eg["i_idx"]), (eg["doc"], eg["j_idx"])):
                ev = eig_artist_map.get(key)
                if ev is not None:
                    ev.set_alpha(1.0)
                    _set_linewidth(ev, 3.0 if g == idx else 2.0)

        e = all_entries[idx]
        length = float(np.linalg.norm(e.get("delta_full", e["delta3"])))
        matched = neighbors[idx]
        max_lines = 8
        if matched:
            lines = [
                f"• {all_entries[m]['doc']}: C{all_entries[m]['i_lab']}→C{all_entries[m]['j_lab']} "
                f"Q={_q_hmean(e.get('edge_quality_hmean', 1.0), all_entries[m].get('edge_quality_hmean', 1.0)):.3f}"
                for m in matched[:max_lines]
            ]
            more = len(matched) - max_lines
            if more > 0:
                lines.append(f"  (+{more} more)")
            match_block = "\n" + "\n".join(lines)
        else:
            match_block = "\n(no directional matches at current Δ/PC1/Q thresholds)"

        src_pc1 = e.get("src_dir_full")
        dst_pc1 = e.get("dst_dir_full")
        pc1_lines = ""
        if src_pc1 is not None and dst_pc1 is not None:
            sp = project_vec3(src_pc1)
            dp = project_vec3(dst_pc1)
            if np.linalg.norm(sp) > 0:
                sp = sp / np.linalg.norm(sp)
            if np.linalg.norm(dp) > 0:
                dp = dp / np.linalg.norm(dp)
            pc1_lines = (
                f"\nsrc_PC1[{projection_state['mode']}]≈({sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f})  "
                f"dst_PC1[{projection_state['mode']}]≈({dp[0]:.2f}, {dp[1]:.2f}, {dp[2]:.2f})"
            )

        selected_text.set_text(
            f"{e['doc']}: C{e['i_lab']}→C{e['j_lab']}\n"
            f"‖Δ‖={length:.3f}, matches={len(matched)}, "
            f"srcQ={e.get('src_quality', 1.0):.3f}, dstQ={e.get('dst_quality', 1.0):.3f}, "
            f"edgeQ={e.get('edge_quality_hmean', 1.0):.3f}\n"
            f"thresholds: Δ≥{current_thr['delta']:.2f}, PC1≥{current_thr['pc1']:.2f}, Q≥{current_thr['quality']:.2f}"
            f"{pc1_lines}{match_block}"
        )
        selected_text.set_visible(True)
        fig.canvas.draw_idle()

        try:
            _show_details_table(idx, group)
        except Exception as ex:
            print(f"[details] error: {ex}")

    # ---------------- Buttons and controls ----------------
    ax_btn_surface = fig.add_axes([0.02, 0.92, 0.18, 0.03]); ax_btn_surface.set_in_layout(False)
    btn_surface = Button(ax_btn_surface, 'Surface')

    def on_cycle_surface(event):
        current = _normalize_surface_mode(surface_state.get("mode", "quality_ellipsoid_mst"))
        try:
            next_idx = (surface_modes.index(current) + 1) % len(surface_modes)
        except ValueError:
            next_idx = 0
        _apply_surface_mode(surface_modes[next_idx], draw=True)

    btn_surface.on_clicked(on_cycle_surface)
    _apply_surface_mode(surface_state["mode"], draw=False)

    ax_btn_restore = fig.add_axes([0.21, 0.92, 0.12, 0.03]); ax_btn_restore.set_in_layout(False)
    btn_restore = Button(ax_btn_restore, 'Restore Styles')

    def on_restore_styles(event):
        _restore_base_styles()

    btn_restore.on_clicked(on_restore_styles)

    ax.set_title(
        "Visualizing Inter-Document Feature Manifold Similarity\n"
        "Through Intra-Document Transform Morphisms + Semantic Quality\n"
        f"Projection: {projection_state['label']}"
    )
    _axis_labels = projection_state.get("axis_labels", ("X", "Y", "Z"))
    ax.set_xlabel(str(_axis_labels[0]))
    ax.set_ylabel(str(_axis_labels[1]))
    ax.set_zlabel(str(_axis_labels[2]))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)
    timer2.stop()
    print(f"Total elapsed time to render visualization: {int(timer2.elapsed())//60}:min {int(timer2.elapsed())%60}:sec")

    init_elev, init_azim = ax.elev, ax.azim

    ax_slider_az = fig.add_axes([0.78, 0.92, 0.20, 0.03]); ax_slider_az.set_in_layout(False)
    ax_slider_el = fig.add_axes([0.78, 0.88, 0.20, 0.03]); ax_slider_el.set_in_layout(False)
    slider_az = Slider(ax_slider_az, "Azim", 0, 360, valinit=ax.azim, valstep=1)
    slider_el = Slider(ax_slider_el, "Elev", -90, 90, valinit=ax.elev, valstep=1)

    def _update_view(_):
        ax.view_init(elev=slider_el.val, azim=slider_az.val)
        fig.canvas.draw_idle()

    slider_az.on_changed(_update_view)
    slider_el.on_changed(_update_view)

    ax_btn_auto = fig.add_axes([0.78, 0.83, 0.08, 0.04]); ax_btn_auto.set_in_layout(False)
    btn_auto = Button(ax_btn_auto, "Auto")
    rotating = [False]
    rot_step = 0.5
    rot_timer = fig.canvas.new_timer(interval=20)

    def _tick(_event=None):
        slider_az.set_val((slider_az.val + rot_step) % 360)

    rot_timer.add_callback(_tick)

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

    ax_btn_reset = fig.add_axes([0.88, 0.83, 0.10, 0.04]); ax_btn_reset.set_in_layout(False)
    btn_reset = Button(ax_btn_reset, "Reset View")

    def _reset_view(_event):
        slider_el.set_val(init_elev)
        slider_az.set_val(init_azim)

    btn_reset.on_clicked(_reset_view)

    # Threshold sliders.  Δ and Q are always active; PC1 appears when the pkl has dirs.
    ax_slider_delta = fig.add_axes([0.78, 0.79, 0.20, 0.03]); ax_slider_delta.set_in_layout(False)
    slider_delta = Slider(ax_slider_delta, "Δ dir cos", 0.0, 1.0, valinit=current_thr["delta"], valstep=0.01)

    def _on_delta_change(val):
        pc1_val = slider_srcdst.val if have_srcdst else None
        _recompute_matches(delta_thr=val, pc1_thr=pc1_val, quality_thr=current_thr["quality"])

    slider_delta.on_changed(_on_delta_change)

    if have_srcdst:
        ax_slider_srcdst = fig.add_axes([0.78, 0.74, 0.20, 0.03]); ax_slider_srcdst.set_in_layout(False)
        slider_srcdst = Slider(ax_slider_srcdst, "PC1 match", 0.0, 1.0, valinit=current_thr["pc1"], valstep=0.01)

        def _on_srcdst_change(val):
            _recompute_matches(delta_thr=slider_delta.val, pc1_thr=val, quality_thr=current_thr["quality"])

        slider_srcdst.on_changed(_on_srcdst_change)
    else:
        slider_srcdst = None
        ax_lbl_srcdst = fig.add_axes([0.78, 0.74, 0.20, 0.03]); ax_lbl_srcdst.set_in_layout(False)
        ax_lbl_srcdst.axis('off')
        ax_lbl_srcdst.text(0.0, 0.5, "PC1 src/dst: n/a", transform=ax_lbl_srcdst.transAxes, va='center')

    ax_slider_quality = fig.add_axes([0.78, 0.69, 0.20, 0.03]); ax_slider_quality.set_in_layout(False)
    slider_quality = Slider(ax_slider_quality, "Q floor", 0.0, 1.0, valinit=current_thr["quality"], valstep=0.01)

    def _on_quality_change(val):
        pc1_val = slider_srcdst.val if have_srcdst else None
        _recompute_matches(delta_thr=slider_delta.val, pc1_thr=pc1_val, quality_thr=val)

    slider_quality.on_changed(_on_quality_change)

    ax_note_q = fig.add_axes([0.78, 0.645, 0.20, 0.035]); ax_note_q.set_in_layout(False)
    ax_note_q.axis('off')
    ax_note_q_text = ax_note_q.text(
        0.0, 0.5,
        f"Q gates matches; projection={projection_state['mode']}; surface={surface_labels.get(surface_state['mode'], surface_state['mode'])}",
        transform=ax_note_q.transAxes,
        va='center', fontsize=8,
    )
    _apply_surface_mode(surface_state["mode"], draw=False)

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
    quality_thresholds = None,
    segments_by_doc: dict | None = None,
    top_k_per_delta: int = 5,         # max matches to keep per Δ at the winning level
    pc1_only_threshold: float = 0.90, # strong PC1 match
    delta_max_for_pc1_only: float = 0.60,  # Δ must be below this to qualify as "PC1-only"
    pc1_only_quality_threshold: float = 0.0,
    compute_acuity_for: str = "aligned_only",  # "aligned_only" | "aligned_plus_pc1_only" | "pc1_only" | "none"
    pc1_match_axis: str = "dst",      # "dst" | "src" | "both"; controls the Y-axis PC1 gate
    source_doc_filter: str | None = None,  # when set, only these source/anchor doc edges are matched
    analyze_scope: str = "full",      # "full" | "anchor"; anchor uses source_doc_filter/doc_id
    require_cross_doc: bool = True,   # only match against other docs
    verbose: bool = True
) -> dict:
    """
    Analyze the morphism match field across documents with independent threshold axes:
        Δ cosine, a selected PC1 concordance axis, and semantic/discursive quality.

    The selected PC1 axis is controlled by pc1_match_axis:
        "dst"  = destination-cluster PC1 concordance only (default)
        "src"  = source-cluster PC1 concordance only
        "both" = min(source PC1, destination PC1), matching the older behavior

    Scope semantics:
      * analyze_scope="full" compares every source morphism edge in the collection
        against every other eligible target edge.
      * analyze_scope="anchor" requires source_doc_filter and compares only that
        selected/anchor document's source edges against the collection. This keeps
        selected-document CSV runs tractable on larger sets.

    Quality semantics:
      * Each cluster has Q_cluster ∈ [0,1]. Legacy 5/6-tuples with no quality
        payload default to Q=1.0 so older saved manifolds remain loadable.
      * Each edge C_i→C_j stores both edge_quality_min = min(Q_i, Q_j)
        and edge_quality_hmean = harmonic_mean(Q_i, Q_j).
      * A cross-document match stores semantic_quality_hmean as the plotted/gated
        Q axis and semantic_quality_min as the conservative weakest-endpoint floor.
      * Aligned match gating requires the selected independent axes to pass:
            Δ >= delta_threshold
            selected_PC1_axis >= pc1_threshold
            semantic_quality_hmean >= quality_threshold
        With the default selected_PC1_axis="dst", the Y-axis is destination-PC1
        concordance only. Source-PC1 values are still computed and exported.

    Acuity semantics:
      * compute_acuity_for controls which accepted match classes receive the
        more expensive lexical-overlap/acuteness computation.  The default,
        "aligned_only", avoids computing lexical metrics for the potentially
        very large PC1-only pool in high-k documents.
    """
    from collections import Counter
    import numpy as np

    if quality_thresholds is None:
        quality_thresholds = tuple(round(i / 100, 2) for i in range(99, -1, -1))

    def _as_desc_unique(vals):
        out = []
        for v in vals:
            try:
                vv = max(0.0, min(1.0, float(v)))
            except Exception:
                continue
            if vv not in out:
                out.append(vv)
        return sorted(out, reverse=True) if out else [0.0]

    delta_thresholds = _as_desc_unique(delta_thresholds)
    pc1_thresholds = _as_desc_unique(pc1_thresholds)
    quality_thresholds = _as_desc_unique(quality_thresholds)

    def _normalize_pc1_match_axis(mode) -> str:
        raw = str(mode or "dst").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "dst": "dst", "dest": "dst", "destination": "dst", "destination_only": "dst", "dst_only": "dst",
            "src": "src", "source": "src", "source_only": "src", "src_only": "src",
            "both": "both", "srcdst": "both", "src_dst": "both", "source_destination": "both",
            "source_and_destination": "both", "min": "both", "min_src_dst": "both", "composite": "both",
        }
        return aliases.get(raw, "dst")

    pc1_match_axis = _normalize_pc1_match_axis(pc1_match_axis)

    def _normalize_compute_acuity_for(mode) -> str:
        raw = str(mode or "aligned_only").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "aligned": "aligned_only",
            "aligned_only": "aligned_only",
            "alignment": "aligned_only",
            "strict": "aligned_only",
            "default": "aligned_only",
            "all": "aligned_plus_pc1_only",
            "both": "aligned_plus_pc1_only",
            "aligned_plus_pc1": "aligned_plus_pc1_only",
            "aligned_plus_pc1_only": "aligned_plus_pc1_only",
            "aligned_and_pc1_only": "aligned_plus_pc1_only",
            "pc1": "pc1_only",
            "pc1_only": "pc1_only",
            "none": "none",
            "no": "none",
            "off": "none",
            "false": "none",
            "0": "none",
            "skip": "none",
        }
        return aliases.get(raw, "aligned_only")

    compute_acuity_for = _normalize_compute_acuity_for(compute_acuity_for)

    def _should_compute_acuity(kind: str) -> bool:
        k = str(kind or "aligned").strip().lower()
        if compute_acuity_for == "none":
            return False
        if compute_acuity_for == "aligned_plus_pc1_only":
            return k in ("aligned", "pc1_only")
        if compute_acuity_for == "pc1_only":
            return k == "pc1_only"
        return k == "aligned"

    def _normalize_analyze_scope(mode) -> str:
        raw = str(mode or "full").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "full": "full", "all": "full", "global": "full", "collection": "full",
            "complete": "full", "everything": "full",
            "anchor": "anchor", "selected": "anchor", "selected_doc": "anchor",
            "selected_document": "anchor", "doc": "anchor", "doc_id": "anchor",
            "focus": "anchor", "source_doc": "anchor", "source_only": "anchor",
        }
        return aliases.get(raw, "full")

    analyze_scope = _normalize_analyze_scope(analyze_scope)
    source_doc_filter_norm = None if source_doc_filter is None else str(source_doc_filter)
    if analyze_scope == "anchor" and not source_doc_filter_norm:
        raise ValueError("analyze_scope='anchor' requires source_doc_filter/doc_id.")
    # Backward-compatible aliases used in returned metadata.
    match_scope = analyze_scope
    source_doc_id_norm = source_doc_filter_norm

    def _pc1_axis_label(mode: str) -> str:
        return {"dst": "dst_pc1", "src": "src_pc1", "both": "min(src_pc1,dst_pc1)"}.get(mode, "dst_pc1")

    def _pc1_axis_from_values(src_val: float, dst_val: float) -> float:
        if pc1_match_axis == "src":
            return float(src_val)
        if pc1_match_axis == "both":
            return float(min(src_val, dst_val))
        return float(dst_val)

    def _pc1_axis_vector(src_vec, dst_vec):
        if pc1_match_axis == "src":
            return src_vec
        if pc1_match_axis == "both":
            return np.minimum(src_vec, dst_vec)
        return dst_vec

    def _unpack_data(data, segments=None):
        """
        Accept 5-, 6-, or 7-tuple document data.
        Returns: delta_matrix, cluster_order, labels, topic_dists, emb_arr, cluster_dirs, quality_map
        """
        if len(data) >= 6:
            delta_matrix, cluster_order, labels, cluster_topic_distributions, cluster_embeddings, cluster_dirs = data[:6]
        else:
            delta_matrix, cluster_order, labels, cluster_topic_distributions, cluster_embeddings = data[:5]
            cluster_dirs = None

        if isinstance(cluster_embeddings, np.ndarray):
            emb_arr = cluster_embeddings
        elif isinstance(cluster_embeddings, dict):
            emb_arr = np.vstack([cluster_embeddings[label] for label in cluster_order])
        else:
            emb_arr = np.asarray(cluster_embeddings)
        if emb_arr.ndim == 1:
            emb_arr = emb_arr[None, :]

        quality_map = _cluster_quality_map_from_cdm(data, default=1.0, segments=segments)
        q_payload = data[6] if isinstance(data, (tuple, list)) and len(data) >= 7 else None
        return delta_matrix, cluster_order, labels, cluster_topic_distributions, emb_arr, cluster_dirs, quality_map, q_payload

    # 1) Collect item-level document embeddings and all Δ entries across docs.
    all_entries = []
    doc_ids = list(document_cluster_data.keys())
    has_quality_payload = False

    # Explicit document-embedding baselines used in Analyze/acuity exports.
    # The residual baseline is the document-centered manifold vector; the raw
    # SBERT baseline is the global pre-centering document anchor.
    manifold_residual_document_embedding_map = {}
    manifold_residual_document_embedding_meta = {}
    raw_sbert_document_embedding_map = {}
    raw_sbert_document_embedding_meta = {}
    for _doc_id, _data in document_cluster_data.items():
        _res_vec, _res_meta = _document_embedding_from_cdm(_data, preferred_key="manifold_residual_document_embedding")
        _raw_vec, _raw_meta = _document_embedding_from_cdm(_data, preferred_key="raw_sbert_document_embedding")
        for _key in (_doc_id, str(_doc_id)):
            manifold_residual_document_embedding_meta[_key] = dict(_res_meta or {})
            raw_sbert_document_embedding_meta[_key] = dict(_raw_meta or {})
        if _res_vec is not None:
            manifold_residual_document_embedding_map[_doc_id] = _res_vec
            manifold_residual_document_embedding_map[str(_doc_id)] = _res_vec
        if _raw_vec is not None:
            raw_sbert_document_embedding_map[_doc_id] = _raw_vec
            raw_sbert_document_embedding_map[str(_doc_id)] = _raw_vec

    for doc_id, data in document_cluster_data.items():
        if isinstance(data, (tuple, list)) and len(data) >= 7:
            has_quality_payload = True
        doc_segments = segments_by_doc.get(doc_id) if isinstance(segments_by_doc, dict) else None
        delta_matrix, cluster_order, _, _, cluster_embeddings, cluster_dirs, quality_map, q_payload = _unpack_data(data, segments=doc_segments)
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

                src_lab = cluster_order[i]
                dst_lab = cluster_order[j]
                src_q = _quality_clamp01(quality_map.get(int(src_lab), 1.0))
                dst_q = _quality_clamp01(quality_map.get(int(dst_lab), 1.0))
                src_q_rec = _cluster_quality_record_from_payload(q_payload, src_lab)
                dst_q_rec = _cluster_quality_record_from_payload(q_payload, dst_lab)
                # Keep both a conservative minimum floor and a smoother harmonic
                # edge score.  The harmonic score gives the Q axis more resolution,
                # while the min score remains available for auditing weak endpoints.
                edge_q_min = min(src_q, dst_q)
                edge_q_hmean = _quality_hmean([src_q, dst_q])

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
                    "i_lab": src_lab,
                    "j_lab": dst_lab,
                    "src_cluster_quality": float(src_q),
                    "dst_cluster_quality": float(dst_q),
                    "src_cluster_lm_fluency": _quality_record_float(src_q_rec, "lm_fluency_score_median"),
                    "dst_cluster_lm_fluency": _quality_record_float(dst_q_rec, "lm_fluency_score_median"),
                    "src_cluster_lm_nll": _quality_record_float(src_q_rec, "lm_fluency_nll_median"),
                    "dst_cluster_lm_nll": _quality_record_float(dst_q_rec, "lm_fluency_nll_median"),
                    "src_quality_model": str(src_q_rec.get("quality_model", "")) if isinstance(src_q_rec, dict) else "",
                    "dst_quality_model": str(dst_q_rec.get("quality_model", "")) if isinstance(dst_q_rec, dict) else "",
                    "edge_quality": float(edge_q_hmean),
                    "edge_quality_hmean": float(edge_q_hmean),
                    "edge_quality_min": float(edge_q_min),
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
                "quality_thresholds": list(quality_thresholds),
                "top_k_per_delta": int(top_k_per_delta),
                "pc1_only_threshold": float(pc1_only_threshold),
                "delta_max_for_pc1_only": float(delta_max_for_pc1_only),
                "pc1_only_quality_threshold": float(pc1_only_quality_threshold),
                "compute_acuity_for": compute_acuity_for,
                "pc1_match_axis": pc1_match_axis,
                "pc1_axis_label": _pc1_axis_label(pc1_match_axis),
                "analyze_scope": analyze_scope,
                "source_doc_filter": source_doc_filter_norm,
                "require_cross_doc": bool(require_cross_doc),
                "gating": f"delta & {_pc1_axis_label(pc1_match_axis)} & semantic_quality (independent axes)"
            }
        }

    docs_arr = np.array([e["doc"] for e in all_entries], dtype=object)

    if analyze_scope == "anchor":
        source_indices = np.array(
            [idx for idx, e in enumerate(all_entries) if str(e["doc"]) == source_doc_filter_norm],
            dtype=int,
        )
        if source_indices.size == 0:
            if verbose:
                print(f"[analyze] No source morphism edges found for anchor/source doc: {source_doc_filter_norm!r}.")
            return {
                "aligned_matches": {}, "pc1_only_matches": {}, "index": {}, "summary": {},
                "shapes": {"num_entries": 0, "source_entries": 0, "target_entries": M, "dim": int(len(all_entries[0]["dir_full"]))},
                "params": {
                    "delta_thresholds": list(delta_thresholds),
                    "pc1_thresholds": list(pc1_thresholds),
                    "quality_thresholds": list(quality_thresholds),
                    "top_k_per_delta": int(top_k_per_delta),
                    "pc1_only_threshold": float(pc1_only_threshold),
                    "delta_max_for_pc1_only": float(delta_max_for_pc1_only),
                    "pc1_only_quality_threshold": float(pc1_only_quality_threshold),
                    "pc1_match_axis": pc1_match_axis,
                    "pc1_axis_label": _pc1_axis_label(pc1_match_axis),
                    "analyze_scope": analyze_scope,
                    "source_doc_filter": source_doc_filter_norm,
                    "require_cross_doc": bool(require_cross_doc),
                    "gating": f"anchor-scoped delta & {_pc1_axis_label(pc1_match_axis)} & semantic_quality",
                },
            }
    else:
        source_indices = np.arange(M, dtype=int)

    S = int(source_indices.size)

    dirs_full = np.vstack([e["dir_full"] for e in all_entries])
    dirs_source = dirs_full[source_indices]
    # Shape is (source_entries, all_target_entries).  In full mode this is M x M;
    # in anchor mode this is anchor_edges x M, avoiding the previous M x M cost.
    delta_sims = dirs_source @ dirs_full.T
    edge_quality = np.asarray([e["edge_quality"] for e in all_entries], dtype=float)
    edge_quality_min = np.asarray([e.get("edge_quality_min", e["edge_quality"]) for e in all_entries], dtype=float)

    def _hmean_pair_vec(a, b, eps: float = 1e-6):
        """Vectorized harmonic mean for one scalar and one vector."""
        a = np.maximum(float(a), eps)
        b = np.maximum(np.asarray(b, dtype=float), eps)
        return 2.0 / ((1.0 / a) + (1.0 / b))

    def _quality_bin(v: float, step: float = 0.01) -> float:
        """Stable floor bin for reporting the quality-axis value in CSVs."""
        try:
            vv = float(np.clip(v, 0.0, 1.0))
        except Exception:
            vv = 0.0
        return float(np.floor((vv + 1e-12) / step) * step)

    have_srcdst = all(e["src_dir_full"] is not None and e["dst_dir_full"] is not None for e in all_entries)
    if have_srcdst:
        src_dirs_full = np.vstack([e["src_dir_full"] for e in all_entries])
        dst_dirs_full = np.vstack([e["dst_dir_full"] for e in all_entries])
        src_dirs_source = src_dirs_full[source_indices]
        dst_dirs_source = dst_dirs_full[source_indices]
        src_sims = np.abs(src_dirs_source @ src_dirs_full.T)
        dst_sims = np.abs(dst_dirs_source @ dst_dirs_full.T)
    else:
        src_sims = None
        dst_sims = None
        if verbose:
            print("[analyze] PC1 directions missing for at least one document; will perform Δ+quality matching where applicable.")

    def _joint_min_axis(delta_vec, pc_vec=None, q_vec=None):
        vals = [delta_vec]
        if pc_vec is not None:
            vals.append(pc_vec)
        if q_vec is not None:
            vals.append(q_vec)
        if len(vals) == 1:
            return vals[0]
        return np.minimum.reduce(vals)

    def _joint_min_legacy_srcdst(delta_vec, src_vec=None, dst_vec=None, q_vec=None):
        vals = [delta_vec]
        if src_vec is not None:
            vals.append(src_vec)
        if dst_vec is not None:
            vals.append(dst_vec)
        if q_vec is not None:
            vals.append(q_vec)
        if len(vals) == 1:
            return vals[0]
        return np.minimum.reduce(vals)

    def _cross_doc_mask(i_global):
        return (docs_arr != docs_arr[int(i_global)]) if require_cross_doc else np.ones(M, dtype=bool)

    # --- Lexical/acuteness helpers -------------------------------------------
    # Lexical overlap is computed only for accepted matches, not for every
    # candidate edge pair.  This keeps anchor-scoped Analyze runs tractable while
    # still giving the plot/CSV enough information to distinguish lexically
    # obvious matches from high-alignment, high-Q, lexically divergent candidates.
    _cluster_counter_cache = {}
    _edge_counter_cache = {}

    def _cluster_counter_for(doc, lab):
        key = (str(doc), int(lab))
        if key in _cluster_counter_cache:
            return _cluster_counter_cache[key]
        cnt = Counter()
        try:
            if isinstance(segments_by_doc, dict) and doc in segments_by_doc and doc in document_cluster_data:
                labels_arr = document_cluster_data[doc][2]
                if hasattr(labels_arr, "tolist"):
                    labels_arr = labels_arr.tolist()
                segs = segments_by_doc.get(doc, [])
                texts = [str(segs[ii]) for ii, ll in enumerate(labels_arr) if int(ll) == int(lab) and ii < len(segs)]
                cnt = lexical_counter_from_texts(texts)
        except Exception:
            cnt = Counter()
        _cluster_counter_cache[key] = cnt
        return cnt

    def _edge_counters_for(e):
        key = (str(e.get("doc")), int(e.get("i_lab")), int(e.get("j_lab")))
        if key in _edge_counter_cache:
            return _edge_counter_cache[key]
        src_cnt = _cluster_counter_for(e.get("doc"), e.get("i_lab"))
        dst_cnt = _cluster_counter_for(e.get("doc"), e.get("j_lab"))
        edge_cnt = _counter_add(src_cnt, dst_cnt)
        rec = (src_cnt, dst_cnt, edge_cnt)
        _edge_counter_cache[key] = rec
        return rec

    def _add_no_acuity_scores(scores: dict, reason: str = "not_computed") -> None:
        """Populate acuity columns without running lexical overlap comparisons."""
        try:
            d = float(scores.get("delta_cos", 0.0) or 0.0)
            pc = float(scores.get("pc1_axis_value", scores.get("dst_pc1", 1.0)) or 0.0)
            qv = float(scores.get("semantic_quality", 0.0) or 0.0)
            alignment_core = float(_quality_hmean([max(0.0, d), max(0.0, pc), max(0.0, qv)]))
        except Exception:
            alignment_core = 0.0
        scores.update({
            "acuity_computed": False,
            "acuity_compute_reason": str(reason),
            "lexical_available": False,
            "lexical_overlap_coefficient": 0.0,
            "lexical_divergence": 0.0,
            "alignment_core": alignment_core,
            "acuity_score": 0.0,
            "acuity_score_count_cosine": 0.0,
            "lexical_dst_overlap_coefficient": 0.0,
            "lexical_dst_jaccard": 0.0,
            "lexical_dst_dice": 0.0,
            "lexical_dst_weighted_jaccard": 0.0,
            "lexical_dst_count_cosine": 0.0,
            "lexical_dst_shared_token_mass_a": 0.0,
            "lexical_dst_shared_token_mass_b": 0.0,
            "lexical_dst_tokens_a": 0,
            "lexical_dst_tokens_b": 0,
            "lexical_dst_unique_a": 0,
            "lexical_dst_unique_b": 0,
            "lexical_dst_shared_unique": 0,
            "lexical_src_overlap_coefficient": 0.0,
            "lexical_src_jaccard": 0.0,
            "lexical_src_count_cosine": 0.0,
            "lexical_edge_overlap_coefficient": 0.0,
            "lexical_edge_jaccard": 0.0,
            "lexical_edge_count_cosine": 0.0,
        })

    def _add_lexical_acuity_scores(scores: dict, i: int, j: int) -> None:
        try:
            e_src = all_entries[int(i)]
            e_tgt = all_entries[int(j)]
            a_src, a_dst, a_edge = _edge_counters_for(e_src)
            b_src, b_dst, b_edge = _edge_counters_for(e_tgt)
            dst_m = lexical_overlap_metrics_from_counters(a_dst, b_dst)
            src_m = lexical_overlap_metrics_from_counters(a_src, b_src)
            edge_m = lexical_overlap_metrics_from_counters(a_edge, b_edge)

            lexical_available = bool(dst_m.get("lexical_available"))
            lexical_overlap = float(dst_m.get("overlap_coefficient", 0.0)) if lexical_available else 0.0
            lexical_divergence = float(1.0 - max(0.0, min(1.0, lexical_overlap))) if lexical_available else 0.0
            d = float(scores.get("delta_cos", 0.0) or 0.0)
            pc = float(scores.get("pc1_axis_value", scores.get("dst_pc1", 1.0)) or 0.0)
            qv = float(scores.get("semantic_quality", 0.0) or 0.0)
            alignment_core = float(_quality_hmean([max(0.0, d), max(0.0, pc), max(0.0, qv)]))
            acuity = float(alignment_core * lexical_divergence) if lexical_available else 0.0
            acuity_cosine = float(alignment_core * (1.0 - max(0.0, min(1.0, float(dst_m.get("count_cosine", 0.0)))))) if lexical_available else 0.0

            scores.update({
                "acuity_computed": bool(lexical_available),
                "acuity_compute_reason": "computed" if lexical_available else "no_lexical_tokens",
                "lexical_available": lexical_available,
                "lexical_overlap_coefficient": lexical_overlap,
                "lexical_divergence": lexical_divergence,
                "alignment_core": alignment_core,
                "acuity_score": acuity,
                "acuity_score_count_cosine": acuity_cosine,

                "lexical_dst_overlap_coefficient": float(dst_m.get("overlap_coefficient", 0.0)),
                "lexical_dst_jaccard": float(dst_m.get("jaccard", 0.0)),
                "lexical_dst_dice": float(dst_m.get("dice", 0.0)),
                "lexical_dst_weighted_jaccard": float(dst_m.get("weighted_jaccard", 0.0)),
                "lexical_dst_count_cosine": float(dst_m.get("count_cosine", 0.0)),
                "lexical_dst_shared_token_mass_a": float(dst_m.get("shared_token_mass_a", 0.0)),
                "lexical_dst_shared_token_mass_b": float(dst_m.get("shared_token_mass_b", 0.0)),
                "lexical_dst_tokens_a": int(dst_m.get("tokens_a", 0)),
                "lexical_dst_tokens_b": int(dst_m.get("tokens_b", 0)),
                "lexical_dst_unique_a": int(dst_m.get("unique_a", 0)),
                "lexical_dst_unique_b": int(dst_m.get("unique_b", 0)),
                "lexical_dst_shared_unique": int(dst_m.get("shared_unique", 0)),

                "lexical_src_overlap_coefficient": float(src_m.get("overlap_coefficient", 0.0)),
                "lexical_src_jaccard": float(src_m.get("jaccard", 0.0)),
                "lexical_src_count_cosine": float(src_m.get("count_cosine", 0.0)),
                "lexical_edge_overlap_coefficient": float(edge_m.get("overlap_coefficient", 0.0)),
                "lexical_edge_jaccard": float(edge_m.get("jaccard", 0.0)),
                "lexical_edge_count_cosine": float(edge_m.get("count_cosine", 0.0)),
            })
        except Exception as ex:
            scores.update({
                "acuity_computed": False,
                "acuity_compute_reason": "error",
                "lexical_available": False,
                "lexical_overlap_coefficient": 0.0,
                "lexical_divergence": 0.0,
                "alignment_core": float(_quality_hmean([
                    max(0.0, float(scores.get("delta_cos", 0.0) or 0.0)),
                    max(0.0, float(scores.get("pc1_axis_value", scores.get("dst_pc1", 1.0)) or 0.0)),
                    max(0.0, float(scores.get("semantic_quality", 0.0) or 0.0)),
                ])),
                "acuity_score": 0.0,
                "acuity_error": str(ex),
            })

    def _make_match(r, i, j, dt=None, pt=None, qt=None, kind="aligned"):
        i = int(i); j = int(j); r = int(r)
        q_pair_hmean = float(_quality_hmean([edge_quality[i], edge_quality[j]]))
        q_pair_min = float(min(edge_quality_min[i], edge_quality_min[j]))
        scores = {
            "delta_cos": float(delta_sims[r, j]),
            "src_edge_quality": float(edge_quality[i]),
            "tgt_edge_quality": float(edge_quality[j]),
            "src_edge_quality_hmean": float(edge_quality[i]),
            "tgt_edge_quality_hmean": float(edge_quality[j]),
            "src_edge_quality_min": float(edge_quality_min[i]),
            "tgt_edge_quality_min": float(edge_quality_min[j]),
            # The plotted/gated Q axis uses a harmonic match score for more
            # resolution than a strict minimum.  The conservative endpoint floor
            # remains in semantic_quality_min.
            "semantic_quality": q_pair_hmean,
            "semantic_quality_hmean": q_pair_hmean,
            "semantic_quality_min": q_pair_min,
            "quality_axis_bin_0p01": _quality_bin(q_pair_hmean, 0.01),
            "src_from_quality": float(all_entries[i]["src_cluster_quality"]),
            "src_to_quality": float(all_entries[i]["dst_cluster_quality"]),
            "tgt_from_quality": float(all_entries[j]["src_cluster_quality"]),
            "tgt_to_quality": float(all_entries[j]["dst_cluster_quality"]),
            "src_from_lm_fluency": all_entries[i].get("src_cluster_lm_fluency", ""),
            "src_to_lm_fluency": all_entries[i].get("dst_cluster_lm_fluency", ""),
            "tgt_from_lm_fluency": all_entries[j].get("src_cluster_lm_fluency", ""),
            "tgt_to_lm_fluency": all_entries[j].get("dst_cluster_lm_fluency", ""),
            "src_from_lm_nll": all_entries[i].get("src_cluster_lm_nll", ""),
            "src_to_lm_nll": all_entries[i].get("dst_cluster_lm_nll", ""),
            "tgt_from_lm_nll": all_entries[j].get("src_cluster_lm_nll", ""),
            "tgt_to_lm_nll": all_entries[j].get("dst_cluster_lm_nll", ""),
            "src_quality_model": all_entries[i].get("src_quality_model", ""),
            "tgt_quality_model": all_entries[j].get("src_quality_model", ""),
        }

        # Document-embedding baselines for the two documents participating in
        # this edge match.  Residual cosine summarizes similarity of the local
        # document-centered manifold baselines; raw SBERT cosine summarizes
        # ordinary global document proximity before centering.
        _res_doc_cos, _res_doc_cos_available = _document_embedding_cosine_from_maps(
            all_entries[i].get("doc"),
            all_entries[j].get("doc"),
            manifold_residual_document_embedding_map,
        )
        _raw_doc_cos, _raw_doc_cos_available = _document_embedding_cosine_from_maps(
            all_entries[i].get("doc"),
            all_entries[j].get("doc"),
            raw_sbert_document_embedding_map,
        )
        scores.update({
            "manifold_residual_doc_cosine": _res_doc_cos,
            "manifold_residual_doc_available": bool(_res_doc_cos_available),
            "raw_sbert_doc_cosine": _raw_doc_cos,
            "raw_sbert_doc_available": bool(_raw_doc_cos_available),
            "src_manifold_residual_doc_embedding_source": (manifold_residual_document_embedding_meta.get(all_entries[i].get("doc"), {}) or {}).get("source", ""),
            "tgt_manifold_residual_doc_embedding_source": (manifold_residual_document_embedding_meta.get(all_entries[j].get("doc"), {}) or {}).get("source", ""),
            "src_manifold_residual_doc_embedding_method": (manifold_residual_document_embedding_meta.get(all_entries[i].get("doc"), {}) or {}).get("method", ""),
            "tgt_manifold_residual_doc_embedding_method": (manifold_residual_document_embedding_meta.get(all_entries[j].get("doc"), {}) or {}).get("method", ""),
            "src_raw_sbert_doc_embedding_source": (raw_sbert_document_embedding_meta.get(all_entries[i].get("doc"), {}) or {}).get("source", ""),
            "tgt_raw_sbert_doc_embedding_source": (raw_sbert_document_embedding_meta.get(all_entries[j].get("doc"), {}) or {}).get("source", ""),
            "src_raw_sbert_doc_embedding_method": (raw_sbert_document_embedding_meta.get(all_entries[i].get("doc"), {}) or {}).get("method", ""),
            "tgt_raw_sbert_doc_embedding_method": (raw_sbert_document_embedding_meta.get(all_entries[j].get("doc"), {}) or {}).get("method", ""),
        })
        if have_srcdst:
            src_val = float(src_sims[r, j])
            dst_val = float(dst_sims[r, j])
            pc_axis_val = _pc1_axis_from_values(src_val, dst_val)
            legacy_srcdst_min = float(min(src_val, dst_val))
            scores.update({
                "src_pc1": src_val,
                "dst_pc1": dst_val,
                "pc1_axis_value": pc_axis_val,
                "pc1_axis_mode": pc1_match_axis,
                "pc1_axis_label": _pc1_axis_label(pc1_match_axis),
                "pc1_composite": legacy_srcdst_min,
                # joint_min and joint_min_4d now follow the plotted/gated PC1 axis.
                # The older source+destination conservative scores remain available.
                "joint_min": float(min(delta_sims[r, j], pc_axis_val)),
                "joint_min_4d": float(min(delta_sims[r, j], pc_axis_val, q_pair_hmean)),
                "joint_min_srcdst": float(min(delta_sims[r, j], src_val, dst_val)),
                "joint_min_srcdst_4d": float(min(delta_sims[r, j], src_val, dst_val, q_pair_hmean)),
            })
        else:
            scores.update({
                "pc1_axis_mode": pc1_match_axis,
                "pc1_axis_label": _pc1_axis_label(pc1_match_axis),
                "joint_min": float(delta_sims[r, j]),
                "joint_min_4d": float(min(delta_sims[r, j], q_pair_hmean)),
            })

        if _should_compute_acuity(kind):
            _add_lexical_acuity_scores(scores, i, j)
        else:
            _add_no_acuity_scores(scores, reason=f"compute_acuity_for={compute_acuity_for};kind={kind}")

        flags = {}
        if dt is not None:
            flags["delta_ok"] = bool(delta_sims[r, j] >= float(dt))
        if pt is not None and have_srcdst:
            src_val = float(src_sims[r, j])
            dst_val = float(dst_sims[r, j])
            pc_axis_val = _pc1_axis_from_values(src_val, dst_val)
            flags["src_pc1_ok"] = bool(src_val >= float(pt))
            flags["dst_pc1_ok"] = bool(dst_val >= float(pt))
            flags["pc1_axis_ok"] = bool(pc_axis_val >= float(pt))
        if qt is not None:
            flags["semantic_quality_ok"] = bool(q_pair_hmean >= float(qt))

        level = {}
        if dt is not None:
            level["delta"] = float(dt)
        if pt is not None:
            level["pc1"] = float(pt)
            level["pc1_axis_mode"] = pc1_match_axis
            level["pc1_axis_label"] = _pc1_axis_label(pc1_match_axis)
        if qt is not None:
            level["quality"] = float(qt)

        return {
            "j": int(j),
            "doc": all_entries[j]["doc"],
            "from": int(all_entries[j]["i_lab"]),
            "to": int(all_entries[j]["j_lab"]),
            "scores": scores,
            "flags": flags,
            "level": level,
        }

    # 4) For each selected/source Δ entry, find best-case matches by threshold levels.
    aligned_matches = {}
    for src_row, i in enumerate(source_indices):
        i = int(i)
        found = []
        cross_doc = _cross_doc_mask(i)
        q_pair_vec = _hmean_pair_vec(edge_quality[i], edge_quality)

        if have_srcdst:
            done = False
            for dt in delta_thresholds:
                if done: break
                for pt in pc1_thresholds:
                    if done: break
                    for qt in quality_thresholds:
                        dt = float(dt); pt = float(pt); qt = float(qt)
                        pc_axis_vec = _pc1_axis_vector(src_sims[src_row], dst_sims[src_row])
                        mask = (delta_sims[src_row] >= dt) & cross_doc
                        mask &= (pc_axis_vec >= pt)
                        mask &= (q_pair_vec >= qt)
                        mask[i] = False

                        idxs = np.where(mask)[0]
                        if idxs.size > 0:
                            joint = _joint_min_axis(delta_sims[src_row, idxs], pc_axis_vec[idxs], q_pair_vec[idxs])
                            order = np.argsort(-joint)
                            keep = idxs[order][:top_k_per_delta]
                            for j in keep:
                                found.append(_make_match(src_row, i, int(j), dt=dt, pt=pt, qt=qt, kind="aligned"))
                            done = True
                            break
        else:
            done = False
            for dt in delta_thresholds:
                if done: break
                for qt in quality_thresholds:
                    dt = float(dt); qt = float(qt)
                    mask = (delta_sims[src_row] >= dt) & cross_doc & (q_pair_vec >= qt)
                    mask[i] = False
                    idxs = np.where(mask)[0]
                    if idxs.size > 0:
                        joint = _joint_min_axis(delta_sims[src_row, idxs], q_vec=q_pair_vec[idxs])
                        order = np.argsort(-joint)
                        keep = idxs[order][:top_k_per_delta]
                        for j in keep:
                            found.append(_make_match(src_row, i, int(j), dt=dt, pt=None, qt=qt, kind="aligned"))
                        done = True
                        break

        aligned_matches[i] = found

    # 5) PC1-only matches: strong PC1 on the selected axis but weak Δ. Add Q as metadata and optional floor.
    pc1_only_matches = {}
    if have_srcdst:
        for src_row, i in enumerate(source_indices):
            i = int(i)
            cross_doc = _cross_doc_mask(i)
            dt_max = float(delta_max_for_pc1_only)
            pt_only = float(pc1_only_threshold)
            qt_only = float(pc1_only_quality_threshold)
            q_pair_vec = _hmean_pair_vec(edge_quality[i], edge_quality)

            pc_axis_vec = _pc1_axis_vector(src_sims[src_row], dst_sims[src_row])
            mask = (delta_sims[src_row] < dt_max) & cross_doc
            mask &= (pc_axis_vec >= pt_only)
            mask &= (q_pair_vec >= qt_only)
            mask[i] = False

            idxs = np.where(mask)[0]
            found = []
            if idxs.size > 0:
                joint = _joint_min_axis(delta_sims[src_row, idxs], pc_axis_vec[idxs], q_pair_vec[idxs])
                order = np.argsort(-joint)
                keep = idxs[order][:top_k_per_delta]
                for j in keep:
                    m = _make_match(src_row, i, int(j), dt=None, pt=pt_only, qt=qt_only, kind="pc1_only")
                    m["criteria"] = {
                        "pc1_only_threshold": pt_only,
                        "delta_max": dt_max,
                        "quality_threshold": qt_only,
                    }
                    found.append(m)
            pc1_only_matches[i] = found
    else:
        if verbose:
            print("[analyze] Skipping PC1-only search (no PC1 directions).")

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
        print(f"\n[analyze] source scope: {analyze_scope}; source_doc_filter={source_doc_filter_norm!r}; source_edges={S:,}; target_pool_edges={M:,}")
        print(f"[analyze] comparison matrix shape: {S:,} × {M:,} = {S * M:,} candidate edge pairs before thresholding")
        print(f"[analyze] compute_acuity_for={compute_acuity_for}; top_k_per_delta={int(top_k_per_delta):,}; pc1_only_threshold={float(pc1_only_threshold):.3f}; delta_max_for_pc1_only={float(delta_max_for_pc1_only):.3f}; pc1_only_quality_threshold={float(pc1_only_quality_threshold):.3f}")
        print(f"\n[analyze] Aligned (Δ & {_pc1_axis_label(pc1_match_axis)} & semantic_quality) match counts by doc→doc:")
        for k, v in aligned_counter.most_common():
            print("  ", k, ":", v)
        if have_srcdst:
            print(f"\n[analyze] PC1-only (weak Δ; {_pc1_axis_label(pc1_match_axis)}; Q annotated) match counts by doc→doc:")
            for k, v in pc1_only_counter.most_common():
                print("  ", k, ":", v)
        qvals = np.asarray([e["edge_quality"] for e in all_entries], dtype=float)
        qq = np.quantile(qvals, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        print("[Q edge hmean] min,25,50,75,90,95,99,max:", np.array2string(qq, precision=3))
        qminvals = np.asarray([e.get("edge_quality_min", e["edge_quality"]) for e in all_entries], dtype=float)
        qmq = np.quantile(qminvals, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        print("[Q edge min]    min,25,50,75,90,95,99,max:", np.array2string(qmq, precision=3))
        cqvals = np.asarray([e["src_cluster_quality"] for e in all_entries] + [e["dst_cluster_quality"] for e in all_entries], dtype=float)
        if cqvals.size:
            cqq = np.quantile(cqvals, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
            print("[Q cluster] min,25,50,75,90,95,99,max:", np.array2string(cqq, precision=3))
        if not has_quality_payload and not isinstance(segments_by_doc, dict):
            print("[analyze] No cluster-quality payload or segments_by_doc found; all cluster qualities defaulted to 1.0. Rebuild manifolds to persist true Q scores.")
        elif isinstance(segments_by_doc, dict):
            print("[analyze] segments_by_doc supplied; near-zero or missing Q payloads can be repaired with text-only quality estimates.")
        print("")

    index = {
        int(i): {
            "doc": all_entries[int(i)]["doc"],
            "from": int(all_entries[int(i)]["i_lab"]),
            "to": int(all_entries[int(i)]["j_lab"]),
            "src_cluster_quality": float(all_entries[int(i)]["src_cluster_quality"]),
            "dst_cluster_quality": float(all_entries[int(i)]["dst_cluster_quality"]),
            "src_cluster_lm_fluency": all_entries[int(i)].get("src_cluster_lm_fluency", ""),
            "dst_cluster_lm_fluency": all_entries[int(i)].get("dst_cluster_lm_fluency", ""),
            "src_cluster_lm_nll": all_entries[int(i)].get("src_cluster_lm_nll", ""),
            "dst_cluster_lm_nll": all_entries[int(i)].get("dst_cluster_lm_nll", ""),
            "edge_quality": float(all_entries[int(i)]["edge_quality"]),
            "edge_quality_hmean": float(all_entries[int(i)].get("edge_quality_hmean", all_entries[int(i)]["edge_quality"])),
            "edge_quality_min": float(all_entries[int(i)].get("edge_quality_min", all_entries[int(i)]["edge_quality"])),
        } for i in source_indices
    }

    return {
        "aligned_matches": aligned_matches,
        "pc1_only_matches": pc1_only_matches,
        "index": index,
        "summary": {
            "aligned_per_docpair": aligned_counter,
            "pc1_only_per_docpair": pc1_only_counter
        },
        "shapes": {
            "num_entries": int(S),
            "source_entries": int(S),
            "target_entries": int(M),
            "dim": int(len(all_entries[0]["dir_full"]))
        },
        "params": {
            "delta_thresholds": list(delta_thresholds),
            "pc1_thresholds": list(pc1_thresholds),
            "quality_thresholds": list(quality_thresholds),
            "top_k_per_delta": int(top_k_per_delta),
            "pc1_only_threshold": float(pc1_only_threshold),
            "delta_max_for_pc1_only": float(delta_max_for_pc1_only),
            "pc1_only_quality_threshold": float(pc1_only_quality_threshold),
            "compute_acuity_for": compute_acuity_for,
            "pc1_match_axis": pc1_match_axis,
            "pc1_axis_label": _pc1_axis_label(pc1_match_axis),
            "match_scope": match_scope,
            "analyze_scope": analyze_scope,
            "source_doc_filter": source_doc_filter_norm,
            "source_doc_id": source_doc_id_norm,
            "source_entries": int(S),
            "target_entries": int(M),
            "require_cross_doc": bool(require_cross_doc),
            "quality_payload_present": bool(has_quality_payload),
            "document_embedding_baseline": "manifold_residual_doc_cosine + raw_sbert_doc_cosine",
            "manifold_residual_document_embedding_available_docs": int(len({str(k) for k in manifold_residual_document_embedding_map.keys()})),
            "raw_sbert_document_embedding_available_docs": int(len({str(k) for k in raw_sbert_document_embedding_map.keys()})),
            "gating": f"delta & {_pc1_axis_label(pc1_match_axis)} & semantic_quality (independent axes); ranking uses joint_min_4d"
        }
    }


# -----------------------------------------------------------------------------
# Anchor null-field comparison helpers
# -----------------------------------------------------------------------------
def _null_normalize_pc1_axis(mode) -> str:
    raw = str(mode or "dst").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "dst": "dst", "dest": "dst", "destination": "dst", "destination_only": "dst", "dst_only": "dst",
        "src": "src", "source": "src", "source_only": "src", "src_only": "src",
        "both": "both", "srcdst": "both", "src_dst": "both", "source_destination": "both",
        "source_and_destination": "both", "min": "both", "min_src_dst": "both", "composite": "both",
    }
    return aliases.get(raw, "dst")


def _null_pc1_axis_label(mode: str) -> str:
    mode = _null_normalize_pc1_axis(mode)
    return {"dst": "dst_pc1", "src": "src_pc1", "both": "min(src_pc1,dst_pc1)"}.get(mode, "dst_pc1")


def _null_unpack_cdm_data(data, segments=None):
    """
    Accept 5-, 6-, or 7-tuple document data and return aligned arrays plus Q.
    This is a module-level version of the unpacking logic used by the Analyze
    endpoint, so null-field analysis can operate directly from saved .pkl CDM
    dictionaries without rebuilding manifolds.
    """
    if len(data) >= 6:
        delta_matrix, cluster_order, labels, cluster_topic_distributions, cluster_embeddings, cluster_dirs = data[:6]
    else:
        delta_matrix, cluster_order, labels, cluster_topic_distributions, cluster_embeddings = data[:5]
        cluster_dirs = None

    if isinstance(cluster_embeddings, np.ndarray):
        emb_arr = cluster_embeddings
    elif isinstance(cluster_embeddings, dict):
        emb_arr = np.vstack([cluster_embeddings[label] for label in cluster_order])
    else:
        emb_arr = np.asarray(cluster_embeddings)
    if emb_arr.ndim == 1:
        emb_arr = emb_arr[None, :]

    quality_map = _cluster_quality_map_from_cdm(data, default=1.0, segments=segments)
    q_payload = data[6] if isinstance(data, (tuple, list)) and len(data) >= 7 else None
    return delta_matrix, list(cluster_order), labels, cluster_topic_distributions, emb_arr, cluster_dirs, quality_map, q_payload


def _collect_morphism_edge_entries(document_cluster_data: dict, segments_by_doc: dict | None = None) -> list[dict]:
    """
    Build edge records for every directed cluster morphism in each document.

    Each record contains full-dimensional Δ direction, optional full-dimensional
    source/destination PC1 directions, harmonic/min edge Q values, endpoint Q
    diagnostics, and doc/cluster identifiers.
    """
    entries = []
    for doc_id, data in (document_cluster_data or {}).items():
        doc_segments = segments_by_doc.get(doc_id) if isinstance(segments_by_doc, dict) else None
        try:
            delta_matrix, cluster_order, _labels, _td, cluster_embeddings, cluster_dirs, quality_map, q_payload = _null_unpack_cdm_data(data, segments=doc_segments)
        except Exception as ex:
            print(f"[null] unable to unpack {doc_id!r}: {ex}", flush=True)
            continue

        cluster_embeddings = np.asarray(cluster_embeddings, dtype=float)
        if cluster_embeddings.ndim == 1:
            cluster_embeddings = cluster_embeddings[None, :]
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
                dfull = np.asarray(delta_matrix[i, j, :], dtype=float)
                nrm = float(np.linalg.norm(dfull))
                if nrm <= 0:
                    continue
                src_lab = cluster_order[i]
                dst_lab = cluster_order[j]
                src_q = _quality_clamp01(quality_map.get(int(src_lab), 1.0))
                dst_q = _quality_clamp01(quality_map.get(int(dst_lab), 1.0))
                src_q_rec = _cluster_quality_record_from_payload(q_payload, src_lab)
                dst_q_rec = _cluster_quality_record_from_payload(q_payload, dst_lab)
                edge_q_min = min(src_q, dst_q)
                edge_q_hmean = _quality_hmean([src_q, dst_q])
                entries.append({
                    "doc": doc_id,
                    "i_idx": int(i),
                    "j_idx": int(j),
                    "i_lab": src_lab,
                    "j_lab": dst_lab,
                    "dir_full": dfull / nrm,
                    "delta_full": dfull,
                    "delta_norm": nrm,
                    "src_dir_full": None if cluster_dirs_unit is None else cluster_dirs_unit[i],
                    "dst_dir_full": None if cluster_dirs_unit is None else cluster_dirs_unit[j],
                    "src_cluster_quality": float(src_q),
                    "dst_cluster_quality": float(dst_q),
                    "edge_quality": float(edge_q_hmean),
                    "edge_quality_hmean": float(edge_q_hmean),
                    "edge_quality_min": float(edge_q_min),
                    "src_cluster_lm_fluency": _quality_record_float(src_q_rec, "lm_fluency_score_median"),
                    "dst_cluster_lm_fluency": _quality_record_float(dst_q_rec, "lm_fluency_score_median"),
                    "src_cluster_lm_nll": _quality_record_float(src_q_rec, "lm_fluency_nll_median"),
                    "dst_cluster_lm_nll": _quality_record_float(dst_q_rec, "lm_fluency_nll_median"),
                    "src_quality_model": str(src_q_rec.get("quality_model", "")) if isinstance(src_q_rec, dict) else "",
                    "dst_quality_model": str(dst_q_rec.get("quality_model", "")) if isinstance(dst_q_rec, dict) else "",
                })
    return entries


def _edge_feature_arrays(entries: list[dict]) -> dict:
    """Convert edge records into aligned numpy arrays for vectorized scoring."""
    if not entries:
        return {
            "dirs": np.zeros((0, 0), dtype=float),
            "src_dirs": None,
            "dst_dirs": None,
            "edge_q": np.zeros((0,), dtype=float),
            "edge_q_min": np.zeros((0,), dtype=float),
            "docs": np.asarray([], dtype=object),
        }
    dirs = np.vstack([np.asarray(e["dir_full"], dtype=float) for e in entries])
    q = np.asarray([float(e.get("edge_quality", 1.0)) for e in entries], dtype=float)
    qmin = np.asarray([float(e.get("edge_quality_min", e.get("edge_quality", 1.0))) for e in entries], dtype=float)
    docs = np.asarray([e.get("doc") for e in entries], dtype=object)
    have_src = all(e.get("src_dir_full") is not None for e in entries)
    have_dst = all(e.get("dst_dir_full") is not None for e in entries)
    if have_src and have_dst:
        src_dirs = np.vstack([np.asarray(e["src_dir_full"], dtype=float) for e in entries])
        dst_dirs = np.vstack([np.asarray(e["dst_dir_full"], dtype=float) for e in entries])
    else:
        src_dirs = None
        dst_dirs = None
    return {"dirs": dirs, "src_dirs": src_dirs, "dst_dirs": dst_dirs, "edge_q": q, "edge_q_min": qmin, "docs": docs}


def _harmonic_pair_vector(a: float, b, eps: float = 1e-6):
    a = max(float(a), eps)
    b = np.maximum(np.asarray(b, dtype=float), eps)
    return 2.0 / ((1.0 / a) + (1.0 / b))


def _null_bin_idx(v, step: float, n_bins: int) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    idx = np.rint((1.0 - arr) / float(step)).astype(int)
    return np.clip(idx, 0, n_bins - 1)


def _null_make_grid_from_arrays(
    anchor_arrays: dict,
    target_arrays: dict,
    pc1_axis_mode: str,
    step: float,
    thresholds_for_edge_support: tuple[float, float, float] | None = None,
    anchor_edge_stats: bool = False,
):
    """
    Build an incremental Δ × PC1 × Q threshold grid for anchor→target edge pairs.

    If `anchor_edge_stats` is True, also return per-anchor-edge support counts at
    the supplied threshold triple and each edge's strongest observed match point.
    """
    pc1_axis_mode = _null_normalize_pc1_axis(pc1_axis_mode)
    n_bins = int(round(1.0 / float(step))) + 1
    grid = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)

    A = anchor_arrays
    T = target_arrays
    if A["dirs"].size == 0 or T["dirs"].size == 0:
        return grid, []

    have_pc1 = A.get("src_dirs") is not None and A.get("dst_dirs") is not None and T.get("src_dirs") is not None and T.get("dst_dirs") is not None
    edge_rows = []
    d_thr, p_thr, q_thr = thresholds_for_edge_support or (0.0, 0.0, 0.0)

    for ai in range(A["dirs"].shape[0]):
        d_scores = np.clip(T["dirs"] @ A["dirs"][ai], -1.0, 1.0)
        # The Analyze endpoint uses signed Δ cosine.  Negative scores cannot pass
        # nonnegative thresholds, but they still get clipped to zero for plotting.
        d_plot = np.clip(d_scores, 0.0, 1.0)

        if have_pc1:
            src_scores = np.abs(T["src_dirs"] @ A["src_dirs"][ai])
            dst_scores = np.abs(T["dst_dirs"] @ A["dst_dirs"][ai])
            if pc1_axis_mode == "src":
                pc_scores = src_scores
            elif pc1_axis_mode == "both":
                pc_scores = np.minimum(src_scores, dst_scores)
            else:
                pc_scores = dst_scores
        else:
            src_scores = None
            dst_scores = None
            pc_scores = np.ones_like(d_plot)

        q_scores = _harmonic_pair_vector(float(A["edge_q"][ai]), T["edge_q"])
        q_scores = np.clip(q_scores, 0.0, 1.0)

        bi = _null_bin_idx(d_plot, step, n_bins)
        bj = _null_bin_idx(pc_scores, step, n_bins)
        bk = _null_bin_idx(q_scores, step, n_bins)
        np.add.at(grid, (bi, bj, bk), 1.0)

        if anchor_edge_stats:
            support_mask = (d_scores >= float(d_thr)) & (pc_scores >= float(p_thr)) & (q_scores >= float(q_thr))
            support = int(np.count_nonzero(support_mask))
            joint = np.minimum.reduce([d_plot, pc_scores, q_scores])
            if joint.size:
                best_idx = int(np.argmax(joint))
                best_delta = float(d_plot[best_idx])
                best_pc1 = float(pc_scores[best_idx])
                best_q = float(q_scores[best_idx])
                best_joint = float(joint[best_idx])
                best_target = best_idx
            else:
                best_delta = best_pc1 = best_q = best_joint = 0.0
                best_target = None
            edge_rows.append({
                "anchor_edge_index": int(ai),
                "observed_support": support,
                "best_delta": best_delta,
                "best_pc1": best_pc1,
                "best_quality": best_q,
                "best_joint": best_joint,
                "best_target_index": best_target,
            })

    return grid, edge_rows



def _collect_anchor_pair_match_rows(
    anchor_entries: list[dict],
    target_entries: list[dict],
    anchor_arrays: dict,
    target_arrays: dict,
    pc1_axis_mode: str,
    thresholds: tuple[float, float, float],
    max_rows: int = 100000,
    match_set: str = "observed",
) -> list[dict]:
    """
    Collect pair-level match examples above one threshold triple for CSV audit.
    For null rows, target identities are retained as edge holders, but the feature
    values may be shuffled depending on the supplied target_arrays.
    """
    pc1_axis_mode = _null_normalize_pc1_axis(pc1_axis_mode)
    d_thr, p_thr, q_thr = thresholds
    A = anchor_arrays
    T = target_arrays
    rows = []
    if A["dirs"].size == 0 or T["dirs"].size == 0:
        return rows
    have_pc1 = A.get("src_dirs") is not None and A.get("dst_dirs") is not None and T.get("src_dirs") is not None and T.get("dst_dirs") is not None
    for ai in range(A["dirs"].shape[0]):
        d_scores = np.clip(T["dirs"] @ A["dirs"][ai], -1.0, 1.0)
        if have_pc1:
            src_scores = np.abs(T["src_dirs"] @ A["src_dirs"][ai])
            dst_scores = np.abs(T["dst_dirs"] @ A["dst_dirs"][ai])
            if pc1_axis_mode == "src":
                pc_scores = src_scores
            elif pc1_axis_mode == "both":
                pc_scores = np.minimum(src_scores, dst_scores)
            else:
                pc_scores = dst_scores
        else:
            src_scores = np.full_like(d_scores, np.nan, dtype=float)
            dst_scores = np.full_like(d_scores, np.nan, dtype=float)
            pc_scores = np.ones_like(d_scores)
        q_scores = np.clip(_harmonic_pair_vector(float(A["edge_q"][ai]), T["edge_q"]), 0.0, 1.0)
        mask = (d_scores >= float(d_thr)) & (pc_scores >= float(p_thr)) & (q_scores >= float(q_thr))
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue
        joint = np.minimum.reduce([np.clip(d_scores[idxs], 0.0, 1.0), pc_scores[idxs], q_scores[idxs]])
        order = np.argsort(-joint)
        # Keep at most max_rows globally; collect a little extra then trim below.
        for local_idx in order:
            ti = int(idxs[int(local_idx)])
            ae = anchor_entries[ai]
            te = target_entries[ti] if ti < len(target_entries) else {}
            rows.append({
                "match_set": match_set,
                "anchor_doc": str(ae.get("doc", "")),
                "anchor_edge_index": int(ai),
                "anchor_from": int(ae.get("i_lab")),
                "anchor_to": int(ae.get("j_lab")),
                "target_doc": str(te.get("doc", "")),
                "target_edge_index": int(ti),
                "target_from": int(te.get("i_lab")) if te else "",
                "target_to": int(te.get("j_lab")) if te else "",
                "delta_cos": float(d_scores[ti]),
                "src_pc1": "" if not np.isfinite(src_scores[ti]) else float(src_scores[ti]),
                "dst_pc1": "" if not np.isfinite(dst_scores[ti]) else float(dst_scores[ti]),
                "pc1_axis_value": float(pc_scores[ti]),
                "semantic_quality": float(q_scores[ti]),
                "joint_min": float(min(max(float(d_scores[ti]), 0.0), float(pc_scores[ti]), float(q_scores[ti]))),
                "anchor_edge_quality": float(ae.get("edge_quality", 1.0)),
                "target_edge_quality": float(T["edge_q"][ti]) if ti < T["edge_q"].shape[0] else "",
                "feature_association": "real" if match_set == "observed" else "shuffled_null",
            })
        if len(rows) > int(max_rows) * 2:
            rows.sort(key=lambda r: float(r.get("joint_min", 0.0)), reverse=True)
            rows = rows[:int(max_rows)]
    rows.sort(key=lambda r: float(r.get("joint_min", 0.0)), reverse=True)
    return rows[:int(max_rows)]


def _null_shuffle_target_arrays_by_doc(target_entries: list[dict], rng: np.random.Generator) -> dict:
    """
    Association-shuffle null: preserve target edge count and marginal feature
    distributions per target document, but break the association among Δ, PC1,
    and Q features by shuffling them independently within each document.
    """
    if not target_entries:
        return _edge_feature_arrays([])

    # Start with copies of real arrays and shuffle blocks per target document.
    arrays = _edge_feature_arrays(target_entries)
    dirs = arrays["dirs"].copy()
    edge_q = arrays["edge_q"].copy()
    edge_q_min = arrays["edge_q_min"].copy()
    src_dirs = None if arrays["src_dirs"] is None else arrays["src_dirs"].copy()
    dst_dirs = None if arrays["dst_dirs"] is None else arrays["dst_dirs"].copy()
    docs = arrays["docs"].copy()

    for doc in sorted(set(docs.tolist())):
        idx = np.where(docs == doc)[0]
        if idx.size <= 1:
            continue
        dirs[idx] = dirs[rng.permutation(idx)]
        edge_q[idx] = edge_q[rng.permutation(idx)]
        edge_q_min[idx] = edge_q_min[rng.permutation(idx)]
        if src_dirs is not None:
            src_dirs[idx] = src_dirs[rng.permutation(idx)]
        if dst_dirs is not None:
            dst_dirs[idx] = dst_dirs[rng.permutation(idx)]

    return {"dirs": dirs, "src_dirs": src_dirs, "dst_dirs": dst_dirs, "edge_q": edge_q, "edge_q_min": edge_q_min, "docs": docs}


def _grid_to_sparse_records(grid, step: float, value_name: str, min_value: float = 0.0, max_records: int | None = None):
    """Convert a dense 3D threshold grid into sparse record dicts."""
    n_bins = int(round(1.0 / float(step))) + 1
    thr = np.round(np.linspace(1.0, 0.0, n_bins), 2)
    idx = np.where(np.asarray(grid) > float(min_value))
    vals = np.asarray(grid)[idx]
    if max_records is not None and vals.size > int(max_records):
        order = np.argsort(-np.abs(vals))[:int(max_records)]
        idx = tuple(a[order] for a in idx)
        vals = vals[order]
    records = []
    for i, j, k, val in zip(idx[0], idx[1], idx[2], vals):
        records.append({
            "delta_threshold": float(thr[i]),
            "pc1_threshold": float(thr[j]),
            "quality_threshold": float(thr[k]),
            value_name: float(val),
        })
    return records


def analyze_anchor_null_match_field(
    document_cluster_data: dict,
    anchor_doc_id: str,
    segments_by_doc: dict | None = None,
    n_null_replicates: int = 50,
    null_strategy: str = "association_shuffle",
    random_seed: int = 0,
    step: float = 0.01,
    pc1_match_axis: str = "dst",
    edge_support_delta_threshold: float = 0.50,
    edge_support_pc1_threshold: float = 0.50,
    edge_support_quality_threshold: float = 0.50,
    max_match_csv_rows: int = 100000,
    verbose: bool = True,
) -> dict:
    """
    Anchor-based observed-vs-null morphism field comparison.

    For one anchor document A, compare A→real target edges against A→null target
    edges.  The null set preserves target edge counts and marginal Δ/PC1/Q feature
    distributions per target document, but shuffles feature associations within
    each target document.  The result contains five views for plotting:
        1. observed field
        2. null mean field
        3. positive residual field (observed > null)
        4. negative residual field (null > observed)
        5. anchor source-edge contribution table
    """
    import numpy as np
    from collections import Counter

    if not isinstance(document_cluster_data, dict) or not document_cluster_data:
        raise ValueError("document_cluster_data must be a non-empty dict loaded from a document delta .pkl file.")
    if anchor_doc_id not in document_cluster_data:
        keys = list(document_cluster_data.keys())[:10]
        raise ValueError(f"anchor_doc_id {anchor_doc_id!r} not found in document_cluster_data. First keys: {keys}")

    pc1_match_axis = _null_normalize_pc1_axis(pc1_match_axis)
    n_null_replicates = max(1, int(n_null_replicates))
    step = max(0.001, float(step))
    rng = np.random.default_rng(int(random_seed))

    entries = _collect_morphism_edge_entries(document_cluster_data, segments_by_doc=segments_by_doc)
    anchor_entries = [e for e in entries if str(e.get("doc")) == str(anchor_doc_id)]
    target_entries = [e for e in entries if str(e.get("doc")) != str(anchor_doc_id)]

    if not anchor_entries:
        raise ValueError(f"No directed morphism edges found for anchor {anchor_doc_id!r}.")
    if not target_entries:
        raise ValueError("No non-anchor target edges found. Load at least two documents.")

    A = _edge_feature_arrays(anchor_entries)
    T = _edge_feature_arrays(target_entries)
    support_thresholds = (float(edge_support_delta_threshold), float(edge_support_pc1_threshold), float(edge_support_quality_threshold))

    observed_grid, edge_rows = _null_make_grid_from_arrays(
        A, T, pc1_match_axis, step,
        thresholds_for_edge_support=support_thresholds,
        anchor_edge_stats=True,
    )
    observed_match_rows = _collect_anchor_pair_match_rows(
        anchor_entries, target_entries, A, T, pc1_match_axis, support_thresholds,
        max_rows=int(max_match_csv_rows), match_set="observed"
    )
    null_example_match_rows = []

    n_bins = observed_grid.shape[0]
    null_sum = np.zeros_like(observed_grid, dtype=np.float64)
    null_sumsq = np.zeros_like(observed_grid, dtype=np.float64)
    null_support_sum = np.zeros((len(anchor_entries),), dtype=np.float64)
    null_support_sumsq = np.zeros((len(anchor_entries),), dtype=np.float64)

    strategy_norm = str(null_strategy or "association_shuffle").strip().lower().replace("-", "_").replace(" ", "_")
    if strategy_norm not in ("association_shuffle", "feature_shuffle", "shuffle"):
        raise ValueError("Currently supported null_strategy values: association_shuffle, feature_shuffle, shuffle")

    if verbose:
        print(
            f"[null] anchor={anchor_doc_id!r}; anchor_edges={len(anchor_entries):,}; "
            f"target_edges={len(target_entries):,}; replicates={n_null_replicates}; "
            f"pc1_axis={_null_pc1_axis_label(pc1_match_axis)}; step={step}",
            flush=True
        )

    for r in range(n_null_replicates):
        null_T = _null_shuffle_target_arrays_by_doc(target_entries, rng)
        if r == 0:
            null_example_match_rows = _collect_anchor_pair_match_rows(
                anchor_entries, target_entries, A, null_T, pc1_match_axis, support_thresholds,
                max_rows=int(max_match_csv_rows), match_set="null_example"
            )
        g, er = _null_make_grid_from_arrays(
            A, null_T, pc1_match_axis, step,
            thresholds_for_edge_support=support_thresholds,
            anchor_edge_stats=True,
        )
        null_sum += g
        null_sumsq += g * g
        if er:
            supp = np.asarray([float(row.get("observed_support", 0.0)) for row in er], dtype=float)
            if supp.shape[0] == null_support_sum.shape[0]:
                null_support_sum += supp
                null_support_sumsq += supp * supp
        if verbose and (r + 1 == 1 or (r + 1) % max(1, n_null_replicates // 5) == 0 or r + 1 == n_null_replicates):
            print(f"[null] replicate {r + 1}/{n_null_replicates} complete", flush=True)

    null_mean = null_sum / float(n_null_replicates)
    null_var = np.maximum(0.0, (null_sumsq / float(n_null_replicates)) - (null_mean * null_mean))
    null_std = np.sqrt(null_var)
    residual = observed_grid - null_mean
    positive_residual = np.maximum(residual, 0.0)
    negative_residual = np.maximum(-residual, 0.0)
    log_enrichment = np.log((observed_grid + 1.0) / (null_mean + 1.0))
    z_score = residual / (null_std + 1e-9)

    null_support_mean = null_support_sum / float(n_null_replicates)
    null_support_var = np.maximum(0.0, (null_support_sumsq / float(n_null_replicates)) - (null_support_mean * null_support_mean))
    null_support_std = np.sqrt(null_support_var)

    for ai, row in enumerate(edge_rows):
        a = anchor_entries[ai]
        row.update({
            "anchor_doc": str(anchor_doc_id),
            "anchor_from": int(a.get("i_lab")),
            "anchor_to": int(a.get("j_lab")),
            "anchor_edge_quality": float(a.get("edge_quality", 1.0)),
            "anchor_src_quality": float(a.get("src_cluster_quality", 1.0)),
            "anchor_dst_quality": float(a.get("dst_cluster_quality", 1.0)),
            "null_support_mean": float(null_support_mean[ai]) if ai < null_support_mean.shape[0] else 0.0,
            "null_support_std": float(null_support_std[ai]) if ai < null_support_std.shape[0] else 0.0,
        })
        row["support_residual"] = float(row.get("observed_support", 0.0) - row.get("null_support_mean", 0.0))
        row["support_z_score"] = float(row["support_residual"] / (row.get("null_support_std", 0.0) + 1e-9))

    observed_total = float(observed_grid.sum())
    null_total = float(null_mean.sum())
    pos_total = float(positive_residual.sum())
    neg_total = float(negative_residual.sum())
    if verbose:
        print(
            f"[null] field totals: observed={observed_total:,.0f}; null_mean={null_total:,.1f}; "
            f"positive_residual={pos_total:,.1f}; negative_residual={neg_total:,.1f}",
            flush=True
        )

    return {
        "kind": "anchor_null_match_field",
        "anchor_doc_id": str(anchor_doc_id),
        "views": {
            "observed": observed_grid,
            "null_mean": null_mean,
            "null_std": null_std,
            "residual": residual,
            "positive_residual": positive_residual,
            "negative_residual": negative_residual,
            "log_enrichment": log_enrichment,
            "z_score": z_score,
        },
        "match_rows": {
            "observed": observed_match_rows,
            "null_example": null_example_match_rows,
        },
        "edge_contributions": edge_rows,
        "anchor_edges": [
            {
                "edge_index": int(i),
                "doc": str(e.get("doc")),
                "from": int(e.get("i_lab")),
                "to": int(e.get("j_lab")),
                "edge_quality": float(e.get("edge_quality", 1.0)),
                "src_quality": float(e.get("src_cluster_quality", 1.0)),
                "dst_quality": float(e.get("dst_cluster_quality", 1.0)),
            }
            for i, e in enumerate(anchor_entries)
        ],
        "target_doc_ids": sorted({str(e.get("doc")) for e in target_entries}),
        "params": {
            "step": float(step),
            "n_bins": int(n_bins),
            "n_null_replicates": int(n_null_replicates),
            "null_strategy": "association_shuffle",
            "random_seed": int(random_seed),
            "pc1_match_axis": pc1_match_axis,
            "pc1_axis_label": _null_pc1_axis_label(pc1_match_axis),
            "edge_support_delta_threshold": float(edge_support_delta_threshold),
            "edge_support_pc1_threshold": float(edge_support_pc1_threshold),
            "edge_support_quality_threshold": float(edge_support_quality_threshold),
            "anchor_edge_count": int(len(anchor_entries)),
            "target_edge_count": int(len(target_entries)),
            "max_match_csv_rows": int(max_match_csv_rows),
            "observed_total": observed_total,
            "null_mean_total": null_total,
            "positive_residual_total": pos_total,
            "negative_residual_total": neg_total,
        },
    }


def save_anchor_null_field_pickle(null_res: dict, path: str) -> str:
    """Save the full anchor null-field result dict to a .pkl file."""
    import pickle, os
    abspath = os.path.abspath(path)
    with open(abspath, "wb") as f:
        pickle.dump(null_res, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[null] saved null field result to: {abspath}", flush=True)
    return abspath




# -----------------------------------------------------------------------------
# Parallel morphism-comparison backend
# -----------------------------------------------------------------------------
# The serial analyze_morphism_match_field() above remains the reference
# implementation and the easiest path for small runs.  The functions below keep
# the same outward result contract while moving the expensive edge×edge scoring
# into chunked worker processes.  The canonical scalable artifact is the compact
# ``morphism_comparison`` payload saved by save_morphism_comparison_pickle().

_MORPHISM_COMPARE_CTX: dict = {}


def recommend_morphism_compare_workers(
    physical_core_hint: int | None = None,
    logical_core_hint: int | None = None,
    reserve_logical_cores: int = 4,
    hard_cap: int = 16,
) -> int:
    """
    Return a conservative default worker count for the parallel comparison pass.

    The comparison workers run large NumPy dot products.  On machines where BLAS
    can use more than one thread, too many Python workers can oversubscribe the
    CPU.  This helper therefore caps the default at 16 workers, which is a safe
    starting point for a 20-core / 40-logical Xeon workstation once BLAS threads
    are limited to one per worker.  Users can raise the worker count explicitly
    after benchmarking.
    """
    import os
    try:
        logical = int(logical_core_hint or (os.cpu_count() or 1))
    except Exception:
        logical = 1
    try:
        physical = int(physical_core_hint) if physical_core_hint is not None else None
    except Exception:
        physical = None

    if physical is not None and physical > 0:
        base = physical
    else:
        # Most workstations expose two logical CPUs per physical core.
        base = max(1, logical // 2) if logical >= 8 else logical

    # Leave several logical CPUs for the OS, UI, disk cache, and BLAS housekeeping.
    reserve_as_physical = max(1, int(round(max(0, reserve_logical_cores) / 2)))
    workers = max(1, base - reserve_as_physical)
    return int(max(1, min(int(hard_cap), workers)))


def _morphism_compare_as_desc_unique(vals, default=(0.0,)):
    out = []
    for v in vals if vals is not None else default:
        try:
            vv = max(0.0, min(1.0, float(v)))
        except Exception:
            continue
        if vv not in out:
            out.append(vv)
    return np.asarray(sorted(out, reverse=True) if out else list(default), dtype=np.float32)


def _morphism_compare_normalize_pc1_axis(mode) -> str:
    raw = str(mode or "dst").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "dst": "dst", "dest": "dst", "destination": "dst", "destination_only": "dst", "dst_only": "dst",
        "src": "src", "source": "src", "source_only": "src", "src_only": "src",
        "both": "both", "srcdst": "both", "src_dst": "both", "source_destination": "both",
        "source_and_destination": "both", "min": "both", "min_src_dst": "both", "composite": "both",
    }
    return aliases.get(raw, "dst")


def _morphism_compare_pc1_axis_label(mode: str) -> str:
    mode = _morphism_compare_normalize_pc1_axis(mode)
    return {"dst": "dst_pc1", "src": "src_pc1", "both": "min(src_pc1,dst_pc1)"}.get(mode, "dst_pc1")


def _morphism_compare_normalize_scope(mode) -> str:
    raw = str(mode or "full").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "full": "full", "all": "full", "global": "full", "collection": "full",
        "complete": "full", "everything": "full",
        "anchor": "anchor", "selected": "anchor", "selected_doc": "anchor",
        "selected_document": "anchor", "doc": "anchor", "doc_id": "anchor",
        "focus": "anchor", "source_doc": "anchor", "source_only": "anchor",
    }
    return aliases.get(raw, "full")


def _morphism_compare_normalize_acuity_mode(mode) -> str:
    raw = str(mode or "aligned_only").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "aligned": "aligned_only", "aligned_only": "aligned_only", "alignment": "aligned_only",
        "strict": "aligned_only", "default": "aligned_only",
        "all": "aligned_plus_pc1_only", "both": "aligned_plus_pc1_only",
        "aligned_plus_pc1": "aligned_plus_pc1_only", "aligned_plus_pc1_only": "aligned_plus_pc1_only",
        "aligned_and_pc1_only": "aligned_plus_pc1_only",
        "pc1": "pc1_only", "pc1_only": "pc1_only",
        "none": "none", "no": "none", "off": "none", "false": "none", "0": "none", "skip": "none",
    }
    return aliases.get(raw, "aligned_only")


def _morphism_match_record_dtype():
    return np.dtype([
        ("match_type", "u1"),           # 0 aligned, 1 pc1_only
        ("src_edge", "i4"),
        ("tgt_edge", "i4"),
        ("delta_cos", "f4"),
        ("src_pc1", "f4"),
        ("dst_pc1", "f4"),
        ("pc1_axis_value", "f4"),
        ("semantic_quality", "f4"),
        ("semantic_quality_min", "f4"),
        ("quality_axis_bin_0p01", "f4"),
        ("joint_min", "f4"),
        ("joint_min_4d", "f4"),
        ("joint_min_srcdst", "f4"),
        ("joint_min_srcdst_4d", "f4"),
        ("detected_delta_thr", "f4"),
        ("detected_pc1_thr", "f4"),
        ("detected_quality_thr", "f4"),
        ("delta_ok", "?"),
        ("src_pc1_ok", "?"),
        ("dst_pc1_ok", "?"),
        ("pc1_axis_ok", "?"),
        ("semantic_quality_ok", "?"),
        ("pc1_only_thr", "f4"),
        ("delta_max_for_pc1_only", "f4"),
        ("pc1_only_quality_thr", "f4"),
    ])


def _morphism_empty_records():
    return np.zeros((0,), dtype=_morphism_match_record_dtype())


def _morphism_records_from_tuples(rows):
    if not rows:
        return _morphism_empty_records()
    return np.asarray(rows, dtype=_morphism_match_record_dtype())


def _morphism_compare_threshold_indices(scores, thresholds):
    """Index of the strictest descending threshold satisfied by each score."""
    thr = np.asarray(thresholds, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    # For descending thresholds, the strictest threshold <= score is the first
    # insertion point in the ascending negated-threshold array.
    idx = np.searchsorted(-thr, -scores, side="left")
    return idx.astype(np.int32, copy=False)


def _morphism_compare_hmean_vec(a, b, eps: float = 1e-6):
    a = max(float(a), eps)
    b = np.maximum(np.asarray(b, dtype=np.float32), eps)
    return (2.0 / ((1.0 / a) + (1.0 / b))).astype(np.float32, copy=False)


def _morphism_compare_pc_axis(src_vals, dst_vals, mode: str):
    mode = _morphism_compare_normalize_pc1_axis(mode)
    if mode == "src":
        return src_vals
    if mode == "both":
        return np.minimum(src_vals, dst_vals)
    return dst_vals


def _morphism_compare_row(
    match_type: int,
    src_edge: int,
    tgt_edge: int,
    delta_val: float,
    src_pc1_val: float,
    dst_pc1_val: float,
    pc_axis_val: float,
    q_val: float,
    q_min_val: float,
    dt: float,
    pt: float,
    qt: float,
    pc1_only_thr: float = np.nan,
    delta_max_for_pc1_only: float = np.nan,
    pc1_only_quality_thr: float = np.nan,
):
    delta_val = float(delta_val)
    src_pc1_val = float(src_pc1_val) if np.isfinite(src_pc1_val) else np.nan
    dst_pc1_val = float(dst_pc1_val) if np.isfinite(dst_pc1_val) else np.nan
    pc_axis_val = float(pc_axis_val) if np.isfinite(pc_axis_val) else np.nan
    q_val = float(q_val)
    q_min_val = float(q_min_val)

    if np.isfinite(pc_axis_val):
        joint_min = min(delta_val, pc_axis_val)
        joint_min_4d = min(delta_val, pc_axis_val, q_val)
    else:
        joint_min = delta_val
        joint_min_4d = min(delta_val, q_val)
    if np.isfinite(src_pc1_val) and np.isfinite(dst_pc1_val):
        joint_min_srcdst = min(delta_val, src_pc1_val, dst_pc1_val)
        joint_min_srcdst_4d = min(delta_val, src_pc1_val, dst_pc1_val, q_val)
    else:
        joint_min_srcdst = np.nan
        joint_min_srcdst_4d = np.nan

    delta_ok = bool(np.isfinite(dt) and delta_val >= float(dt))
    src_ok = bool(np.isfinite(pt) and np.isfinite(src_pc1_val) and src_pc1_val >= float(pt))
    dst_ok = bool(np.isfinite(pt) and np.isfinite(dst_pc1_val) and dst_pc1_val >= float(pt))
    pc_axis_ok = bool(np.isfinite(pt) and np.isfinite(pc_axis_val) and pc_axis_val >= float(pt))
    q_ok = bool(np.isfinite(qt) and q_val >= float(qt))
    return (
        int(match_type), int(src_edge), int(tgt_edge),
        delta_val, src_pc1_val, dst_pc1_val, pc_axis_val,
        q_val, q_min_val, float(np.floor((max(0.0, min(1.0, q_val)) + 1e-12) / 0.01) * 0.01),
        float(joint_min), float(joint_min_4d), float(joint_min_srcdst), float(joint_min_srcdst_4d),
        float(dt) if np.isfinite(dt) else np.nan,
        float(pt) if np.isfinite(pt) else np.nan,
        float(qt) if np.isfinite(qt) else np.nan,
        delta_ok, src_ok, dst_ok, pc_axis_ok, q_ok,
        float(pc1_only_thr) if np.isfinite(pc1_only_thr) else np.nan,
        float(delta_max_for_pc1_only) if np.isfinite(delta_max_for_pc1_only) else np.nan,
        float(pc1_only_quality_thr) if np.isfinite(pc1_only_quality_thr) else np.nan,
    )


def _init_morphism_compare_worker(array_paths: dict, params: dict):
    """Initializer for source-chunk comparison workers."""
    import os
    import numpy as _np

    # These environment variables are most effective before NumPy loads, but the
    # runtime threadpoolctl limit below handles the common already-loaded case.
    os.environ.setdefault("OMP_NUM_THREADS", str(params.get("blas_threads", 1)))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(params.get("blas_threads", 1)))
    os.environ.setdefault("MKL_NUM_THREADS", str(params.get("blas_threads", 1)))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(params.get("blas_threads", 1)))

    global _MORPHISM_COMPARE_CTX
    _MORPHISM_COMPARE_CTX = {
        "delta_dir": _np.load(array_paths["delta_dir"], mmap_mode="r"),
        "edge_quality": _np.load(array_paths["edge_quality"], mmap_mode="r"),
        "edge_quality_min": _np.load(array_paths["edge_quality_min"], mmap_mode="r"),
        "doc_code": _np.load(array_paths["doc_code"], mmap_mode="r"),
        "source_indices": _np.load(array_paths["source_indices"], mmap_mode="r"),
        "src_pc1": None,
        "dst_pc1": None,
        "params": dict(params or {}),
        "_threadpool_controller": None,
    }
    if array_paths.get("src_pc1") and array_paths.get("dst_pc1"):
        _MORPHISM_COMPARE_CTX["src_pc1"] = _np.load(array_paths["src_pc1"], mmap_mode="r")
        _MORPHISM_COMPARE_CTX["dst_pc1"] = _np.load(array_paths["dst_pc1"], mmap_mode="r")

    try:
        from threadpoolctl import threadpool_limits
        _MORPHISM_COMPARE_CTX["_threadpool_controller"] = threadpool_limits(
            limits=int(params.get("blas_threads", 1)),
            user_api="blas",
        )
    except Exception:
        _MORPHISM_COMPARE_CTX["_threadpool_controller"] = None


def _morphism_compare_source_chunk_worker(source_pos_start: int, source_pos_stop: int):
    """Compare a chunk of source-edge positions against the full target pool."""
    import numpy as _np

    ctx = _MORPHISM_COMPARE_CTX
    delta_dir = ctx["delta_dir"]
    src_pc1 = ctx.get("src_pc1")
    dst_pc1 = ctx.get("dst_pc1")
    edge_q = ctx["edge_quality"]
    edge_q_min = ctx["edge_quality_min"]
    doc_code = ctx["doc_code"]
    source_indices = ctx["source_indices"]
    params = ctx["params"]

    target_block_size = max(1, int(params.get("target_block_size", 8192)))
    top_k = max(1, int(params.get("top_k_per_delta", 5)))
    require_cross_doc = bool(params.get("require_cross_doc", True))
    pc1_axis_mode = _morphism_compare_normalize_pc1_axis(params.get("pc1_match_axis", "dst"))
    have_pc1 = src_pc1 is not None and dst_pc1 is not None

    delta_thresholds = _np.asarray(params.get("delta_thresholds", [0.0]), dtype=_np.float32)
    pc1_thresholds = _np.asarray(params.get("pc1_thresholds", [0.0]), dtype=_np.float32)
    quality_thresholds = _np.asarray(params.get("quality_thresholds", [0.0]), dtype=_np.float32)
    pc1_only_threshold = float(params.get("pc1_only_threshold", 0.90))
    delta_max_for_pc1_only = float(params.get("delta_max_for_pc1_only", 0.60))
    pc1_only_quality_threshold = float(params.get("pc1_only_quality_threshold", 0.0))
    include_pc1_only = bool(params.get("include_pc1_only", True)) and have_pc1

    n_targets = int(delta_dir.shape[0])
    rows = []

    def _append_top_from_block(row_list, joint_vals, idxs_global, dvals, spvals, dpvals, pcvals, qvals, qminvals, dt, pt, qt, mtype, pthr=_np.nan, dmax=_np.nan, qthr=_np.nan):
        if idxs_global.size == 0:
            return row_list
        if idxs_global.size > top_k:
            keep_local = _np.argpartition(-joint_vals, top_k - 1)[:top_k]
        else:
            keep_local = _np.arange(idxs_global.size)
        for kk in keep_local:
            row_list.append((
                float(joint_vals[kk]),
                _morphism_compare_row(
                    mtype,
                    int(src_global),
                    int(idxs_global[kk]),
                    float(dvals[kk]),
                    float(spvals[kk]) if spvals is not None else _np.nan,
                    float(dpvals[kk]) if dpvals is not None else _np.nan,
                    float(pcvals[kk]) if pcvals is not None else _np.nan,
                    float(qvals[kk]),
                    float(qminvals[kk]),
                    float(dt), float(pt), float(qt),
                    pc1_only_thr=pthr,
                    delta_max_for_pc1_only=dmax,
                    pc1_only_quality_thr=qthr,
                )
            ))
        return row_list

    for source_pos in range(int(source_pos_start), int(source_pos_stop)):
        src_global = int(source_indices[source_pos])
        src_doc_code = int(doc_code[src_global])
        src_q = float(edge_q[src_global])
        src_q_min = float(edge_q_min[src_global])

        aligned_best_key = None
        aligned_candidates = []
        pc1_only_candidates = []

        for t0 in range(0, n_targets, target_block_size):
            t1 = min(t0 + target_block_size, n_targets)
            target_slice = slice(t0, t1)
            idxs_global_all = _np.arange(t0, t1, dtype=_np.int32)

            dvals_all = (delta_dir[target_slice] @ delta_dir[src_global]).astype(_np.float32, copy=False)
            qvals_all = _morphism_compare_hmean_vec(src_q, edge_q[target_slice])
            qmin_all = _np.minimum(src_q_min, edge_q_min[target_slice]).astype(_np.float32, copy=False)

            if have_pc1:
                spvals_all = _np.abs(src_pc1[target_slice] @ src_pc1[src_global]).astype(_np.float32, copy=False)
                dpvals_all = _np.abs(dst_pc1[target_slice] @ dst_pc1[src_global]).astype(_np.float32, copy=False)
                pcvals_all = _morphism_compare_pc_axis(spvals_all, dpvals_all, pc1_axis_mode).astype(_np.float32, copy=False)
            else:
                spvals_all = None
                dpvals_all = None
                pcvals_all = None

            if require_cross_doc:
                valid_base = (doc_code[target_slice] != src_doc_code)
            else:
                valid_base = _np.ones((t1 - t0,), dtype=bool)
            if t0 <= src_global < t1:
                valid_base[int(src_global - t0)] = False

            # Aligned threshold search.  We keep the lexicographically strictest
            # (Δ, PC1, Q) threshold tuple seen across target blocks, matching the
            # serial strict→loose loop without materializing a full M×M matrix.
            d_idx = _morphism_compare_threshold_indices(dvals_all, delta_thresholds)
            q_idx = _morphism_compare_threshold_indices(qvals_all, quality_thresholds)
            valid = valid_base & (d_idx < len(delta_thresholds)) & (q_idx < len(quality_thresholds))

            if have_pc1:
                p_idx = _morphism_compare_threshold_indices(pcvals_all, pc1_thresholds)
                valid &= (p_idx < len(pc1_thresholds))
            else:
                p_idx = _np.zeros_like(d_idx)

            if _np.any(valid):
                loc_valid = _np.where(valid)[0]
                if have_pc1:
                    order = _np.lexsort((q_idx[loc_valid], p_idx[loc_valid], d_idx[loc_valid]))
                    best_loc = loc_valid[order[0]]
                    block_key = (int(d_idx[best_loc]), int(p_idx[best_loc]), int(q_idx[best_loc]))
                else:
                    order = _np.lexsort((q_idx[loc_valid], d_idx[loc_valid]))
                    best_loc = loc_valid[order[0]]
                    block_key = (int(d_idx[best_loc]), 0, int(q_idx[best_loc]))

                if aligned_best_key is None or block_key < aligned_best_key:
                    aligned_best_key = block_key
                    aligned_candidates = []
                if block_key == aligned_best_key:
                    dt = float(delta_thresholds[block_key[0]])
                    pt = float(pc1_thresholds[block_key[1]]) if have_pc1 else _np.nan
                    qt = float(quality_thresholds[block_key[2]])
                    same_level = valid & (d_idx == block_key[0]) & (q_idx == block_key[2])
                    if have_pc1:
                        same_level &= (p_idx == block_key[1])
                    loc = _np.where(same_level)[0]
                    if loc.size:
                        if have_pc1:
                            joint = _np.minimum(_np.minimum(dvals_all[loc], pcvals_all[loc]), qvals_all[loc])
                            spv = spvals_all[loc]; dpv = dpvals_all[loc]; pcv = pcvals_all[loc]
                        else:
                            joint = _np.minimum(dvals_all[loc], qvals_all[loc])
                            spv = dpv = pcv = None
                        aligned_candidates = _append_top_from_block(
                            aligned_candidates,
                            joint,
                            idxs_global_all[loc],
                            dvals_all[loc],
                            spv,
                            dpv,
                            pcv,
                            qvals_all[loc],
                            qmin_all[loc],
                            dt, pt, qt,
                            0,
                        )

            # PC1-only search is independent of the strict aligned threshold grid.
            if include_pc1_only:
                pc1_mask = valid_base & (dvals_all < delta_max_for_pc1_only) & (pcvals_all >= pc1_only_threshold) & (qvals_all >= pc1_only_quality_threshold)
                loc = _np.where(pc1_mask)[0]
                if loc.size:
                    joint = _np.minimum(_np.minimum(dvals_all[loc], pcvals_all[loc]), qvals_all[loc])
                    pc1_only_candidates = _append_top_from_block(
                        pc1_only_candidates,
                        joint,
                        idxs_global_all[loc],
                        dvals_all[loc],
                        spvals_all[loc],
                        dpvals_all[loc],
                        pcvals_all[loc],
                        qvals_all[loc],
                        qmin_all[loc],
                        _np.nan,
                        pc1_only_threshold,
                        pc1_only_quality_threshold,
                        1,
                        pthr=pc1_only_threshold,
                        dmax=delta_max_for_pc1_only,
                        qthr=pc1_only_quality_threshold,
                    )

        if aligned_candidates:
            aligned_candidates.sort(key=lambda x: x[0], reverse=True)
            rows.extend([r for _joint, r in aligned_candidates[:top_k]])
        if pc1_only_candidates:
            pc1_only_candidates.sort(key=lambda x: x[0], reverse=True)
            rows.extend([r for _joint, r in pc1_only_candidates[:top_k]])

    return _morphism_records_from_tuples(rows)


def _morphism_edge_index_payload(entries: list[dict]) -> dict:
    docs = []
    doc_to_code = {}
    doc_codes = []
    src_labels = []
    dst_labels = []
    src_q = []
    dst_q = []
    edge_q = []
    edge_q_min = []
    src_lm_fluency = []
    dst_lm_fluency = []
    src_lm_nll = []
    dst_lm_nll = []
    src_quality_model = []
    dst_quality_model = []
    delta_norm = []
    for e in entries:
        d = str(e.get("doc"))
        if d not in doc_to_code:
            doc_to_code[d] = len(docs)
            docs.append(d)
        doc_codes.append(doc_to_code[d])
        src_labels.append(int(e.get("i_lab")))
        dst_labels.append(int(e.get("j_lab")))
        src_q.append(float(e.get("src_cluster_quality", 1.0)))
        dst_q.append(float(e.get("dst_cluster_quality", 1.0)))
        edge_q.append(float(e.get("edge_quality", 1.0)))
        edge_q_min.append(float(e.get("edge_quality_min", e.get("edge_quality", 1.0))))
        src_lm_fluency.append(e.get("src_cluster_lm_fluency", ""))
        dst_lm_fluency.append(e.get("dst_cluster_lm_fluency", ""))
        src_lm_nll.append(e.get("src_cluster_lm_nll", ""))
        dst_lm_nll.append(e.get("dst_cluster_lm_nll", ""))
        src_quality_model.append(str(e.get("src_quality_model", "")))
        dst_quality_model.append(str(e.get("dst_quality_model", "")))
        delta_norm.append(float(e.get("delta_norm", np.linalg.norm(e.get("delta_full", 0.0)) if e.get("delta_full") is not None else 0.0)))
    return {
        "doc_ids": docs,
        "doc_code": np.asarray(doc_codes, dtype=np.int32),
        "src_label": np.asarray(src_labels, dtype=np.int32),
        "dst_label": np.asarray(dst_labels, dtype=np.int32),
        "src_cluster_quality": np.asarray(src_q, dtype=np.float32),
        "dst_cluster_quality": np.asarray(dst_q, dtype=np.float32),
        "edge_quality": np.asarray(edge_q, dtype=np.float32),
        "edge_quality_min": np.asarray(edge_q_min, dtype=np.float32),
        "src_cluster_lm_fluency": np.asarray(src_lm_fluency, dtype=object),
        "dst_cluster_lm_fluency": np.asarray(dst_lm_fluency, dtype=object),
        "src_cluster_lm_nll": np.asarray(src_lm_nll, dtype=object),
        "dst_cluster_lm_nll": np.asarray(dst_lm_nll, dtype=object),
        "src_quality_model": np.asarray(src_quality_model, dtype=object),
        "dst_quality_model": np.asarray(dst_quality_model, dtype=object),
        "delta_norm": np.asarray(delta_norm, dtype=np.float32),
    }


def _morphism_edge_vectors_payload(entries: list[dict], include_delta_full: bool = False) -> dict:
    if not entries:
        return {}
    payload = {
        "delta_dir": np.vstack([np.asarray(e.get("dir_full"), dtype=np.float32) for e in entries]).astype(np.float32, copy=False),
    }
    if all(e.get("src_dir_full") is not None and e.get("dst_dir_full") is not None for e in entries):
        payload["src_pc1"] = np.vstack([np.asarray(e.get("src_dir_full"), dtype=np.float32) for e in entries]).astype(np.float32, copy=False)
        payload["dst_pc1"] = np.vstack([np.asarray(e.get("dst_dir_full"), dtype=np.float32) for e in entries]).astype(np.float32, copy=False)
    if include_delta_full:
        payload["delta_full"] = np.vstack([np.asarray(e.get("delta_full"), dtype=np.float32) for e in entries]).astype(np.float32, copy=False)
    return payload


def _morphism_add_no_acuity_scores(scores: dict, reason: str = "not_computed") -> None:
    try:
        d = float(scores.get("delta_cos", 0.0) or 0.0)
        pc = float(scores.get("pc1_axis_value", scores.get("dst_pc1", 1.0)) or 0.0)
        qv = float(scores.get("semantic_quality", 0.0) or 0.0)
        alignment_core = float(_quality_hmean([max(0.0, d), max(0.0, pc), max(0.0, qv)]))
    except Exception:
        alignment_core = 0.0
    scores.update({
        "acuity_computed": False,
        "acuity_compute_reason": str(reason),
        "lexical_available": False,
        "lexical_overlap_coefficient": 0.0,
        "lexical_divergence": 0.0,
        "alignment_core": alignment_core,
        "acuity_score": 0.0,
        "acuity_score_count_cosine": 0.0,
        "lexical_dst_overlap_coefficient": 0.0,
        "lexical_dst_jaccard": 0.0,
        "lexical_dst_dice": 0.0,
        "lexical_dst_weighted_jaccard": 0.0,
        "lexical_dst_count_cosine": 0.0,
        "lexical_dst_shared_token_mass_a": 0.0,
        "lexical_dst_shared_token_mass_b": 0.0,
        "lexical_dst_tokens_a": 0,
        "lexical_dst_tokens_b": 0,
        "lexical_dst_unique_a": 0,
        "lexical_dst_unique_b": 0,
        "lexical_dst_shared_unique": 0,
        "lexical_src_overlap_coefficient": 0.0,
        "lexical_src_jaccard": 0.0,
        "lexical_src_count_cosine": 0.0,
        "lexical_edge_overlap_coefficient": 0.0,
        "lexical_edge_jaccard": 0.0,
        "lexical_edge_count_cosine": 0.0,
    })


def _morphism_records_to_legacy_result(
    records,
    entries: list[dict],
    source_indices,
    params: dict,
    document_cluster_data: dict | None = None,
    segments_by_doc: dict | None = None,
) -> dict:
    """Convert compact match records to the existing Analyze result contract."""
    from collections import Counter

    records = np.asarray(records, dtype=_morphism_match_record_dtype())
    source_indices = np.asarray(source_indices, dtype=np.int32)
    pc1_axis_mode = _morphism_compare_normalize_pc1_axis(params.get("pc1_match_axis", "dst"))
    pc1_axis_label = _morphism_compare_pc1_axis_label(pc1_axis_mode)
    compute_acuity_for = _morphism_compare_normalize_acuity_mode(params.get("compute_acuity_for", "aligned_only"))

    def _should_compute_acuity(kind: str) -> bool:
        if compute_acuity_for == "none":
            return False
        if compute_acuity_for == "aligned_plus_pc1_only":
            return kind in ("aligned", "pc1_only")
        if compute_acuity_for == "pc1_only":
            return kind == "pc1_only"
        return kind == "aligned"

    manifold_residual_document_embedding_map = {}
    manifold_residual_document_embedding_meta = {}
    raw_sbert_document_embedding_map = {}
    raw_sbert_document_embedding_meta = {}
    for doc_id, data in (document_cluster_data or {}).items():
        res_vec, res_meta = _document_embedding_from_cdm(data, preferred_key="manifold_residual_document_embedding")
        raw_vec, raw_meta = _document_embedding_from_cdm(data, preferred_key="raw_sbert_document_embedding")
        for key in (doc_id, str(doc_id)):
            manifold_residual_document_embedding_meta[key] = dict(res_meta or {})
            raw_sbert_document_embedding_meta[key] = dict(raw_meta or {})
        if res_vec is not None:
            manifold_residual_document_embedding_map[doc_id] = res_vec
            manifold_residual_document_embedding_map[str(doc_id)] = res_vec
        if raw_vec is not None:
            raw_sbert_document_embedding_map[doc_id] = raw_vec
            raw_sbert_document_embedding_map[str(doc_id)] = raw_vec

    cluster_counter_cache = {}
    edge_counter_cache = {}

    def _cluster_counter_for(doc, lab):
        key = (str(doc), int(lab))
        if key in cluster_counter_cache:
            return cluster_counter_cache[key]
        cnt = Counter()
        try:
            if isinstance(segments_by_doc, dict) and doc in segments_by_doc and doc in (document_cluster_data or {}):
                labels_arr = document_cluster_data[doc][2]
                if hasattr(labels_arr, "tolist"):
                    labels_arr = labels_arr.tolist()
                segs = segments_by_doc.get(doc, [])
                texts = [str(segs[ii]) for ii, ll in enumerate(labels_arr) if int(ll) == int(lab) and ii < len(segs)]
                cnt = lexical_counter_from_texts(texts)
        except Exception:
            cnt = Counter()
        cluster_counter_cache[key] = cnt
        return cnt

    def _edge_counters_for(e):
        key = (str(e.get("doc")), int(e.get("i_lab")), int(e.get("j_lab")))
        if key in edge_counter_cache:
            return edge_counter_cache[key]
        src_cnt = _cluster_counter_for(e.get("doc"), e.get("i_lab"))
        dst_cnt = _cluster_counter_for(e.get("doc"), e.get("j_lab"))
        edge_cnt = _counter_add(src_cnt, dst_cnt)
        rec = (src_cnt, dst_cnt, edge_cnt)
        edge_counter_cache[key] = rec
        return rec

    def _add_lexical_acuity(scores, i, j):
        try:
            e_src = entries[int(i)]
            e_tgt = entries[int(j)]
            a_src, a_dst, a_edge = _edge_counters_for(e_src)
            b_src, b_dst, b_edge = _edge_counters_for(e_tgt)
            dst_m = lexical_overlap_metrics_from_counters(a_dst, b_dst)
            src_m = lexical_overlap_metrics_from_counters(a_src, b_src)
            edge_m = lexical_overlap_metrics_from_counters(a_edge, b_edge)
            lexical_available = bool(dst_m.get("lexical_available"))
            lexical_overlap = float(dst_m.get("overlap_coefficient", 0.0)) if lexical_available else 0.0
            lexical_divergence = float(1.0 - max(0.0, min(1.0, lexical_overlap))) if lexical_available else 0.0
            d = float(scores.get("delta_cos", 0.0) or 0.0)
            pc = float(scores.get("pc1_axis_value", scores.get("dst_pc1", 1.0)) or 0.0)
            qv = float(scores.get("semantic_quality", 0.0) or 0.0)
            alignment_core = float(_quality_hmean([max(0.0, d), max(0.0, pc), max(0.0, qv)]))
            acuity = float(alignment_core * lexical_divergence) if lexical_available else 0.0
            acuity_cosine = float(alignment_core * (1.0 - max(0.0, min(1.0, float(dst_m.get("count_cosine", 0.0)))))) if lexical_available else 0.0
            scores.update({
                "acuity_computed": bool(lexical_available),
                "acuity_compute_reason": "computed" if lexical_available else "no_lexical_tokens",
                "lexical_available": lexical_available,
                "lexical_overlap_coefficient": lexical_overlap,
                "lexical_divergence": lexical_divergence,
                "alignment_core": alignment_core,
                "acuity_score": acuity,
                "acuity_score_count_cosine": acuity_cosine,
                "lexical_dst_overlap_coefficient": float(dst_m.get("overlap_coefficient", 0.0)),
                "lexical_dst_jaccard": float(dst_m.get("jaccard", 0.0)),
                "lexical_dst_dice": float(dst_m.get("dice", 0.0)),
                "lexical_dst_weighted_jaccard": float(dst_m.get("weighted_jaccard", 0.0)),
                "lexical_dst_count_cosine": float(dst_m.get("count_cosine", 0.0)),
                "lexical_dst_shared_token_mass_a": float(dst_m.get("shared_token_mass_a", 0.0)),
                "lexical_dst_shared_token_mass_b": float(dst_m.get("shared_token_mass_b", 0.0)),
                "lexical_dst_tokens_a": int(dst_m.get("tokens_a", 0)),
                "lexical_dst_tokens_b": int(dst_m.get("tokens_b", 0)),
                "lexical_dst_unique_a": int(dst_m.get("unique_a", 0)),
                "lexical_dst_unique_b": int(dst_m.get("unique_b", 0)),
                "lexical_dst_shared_unique": int(dst_m.get("shared_unique", 0)),
                "lexical_src_overlap_coefficient": float(src_m.get("overlap_coefficient", 0.0)),
                "lexical_src_jaccard": float(src_m.get("jaccard", 0.0)),
                "lexical_src_count_cosine": float(src_m.get("count_cosine", 0.0)),
                "lexical_edge_overlap_coefficient": float(edge_m.get("overlap_coefficient", 0.0)),
                "lexical_edge_jaccard": float(edge_m.get("jaccard", 0.0)),
                "lexical_edge_count_cosine": float(edge_m.get("count_cosine", 0.0)),
            })
        except Exception:
            _morphism_add_no_acuity_scores(scores, reason="error")

    index = {
        int(i): {
            "doc": entries[int(i)]["doc"],
            "from": int(entries[int(i)]["i_lab"]),
            "to": int(entries[int(i)]["j_lab"]),
            "src_cluster_quality": float(entries[int(i)].get("src_cluster_quality", 1.0)),
            "dst_cluster_quality": float(entries[int(i)].get("dst_cluster_quality", 1.0)),
            "src_cluster_lm_fluency": entries[int(i)].get("src_cluster_lm_fluency", ""),
            "dst_cluster_lm_fluency": entries[int(i)].get("dst_cluster_lm_fluency", ""),
            "src_cluster_lm_nll": entries[int(i)].get("src_cluster_lm_nll", ""),
            "dst_cluster_lm_nll": entries[int(i)].get("dst_cluster_lm_nll", ""),
            "edge_quality": float(entries[int(i)].get("edge_quality", 1.0)),
            "edge_quality_hmean": float(entries[int(i)].get("edge_quality_hmean", entries[int(i)].get("edge_quality", 1.0))),
            "edge_quality_min": float(entries[int(i)].get("edge_quality_min", entries[int(i)].get("edge_quality", 1.0))),
        }
        for i in source_indices
    }
    aligned_matches = {int(i): [] for i in source_indices}
    pc1_only_matches = {int(i): [] for i in source_indices}
    aligned_counter = Counter()
    pc1_only_counter = Counter()

    def _clean_float(x, blank_if_nan=False):
        try:
            fx = float(x)
            if not np.isfinite(fx):
                return "" if blank_if_nan else fx
            return fx
        except Exception:
            return "" if blank_if_nan else np.nan

    for rec in records:
        i = int(rec["src_edge"]); j = int(rec["tgt_edge"])
        if i < 0 or j < 0 or i >= len(entries) or j >= len(entries):
            continue
        kind = "aligned" if int(rec["match_type"]) == 0 else "pc1_only"
        e_src = entries[i]
        e_tgt = entries[j]
        q_pair_hmean = float(rec["semantic_quality"])
        scores = {
            "delta_cos": float(rec["delta_cos"]),
            "src_edge_quality": float(e_src.get("edge_quality", 1.0)),
            "tgt_edge_quality": float(e_tgt.get("edge_quality", 1.0)),
            "src_edge_quality_hmean": float(e_src.get("edge_quality", 1.0)),
            "tgt_edge_quality_hmean": float(e_tgt.get("edge_quality", 1.0)),
            "src_edge_quality_min": float(e_src.get("edge_quality_min", e_src.get("edge_quality", 1.0))),
            "tgt_edge_quality_min": float(e_tgt.get("edge_quality_min", e_tgt.get("edge_quality", 1.0))),
            "semantic_quality": q_pair_hmean,
            "semantic_quality_hmean": q_pair_hmean,
            "semantic_quality_min": float(rec["semantic_quality_min"]),
            "quality_axis_bin_0p01": float(rec["quality_axis_bin_0p01"]),
            "src_from_quality": float(e_src.get("src_cluster_quality", 1.0)),
            "src_to_quality": float(e_src.get("dst_cluster_quality", 1.0)),
            "tgt_from_quality": float(e_tgt.get("src_cluster_quality", 1.0)),
            "tgt_to_quality": float(e_tgt.get("dst_cluster_quality", 1.0)),
            "src_from_lm_fluency": e_src.get("src_cluster_lm_fluency", ""),
            "src_to_lm_fluency": e_src.get("dst_cluster_lm_fluency", ""),
            "tgt_from_lm_fluency": e_tgt.get("src_cluster_lm_fluency", ""),
            "tgt_to_lm_fluency": e_tgt.get("dst_cluster_lm_fluency", ""),
            "src_from_lm_nll": e_src.get("src_cluster_lm_nll", ""),
            "src_to_lm_nll": e_src.get("dst_cluster_lm_nll", ""),
            "tgt_from_lm_nll": e_tgt.get("src_cluster_lm_nll", ""),
            "tgt_to_lm_nll": e_tgt.get("dst_cluster_lm_nll", ""),
            "src_quality_model": e_src.get("src_quality_model", ""),
            "tgt_quality_model": e_tgt.get("src_quality_model", ""),
            "src_pc1": _clean_float(rec["src_pc1"], blank_if_nan=True),
            "dst_pc1": _clean_float(rec["dst_pc1"], blank_if_nan=True),
            "pc1_axis_value": _clean_float(rec["pc1_axis_value"], blank_if_nan=True),
            "pc1_axis_mode": pc1_axis_mode,
            "pc1_axis_label": pc1_axis_label,
            "pc1_composite": (min(float(rec["src_pc1"]), float(rec["dst_pc1"])) if np.isfinite(float(rec["src_pc1"])) and np.isfinite(float(rec["dst_pc1"])) else ""),
            "joint_min": _clean_float(rec["joint_min"], blank_if_nan=True),
            "joint_min_4d": _clean_float(rec["joint_min_4d"], blank_if_nan=True),
            "joint_min_srcdst": _clean_float(rec["joint_min_srcdst"], blank_if_nan=True),
            "joint_min_srcdst_4d": _clean_float(rec["joint_min_srcdst_4d"], blank_if_nan=True),
        }
        res_doc_cos, res_doc_available = _document_embedding_cosine_from_maps(e_src.get("doc"), e_tgt.get("doc"), manifold_residual_document_embedding_map)
        raw_doc_cos, raw_doc_available = _document_embedding_cosine_from_maps(e_src.get("doc"), e_tgt.get("doc"), raw_sbert_document_embedding_map)
        scores.update({
            "manifold_residual_doc_cosine": res_doc_cos,
            "manifold_residual_doc_available": bool(res_doc_available),
            "raw_sbert_doc_cosine": raw_doc_cos,
            "raw_sbert_doc_available": bool(raw_doc_available),
            "src_manifold_residual_doc_embedding_source": (manifold_residual_document_embedding_meta.get(e_src.get("doc"), {}) or {}).get("source", ""),
            "tgt_manifold_residual_doc_embedding_source": (manifold_residual_document_embedding_meta.get(e_tgt.get("doc"), {}) or {}).get("source", ""),
            "src_manifold_residual_doc_embedding_method": (manifold_residual_document_embedding_meta.get(e_src.get("doc"), {}) or {}).get("method", ""),
            "tgt_manifold_residual_doc_embedding_method": (manifold_residual_document_embedding_meta.get(e_tgt.get("doc"), {}) or {}).get("method", ""),
            "src_raw_sbert_doc_embedding_source": (raw_sbert_document_embedding_meta.get(e_src.get("doc"), {}) or {}).get("source", ""),
            "tgt_raw_sbert_doc_embedding_source": (raw_sbert_document_embedding_meta.get(e_tgt.get("doc"), {}) or {}).get("source", ""),
            "src_raw_sbert_doc_embedding_method": (raw_sbert_document_embedding_meta.get(e_src.get("doc"), {}) or {}).get("method", ""),
            "tgt_raw_sbert_doc_embedding_method": (raw_sbert_document_embedding_meta.get(e_tgt.get("doc"), {}) or {}).get("method", ""),
        })
        if _should_compute_acuity(kind):
            _add_lexical_acuity(scores, i, j)
        else:
            _morphism_add_no_acuity_scores(scores, reason=f"compute_acuity_for={compute_acuity_for};kind={kind}")

        flags = {
            "delta_ok": bool(rec["delta_ok"]),
            "src_pc1_ok": bool(rec["src_pc1_ok"]),
            "dst_pc1_ok": bool(rec["dst_pc1_ok"]),
            "pc1_axis_ok": bool(rec["pc1_axis_ok"]),
            "semantic_quality_ok": bool(rec["semantic_quality_ok"]),
        }
        level = {}
        if np.isfinite(float(rec["detected_delta_thr"])):
            level["delta"] = float(rec["detected_delta_thr"])
        if np.isfinite(float(rec["detected_pc1_thr"])):
            level["pc1"] = float(rec["detected_pc1_thr"])
            level["pc1_axis_mode"] = pc1_axis_mode
            level["pc1_axis_label"] = pc1_axis_label
        if np.isfinite(float(rec["detected_quality_thr"])):
            level["quality"] = float(rec["detected_quality_thr"])
        m = {
            "j": int(j),
            "doc": e_tgt.get("doc"),
            "from": int(e_tgt.get("i_lab")),
            "to": int(e_tgt.get("j_lab")),
            "scores": scores,
            "flags": flags,
            "level": level,
        }
        if kind == "pc1_only":
            m["criteria"] = {
                "pc1_only_threshold": float(rec["pc1_only_thr"]),
                "delta_max": float(rec["delta_max_for_pc1_only"]),
                "quality_threshold": float(rec["pc1_only_quality_thr"]),
            }
            pc1_only_matches.setdefault(i, []).append(m)
            pc1_only_counter[(str(e_src.get("doc")), str(e_tgt.get("doc")))] += 1
        else:
            aligned_matches.setdefault(i, []).append(m)
            aligned_counter[(str(e_src.get("doc")), str(e_tgt.get("doc")))] += 1

    return {
        "aligned_matches": aligned_matches,
        "pc1_only_matches": pc1_only_matches,
        "index": index,
        "summary": {
            "aligned_per_docpair": aligned_counter,
            "pc1_only_per_docpair": pc1_only_counter,
        },
        "shapes": {
            "num_entries": int(source_indices.size),
            "source_entries": int(source_indices.size),
            "target_entries": int(len(entries)),
            "dim": int(len(entries[0]["dir_full"])) if entries else 0,
        },
        "params": dict(params),
    }


def analyze_morphism_match_field_parallel(
    document_cluster_data: dict,
    delta_thresholds=(0.98, 0.95, 0.92, 0.90, 0.88, 0.85),
    pc1_thresholds=(0.98, 0.95, 0.92, 0.90, 0.88, 0.85),
    quality_thresholds=None,
    segments_by_doc: dict | None = None,
    top_k_per_delta: int = 5,
    pc1_only_threshold: float = 0.90,
    delta_max_for_pc1_only: float = 0.60,
    pc1_only_quality_threshold: float = 0.0,
    compute_acuity_for: str = "aligned_only",
    pc1_match_axis: str = "dst",
    source_doc_filter: str | None = None,
    analyze_scope: str = "full",
    require_cross_doc: bool = True,
    max_workers: int | None = None,
    source_chunk_size: int = 384,
    target_block_size: int = 8192,
    blas_threads: int = 1,
    work_dir: str | None = None,
    keep_work_dir: bool = False,
    include_edge_vectors_in_result: bool = True,
    include_delta_full_in_vectors: bool = False,
    compact_only: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Parallel, array-backed version of analyze_morphism_match_field().

    The result preserves the existing Analyze dict contract unless
    ``compact_only=True``.  In every mode it also includes a
    ``morphism_comparison`` payload containing compact structured match records
    and an edge table suitable for direct pickle-based analysis.
    """
    import os
    import tempfile
    import shutil
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if quality_thresholds is None:
        quality_thresholds = tuple(round(i / 100, 2) for i in range(99, -1, -1))

    delta_thresholds = _morphism_compare_as_desc_unique(delta_thresholds, default=(0.0,))
    pc1_thresholds = _morphism_compare_as_desc_unique(pc1_thresholds, default=(0.0,))
    quality_thresholds = _morphism_compare_as_desc_unique(quality_thresholds, default=(0.0,))
    pc1_match_axis = _morphism_compare_normalize_pc1_axis(pc1_match_axis)
    analyze_scope = _morphism_compare_normalize_scope(analyze_scope)
    compute_acuity_for = _morphism_compare_normalize_acuity_mode(compute_acuity_for)
    source_doc_filter_norm = None if source_doc_filter is None else str(source_doc_filter)
    if analyze_scope == "anchor" and not source_doc_filter_norm:
        raise ValueError("analyze_scope='anchor' requires source_doc_filter/doc_id.")

    entries = _collect_morphism_edge_entries(document_cluster_data, segments_by_doc=segments_by_doc)
    M = len(entries)
    if M == 0:
        payload = {
            "kind": "morphism_comparison",
            "version": 1,
            "edge_index": _morphism_edge_index_payload([]),
            "matches": _morphism_empty_records(),
            "match_type_codes": {"aligned": 0, "pc1_only": 1},
            "params": {},
            "summary": {"num_edges": 0, "num_matches": 0},
        }
        return {
            "aligned_matches": {}, "pc1_only_matches": {}, "index": {}, "summary": {},
            "shapes": {"num_entries": 0, "source_entries": 0, "target_entries": 0, "dim": 0},
            "params": {}, "morphism_comparison": payload,
        }

    edge_index = _morphism_edge_index_payload(entries)
    docs_arr = np.asarray([str(e.get("doc")) for e in entries], dtype=object)
    if analyze_scope == "anchor":
        source_indices = np.asarray([i for i, d in enumerate(docs_arr) if str(d) == source_doc_filter_norm], dtype=np.int32)
        if source_indices.size == 0:
            raise ValueError(f"No source morphism edges found for source_doc_filter={source_doc_filter_norm!r}.")
    else:
        source_indices = np.arange(M, dtype=np.int32)

    have_pc1 = all(e.get("src_dir_full") is not None and e.get("dst_dir_full") is not None for e in entries)
    dim = int(len(entries[0].get("dir_full"))) if entries else 0
    workers = int(max_workers) if max_workers is not None else recommend_morphism_compare_workers()
    workers = max(1, min(workers, int(source_indices.size)))
    source_chunk_size = max(1, int(source_chunk_size))
    target_block_size = max(1, int(target_block_size))
    blas_threads = max(1, int(blas_threads))

    params = {
        "delta_thresholds": [float(x) for x in delta_thresholds.tolist()],
        "pc1_thresholds": [float(x) for x in pc1_thresholds.tolist()],
        "quality_thresholds": [float(x) for x in quality_thresholds.tolist()],
        "top_k_per_delta": int(top_k_per_delta),
        "pc1_only_threshold": float(pc1_only_threshold),
        "delta_max_for_pc1_only": float(delta_max_for_pc1_only),
        "pc1_only_quality_threshold": float(pc1_only_quality_threshold),
        "compute_acuity_for": compute_acuity_for,
        "pc1_match_axis": pc1_match_axis,
        "pc1_axis_label": _morphism_compare_pc1_axis_label(pc1_match_axis),
        "match_scope": analyze_scope,
        "analyze_scope": analyze_scope,
        "source_doc_filter": source_doc_filter_norm,
        "source_doc_id": source_doc_filter_norm,
        "source_entries": int(source_indices.size),
        "target_entries": int(M),
        "require_cross_doc": bool(require_cross_doc),
        "comparison_engine": "parallel_chunked_numpy",
        "parallel_workers": int(workers),
        "source_chunk_size": int(source_chunk_size),
        "target_block_size": int(target_block_size),
        "blas_threads_per_worker": int(blas_threads),
        "edge_vector_dim": int(dim),
        "quality_payload_present": bool(any(isinstance(v, (tuple, list)) and len(v) >= 7 for v in (document_cluster_data or {}).values())),
        "document_embedding_baseline": "manifold_residual_doc_cosine + raw_sbert_doc_cosine",
        "gating": f"delta & {_morphism_compare_pc1_axis_label(pc1_match_axis)} & semantic_quality (independent axes); ranking uses joint_min_4d",
    }

    temp_created = False
    temp_dir = None
    if work_dir:
        temp_dir = os.path.abspath(work_dir)
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="morphism_compare_")
        temp_created = True

    array_paths = {}
    try:
        delta_dir = np.vstack([np.asarray(e["dir_full"], dtype=np.float32) for e in entries]).astype(np.float32, copy=False)
        np.save(os.path.join(temp_dir, "delta_dir.npy"), delta_dir)
        array_paths["delta_dir"] = os.path.join(temp_dir, "delta_dir.npy")
        np.save(os.path.join(temp_dir, "edge_quality.npy"), edge_index["edge_quality"].astype(np.float32, copy=False))
        array_paths["edge_quality"] = os.path.join(temp_dir, "edge_quality.npy")
        np.save(os.path.join(temp_dir, "edge_quality_min.npy"), edge_index["edge_quality_min"].astype(np.float32, copy=False))
        array_paths["edge_quality_min"] = os.path.join(temp_dir, "edge_quality_min.npy")
        np.save(os.path.join(temp_dir, "doc_code.npy"), edge_index["doc_code"].astype(np.int32, copy=False))
        array_paths["doc_code"] = os.path.join(temp_dir, "doc_code.npy")
        np.save(os.path.join(temp_dir, "source_indices.npy"), source_indices.astype(np.int32, copy=False))
        array_paths["source_indices"] = os.path.join(temp_dir, "source_indices.npy")
        if have_pc1:
            src_pc1_arr = np.vstack([np.asarray(e["src_dir_full"], dtype=np.float32) for e in entries]).astype(np.float32, copy=False)
            dst_pc1_arr = np.vstack([np.asarray(e["dst_dir_full"], dtype=np.float32) for e in entries]).astype(np.float32, copy=False)
            np.save(os.path.join(temp_dir, "src_pc1.npy"), src_pc1_arr)
            np.save(os.path.join(temp_dir, "dst_pc1.npy"), dst_pc1_arr)
            array_paths["src_pc1"] = os.path.join(temp_dir, "src_pc1.npy")
            array_paths["dst_pc1"] = os.path.join(temp_dir, "dst_pc1.npy")

        # Set before ProcessPoolExecutor spawns workers so Windows/spawn workers
        # import NumPy/OpenBLAS with the requested thread cap.
        for _env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[_env_name] = str(int(blas_threads))

        worker_params = dict(params)
        worker_params.update({
            "delta_thresholds": delta_thresholds,
            "pc1_thresholds": pc1_thresholds,
            "quality_thresholds": quality_thresholds,
            "target_block_size": int(target_block_size),
            "include_pc1_only": bool(have_pc1),
            "blas_threads": int(blas_threads),
        })

        chunks = [(s, min(s + source_chunk_size, int(source_indices.size))) for s in range(0, int(source_indices.size), source_chunk_size)]
        if verbose:
            print(
                f"[analyze-parallel] source_edges={source_indices.size:,}; target_edges={M:,}; "
                f"candidate_pairs={int(source_indices.size) * int(M):,}; workers={workers}; "
                f"source_chunk={source_chunk_size}; target_block={target_block_size}; have_pc1={have_pc1}",
                flush=True,
            )

        arrays = []
        if workers <= 1 or len(chunks) <= 1:
            _init_morphism_compare_worker(array_paths, worker_params)
            for k, (s, t) in enumerate(chunks, 1):
                arrays.append(_morphism_compare_source_chunk_worker(s, t))
                if verbose and (k == 1 or k == len(chunks) or k % max(1, len(chunks)//10) == 0):
                    print(f"[analyze-parallel] chunk {k}/{len(chunks)} complete", flush=True)
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_morphism_compare_worker,
                initargs=(array_paths, worker_params),
            ) as pool:
                futs = {pool.submit(_morphism_compare_source_chunk_worker, s, t): (s, t) for s, t in chunks}
                done = 0
                for fut in as_completed(futs):
                    arrays.append(fut.result())
                    done += 1
                    if verbose and (done == 1 or done == len(futs) or done % max(1, len(futs)//10) == 0):
                        print(f"[analyze-parallel] chunk {done}/{len(futs)} complete", flush=True)
        if arrays:
            nonempty = [a for a in arrays if getattr(a, "size", 0) > 0]
            records = np.concatenate(nonempty).astype(_morphism_match_record_dtype(), copy=False) if nonempty else _morphism_empty_records()
        else:
            records = _morphism_empty_records()
    finally:
        if temp_created and not keep_work_dir:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    n_aligned = int(np.count_nonzero(records["match_type"] == 0)) if records.size else 0
    n_pc1_only = int(np.count_nonzero(records["match_type"] == 1)) if records.size else 0
    if verbose:
        print(
            f"[analyze-parallel] retained matches: aligned={n_aligned:,}; pc1_only={n_pc1_only:,}; total={records.size:,}",
            flush=True,
        )

    payload = {
        "kind": "morphism_comparison",
        "version": 1,
        "match_type_codes": {"aligned": 0, "pc1_only": 1},
        "edge_index": edge_index,
        "matches": records,
        "params": dict(params),
        "summary": {
            "num_edges": int(M),
            "source_edges": int(source_indices.size),
            "target_edges": int(M),
            "num_matches": int(records.size),
            "num_aligned_matches": int(n_aligned),
            "num_pc1_only_matches": int(n_pc1_only),
            "comparison_matrix_shape": (int(source_indices.size), int(M)),
            "candidate_edge_pairs": int(source_indices.size) * int(M),
            "have_pc1": bool(have_pc1),
            "edge_vector_dim": int(dim),
        },
    }
    if include_edge_vectors_in_result:
        payload["edge_vectors"] = _morphism_edge_vectors_payload(entries, include_delta_full=include_delta_full_in_vectors)

    if compact_only:
        return {
            "aligned_matches": {},
            "pc1_only_matches": {},
            "index": {},
            "summary": {},
            "shapes": {"num_entries": int(source_indices.size), "source_entries": int(source_indices.size), "target_entries": int(M), "dim": int(dim)},
            "params": dict(params),
            "morphism_comparison": payload,
        }

    res = _morphism_records_to_legacy_result(
        records,
        entries,
        source_indices,
        params,
        document_cluster_data=document_cluster_data,
        segments_by_doc=segments_by_doc,
    )
    res["morphism_comparison"] = payload
    return res


def save_morphism_comparison_pickle(res_or_payload: dict, path: str, include_legacy_result: bool = False) -> str:
    """
    Save the compact morphism-comparison payload to a .pkl file.

    Pass either the full Analyze result returned by
    analyze_morphism_match_field_parallel() or the payload in its
    ``morphism_comparison`` key.  By default the legacy nested match maps are not
    stored, keeping the pickle substantially smaller than a dict-of-dicts export.
    """
    import os
    import pickle

    if not isinstance(res_or_payload, dict):
        raise ValueError("res_or_payload must be a dict returned by the Analyze comparison backend.")
    if res_or_payload.get("kind") == "morphism_comparison":
        payload = dict(res_or_payload)
    else:
        payload = dict(res_or_payload.get("morphism_comparison") or {})
        if not payload:
            # Serial results do not have the compact payload.  Store the full
            # result as a compatibility fallback rather than silently failing.
            payload = {"kind": "morphism_comparison_legacy_result", "version": 1, "legacy_result": res_or_payload}
    if include_legacy_result and res_or_payload.get("kind") != "morphism_comparison":
        payload["legacy_result"] = {
            "aligned_matches": res_or_payload.get("aligned_matches", {}),
            "pc1_only_matches": res_or_payload.get("pc1_only_matches", {}),
            "index": res_or_payload.get("index", {}),
            "summary": res_or_payload.get("summary", {}),
            "shapes": res_or_payload.get("shapes", {}),
            "params": res_or_payload.get("params", {}),
        }
    abspath = os.path.abspath(path)
    with open(abspath, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[analyze] saved morphism comparison PKL to: {abspath}", flush=True)
    return abspath

def output_anchor_null_match_csvs(
    null_res: dict,
    field_csv_path: str | None = None,
    edge_csv_path: str | None = None,
    match_csv_path: str | None = None,
    max_field_rows: int = 250000,
):
    """
    Save CSVs for explainability:
      * field CSV: sparse threshold voxels with observed/null/residual statistics
      * edge CSV: anchor source-edge contribution summary
      * match CSV: pair-level observed and one null-example match rows above the endpoint thresholds
    """
    import csv, os
    if not null_res or null_res.get("kind") != "anchor_null_match_field":
        raise ValueError("null_res must be an anchor_null_match_field result returned by analyze_anchor_null_match_field().")

    step = float(null_res.get("params", {}).get("step", 0.01))
    views = null_res.get("views", {}) or {}
    obs = np.asarray(views.get("observed"), dtype=float)
    nmean = np.asarray(views.get("null_mean"), dtype=float)
    nstd = np.asarray(views.get("null_std"), dtype=float)
    resid = np.asarray(views.get("residual"), dtype=float)
    loge = np.asarray(views.get("log_enrichment"), dtype=float)
    z = np.asarray(views.get("z_score"), dtype=float)
    if obs.size == 0:
        print("[null] no field rows to save", flush=True)
    elif field_csv_path:
        n_bins = obs.shape[0]
        thr = np.round(np.linspace(1.0, 0.0, n_bins), 2)
        mask = (obs > 0) | (nmean > 0) | (np.abs(resid) > 1e-12)
        idx = np.where(mask)
        vals_abs = np.abs(resid[idx])
        if vals_abs.size > int(max_field_rows):
            order = np.argsort(-vals_abs)[:int(max_field_rows)]
            idx = tuple(a[order] for a in idx)
        abspath = os.path.abspath(field_csv_path)
        with open(abspath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                "anchor_doc", "delta_threshold", "pc1_threshold", "quality_threshold",
                "observed_count", "null_mean_count", "null_std_count", "residual_observed_minus_null",
                "positive_residual", "negative_residual", "log_enrichment", "z_score",
                "pc1_axis_label", "n_null_replicates", "null_strategy"
            ])
            for i, j, k in zip(idx[0], idx[1], idx[2]):
                rv = float(resid[i, j, k])
                writer.writerow([
                    null_res.get("anchor_doc_id", ""),
                    float(thr[i]), float(thr[j]), float(thr[k]),
                    float(obs[i, j, k]), float(nmean[i, j, k]), float(nstd[i, j, k]), rv,
                    max(rv, 0.0), max(-rv, 0.0), float(loge[i, j, k]), float(z[i, j, k]),
                    null_res.get("params", {}).get("pc1_axis_label", ""),
                    null_res.get("params", {}).get("n_null_replicates", ""),
                    null_res.get("params", {}).get("null_strategy", ""),
                ])
        print(f"[null] field CSV written to: {abspath}", flush=True)

    if edge_csv_path:
        abspath = os.path.abspath(edge_csv_path)
        rows = list(null_res.get("edge_contributions", []) or [])
        with open(abspath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            header = [
                "anchor_doc", "anchor_edge_index", "anchor_from", "anchor_to",
                "anchor_edge_quality", "anchor_src_quality", "anchor_dst_quality",
                "observed_support", "null_support_mean", "null_support_std",
                "support_residual", "support_z_score",
                "best_delta", "best_pc1", "best_quality", "best_joint",
                "edge_support_delta_threshold", "edge_support_pc1_threshold", "edge_support_quality_threshold",
            ]
            writer.writerow(header)
            params = null_res.get("params", {}) or {}
            for row in rows:
                writer.writerow([
                    row.get("anchor_doc", null_res.get("anchor_doc_id", "")),
                    row.get("anchor_edge_index", ""), row.get("anchor_from", ""), row.get("anchor_to", ""),
                    row.get("anchor_edge_quality", ""), row.get("anchor_src_quality", ""), row.get("anchor_dst_quality", ""),
                    row.get("observed_support", ""), row.get("null_support_mean", ""), row.get("null_support_std", ""),
                    row.get("support_residual", ""), row.get("support_z_score", ""),
                    row.get("best_delta", ""), row.get("best_pc1", ""), row.get("best_quality", ""), row.get("best_joint", ""),
                    params.get("edge_support_delta_threshold", ""), params.get("edge_support_pc1_threshold", ""), params.get("edge_support_quality_threshold", ""),
                ])
        print(f"[null] edge contribution CSV written to: {abspath}", flush=True)

    if match_csv_path:
        abspath = os.path.abspath(match_csv_path)
        match_rows = []
        mr = null_res.get("match_rows", {}) or {}
        match_rows.extend(list(mr.get("observed", []) or []))
        match_rows.extend(list(mr.get("null_example", []) or []))
        with open(abspath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            header = [
                "match_set", "feature_association",
                "anchor_doc", "anchor_edge_index", "anchor_from", "anchor_to",
                "target_doc", "target_edge_index", "target_from", "target_to",
                "delta_cos", "src_pc1", "dst_pc1", "pc1_axis_value", "semantic_quality", "joint_min",
                "anchor_edge_quality", "target_edge_quality",
                "edge_support_delta_threshold", "edge_support_pc1_threshold", "edge_support_quality_threshold",
                "pc1_axis_label", "null_strategy"
            ]
            writer.writerow(header)
            params = null_res.get("params", {}) or {}
            for row in match_rows:
                writer.writerow([
                    row.get("match_set", ""), row.get("feature_association", ""),
                    row.get("anchor_doc", ""), row.get("anchor_edge_index", ""), row.get("anchor_from", ""), row.get("anchor_to", ""),
                    row.get("target_doc", ""), row.get("target_edge_index", ""), row.get("target_from", ""), row.get("target_to", ""),
                    row.get("delta_cos", ""), row.get("src_pc1", ""), row.get("dst_pc1", ""), row.get("pc1_axis_value", ""), row.get("semantic_quality", ""), row.get("joint_min", ""),
                    row.get("anchor_edge_quality", ""), row.get("target_edge_quality", ""),
                    params.get("edge_support_delta_threshold", ""), params.get("edge_support_pc1_threshold", ""), params.get("edge_support_quality_threshold", ""),
                    params.get("pc1_axis_label", ""), params.get("null_strategy", ""),
                ])
        print(f"[null] pair-level match CSV written to: {abspath}", flush=True)


def plot_anchor_null_match_field_3d(
    null_res: dict,
    figsize=(12, 9),
    max_plot_points: int = 150000,
    initial_view: str = "positive_residual",
    log_size: bool = True,
):
    """
    Five-view toggle plot for anchor null-field comparison.

    Views:
      1. Observed field
      2. Null mean field
      3. Positive residual field: observed > null
      4. Negative residual field: null > observed
      5. Anchor source-edge contribution
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not null_res or null_res.get("kind") != "anchor_null_match_field":
        print("[plot_anchor_null_match_field_3d] invalid null result")
        return

    params = null_res.get("params", {}) or {}
    step = float(params.get("step", 0.01))
    views = null_res.get("views", {}) or {}
    obs = np.asarray(views.get("observed"), dtype=float)
    if obs.size == 0:
        print("[plot_anchor_null_match_field_3d] empty field")
        return
    n_bins = obs.shape[0]
    thr = np.round(np.linspace(1.0, 0.0, n_bins), 2)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    state = {"view": str(initial_view or "positive_residual"), "q_floor": 0.0, "scat": None, "legend": None}

    view_order = ["observed", "null_mean", "positive_residual", "negative_residual", "edge_contribution"]
    view_labels = {
        "observed": "Observed: A ↔ real targets",
        "null_mean": "Null mean: A ↔ shuffled targets",
        "positive_residual": "Positive residual: observed > null",
        "negative_residual": "Negative residual: null > observed",
        "edge_contribution": "Anchor source-edge contribution",
    }
    view_colors = {
        "observed": "#2ca02c",
        "null_mean": "#1f77b4",
        "positive_residual": "#00b050",
        "negative_residual": "#1f4fff",
        "edge_positive": "#00b050",
        "edge_negative": "#1f4fff",
        "edge_neutral": "#777777",
    }

    def _grid_for_view(name):
        if name == "observed":
            return np.asarray(views.get("observed"), dtype=float)
        if name == "null_mean":
            return np.asarray(views.get("null_mean"), dtype=float)
        if name == "negative_residual":
            return np.asarray(views.get("negative_residual"), dtype=float)
        return np.asarray(views.get("positive_residual"), dtype=float)

    def _select_grid_points(grid):
        mask = np.asarray(grid) > 0
        q_mask = thr >= float(state.get("q_floor", 0.0))
        if q_mask.size == grid.shape[2]:
            mask[:, :, ~q_mask] = False
        idx = np.where(mask)
        vals = grid[idx]
        if vals.size > int(max_plot_points):
            order = np.argsort(-vals)[:int(max_plot_points)]
            idx = tuple(a[order] for a in idx)
            vals = vals[order]
        return idx, vals

    def _size_from_vals(vals, base=14.0, scale=60.0):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            return vals
        if log_size:
            vv = np.log1p(np.maximum(vals, 0.0))
        else:
            vv = np.maximum(vals, 0.0)
        mx = float(vv.max()) if vv.size else 1.0
        if mx <= 0:
            return np.full(vals.shape, base)
        return base + scale * (vv / mx)

    def _refresh(_evt=None):
        if state["scat"] is not None:
            try:
                state["scat"].remove()
            except Exception:
                pass
            state["scat"] = None
        if state["legend"] is not None:
            try:
                state["legend"].remove()
            except Exception:
                pass
            state["legend"] = None

        ax.cla()
        view = state["view"]
        anchor = null_res.get("anchor_doc_id", "")
        axis_label = params.get("pc1_axis_label", "dst_pc1")

        if view == "edge_contribution":
            rows = list(null_res.get("edge_contributions", []) or [])
            rows = [r for r in rows if float(r.get("best_quality", 0.0) or 0.0) >= float(state.get("q_floor", 0.0))]
            if rows:
                x = np.asarray([float(r.get("best_delta", 0.0) or 0.0) for r in rows], dtype=float)
                y = np.asarray([float(r.get("best_pc1", 0.0) or 0.0) for r in rows], dtype=float)
                z = np.asarray([float(r.get("best_quality", 0.0) or 0.0) for r in rows], dtype=float)
                residuals = np.asarray([float(r.get("support_residual", 0.0) or 0.0) for r in rows], dtype=float)
                colors = [view_colors["edge_positive"] if v > 0 else view_colors["edge_negative"] if v < 0 else view_colors["edge_neutral"] for v in residuals]
                sizes = _size_from_vals(np.abs(residuals), base=20, scale=80)
                state["scat"] = ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.85, edgecolors="k", linewidths=0.4, depthshade=True)
                try:
                    import mplcursors
                    cursor = mplcursors.cursor(state["scat"], hover=True, annotation_kwargs=dict(arrowprops=None))
                    @cursor.connect("add")
                    def _on_add(sel):
                        idx = int(sel.index)
                        if 0 <= idx < len(rows):
                            r = rows[idx]
                            sel.annotation.set_text(
                                f"{anchor}: C{r.get('anchor_from')}→C{r.get('anchor_to')}\n"
                                f"obs={r.get('observed_support')}, null≈{float(r.get('null_support_mean',0.0)):.2f}\n"
                                f"resid={float(r.get('support_residual',0.0)):+.2f}, z={float(r.get('support_z_score',0.0)):+.2f}\n"
                                f"best Δ={float(r.get('best_delta',0.0)):.2f}, PC1={float(r.get('best_pc1',0.0)):.2f}, Q={float(r.get('best_quality',0.0)):.2f}"
                            )
                except Exception:
                    pass
            handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=view_colors["edge_positive"], markeredgecolor='k', markersize=8, linewidth=0, label="real-enriched anchor edge"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=view_colors["edge_negative"], markeredgecolor='k', markersize=8, linewidth=0, label="null-enriched anchor edge"),
            ]
            state["legend"] = ax.legend(handles=handles, loc="upper right")
        else:
            grid = _grid_for_view(view)
            idx, vals = _select_grid_points(grid)
            if vals.size:
                x = thr[idx[0]]; y = thr[idx[1]]; z = thr[idx[2]]
                color = view_colors.get(view, "#00b050")
                sizes = _size_from_vals(vals, base=10, scale=70)
                state["scat"] = ax.scatter(x, y, z, c=color, s=sizes, alpha=0.70, edgecolors="none", depthshade=True)
            handles = [Patch(facecolor=view_colors.get(view, "#00b050"), edgecolor="k", label=view_labels.get(view, view))]
            if vals.size >= int(max_plot_points):
                handles.append(Patch(facecolor=(0.75,0.75,0.75,1), edgecolor="k", label=f"display capped at {int(max_plot_points):,} pts"))
            state["legend"] = ax.legend(handles=handles, loc="upper right")

        ax.set_title(f"Anchor null-field comparison for {anchor}\n{view_labels.get(view, view)}; Q ≥ {state['q_floor']:.2f}")
        ax.set_xlabel("Δ direction cosine threshold / best Δ")
        ax.set_ylabel(f"PC1 threshold / best PC1 ({axis_label})")
        ax.set_zlabel("Semantic quality threshold Q / best Q")
        ax.set_xlim(1.0, 0.0)
        ax.set_ylim(1.0, 0.0)
        ax.set_zlim(0.0, 1.0)
        fig.canvas.draw_idle()

    _refresh()

    ax_q = fig.add_axes([0.76, 0.92, 0.20, 0.03]); ax_q.set_in_layout(False)
    slider_q = Slider(ax_q, "Q floor", 0.0, 1.0, valinit=0.0, valstep=step)
    def _on_q(val):
        state["q_floor"] = float(val)
        _refresh()
    slider_q.on_changed(_on_q)

    # Five view buttons.
    buttons = []
    labels = [
        ("observed", "Observed"),
        ("null_mean", "Null"),
        ("positive_residual", "+Residual"),
        ("negative_residual", "-Residual"),
        ("edge_contribution", "A-edge"),
    ]
    for ix, (key, lab) in enumerate(labels):
        axb = fig.add_axes([0.02 + ix * 0.105, 0.92, 0.095, 0.035]); axb.set_in_layout(False)
        b = Button(axb, lab)
        def _make_cb(k):
            def _cb(_evt):
                state["view"] = k
                _refresh()
            return _cb
        b.on_clicked(_make_cb(key))
        buttons.append(b)

    # Keep widget references alive.
    fig._anchor_null_widgets = {"buttons": buttons, "slider_q": slider_q}
    plt.show()


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
    csv_path=None,
    csv_mode: str = "full",
    print_summary: bool = True
):
    """
    Print analysis and write a tab-delimited CSV with one row per unique match.

    CSV includes the independent axes:
        delta_cos, selected PC1 axis, semantic_quality
    while still exporting both raw src_pc1 and dst_pc1 values. In v10 the
    default selected PC1 axis is destination-PC1.

    csv_mode controls file output. print_summary controls the legacy console summaries:
        "full"     -> write every deduplicated inter-document match row
        "selected" -> write only rows where doc_id is source or target document
        "none"     -> do not prompt for or write an Analyze CSV
    """
    import csv
    from tkinter import filedialog
    import tkinter as tk

    def _fmt(x):
        return "—" if x is None or x == "" else f"{float(x):.3f}"

    def _get_param(name, default_val):
        try:
            params = res.get("params", {})
            if name in params:
                return float(params[name])
        except Exception:
            pass
        if name in {"pc1_only_threshold", "delta_max_for_pc1_only", "pc1_only_quality_threshold"}:
            try:
                for _i, lst in res.get("pc1_only_matches", {}).items():
                    for m in lst or []:
                        crit = m.get("criteria") or {}
                        key = "delta_max" if name == "delta_max_for_pc1_only" else ("quality_threshold" if name == "pc1_only_quality_threshold" else name)
                        if key in crit:
                            return float(crit[key])
            except Exception:
                pass
        return float(default_val)

    def _pc1_axis_label_for_output() -> str:
        try:
            params = res.get("params", {}) or {}
            return str(params.get("pc1_axis_label") or params.get("pc1_match_axis") or "dst_pc1")
        except Exception:
            return "dst_pc1"

    pc1_axis_label_out = _pc1_axis_label_for_output()

    def _normalize_csv_mode(mode) -> str:
        raw = str(mode or "full").strip().lower().replace("-", "_").replace(" ", "_")
        if raw in {"full", "all", "everything", "complete"}:
            return "full"
        if raw in {"selected", "selected_doc", "selected_document", "doc", "doc_id", "anchor", "involving", "focus"}:
            return "selected"
        if raw in {"none", "no", "no_csv", "off", "false", "0", "skip"}:
            return "none"
        print(f"[output_analysis] Unknown csv_mode={mode!r}; using 'full'.")
        return "full"

    csv_mode_norm = _normalize_csv_mode(csv_mode)

    def _choose_csv_save_path(default_name: str = "morphism_matches.csv", parent_window=None):
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

    def _get_labels_array(doc):
        if not document_cluster_data or doc not in document_cluster_data:
            return None
        try:
            return document_cluster_data[doc][2]
        except Exception:
            return None

    def _get_segments_list(doc):
        if segments_by_doc and doc in segments_by_doc:
            return segments_by_doc[doc]
        return None

    def _cluster_text(doc, cluster_label):
        labels_arr = _get_labels_array(doc)
        segs = _get_segments_list(doc)
        if labels_arr is None or segs is None:
            return "N/A"
        try:
            idxs = [i for i, lab in enumerate(labels_arr) if int(lab) == int(cluster_label)]
            return "\n\n".join(str(segs[i]) for i in idxs)
        except Exception:
            return "N/A"

    if not res or "index" not in res or not res["index"]:
        print("[output_analysis] No entries in result index.")
        return

    if print_summary:
        first_idx, first_meta = next(iter(res["index"].items()))
        display_doc = first_meta["doc"] if doc_id is None else doc_id
        entries = [(i, meta) for i, meta in res["index"].items() if meta["doc"] == display_doc]
        entries.sort(key=lambda t: (t[1]["from"], t[1]["to"]))

        print(f"\n=== Matches for document: {display_doc} ===\n")
        for i, meta in entries:
            src_lab = meta["from"]
            dst_lab = meta["to"]
            edge_q = meta.get("edge_quality", "")
            print(f"C{src_lab} \u2192 C{dst_lab} | edge_Q={_fmt(edge_q)}")

            aligned = res.get("aligned_matches", {}).get(i, [])
            pc1_only = res.get("pc1_only_matches", {}).get(i, [])

            if aligned:
                print(f"  Δ & {pc1_axis_label_out} & Q aligned:")
                for m in aligned:
                    sc = m.get("scores", {}) or {}
                    print(
                        "    → {doc}: C{f}→C{t} | Δ={d} pc_axis={pa} src={s} dst={p} Q={q} joint4={j}".format(
                            doc=m["doc"], f=m["from"], t=m["to"],
                            d=_fmt(sc.get("delta_cos")),
                            pa=_fmt(sc.get("pc1_axis_value", sc.get("dst_pc1"))),
                            s=_fmt(sc.get("src_pc1")),
                            p=_fmt(sc.get("dst_pc1")),
                            q=_fmt(sc.get("semantic_quality")),
                            j=_fmt(sc.get("joint_min_4d")),
                        )
                    )
            else:
                print(f"  Δ & {pc1_axis_label_out} & Q aligned: (none)")

            if pc1_only:
                print("  PC1-only:")
                for m in pc1_only:
                    sc = m.get("scores", {}) or {}
                    print(
                        "    → {doc}: C{f}→C{t} | Δ={d} pc_axis={pa} src={s} dst={p} Q={q} pc1_comp={pc}".format(
                            doc=m["doc"], f=m["from"], t=m["to"],
                            d=_fmt(sc.get("delta_cos")),
                            pa=_fmt(sc.get("pc1_axis_value", sc.get("dst_pc1"))),
                            s=_fmt(sc.get("src_pc1")),
                            p=_fmt(sc.get("dst_pc1")),
                            q=_fmt(sc.get("semantic_quality")),
                            pc=_fmt(sc.get("pc1_composite")),
                        )
                    )
            else:
                print("  PC1-only: (none)")
            print()

        print("\n=== PC1-only matches across ALL documents ===\n")
        any_pc1_only = False
        for i, meta in sorted(res["index"].items(), key=lambda t: (t[1]["doc"], t[1]["from"], t[1]["to"])):
            pc1_only_list = res.get("pc1_only_matches", {}).get(i, [])
            if not pc1_only_list:
                continue
            any_pc1_only = True
            src_doc = meta["doc"]; src_lab = meta["from"]; dst_lab = meta["to"]
            print(f"{src_doc}: C{src_lab} \u2192 C{dst_lab} | edge_Q={_fmt(meta.get('edge_quality', ''))}")
            for m in pc1_only_list:
                sc = m.get("scores", {}) or {}
                print(
                    "  → {doc}: C{f}→C{t} | Δ={d} pc_axis={pa} src={s} dst={p} Q={q} pc1_comp={pc}".format(
                        doc=m["doc"], f=m["from"], t=m["to"],
                        d=_fmt(sc.get("delta_cos")),
                        pa=_fmt(sc.get("pc1_axis_value", sc.get("dst_pc1"))),
                        s=_fmt(sc.get("src_pc1")),
                        p=_fmt(sc.get("dst_pc1")),
                        q=_fmt(sc.get("semantic_quality")),
                        pc=_fmt(sc.get("pc1_composite")),
                    )
                )
            print()
        if not any_pc1_only:
            print("  (none found)\n")

        dst_thr = _get_param("pc1_only_threshold", 0.90)
        print(f"\n=== Dst PC1-only across ALL documents (dst_pc1 ≥ {dst_thr:.2f}) ===\n")
        any_dst_only = False
        aligned_map = res.get("aligned_matches", {}) or {}
        pc1_only_map = res.get("pc1_only_matches", {}) or {}
        index_map = res.get("index", {}) or {}

        for i, meta in sorted(index_map.items(), key=lambda t: (t[1]["doc"], t[1]["from"], t[1]["to"])):
            cand = []
            cand.extend(aligned_map.get(i, []))
            cand.extend(pc1_only_map.get(i, []))
            seen_j = set()
            dst_only_hits = []
            for m in cand:
                sc = (m.get("scores", {}) or {})
                dst_val = sc.get("dst_pc1", None)
                j_idx = m.get("j", None)
                try:
                    if dst_val is not None and float(dst_val) >= dst_thr:
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
                    "  → {doc}: C{f}→C{t} | dst={dst} Δ={d} src={s} Q={q}".format(
                        doc=m["doc"], f=m["from"], t=m["to"],
                        dst=_fmt(sc.get("dst_pc1")),
                        d=_fmt(sc.get("delta_cos")),
                        s=_fmt(sc.get("src_pc1")),
                        q=_fmt(sc.get("semantic_quality")),
                    )
                )
            print()
        if not any_dst_only:
            print("  (none found)\n")

    else:
        print("[output_analysis] Human-readable summaries skipped because print_summary=False.")

    if csv_mode_norm == "none":
        print("[output_analysis] Analyze CSV skipped because csv_mode='none'.")
        return

    selected_doc_for_csv = None
    if csv_mode_norm == "selected":
        if doc_id is None or str(doc_id).strip() == "":
            print("[output_analysis] Selected-document CSV requested, but no doc_id was provided; CSV skipped.")
            return
        selected_doc_for_csv = str(doc_id)
        print(f"[output_analysis] CSV mode: selected-document only for doc_id={selected_doc_for_csv!r}.")
    else:
        print("[output_analysis] CSV mode: full match export.")

    header = [
        "match_type",
        "src_doc", "src_from", "src_to",
        "tgt_doc", "tgt_from", "tgt_to",
        "delta_cos", "src_pc1", "dst_pc1", "pc1_axis_value", "pc1_axis_mode", "pc1_axis_label",
        "manifold_residual_doc_cosine", "manifold_residual_doc_available",
        "raw_sbert_doc_cosine", "raw_sbert_doc_available",
        "src_manifold_residual_doc_embedding_source", "tgt_manifold_residual_doc_embedding_source",
        "src_manifold_residual_doc_embedding_method", "tgt_manifold_residual_doc_embedding_method",
        "src_raw_sbert_doc_embedding_source", "tgt_raw_sbert_doc_embedding_source",
        "src_raw_sbert_doc_embedding_method", "tgt_raw_sbert_doc_embedding_method",
        "semantic_quality", "semantic_quality_hmean", "semantic_quality_min", "quality_axis_bin_0p01",
        "lexical_available", "acuity_computed", "acuity_compute_reason",
        "lexical_overlap_coefficient", "lexical_divergence", "alignment_core", "acuity_score", "acuity_score_count_cosine",
        "lexical_dst_overlap_coefficient", "lexical_dst_jaccard", "lexical_dst_dice",
        "lexical_dst_weighted_jaccard", "lexical_dst_count_cosine",
        "lexical_dst_tokens_a", "lexical_dst_tokens_b", "lexical_dst_unique_a", "lexical_dst_unique_b", "lexical_dst_shared_unique",
        "lexical_src_overlap_coefficient", "lexical_src_jaccard", "lexical_src_count_cosine",
        "lexical_edge_overlap_coefficient", "lexical_edge_jaccard", "lexical_edge_count_cosine",
        "src_edge_quality", "tgt_edge_quality",
        "src_edge_quality_min", "tgt_edge_quality_min",
        "src_from_quality", "src_to_quality",
        "tgt_from_quality", "tgt_to_quality",
        "src_from_lm_fluency", "src_to_lm_fluency",
        "tgt_from_lm_fluency", "tgt_to_lm_fluency",
        "src_from_lm_nll", "src_to_lm_nll",
        "tgt_from_lm_nll", "tgt_to_lm_nll",
        "src_quality_model", "tgt_quality_model",
        "pc1_composite",
        "joint_min", "joint_min_4d", "joint_min_srcdst", "joint_min_srcdst_4d",
        "delta_ok", "src_pc1_ok", "dst_pc1_ok", "pc1_axis_ok", "semantic_quality_ok",
        "detected_delta_thr", "detected_pc1_thr", "detected_quality_thr",
        "pc1_only_thr", "delta_max_for_pc1_only", "pc1_only_quality_thr",
        # Optional full-text columns can be re-enabled later:
        # "src_doc_src_cluster_text", "src_doc_dst_cluster_text",
        # "tgt_doc_src_cluster_text", "tgt_doc_dst_cluster_text",
    ]

    if csv_path is None:
        csv_path = _choose_csv_save_path(parent_window=globals().get("root"))
    if not csv_path:
        print("[output_analysis] CSV save cancelled (no file chosen).")
        return

    emitted = set()
    rows_written = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
        writer.writerow(header)

        def _emit_rows(kind: str, mmap: dict):
            nonlocal rows_written
            for i_src, lst in mmap.items():
                i_src = int(i_src)
                src_meta = res["index"].get(i_src, {})
                src_doc = src_meta.get("doc")
                src_from = src_meta.get("from")
                src_to = src_meta.get("to")
                for m in lst or []:
                    j_tgt = m.get("j")
                    if j_tgt is None:
                        continue
                    key = (i_src, int(j_tgt), kind)
                    if key in emitted:
                        continue
                    emitted.add(key)

                    tgt_doc = m.get("doc")
                    tgt_from = m.get("from")
                    tgt_to = m.get("to")

                    if selected_doc_for_csv is not None:
                        if str(src_doc) != selected_doc_for_csv and str(tgt_doc) != selected_doc_for_csv:
                            continue

                    sc = m.get("scores", {}) or {}
                    flags = m.get("flags", {}) or {}
                    level = m.get("level", {}) or {}
                    crit = m.get("criteria", {}) or {}

                    # compute fallback joint scores if absent
                    d = sc.get("delta_cos", None)
                    s = sc.get("src_pc1", None)
                    p = sc.get("dst_pc1", None)
                    q = sc.get("semantic_quality", None)
                    try:
                        joint_min = min([v for v in (d, s, p) if v is not None])
                    except Exception:
                        joint_min = ""
                    try:
                        joint_min_4d = min([v for v in (d, s, p, q) if v is not None])
                    except Exception:
                        joint_min_4d = ""

                    row = [
                        kind,
                        src_doc, src_from, src_to,
                        tgt_doc, tgt_from, tgt_to,
                        sc.get("delta_cos", ""),
                        sc.get("src_pc1", ""),
                        sc.get("dst_pc1", ""),
                        sc.get("pc1_axis_value", ""),
                        sc.get("pc1_axis_mode", ""),
                        sc.get("pc1_axis_label", ""),
                        sc.get("manifold_residual_doc_cosine", ""),
                        sc.get("manifold_residual_doc_available", ""),
                        sc.get("raw_sbert_doc_cosine", ""),
                        sc.get("raw_sbert_doc_available", ""),
                        sc.get("src_manifold_residual_doc_embedding_source", ""),
                        sc.get("tgt_manifold_residual_doc_embedding_source", ""),
                        sc.get("src_manifold_residual_doc_embedding_method", ""),
                        sc.get("tgt_manifold_residual_doc_embedding_method", ""),
                        sc.get("src_raw_sbert_doc_embedding_source", ""),
                        sc.get("tgt_raw_sbert_doc_embedding_source", ""),
                        sc.get("src_raw_sbert_doc_embedding_method", ""),
                        sc.get("tgt_raw_sbert_doc_embedding_method", ""),
                        sc.get("semantic_quality", ""),
                        sc.get("semantic_quality_hmean", sc.get("semantic_quality", "")),
                        sc.get("semantic_quality_min", ""),
                        sc.get("quality_axis_bin_0p01", ""),
                        sc.get("lexical_available", ""),
                        sc.get("acuity_computed", ""),
                        sc.get("acuity_compute_reason", ""),
                        sc.get("lexical_overlap_coefficient", ""),
                        sc.get("lexical_divergence", ""),
                        sc.get("alignment_core", ""),
                        sc.get("acuity_score", ""),
                        sc.get("acuity_score_count_cosine", ""),
                        sc.get("lexical_dst_overlap_coefficient", ""),
                        sc.get("lexical_dst_jaccard", ""),
                        sc.get("lexical_dst_dice", ""),
                        sc.get("lexical_dst_weighted_jaccard", ""),
                        sc.get("lexical_dst_count_cosine", ""),
                        sc.get("lexical_dst_tokens_a", ""),
                        sc.get("lexical_dst_tokens_b", ""),
                        sc.get("lexical_dst_unique_a", ""),
                        sc.get("lexical_dst_unique_b", ""),
                        sc.get("lexical_dst_shared_unique", ""),
                        sc.get("lexical_src_overlap_coefficient", ""),
                        sc.get("lexical_src_jaccard", ""),
                        sc.get("lexical_src_count_cosine", ""),
                        sc.get("lexical_edge_overlap_coefficient", ""),
                        sc.get("lexical_edge_jaccard", ""),
                        sc.get("lexical_edge_count_cosine", ""),
                        sc.get("src_edge_quality", ""),
                        sc.get("tgt_edge_quality", ""),
                        sc.get("src_edge_quality_min", ""),
                        sc.get("tgt_edge_quality_min", ""),
                        sc.get("src_from_quality", ""),
                        sc.get("src_to_quality", ""),
                        sc.get("tgt_from_quality", ""),
                        sc.get("tgt_to_quality", ""),
                        sc.get("src_from_lm_fluency", ""),
                        sc.get("src_to_lm_fluency", ""),
                        sc.get("tgt_from_lm_fluency", ""),
                        sc.get("tgt_to_lm_fluency", ""),
                        sc.get("src_from_lm_nll", ""),
                        sc.get("src_to_lm_nll", ""),
                        sc.get("tgt_from_lm_nll", ""),
                        sc.get("tgt_to_lm_nll", ""),
                        sc.get("src_quality_model", ""),
                        sc.get("tgt_quality_model", ""),
                        sc.get("pc1_composite", ""),
                        sc.get("joint_min", joint_min),
                        sc.get("joint_min_4d", joint_min_4d),
                        sc.get("joint_min_srcdst", ""),
                        sc.get("joint_min_srcdst_4d", ""),
                        flags.get("delta_ok", ""),
                        flags.get("src_pc1_ok", ""),
                        flags.get("dst_pc1_ok", ""),
                        flags.get("pc1_axis_ok", ""),
                        flags.get("semantic_quality_ok", ""),
                        level.get("delta", ""),
                        level.get("pc1", ""),
                        level.get("quality", ""),
                        crit.get("pc1_only_threshold", ""),
                        crit.get("delta_max", ""),
                        crit.get("quality_threshold", ""),
                    ]
                    writer.writerow(row)
                    rows_written += 1

        _emit_rows("aligned", res.get("aligned_matches", {}) or {})
        _emit_rows("pc1_only", res.get("pc1_only_matches", {}) or {})

    print(f"[output_analysis] CSV written to: {csv_path} ({rows_written:,} data rows; mode={csv_mode_norm})")


# -----------------------------------------------------------------------------
# Function: output_acuity_candidates_csv
# Summary:
#   Writes a compact candidate table sorted by semantic/lexical acuity.  It is
#   designed to surface individual high-Δ, high-PC1, high-Q, lexically divergent
#   morphism matches that can be hidden inside averaged threshold bins.
# -----------------------------------------------------------------------------
def output_acuity_candidates_csv(
    res: dict,
    csv_path: str | None = None,
    doc_id: str | None = None,
    csv_mode: str = "selected",       # "selected" | "full" | "none"
    top_n: int = 500,
    step: float = 0.01,
    peak_per_bin: bool = True,
):
    import csv, os
    import numpy as np
    from tkinter import filedialog
    import tkinter as tk

    def _norm_mode(mode):
        raw = str(mode or "selected").strip().lower().replace("-", "_").replace(" ", "_")
        if raw in ("all", "everything", "complete"):
            return "full"
        if raw in ("no", "none", "no_csv", "off", "false", "0", "skip"):
            return "none"
        if raw in ("selected_doc", "selected_document", "doc", "doc_id", "anchor", "involving", "focus"):
            return "selected"
        return raw if raw in ("selected", "full", "none") else "selected"

    csv_mode = _norm_mode(csv_mode)
    if csv_mode == "none":
        print("[acuity] Top-candidates CSV skipped because csv_mode='none'.")
        return
    if not res or "index" not in res or not res.get("index"):
        print("[acuity] No Analyze result rows available for top-candidates CSV.")
        return
    if csv_mode == "selected" and not doc_id:
        print("[acuity] Selected top-candidates CSV requested, but no doc_id was supplied; skipped.")
        return

    def _sf(x, default=0.0):
        try:
            if x is None or x == "":
                return default
            v = float(x)
            if not np.isfinite(v):
                return default
            return v
        except Exception:
            return default

    def _bin(v):
        v = max(0.0, min(1.0, _sf(v, 0.0)))
        return round(np.floor((v + 1e-12) / max(1e-9, float(step))) * float(step), 6)

    rows = []
    seen = set()
    for kind, mmap in (("aligned", res.get("aligned_matches", {}) or {}), ("pc1_only", res.get("pc1_only_matches", {}) or {})):
        for i_src, lst in mmap.items():
            i_src = int(i_src)
            src_meta = res.get("index", {}).get(i_src, {}) or {}
            src_doc = src_meta.get("doc")
            for m in (lst or []):
                j_tgt = m.get("j")
                if j_tgt is None:
                    continue
                key = (kind, int(i_src), int(j_tgt))
                if key in seen:
                    continue
                seen.add(key)
                tgt_doc = m.get("doc")
                if csv_mode == "selected" and str(src_doc) != str(doc_id) and str(tgt_doc) != str(doc_id):
                    continue
                sc = m.get("scores", {}) or {}
                # Top-candidate output is intended to surface matches for which
                # lexical/acuteness diagnostics were actually computed.  This
                # avoids filling the top-N table with zero-acuity PC1-only rows
                # when compute_acuity_for="aligned_only".
                _acuity_flag = str(sc.get("acuity_computed", "")).strip().lower() in ("true", "1", "yes", "y", "on")
                if (not _acuity_flag) and _sf(sc.get("acuity_score"), 0.0) <= 0.0:
                    continue
                d = _sf(sc.get("delta_cos"), 0.0)
                pc = _sf(sc.get("pc1_axis_value", sc.get("dst_pc1", 0.0)), 0.0)
                q = _sf(sc.get("semantic_quality"), 0.0)
                lex = _sf(sc.get("lexical_overlap_coefficient"), 0.0)
                div = _sf(sc.get("lexical_divergence"), 0.0)
                core = _sf(sc.get("alignment_core"), _quality_hmean([d, pc, q]))
                acuity = _sf(sc.get("acuity_score"), core * div)
                row = {
                    "match_type": kind,
                    "acuity_computed": sc.get("acuity_computed", ""),
                    "acuity_compute_reason": sc.get("acuity_compute_reason", ""),
                    "src_entry_index": i_src,
                    "tgt_entry_index": int(j_tgt),
                    "src_doc": src_doc,
                    "src_from": src_meta.get("from"),
                    "src_to": src_meta.get("to"),
                    "tgt_doc": tgt_doc,
                    "tgt_from": m.get("from"),
                    "tgt_to": m.get("to"),
                    "delta_cos": d,
                    "pc1_axis_value": pc,
                    "pc1_axis_mode": sc.get("pc1_axis_mode", ""),
                    "pc1_axis_label": sc.get("pc1_axis_label", ""),
                    "manifold_residual_doc_cosine": _sf(sc.get("manifold_residual_doc_cosine"), ""),
                    "manifold_residual_doc_available": sc.get("manifold_residual_doc_available", ""),
                    "raw_sbert_doc_cosine": _sf(sc.get("raw_sbert_doc_cosine"), ""),
                    "raw_sbert_doc_available": sc.get("raw_sbert_doc_available", ""),
                    "src_manifold_residual_doc_embedding_source": sc.get("src_manifold_residual_doc_embedding_source", ""),
                    "tgt_manifold_residual_doc_embedding_source": sc.get("tgt_manifold_residual_doc_embedding_source", ""),
                    "src_raw_sbert_doc_embedding_source": sc.get("src_raw_sbert_doc_embedding_source", ""),
                    "tgt_raw_sbert_doc_embedding_source": sc.get("tgt_raw_sbert_doc_embedding_source", ""),
                    "semantic_quality": q,
                    "lexical_overlap_coefficient": lex,
                    "lexical_divergence": div,
                    "alignment_core": core,
                    "acuity_score": acuity,
                    "acuity_score_count_cosine": _sf(sc.get("acuity_score_count_cosine"), 0.0),
                    "lexical_dst_count_cosine": _sf(sc.get("lexical_dst_count_cosine"), 0.0),
                    "lexical_dst_jaccard": _sf(sc.get("lexical_dst_jaccard"), 0.0),
                    "lexical_dst_weighted_jaccard": _sf(sc.get("lexical_dst_weighted_jaccard"), 0.0),
                    "lexical_src_overlap_coefficient": _sf(sc.get("lexical_src_overlap_coefficient"), 0.0),
                    "lexical_edge_overlap_coefficient": _sf(sc.get("lexical_edge_overlap_coefficient"), 0.0),
                    "lexical_available": sc.get("lexical_available", ""),
                    "joint_min_4d": _sf(sc.get("joint_min_4d"), min(d, pc, q)),
                    "delta_bin": _bin(d),
                    "pc1_bin": _bin(pc),
                    "quality_bin": _bin(q),
                    "detected_delta_thr": (m.get("level", {}) or {}).get("delta", ""),
                    "detected_pc1_thr": (m.get("level", {}) or {}).get("pc1", ""),
                    "detected_quality_thr": (m.get("level", {}) or {}).get("quality", ""),
                }
                rows.append(row)

    if not rows:
        print("[acuity] No candidate rows survived the selected/full filter.")
        return

    # Bin summaries let a single acute candidate remain visible even inside a
    # large count bin; the CSV reports each candidate's local bin context.
    by_bin = {}
    for r in rows:
        key = (r["delta_bin"], r["pc1_bin"], r["quality_bin"])
        by_bin.setdefault(key, []).append(r)
    peak_rows = set()
    for key, lst in by_bin.items():
        lst.sort(key=lambda rr: rr.get("acuity_score", 0.0), reverse=True)
        peak_rows.add(id(lst[0]))
        count = len(lst)
        mean_lex = sum(rr.get("lexical_overlap_coefficient", 0.0) for rr in lst) / max(1, count)
        mean_div = sum(rr.get("lexical_divergence", 0.0) for rr in lst) / max(1, count)
        max_acuity = lst[0].get("acuity_score", 0.0)
        for rr in lst:
            rr["bin_count"] = int(count)
            rr["bin_mean_lexical_overlap"] = float(mean_lex)
            rr["bin_mean_lexical_divergence"] = float(mean_div)
            rr["bin_max_acuity"] = float(max_acuity)
            rr["is_bin_peak_acuity"] = bool(id(rr) in peak_rows)
            rr["density_adjusted_acuity_log"] = float(rr.get("acuity_score", 0.0) / np.log2(2.0 + count))
            rr["density_adjusted_acuity_sqrt"] = float(rr.get("acuity_score", 0.0) / np.sqrt(max(1.0, count)))

    if peak_per_bin:
        rows_out = [r for r in rows if r.get("is_bin_peak_acuity")]
    else:
        rows_out = rows
    rows_out.sort(key=lambda rr: rr.get("acuity_score", 0.0), reverse=True)
    rows_out = rows_out[:max(1, int(top_n))]
    for rank, rr in enumerate(rows_out, 1):
        rr["acuity_rank"] = rank

    if csv_path is None:
        try:
            parent = globals().get("root") or tk._get_default_root()
        except Exception:
            parent = None
        csv_path = filedialog.asksaveasfilename(
            parent=parent,
            title="Save top acuity candidates CSV",
            defaultextension=".txt",
            initialfile="top_acuity_candidates.txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
        ) or None
    if not csv_path:
        print("[acuity] CSV save cancelled.")
        return

    header = [
        "acuity_rank", "match_type", "acuity_computed", "acuity_compute_reason",
        "src_doc", "src_from", "src_to", "tgt_doc", "tgt_from", "tgt_to",
        "delta_cos", "pc1_axis_value", "pc1_axis_mode", "pc1_axis_label",
        "manifold_residual_doc_cosine", "manifold_residual_doc_available",
        "raw_sbert_doc_cosine", "raw_sbert_doc_available",
        "src_manifold_residual_doc_embedding_source", "tgt_manifold_residual_doc_embedding_source",
        "src_raw_sbert_doc_embedding_source", "tgt_raw_sbert_doc_embedding_source",
        "semantic_quality",
        "lexical_overlap_coefficient", "lexical_divergence", "alignment_core", "acuity_score", "acuity_score_count_cosine",
        "lexical_dst_count_cosine", "lexical_dst_jaccard", "lexical_dst_weighted_jaccard",
        "lexical_src_overlap_coefficient", "lexical_edge_overlap_coefficient", "lexical_available",
        "joint_min_4d", "delta_bin", "pc1_bin", "quality_bin", "bin_count", "bin_mean_lexical_overlap",
        "bin_mean_lexical_divergence", "bin_max_acuity", "is_bin_peak_acuity",
        "density_adjusted_acuity_log", "density_adjusted_acuity_sqrt",
        "detected_delta_thr", "detected_pc1_thr", "detected_quality_thr",
        "src_entry_index", "tgt_entry_index",
    ]
    abspath = os.path.abspath(csv_path)
    with open(abspath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for rr in rows_out:
            writer.writerow(rr)
    print(f"[acuity] Top acuity candidates CSV written to: {abspath} ({len(rows_out):,} rows; mode={csv_mode}; peak_per_bin={peak_per_bin})")


# -----------------------------------------------------------------------------
# Function: plot_morphism_match_field_3d
# Summary:
#   Builds an interactive 3D view of the *incremental* match counts as a
#   function of thresholds (X=Δ cosine, Y=PC1 concordance). Supports three
#   modes: AND (Δ & src & dst), Δ&dst only, and src&dst only (Δ ignored and
#   drawn at X=1.00). Uses a persistent colorbar and a convex‑hull edge curve on the XY
#   projection to summarize the frontier of non‑zero cells.
# Effect:
#   Visual “phase diagram” for threshold selection—makes it easy to spot
#   stable regimes where many new matches appear (or don’t) as criteria relax.
# -----------------------------------------------------------------------------


def _morphism_comparison_payload_from_result(res: dict):
    """Return a compact morphism_comparison payload from either direct payload or Analyze result."""
    if not isinstance(res, dict):
        return None
    if res.get("kind") == "morphism_comparison":
        return res
    payload = res.get("morphism_comparison")
    if isinstance(payload, dict) and payload.get("kind") == "morphism_comparison":
        return payload
    return None



def _normalize_compact_diagnostics_mode(mode: str | None) -> str:
    """Normalize compact morphism-comparison enrichment mode names."""
    raw = str(mode or "plot_cache").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "plot_cache",
        "default": "plot_cache",
        "cache": "plot_cache",
        "plot": "plot_cache",
        "plotcache": "plot_cache",
        "plot_cache": "plot_cache",
        "graph": "plot_cache",
        "graph_cache": "plot_cache",
        "top": "top_candidates",
        "top_candidate": "top_candidates",
        "top_candidates": "top_candidates",
        "candidate": "top_candidates",
        "candidates": "top_candidates",
        "all": "all_matches",
        "all_match": "all_matches",
        "all_matches": "all_matches",
        "full": "all_matches",
        "full_match": "all_matches",
        "full_matches": "all_matches",
        "none": "none",
        "off": "none",
        "no": "none",
        "false": "none",
        "0": "none",
        "skip": "none",
    }
    return aliases.get(raw, "plot_cache")


def _compact_counter_signature(counter_obj):
    """Return a small reusable lexical signature for a Counter-like object."""
    cnt = Counter(counter_obj or {})
    keys = set(cnt)
    total = int(sum(cnt.values()))
    norm = math.sqrt(sum(float(v) * float(v) for v in cnt.values()))
    return {
        "counter": cnt,
        "keys": keys,
        "total": total,
        "unique": int(len(keys)),
        "norm": float(norm),
    }


def _compact_lexical_metrics_from_signatures(sig_a, sig_b, detailed: bool = False) -> dict:
    """Fast lexical-overlap metrics for precomputed counter signatures."""
    if not sig_a or not sig_b:
        return {
            "lexical_available": False,
            "tokens_a": 0, "tokens_b": 0, "unique_a": 0, "unique_b": 0, "shared_unique": 0,
            "overlap_coefficient": 0.0, "jaccard": 0.0, "dice": 0.0,
            "weighted_jaccard": 0.0, "count_cosine": 0.0,
            "shared_token_mass_a": 0.0, "shared_token_mass_b": 0.0,
        }
    ca = sig_a.get("counter") or Counter()
    cb = sig_b.get("counter") or Counter()
    sa = sig_a.get("keys") or set()
    sb = sig_b.get("keys") or set()
    total_a = int(sig_a.get("total", 0) or 0)
    total_b = int(sig_b.get("total", 0) or 0)
    unique_a = int(sig_a.get("unique", len(sa)) or 0)
    unique_b = int(sig_b.get("unique", len(sb)) or 0)
    if total_a <= 0 or total_b <= 0 or unique_a <= 0 or unique_b <= 0:
        return {
            "lexical_available": False,
            "tokens_a": total_a, "tokens_b": total_b,
            "unique_a": unique_a, "unique_b": unique_b, "shared_unique": 0,
            "overlap_coefficient": 0.0, "jaccard": 0.0, "dice": 0.0,
            "weighted_jaccard": 0.0, "count_cosine": 0.0,
            "shared_token_mass_a": 0.0, "shared_token_mass_b": 0.0,
        }

    inter = sa & sb
    shared_unique = int(len(inter))
    overlap = float(shared_unique / max(1, min(unique_a, unique_b)))

    # Dot product over the smaller vocabulary side.
    if unique_a <= unique_b:
        dot = sum(float(v) * float(cb.get(t, 0)) for t, v in ca.items())
    else:
        dot = sum(float(v) * float(ca.get(t, 0)) for t, v in cb.items())
    norm_a = float(sig_a.get("norm", 0.0) or 0.0)
    norm_b = float(sig_b.get("norm", 0.0) or 0.0)
    count_cos = float(dot / max(1e-12, norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0

    out = {
        "lexical_available": True,
        "tokens_a": total_a,
        "tokens_b": total_b,
        "unique_a": unique_a,
        "unique_b": unique_b,
        "shared_unique": shared_unique,
        "overlap_coefficient": overlap,
        "count_cosine": count_cos,
    }
    if detailed:
        union = sa | sb
        jacc = float(shared_unique / max(1, len(union))) if union else 0.0
        dice = float(2 * shared_unique / max(1, unique_a + unique_b)) if (unique_a or unique_b) else 0.0
        min_sum = 0
        max_sum = 0
        for t in union:
            av = int(ca.get(t, 0)); bv = int(cb.get(t, 0))
            min_sum += av if av <= bv else bv
            max_sum += av if av >= bv else bv
        mass_a = sum(int(ca.get(t, 0)) for t in inter)
        mass_b = sum(int(cb.get(t, 0)) for t in inter)
        out.update({
            "jaccard": jacc,
            "dice": dice,
            "weighted_jaccard": float(min_sum / max(1, max_sum)),
            "shared_token_mass_a": float(mass_a / max(1, total_a)),
            "shared_token_mass_b": float(mass_b / max(1, total_b)),
        })
    else:
        out.update({
            "jaccard": 0.0,
            "dice": 0.0,
            "weighted_jaccard": 0.0,
            "shared_token_mass_a": 0.0,
            "shared_token_mass_b": 0.0,
        })
    return out


def _compact_cluster_counter_signatures(document_cluster_data: dict | None, segments_by_doc: dict | None) -> dict:
    """Precompute lexical counter signatures keyed by (doc_id_as_str, cluster_label_int)."""
    out: dict[tuple[str, int], dict] = {}
    if not isinstance(document_cluster_data, dict) or not isinstance(segments_by_doc, dict):
        return out

    # Let string ids resolve to original keys, but store signatures under str(doc_id)
    # because compact edge_index doc_ids are string-normalized.
    for doc_id, data in document_cluster_data.items():
        try:
            segs = segments_by_doc.get(doc_id)
            if segs is None:
                segs = segments_by_doc.get(str(doc_id))
            if segs is None:
                continue
            segs = list(segs)
            labels = data[2]
            labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        except Exception:
            continue
        by_lab = defaultdict(list)
        for ii, lab in enumerate(labels):
            if ii >= len(segs):
                continue
            try:
                lab_i = int(lab)
            except Exception:
                continue
            by_lab[lab_i].append(str(segs[ii]))
        for lab_i, texts in by_lab.items():
            out[(str(doc_id), int(lab_i))] = _compact_counter_signature(lexical_counter_from_texts(texts))
    return out


def _compact_document_embedding_table(document_cluster_data: dict | None, doc_ids: list[str], preferred_key: str = "manifold_residual_document_embedding") -> dict:
    """Build a compact document-embedding table aligned to edge_index['doc_ids']."""
    doc_ids = [str(d) for d in (doc_ids or [])]
    key_by_str = {str(k): k for k in (document_cluster_data or {}).keys()} if isinstance(document_cluster_data, dict) else {}
    vectors = []
    available = []
    source = []
    method = []
    dims = []
    common_dim = None
    raw_vecs = []
    for d in doc_ids:
        vec = None
        meta = {"available": False, "source": "missing", "method": "", "dim": 0}
        try:
            orig_key = key_by_str.get(str(d), d)
            if isinstance(document_cluster_data, dict) and orig_key in document_cluster_data:
                vec, meta = _document_embedding_from_cdm(document_cluster_data[orig_key], preferred_key=preferred_key)
        except Exception as ex:
            meta = {"available": False, "source": "error", "method": "", "dim": 0, "error": str(ex)}
            vec = None
        if vec is not None:
            vv = np.asarray(vec, dtype=np.float32).reshape(-1)
            if common_dim is None:
                common_dim = int(vv.shape[0])
            if int(vv.shape[0]) != int(common_dim):
                vec = None
                meta = dict(meta or {})
                meta.update({"available": False, "source": "dimension_mismatch", "dim": int(vv.shape[0])})
        raw_vecs.append(None if vec is None else np.asarray(vec, dtype=np.float32).reshape(-1))
        available.append(bool(vec is not None))
        source.append(str((meta or {}).get("source", "")))
        method.append(str((meta or {}).get("method", "")))
        dims.append(int((meta or {}).get("dim", 0) or (0 if vec is None else len(vec))))
    dim = int(common_dim or max([0] + [int(d) for d in dims]))
    if dim > 0:
        arr = np.zeros((len(doc_ids), dim), dtype=np.float32)
        for i, v in enumerate(raw_vecs):
            if v is not None and v.shape[0] == dim:
                arr[i] = v.astype(np.float32, copy=False)
    else:
        arr = np.zeros((len(doc_ids), 0), dtype=np.float32)
    return {
        "doc_ids": np.asarray(doc_ids, dtype=object),
        "vectors": arr,
        "available": np.asarray(available, dtype=np.uint8),
        "source": np.asarray(source, dtype=object),
        "method": np.asarray(method, dtype=object),
        "dim": np.asarray(dims, dtype=np.int32),
        "preferred_key": str(preferred_key),
    }


def _compact_top_candidate_dtype():
    return np.dtype([
        ("match_row", "i8"),
        ("match_type", "u1"),
        ("src_edge", "i4"),
        ("tgt_edge", "i4"),
        ("src_doc_code", "i4"),
        ("tgt_doc_code", "i4"),
        ("src_from_label", "i4"),
        ("src_to_label", "i4"),
        ("tgt_from_label", "i4"),
        ("tgt_to_label", "i4"),
        ("delta_cos", "f4"),
        ("pc1_axis_value", "f4"),
        ("semantic_quality", "f4"),
        ("lexical_overlap_coefficient", "f4"),
        ("lexical_divergence", "f4"),
        ("alignment_core", "f4"),
        ("acuity_score", "f4"),
        ("manifold_residual_doc_cosine", "f4"),
        ("manifold_residual_doc_available", "u1"),
        ("raw_sbert_doc_cosine", "f4"),
        ("raw_sbert_doc_available", "u1"),
    ])


def enrich_morphism_comparison_diagnostics(
    res_or_payload: dict,
    document_cluster_data: dict | None = None,
    segments_by_doc: dict | None = None,
    diagnostics_mode: str = "plot_cache",
    step: float = 0.01,
    top_candidates: int = 5000,
    high_acuity_threshold: float = 0.50,
    verbose: bool = True,
) -> dict:
    """
    Add compact interpretive diagnostics to a morphism_comparison payload.

    The enrichment is array-based and does not build the legacy nested Analyze
    dictionaries.  It stores per-retained-match primary diagnostics in
    ``match_diagnostics`` and binned graph data in ``plot_cache``.

    diagnostics_mode:
        none           -> leave payload unchanged except metadata
        plot_cache     -> compute/store primary arrays + plot_cache + top candidates
        top_candidates -> same as plot_cache, emphasizing top-candidate retention
        all_matches    -> additionally store detailed source/destination/edge
                          lexical metric arrays for every retained match
    """
    mode = _normalize_compact_diagnostics_mode(diagnostics_mode)
    payload = _morphism_comparison_payload_from_result(res_or_payload)
    if payload is None:
        return res_or_payload
    if mode == "none":
        payload.setdefault("diagnostics", {})["mode"] = "none"
        payload.setdefault("summary", {})["compact_diagnostics_mode"] = "none"
        return res_or_payload

    records = payload.get("matches")
    if records is None:
        payload.setdefault("diagnostics", {})["error"] = "missing matches array"
        return res_or_payload
    records = np.asarray(records)
    n = int(records.size)
    if n == 0:
        payload["match_diagnostics"] = {
            "mode": mode,
            "row_aligned_to_matches": True,
            "lexical_available": np.zeros((0,), dtype=np.uint8),
            "manifold_residual_doc_available": np.zeros((0,), dtype=np.uint8),
            "raw_sbert_doc_available": np.zeros((0,), dtype=np.uint8),
        }
        payload["plot_cache"] = {"step": float(step), "empty": True}
        payload.setdefault("summary", {})["compact_diagnostics_mode"] = mode
        return res_or_payload

    edge_index = payload.get("edge_index") or {}
    doc_ids = [str(d) for d in edge_index.get("doc_ids", [])]
    doc_code = np.asarray(edge_index.get("doc_code"), dtype=np.int32)
    src_label = np.asarray(edge_index.get("src_label"), dtype=np.int32)
    dst_label = np.asarray(edge_index.get("dst_label"), dtype=np.int32)
    num_edges = int(doc_code.shape[0])

    if not (records.dtype.names and "src_edge" in records.dtype.names and "tgt_edge" in records.dtype.names):
        raise ValueError("morphism_comparison.matches must be a structured array with src_edge/tgt_edge fields")

    src_edges = np.asarray(records["src_edge"], dtype=np.int64)
    tgt_edges = np.asarray(records["tgt_edge"], dtype=np.int64)
    valid_edges = (src_edges >= 0) & (tgt_edges >= 0) & (src_edges < num_edges) & (tgt_edges < num_edges)

    step = max(0.001, float(step))
    n_bins = int(round(1.0 / step)) + 1
    thr = np.round(np.linspace(1.0, 0.0, n_bins), 6).astype(np.float32)

    def _field(name, default=0.0, clip01=True):
        if records.dtype.names and name in records.dtype.names:
            arr = np.asarray(records[name], dtype=np.float32)
        else:
            arr = np.full(n, float(default), dtype=np.float32)
        arr = np.nan_to_num(arr, nan=float(default), posinf=1.0 if clip01 else np.nan, neginf=0.0 if clip01 else np.nan)
        if clip01:
            arr = np.clip(arr, 0.0, 1.0)
        return arr.astype(np.float32, copy=False)

    delta_vals = _field("delta_cos", 0.0)
    pc_vals = _field("pc1_axis_value", 1.0)
    q_vals = _field("semantic_quality", 1.0)
    eps = np.float32(1e-6)
    alignment_core = (3.0 / ((1.0 / np.maximum(delta_vals, eps)) + (1.0 / np.maximum(pc_vals, eps)) + (1.0 / np.maximum(q_vals, eps)))).astype(np.float32)

    # Document embedding cosines are vectorized by document code when payloads
    # are available.  Residual and raw SBERT baselines are kept separate.
    manifold_doc_table = _compact_document_embedding_table(
        document_cluster_data,
        doc_ids,
        preferred_key="manifold_residual_document_embedding",
    )
    raw_sbert_doc_table = _compact_document_embedding_table(
        document_cluster_data,
        doc_ids,
        preferred_key="raw_sbert_document_embedding",
    )
    payload["document_embeddings"] = {
        "kind": "document_embedding_tables",
        "version": 2,
        "manifold_residual": manifold_doc_table,
        "raw_sbert": raw_sbert_doc_table,
    }

    def _cosines_from_doc_table(doc_table: dict, label: str) -> tuple[np.ndarray, np.ndarray]:
        cos = np.full(n, np.nan, dtype=np.float32)
        avail_out = np.zeros(n, dtype=np.uint8)
        try:
            vectors = np.asarray(doc_table.get("vectors"), dtype=np.float32)
            available = np.asarray(doc_table.get("available"), dtype=np.uint8)
            src_doc_codes = doc_code[src_edges]
            tgt_doc_codes = doc_code[tgt_edges]
            ok = valid_edges & (src_doc_codes >= 0) & (tgt_doc_codes >= 0) & (src_doc_codes < available.shape[0]) & (tgt_doc_codes < available.shape[0])
            ok &= (available[src_doc_codes] > 0) & (available[tgt_doc_codes] > 0) if available.size else False
            if vectors.ndim == 2 and vectors.shape[1] > 0 and np.any(ok):
                cos[ok] = np.einsum("ij,ij->i", vectors[src_doc_codes[ok]], vectors[tgt_doc_codes[ok]]).astype(np.float32)
                cos = np.clip(cos, -1.0, 1.0)
                avail_out[ok] = 1
        except Exception as ex:
            if verbose:
                print(f"[analyze-enrich] {label} document cosine unavailable: {ex}", flush=True)
        return cos, avail_out

    manifold_residual_doc_cos, manifold_residual_doc_avail = _cosines_from_doc_table(manifold_doc_table, "manifold_residual")
    raw_sbert_doc_cos, raw_sbert_doc_avail = _cosines_from_doc_table(raw_sbert_doc_table, "raw_sbert")

    # Lexical signatures are small and reusable.  Missing text simply yields no
    # lexical availability, but the count/Q/doc-cos views still work.
    if verbose:
        print(f"[analyze-enrich] mode={mode}; retained_matches={n:,}; building lexical signatures...", flush=True)
    cluster_sigs = _compact_cluster_counter_signatures(document_cluster_data, segments_by_doc)
    empty_sig = _compact_counter_signature(Counter())

    src_dst_sigs = [empty_sig] * num_edges
    tgt_dst_sigs = [empty_sig] * num_edges
    src_src_sigs = [empty_sig] * num_edges
    tgt_src_sigs = [empty_sig] * num_edges
    edge_sigs = None
    detailed = mode == "all_matches"

    # Per-edge signatures used by the retained-match loop.
    src_sig_by_edge = [empty_sig] * num_edges
    dst_sig_by_edge = [empty_sig] * num_edges
    if num_edges > 0:
        for ei in range(num_edges):
            dcode = int(doc_code[ei]) if ei < doc_code.shape[0] else -1
            doc_s = doc_ids[dcode] if 0 <= dcode < len(doc_ids) else ""
            src_sig_by_edge[ei] = cluster_sigs.get((doc_s, int(src_label[ei])), empty_sig)
            dst_sig_by_edge[ei] = cluster_sigs.get((doc_s, int(dst_label[ei])), empty_sig)
        if detailed:
            edge_sigs = []
            for ei in range(num_edges):
                edge_sigs.append(_compact_counter_signature(_counter_add(src_sig_by_edge[ei].get("counter"), dst_sig_by_edge[ei].get("counter"))))

    lexical_available = np.zeros(n, dtype=np.uint8)
    lexical_overlap = np.zeros(n, dtype=np.float32)
    lexical_divergence = np.zeros(n, dtype=np.float32)
    lexical_dst_count_cosine = np.zeros(n, dtype=np.float32)
    acuity_score_count_cosine = np.zeros(n, dtype=np.float32)

    if detailed:
        dst_jaccard = np.zeros(n, dtype=np.float32)
        dst_dice = np.zeros(n, dtype=np.float32)
        dst_weighted_jaccard = np.zeros(n, dtype=np.float32)
        dst_tokens_a = np.zeros(n, dtype=np.int32)
        dst_tokens_b = np.zeros(n, dtype=np.int32)
        dst_unique_a = np.zeros(n, dtype=np.int32)
        dst_unique_b = np.zeros(n, dtype=np.int32)
        dst_shared_unique = np.zeros(n, dtype=np.int32)
        src_overlap = np.zeros(n, dtype=np.float32)
        src_jaccard = np.zeros(n, dtype=np.float32)
        src_count_cosine = np.zeros(n, dtype=np.float32)
        edge_overlap = np.zeros(n, dtype=np.float32)
        edge_jaccard = np.zeros(n, dtype=np.float32)
        edge_count_cosine = np.zeros(n, dtype=np.float32)

    progress_every = max(250000, n // 8 if n >= 1000000 else n + 1)
    for rr in range(n):
        if not valid_edges[rr]:
            continue
        se = int(src_edges[rr]); te = int(tgt_edges[rr])
        dst_m = _compact_lexical_metrics_from_signatures(dst_sig_by_edge[se], dst_sig_by_edge[te], detailed=detailed)
        if dst_m.get("lexical_available", False):
            lexical_available[rr] = 1
            ov = float(dst_m.get("overlap_coefficient", 0.0))
            lexical_overlap[rr] = np.float32(max(0.0, min(1.0, ov)))
            lexical_divergence[rr] = np.float32(1.0 - max(0.0, min(1.0, ov)))
            cc = float(dst_m.get("count_cosine", 0.0))
            lexical_dst_count_cosine[rr] = np.float32(max(0.0, min(1.0, cc)))
        if detailed:
            dst_jaccard[rr] = np.float32(dst_m.get("jaccard", 0.0))
            dst_dice[rr] = np.float32(dst_m.get("dice", 0.0))
            dst_weighted_jaccard[rr] = np.float32(dst_m.get("weighted_jaccard", 0.0))
            dst_tokens_a[rr] = int(dst_m.get("tokens_a", 0))
            dst_tokens_b[rr] = int(dst_m.get("tokens_b", 0))
            dst_unique_a[rr] = int(dst_m.get("unique_a", 0))
            dst_unique_b[rr] = int(dst_m.get("unique_b", 0))
            dst_shared_unique[rr] = int(dst_m.get("shared_unique", 0))
            src_m = _compact_lexical_metrics_from_signatures(src_sig_by_edge[se], src_sig_by_edge[te], detailed=True)
            edge_m = _compact_lexical_metrics_from_signatures(edge_sigs[se], edge_sigs[te], detailed=True) if edge_sigs is not None else {}
            src_overlap[rr] = np.float32(src_m.get("overlap_coefficient", 0.0))
            src_jaccard[rr] = np.float32(src_m.get("jaccard", 0.0))
            src_count_cosine[rr] = np.float32(src_m.get("count_cosine", 0.0))
            edge_overlap[rr] = np.float32(edge_m.get("overlap_coefficient", 0.0))
            edge_jaccard[rr] = np.float32(edge_m.get("jaccard", 0.0))
            edge_count_cosine[rr] = np.float32(edge_m.get("count_cosine", 0.0))
        if verbose and (rr + 1) % progress_every == 0:
            print(f"[analyze-enrich] lexical diagnostics {rr + 1:,}/{n:,} retained matches", flush=True)

    acuity_score = (alignment_core * lexical_divergence).astype(np.float32)
    acuity_score_count_cosine = (alignment_core * (1.0 - lexical_dst_count_cosine)).astype(np.float32)
    acuity_score[lexical_available == 0] = 0.0
    acuity_score_count_cosine[lexical_available == 0] = 0.0

    bi = np.rint((1.0 - np.clip(delta_vals, 0.0, 1.0)) / step).astype(np.int32)
    bj = np.rint((1.0 - np.clip(pc_vals, 0.0, 1.0)) / step).astype(np.int32)
    bk = np.rint((1.0 - np.clip(q_vals, 0.0, 1.0)) / step).astype(np.int32)
    bi = np.clip(bi, 0, n_bins - 1); bj = np.clip(bj, 0, n_bins - 1); bk = np.clip(bk, 0, n_bins - 1)

    count_grid = np.zeros((n_bins, n_bins, n_bins), dtype=np.uint32)
    np.add.at(count_grid, (bi, bj, bk), 1)

    lex_sum = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    lex_n = np.zeros((n_bins, n_bins, n_bins), dtype=np.uint32)
    div_sum = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    acuity_max = np.zeros((n_bins, n_bins, n_bins), dtype=np.float32)
    manifold_residual_doc_cos_sum = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    manifold_residual_doc_cos_n = np.zeros((n_bins, n_bins, n_bins), dtype=np.uint32)
    raw_sbert_doc_cos_sum = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    raw_sbert_doc_cos_n = np.zeros((n_bins, n_bins, n_bins), dtype=np.uint32)
    high_acuity_count = np.zeros((n_bins, n_bins, n_bins), dtype=np.uint32)

    lm = lexical_available.astype(bool)
    if np.any(lm):
        np.add.at(lex_sum, (bi[lm], bj[lm], bk[lm]), lexical_overlap[lm])
        np.add.at(div_sum, (bi[lm], bj[lm], bk[lm]), lexical_divergence[lm])
        np.add.at(lex_n, (bi[lm], bj[lm], bk[lm]), 1)
    np.maximum.at(acuity_max, (bi, bj, bk), acuity_score)
    high_mask = acuity_score >= float(high_acuity_threshold)
    if np.any(high_mask):
        np.add.at(high_acuity_count, (bi[high_mask], bj[high_mask], bk[high_mask]), 1)
    dm_res = manifold_residual_doc_avail.astype(bool) & np.isfinite(manifold_residual_doc_cos)
    if np.any(dm_res):
        np.add.at(manifold_residual_doc_cos_sum, (bi[dm_res], bj[dm_res], bk[dm_res]), manifold_residual_doc_cos[dm_res])
        np.add.at(manifold_residual_doc_cos_n, (bi[dm_res], bj[dm_res], bk[dm_res]), 1)
    dm_raw = raw_sbert_doc_avail.astype(bool) & np.isfinite(raw_sbert_doc_cos)
    if np.any(dm_raw):
        np.add.at(raw_sbert_doc_cos_sum, (bi[dm_raw], bj[dm_raw], bk[dm_raw]), raw_sbert_doc_cos[dm_raw])
        np.add.at(raw_sbert_doc_cos_n, (bi[dm_raw], bj[dm_raw], bk[dm_raw]), 1)

    # Per-source-edge compact support stats for anchor/source-edge view.
    edge_match_count = np.zeros(num_edges, dtype=np.uint32)
    edge_high_acuity_count = np.zeros(num_edges, dtype=np.uint32)
    edge_best_acuity = np.zeros(num_edges, dtype=np.float32)
    if valid_edges.any():
        np.add.at(edge_match_count, src_edges[valid_edges], 1)
        high_valid = valid_edges & high_mask
        if np.any(high_valid):
            np.add.at(edge_high_acuity_count, src_edges[high_valid], 1)
        np.maximum.at(edge_best_acuity, src_edges[valid_edges], acuity_score[valid_edges])

    topn = max(0, int(top_candidates or 0))
    if topn > 0 and n > 0:
        candidate_score = acuity_score.copy()
        if not np.any(candidate_score > 0):
            candidate_score = alignment_core.copy()
        k = min(topn, n)
        order = np.argpartition(-candidate_score, k - 1)[:k] if n > k else np.arange(n)
        order = order[np.argsort(-candidate_score[order])]
        cand = np.zeros(order.shape[0], dtype=_compact_top_candidate_dtype())
        cand["match_row"] = order.astype(np.int64)
        cand["match_type"] = np.asarray(records["match_type"][order], dtype=np.uint8)
        cand["src_edge"] = src_edges[order].astype(np.int32)
        cand["tgt_edge"] = tgt_edges[order].astype(np.int32)
        cand["src_doc_code"] = doc_code[src_edges[order]].astype(np.int32)
        cand["tgt_doc_code"] = doc_code[tgt_edges[order]].astype(np.int32)
        cand["src_from_label"] = src_label[src_edges[order]].astype(np.int32)
        cand["src_to_label"] = dst_label[src_edges[order]].astype(np.int32)
        cand["tgt_from_label"] = src_label[tgt_edges[order]].astype(np.int32)
        cand["tgt_to_label"] = dst_label[tgt_edges[order]].astype(np.int32)
        cand["delta_cos"] = delta_vals[order]
        cand["pc1_axis_value"] = pc_vals[order]
        cand["semantic_quality"] = q_vals[order]
        cand["lexical_overlap_coefficient"] = lexical_overlap[order]
        cand["lexical_divergence"] = lexical_divergence[order]
        cand["alignment_core"] = alignment_core[order]
        cand["acuity_score"] = acuity_score[order]
        cand["manifold_residual_doc_cosine"] = np.nan_to_num(manifold_residual_doc_cos[order], nan=np.nan).astype(np.float32)
        cand["manifold_residual_doc_available"] = manifold_residual_doc_avail[order]
        cand["raw_sbert_doc_cosine"] = np.nan_to_num(raw_sbert_doc_cos[order], nan=np.nan).astype(np.float32)
        cand["raw_sbert_doc_available"] = raw_sbert_doc_avail[order]
    else:
        cand = np.zeros((0,), dtype=_compact_top_candidate_dtype())

    match_diag = {
        "mode": mode,
        "row_aligned_to_matches": True,
        "lexical_available": lexical_available,
        "lexical_overlap_coefficient": lexical_overlap,
        "lexical_divergence": lexical_divergence,
        "alignment_core": alignment_core,
        "acuity_score": acuity_score,
        "acuity_score_count_cosine": acuity_score_count_cosine,
        "lexical_dst_count_cosine": lexical_dst_count_cosine,
        "manifold_residual_doc_cosine": manifold_residual_doc_cos,
        "manifold_residual_doc_available": manifold_residual_doc_avail,
        "raw_sbert_doc_cosine": raw_sbert_doc_cos,
        "raw_sbert_doc_available": raw_sbert_doc_avail,
        "semantic_quality": q_vals,
        "semantic_quality_min": _field("semantic_quality_min", 0.0),
    }
    if detailed:
        match_diag.update({
            "lexical_dst_jaccard": dst_jaccard,
            "lexical_dst_dice": dst_dice,
            "lexical_dst_weighted_jaccard": dst_weighted_jaccard,
            "lexical_dst_tokens_a": dst_tokens_a,
            "lexical_dst_tokens_b": dst_tokens_b,
            "lexical_dst_unique_a": dst_unique_a,
            "lexical_dst_unique_b": dst_unique_b,
            "lexical_dst_shared_unique": dst_shared_unique,
            "lexical_src_overlap_coefficient": src_overlap,
            "lexical_src_jaccard": src_jaccard,
            "lexical_src_count_cosine": src_count_cosine,
            "lexical_edge_overlap_coefficient": edge_overlap,
            "lexical_edge_jaccard": edge_jaccard,
            "lexical_edge_count_cosine": edge_count_cosine,
        })
    payload["match_diagnostics"] = match_diag
    payload["plot_cache"] = {
        "kind": "morphism_comparison_plot_cache",
        "version": 1,
        "step": float(step),
        "thresholds": thr,
        "count": count_grid,
        "lex_sum": lex_sum.astype(np.float32),
        "lex_n": lex_n,
        "div_sum": div_sum.astype(np.float32),
        "acuity_max": acuity_max,
        "manifold_residual_doc_cosine_sum": manifold_residual_doc_cos_sum.astype(np.float32),
        "manifold_residual_doc_cosine_n": manifold_residual_doc_cos_n,
        "raw_sbert_doc_cosine_sum": raw_sbert_doc_cos_sum.astype(np.float32),
        "raw_sbert_doc_cosine_n": raw_sbert_doc_cos_n,
        "high_acuity_count": high_acuity_count,
        "top_candidates": cand,
        "edge_stats": {
            "match_count_by_src_edge": edge_match_count,
            "high_acuity_count_by_src_edge": edge_high_acuity_count,
            "best_acuity_by_src_edge": edge_best_acuity,
            "high_acuity_threshold": float(high_acuity_threshold),
        },
    }
    payload.setdefault("diagnostics", {})
    payload["diagnostics"].update({
        "mode": mode,
        "step": float(step),
        "top_candidates": int(topn),
        "lexical_signature_count": int(len(cluster_sigs)),
        "lexical_available_matches": int(np.count_nonzero(lexical_available)),
        "manifold_residual_doc_available_matches": int(np.count_nonzero(manifold_residual_doc_avail)),
        "raw_sbert_doc_available_matches": int(np.count_nonzero(raw_sbert_doc_avail)),
        "high_acuity_threshold": float(high_acuity_threshold),
        "stored_match_diagnostics": True,
        "stored_detailed_lexical_diagnostics": bool(detailed),
    })
    payload["version"] = max(2, int(payload.get("version", 1) or 1))
    payload.setdefault("summary", {})
    payload["summary"].update({
        "compact_diagnostics_mode": mode,
        "lexical_available_matches": int(np.count_nonzero(lexical_available)),
        "manifold_residual_doc_available_matches": int(np.count_nonzero(manifold_residual_doc_avail)),
        "raw_sbert_doc_available_matches": int(np.count_nonzero(raw_sbert_doc_avail)),
        "max_acuity_score": float(np.nanmax(acuity_score)) if acuity_score.size else 0.0,
        "mean_manifold_residual_doc_cosine_available": float(np.nanmean(manifold_residual_doc_cos[manifold_residual_doc_avail.astype(bool)])) if np.any(manifold_residual_doc_avail) else "",
        "mean_raw_sbert_doc_cosine_available": float(np.nanmean(raw_sbert_doc_cos[raw_sbert_doc_avail.astype(bool)])) if np.any(raw_sbert_doc_avail) else "",
        "plot_cache_step": float(step),
    })
    if verbose:
        print(
            f"[analyze-enrich] complete: lexical_available={int(np.count_nonzero(lexical_available)):,}/{n:,}; "
            f"manifold_residual_doc_available={int(np.count_nonzero(manifold_residual_doc_avail)):,}/{n:,}; "
            f"raw_sbert_doc_available={int(np.count_nonzero(raw_sbert_doc_avail)):,}/{n:,}; "
            f"max_acuity={payload['summary']['max_acuity_score']:.4f}; top_candidates={int(cand.shape[0]):,}",
            flush=True,
        )
    return res_or_payload


def _plot_morphism_match_field_3d_compact(
    payload: dict,
    step: float = 0.01,
    figsize=(12, 9),
    cmap_name: str = "YlOrRd",
    doc_id: str | None = None,
    log_colors=True,
    initial_quality_floor: float = 0.0,
    cumulative: bool = False,
    include_cumulative: bool = False,
    max_plot_points: int = 150000,
    top_candidate_points: int = 2000,
) -> None:
    """
    Compact-payload graph for Analyze results produced with compact_only=True.

    If the payload contains an enrichment ``plot_cache``, this restores the
    lexical/acuteness/document-baseline views without rebuilding legacy match
    dictionaries.  If no cache is present, it falls back to the count field from
    the structured ``matches`` array.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors
    from matplotlib.patches import Patch
    from matplotlib.widgets import Slider, Button as _Btn

    if not isinstance(payload, dict):
        print("[plot_morphism_match_field_3d] Empty compact payload.")
        return
    records = payload.get("matches")
    if records is None:
        print("[plot_morphism_match_field_3d] Compact payload has no matches array.")
        return
    records = np.asarray(records)
    if records.size == 0:
        print("[plot_morphism_match_field_3d] Compact payload contains zero retained matches.")
        return

    cache = payload.get("plot_cache") if isinstance(payload.get("plot_cache"), dict) else None
    diag = payload.get("match_diagnostics") if isinstance(payload.get("match_diagnostics"), dict) else {}
    edge_index = payload.get("edge_index") or {}
    doc_ids = [str(d) for d in edge_index.get("doc_ids", [])]
    doc_code = np.asarray(edge_index.get("doc_code", []), dtype=np.int32)
    src_label = np.asarray(edge_index.get("src_label", []), dtype=np.int32)
    dst_label = np.asarray(edge_index.get("dst_label", []), dtype=np.int32)

    step = max(0.001, float((cache or {}).get("step", step)))
    n_bins = int(round(1.0 / step)) + 1
    thr = np.asarray((cache or {}).get("thresholds", np.round(np.linspace(1.0, 0.0, n_bins), 6)), dtype=float)
    if thr.shape[0] != n_bins:
        thr = np.round(np.linspace(1.0, 0.0, n_bins), 6)

    def _record_field(name, default=0.0, clip01=True):
        try:
            if records.dtype.names and name in records.dtype.names:
                arr = np.asarray(records[name], dtype=float)
            else:
                arr = np.full(records.shape[0], float(default), dtype=float)
        except Exception:
            arr = np.full(records.shape[0], float(default), dtype=float)
        arr = np.nan_to_num(arr, nan=float(default), posinf=1.0 if clip01 else float(default), neginf=0.0 if clip01 else float(default))
        return np.clip(arr, 0.0, 1.0) if clip01 else arr

    def _diag_field(name, default=0.0, clip01=True):
        try:
            if isinstance(diag, dict) and name in diag:
                arr = np.asarray(diag[name], dtype=float)
                if arr.shape[0] == records.shape[0]:
                    arr = np.nan_to_num(arr, nan=float(default), posinf=1.0 if clip01 else float(default), neginf=0.0 if clip01 else float(default))
                    return np.clip(arr, 0.0, 1.0) if clip01 else arr
        except Exception:
            pass
        return np.full(records.shape[0], float(default), dtype=float)

    def _bin_idx(arr):
        arr = np.clip(np.asarray(arr, dtype=float), 0.0, 1.0)
        idx = np.rint((1.0 - arr) / step).astype(int)
        return np.clip(idx, 0, n_bins - 1)

    if cache and isinstance(cache.get("count"), np.ndarray):
        inc = np.asarray(cache.get("count"), dtype=np.uint32)
        lex_sum = np.asarray(cache.get("lex_sum", np.zeros_like(inc, dtype=np.float32)), dtype=float)
        lex_n = np.asarray(cache.get("lex_n", np.zeros_like(inc, dtype=np.uint32)), dtype=np.uint32)
        div_sum = np.asarray(cache.get("div_sum", np.zeros_like(inc, dtype=np.float32)), dtype=float)
        acuity_max = np.asarray(cache.get("acuity_max", np.zeros_like(inc, dtype=np.float32)), dtype=float)
        manifold_residual_doc_cos_sum = np.asarray(cache.get("manifold_residual_doc_cosine_sum", cache.get("doc_embedding_cosine_sum", np.zeros_like(inc, dtype=np.float32))), dtype=float)
        manifold_residual_doc_cos_n = np.asarray(cache.get("manifold_residual_doc_cosine_n", cache.get("doc_embedding_cosine_n", np.zeros_like(inc, dtype=np.uint32))), dtype=np.uint32)
        raw_sbert_doc_cos_sum = np.asarray(cache.get("raw_sbert_doc_cosine_sum", np.zeros_like(inc, dtype=np.float32)), dtype=float)
        raw_sbert_doc_cos_n = np.asarray(cache.get("raw_sbert_doc_cosine_n", np.zeros_like(inc, dtype=np.uint32)), dtype=np.uint32)
        high_acuity_count = np.asarray(cache.get("high_acuity_count", np.zeros_like(inc, dtype=np.uint32)), dtype=np.uint32)
        top_candidates = np.asarray(cache.get("top_candidates", np.zeros((0,), dtype=_compact_top_candidate_dtype())))
        edge_stats = cache.get("edge_stats", {}) if isinstance(cache.get("edge_stats"), dict) else {}
        print(
            "[plot_morphism_match_field_3d] Using enriched compact morphism_comparison payload: "
            f"matches={int(records.size):,}; occupied_cells={int(np.count_nonzero(inc)):,}; "
            f"diagnostics_mode={(payload.get('diagnostics') or {}).get('mode', 'unknown')}; step={step:.3f}."
        )
    else:
        # Fallback: build only the count field from matches.
        delta_vals = _record_field("delta_cos", 0.0)
        pc_vals = _record_field("pc1_axis_value", 1.0)
        q_vals = _record_field("semantic_quality", 1.0)
        bi = _bin_idx(delta_vals); bj = _bin_idx(pc_vals); bk = _bin_idx(q_vals)
        inc = np.zeros((n_bins, n_bins, n_bins), dtype=np.uint32)
        np.add.at(inc, (bi, bj, bk), 1)
        lex_sum = np.zeros_like(inc, dtype=float)
        lex_n = np.zeros_like(inc, dtype=np.uint32)
        div_sum = np.zeros_like(inc, dtype=float)
        acuity_max = np.zeros_like(inc, dtype=float)
        manifold_residual_doc_cos_sum = np.zeros_like(inc, dtype=float)
        manifold_residual_doc_cos_n = np.zeros_like(inc, dtype=np.uint32)
        raw_sbert_doc_cos_sum = np.zeros_like(inc, dtype=float)
        raw_sbert_doc_cos_n = np.zeros_like(inc, dtype=np.uint32)
        high_acuity_count = np.zeros_like(inc, dtype=np.uint32)
        top_candidates = np.zeros((0,), dtype=_compact_top_candidate_dtype())
        edge_stats = {}
        print(
            "[plot_morphism_match_field_3d] Using compact morphism_comparison payload without diagnostics cache: "
            f"matches={int(records.size):,}; step={step:.3f}; count-field only."
        )

    if inc.sum() <= 0:
        print("[plot_morphism_match_field_3d] No scored matches to plot.")
        return

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_lex = np.divide(lex_sum, np.maximum(lex_n, 1), where=(lex_n > 0))
        mean_div = np.divide(div_sum, np.maximum(lex_n, 1), where=(lex_n > 0))
        mean_manifold_residual_doc_cos = np.divide(manifold_residual_doc_cos_sum, np.maximum(manifold_residual_doc_cos_n, 1), where=(manifold_residual_doc_cos_n > 0))
        mean_raw_sbert_doc_cos = np.divide(raw_sbert_doc_cos_sum, np.maximum(raw_sbert_doc_cos_n, 1), where=(raw_sbert_doc_cos_n > 0))
    mean_lex[lex_n <= 0] = 0.0
    mean_div[lex_n <= 0] = 0.0
    mean_manifold_residual_doc_cos[manifold_residual_doc_cos_n <= 0] = np.nan
    mean_raw_sbert_doc_cos[raw_sbert_doc_cos_n <= 0] = np.nan

    # Highlight bins involving a selected document can be rebuilt cheaply from
    # match records and document codes.  It is deliberately separate from the
    # saved plot_cache because doc_id is a viewer-time choice.
    hl_inc = np.zeros_like(inc, dtype=np.uint32)
    if doc_id and doc_code.size and records.dtype.names and "src_edge" in records.dtype.names and "tgt_edge" in records.dtype.names:
        try:
            src_edges = np.asarray(records["src_edge"], dtype=np.int64)
            tgt_edges = np.asarray(records["tgt_edge"], dtype=np.int64)
            doc_lookup = {d: i for i, d in enumerate(doc_ids)}
            code = doc_lookup.get(str(doc_id))
            if code is not None:
                delta_vals = _record_field("delta_cos", 0.0)
                pc_vals = _record_field("pc1_axis_value", 1.0)
                q_vals = _record_field("semantic_quality", 1.0)
                bi = _bin_idx(delta_vals); bj = _bin_idx(pc_vals); bk = _bin_idx(q_vals)
                valid = (src_edges >= 0) & (tgt_edges >= 0) & (src_edges < doc_code.shape[0]) & (tgt_edges < doc_code.shape[0])
                involved = valid & ((doc_code[src_edges] == int(code)) | (doc_code[tgt_edges] == int(code)))
                if np.any(involved):
                    np.add.at(hl_inc, (bi[involved], bj[involved], bk[involved]), 1)
        except Exception:
            pass

    if cache and cache.get("edge_stats"):
        try:
            src_edges_for_best = np.asarray(records["src_edge"], dtype=np.int64) if records.dtype.names and "src_edge" in records.dtype.names else np.zeros((records.size,), dtype=np.int64)
        except Exception:
            src_edges_for_best = np.zeros((records.size,), dtype=np.int64)
    else:
        src_edges_for_best = np.zeros((records.size,), dtype=np.int64)

    if include_cumulative:
        cum_grid = inc.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
        cum_hl_grid = hl_inc.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    else:
        cum_grid = None
        cum_hl_grid = None

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    cmap = cm.get_cmap(cmap_name)
    sm = cm.ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    sm.set_array([])
    cbar_obj = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.08)

    # Include the document-baseline view only when at least one bin has a value.
    view_order = ["count"]
    if np.count_nonzero(lex_n) > 0:
        view_order += ["mean_lexical_overlap", "lexical_divergence", "peak_acuity", "acute_candidate", "lexical_z_quality"]
    if np.count_nonzero(manifold_residual_doc_cos_n) > 0:
        view_order.append("mean_manifold_residual_doc_cosine")
    if np.count_nonzero(raw_sbert_doc_cos_n) > 0:
        view_order.append("mean_raw_sbert_doc_cosine")
    if cache and isinstance(edge_stats, dict) and np.asarray(edge_stats.get("match_count_by_src_edge", [])).size:
        view_order.append("anchor_edge")

    view_labels = {
        "count": "Match count field",
        "mean_lexical_overlap": "Mean destination lexical overlap",
        "lexical_divergence": "Mean lexical divergence field",
        "peak_acuity": "Peak acuity field",
        "acute_candidate": "Acute candidate scatter",
        "lexical_z_quality": "Lexical-overlap Z / peak acuity color",
        "mean_manifold_residual_doc_cosine": "Mean manifold-residual document cosine",
        "mean_raw_sbert_doc_cosine": "Mean raw SBERT document cosine",
        "anchor_edge": "Anchor-edge contribution",
    }
    state = {
        "view": "count",
        "cumulative": bool(cumulative),
        "q_floor": float(np.clip(initial_quality_floor, 0.0, 1.0)),
        "q_mode": "floor",
        "scat": None,
        "hl": None,
        "legend": None,
    }

    def _clear_artists():
        for key in ("scat", "hl"):
            art = state.get(key)
            if art is not None:
                try: art.remove()
                except Exception: pass
                state[key] = None
        if state.get("legend") is not None:
            try: state["legend"].remove()
            except Exception: pass
            state["legend"] = None

    def _bin_idx_scalar(v: float) -> int:
        v = float(np.clip(v, 0.0, 1.0))
        return int(np.clip(int(np.rint((1.0 - v) / step)), 0, n_bins - 1))

    def _q_keep(kidx):
        if state.get("q_mode") == "slice":
            return kidx == _bin_idx_scalar(state["q_floor"])
        return thr[kidx] >= state["q_floor"]

    def _grid_for_count():
        if state["cumulative"] and include_cumulative and cum_grid is not None:
            return cum_grid, cum_hl_grid
        return inc, hl_inc

    def _grid_points(value_grid, support_grid, hgrid=None, support_min=1):
        mask = np.asarray(support_grid) >= int(support_min)
        q_mask = np.array([_q_keep(k) for k in range(mask.shape[2])], dtype=bool)
        mask[:, :, ~q_mask] = False
        idx = np.where(mask)
        vals = np.asarray(value_grid[idx], dtype=float)
        hvals = np.asarray(hgrid[idx], dtype=float) if hgrid is not None else np.zeros(vals.shape, dtype=float)
        counts = np.asarray(support_grid[idx], dtype=float)
        if vals.size > max_plot_points:
            order = np.argsort(-vals if state["view"] != "count" else -counts)[:max_plot_points]
            idx = tuple(a[order] for a in idx)
            vals = vals[order]; hvals = hvals[order]; counts = counts[order]
        return idx, vals, hvals, counts

    def _candidate_points():
        cand = np.asarray(top_candidates)
        if cand.size == 0 or not cand.dtype.names:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []
        qvals = np.asarray(cand["semantic_quality"], dtype=float)
        keep = np.array([_q_keep(_bin_idx_scalar(q)) for q in qvals], dtype=bool)
        rows = cand[keep]
        if rows.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []
        rows = rows[np.argsort(-np.asarray(rows["acuity_score"], dtype=float))]
        rows = rows[:max(1, int(top_candidate_points))]
        return (
            np.asarray(rows["delta_cos"], dtype=float),
            np.asarray(rows["pc1_axis_value"], dtype=float),
            np.asarray(rows["semantic_quality"], dtype=float),
            np.asarray(rows["acuity_score"], dtype=float),
            np.ones(rows.shape[0], dtype=float),
            rows,
        )

    def _lexical_z_quality_points():
        idx, vals, hvals, counts = _grid_points(acuity_max, inc, hl_inc, support_min=1)
        if vals.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []
        lex_z = np.asarray(mean_lex[idx], dtype=float)
        lex_z = np.clip(np.nan_to_num(lex_z, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        return (
            np.asarray(thr[idx[0]], dtype=float),
            np.asarray(thr[idx[1]], dtype=float),
            lex_z,
            np.asarray(vals, dtype=float),
            np.asarray(counts, dtype=float),
            [],
        )

    def _edge_points():
        es = edge_stats if isinstance(edge_stats, dict) else {}
        best = np.asarray(es.get("best_acuity_by_src_edge", []), dtype=float)
        match_count = np.asarray(es.get("match_count_by_src_edge", []), dtype=float)
        high_count = np.asarray(es.get("high_acuity_count_by_src_edge", []), dtype=float)
        if best.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []
        edge_ids = np.where(match_count > 0)[0]
        if edge_ids.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []
        # Locate one top candidate row per edge when possible so the point has a
        # concrete Δ/PC1/Q coordinate rather than an edge-only summary.
        cand = np.asarray(top_candidates)
        rows = []
        if cand.size and cand.dtype.names:
            seen_edges = set()
            for row in cand[np.argsort(-np.asarray(cand["acuity_score"], dtype=float))]:
                se = int(row["src_edge"])
                if se in seen_edges:
                    continue
                if se < best.size and match_count[se] > 0 and _q_keep(_bin_idx_scalar(float(row["semantic_quality"]))):
                    rows.append(row)
                    seen_edges.add(se)
                if len(rows) >= max_plot_points:
                    break
        if not rows:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []
        rows = np.asarray(rows, dtype=cand.dtype)
        return (
            np.asarray(rows["delta_cos"], dtype=float),
            np.asarray(rows["pc1_axis_value"], dtype=float),
            np.asarray(rows["semantic_quality"], dtype=float),
            np.asarray(rows["acuity_score"], dtype=float),
            np.asarray([max(1.0, high_count[int(r["src_edge"])]) for r in rows], dtype=float),
            rows,
        )

    def _refresh(_evt=None):
        _clear_artists()
        view = state["view"]
        rows = []

        if view == "count":
            grid, hgrid = _grid_for_count()
            idx, vals, hvals, counts = _grid_points(grid.astype(float), grid, hgrid, support_min=1)
            label = "match count"
            norm = mcolors.LogNorm(vmin=1.0, vmax=max(1.0, float(vals.max()) if vals.size else 1.0)) if log_colors else mcolors.Normalize(vmin=1.0, vmax=max(1.0, float(vals.max()) if vals.size else 1.0))
            colors = cmap(norm(np.maximum(vals, 1.0))) if vals.size else []
            sizes = np.full(vals.size, 12.0)
            x, y, z = thr[idx[0]], thr[idx[1]], thr[idx[2]]
        elif view == "mean_lexical_overlap":
            idx, vals, hvals, counts = _grid_points(mean_lex, lex_n, hl_inc, support_min=1)
            label = "mean destination lexical overlap coefficient"
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            colors = cmap(norm(vals)) if vals.size else []
            sizes = 10.0 + 6.0 * np.sqrt(np.maximum(counts, 1.0))
            x, y, z = thr[idx[0]], thr[idx[1]], thr[idx[2]]
        elif view == "lexical_divergence":
            idx, vals, hvals, counts = _grid_points(mean_div, lex_n, hl_inc, support_min=1)
            label = "mean lexical divergence (1 - overlap coefficient)"
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            colors = cmap(norm(vals)) if vals.size else []
            sizes = 10.0 + 6.0 * np.sqrt(np.maximum(counts, 1.0))
            x, y, z = thr[idx[0]], thr[idx[1]], thr[idx[2]]
        elif view == "peak_acuity":
            idx, vals, hvals, counts = _grid_points(acuity_max, inc, hl_inc, support_min=1)
            label = "peak semantic/lexical acuity per bin"
            norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, float(vals.max()) if vals.size else 1.0))
            colors = cmap(norm(vals)) if vals.size else []
            sizes = 14.0 + 8.0 * np.sqrt(np.maximum(counts, 1.0))
            x, y, z = thr[idx[0]], thr[idx[1]], thr[idx[2]]
        elif view == "acute_candidate":
            x, y, z, vals, hvals, rows = _candidate_points()
            label = "individual candidate acuity score"
            norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, float(vals.max()) if vals.size else 1.0))
            colors = cmap(norm(vals)) if vals.size else []
            sizes = 18.0 + 80.0 * np.maximum(vals, 0.0)
            counts = np.ones_like(vals)
        elif view == "lexical_z_quality":
            x, y, z, vals, hvals, rows = _lexical_z_quality_points()
            label = "peak acuity value per retained-match bin"
            norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, float(vals.max()) if vals.size else 1.0))
            colors = cmap(norm(vals)) if vals.size else []
            sizes = 12.0 + 8.0 * np.sqrt(np.maximum(hvals, 1.0))
            counts = hvals
        elif view == "mean_manifold_residual_doc_cosine":
            idx, vals, hvals, counts = _grid_points(mean_manifold_residual_doc_cos, manifold_residual_doc_cos_n, hl_inc, support_min=1)
            label = "mean manifold-residual document cosine"
            # Values may be negative; keep full cosine scale.
            vals = np.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=-1.0)
            norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
            colors = cmap(norm(vals)) if vals.size else []
            sizes = 10.0 + 6.0 * np.sqrt(np.maximum(counts, 1.0))
            x, y, z = thr[idx[0]], thr[idx[1]], thr[idx[2]]
        elif view == "mean_raw_sbert_doc_cosine":
            idx, vals, hvals, counts = _grid_points(mean_raw_sbert_doc_cos, raw_sbert_doc_cos_n, hl_inc, support_min=1)
            label = "mean raw SBERT document cosine"
            # Values may be negative; keep full cosine scale.
            vals = np.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=-1.0)
            norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
            colors = cmap(norm(vals)) if vals.size else []
            sizes = 10.0 + 6.0 * np.sqrt(np.maximum(counts, 1.0))
            x, y, z = thr[idx[0]], thr[idx[1]], thr[idx[2]]
        else:  # anchor_edge
            x, y, z, vals, counts, rows = _edge_points()
            hvals = np.ones_like(vals) if doc_id else np.zeros_like(vals)
            label = "best acuity per source edge"
            norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, float(vals.max()) if vals.size else 1.0))
            colors = cmap(norm(vals)) if vals.size else []
            sizes = 24.0 + 18.0 * np.sqrt(np.maximum(counts, 1.0))

        cbar_obj.update_normal(cm.ScalarMappable(norm=norm, cmap=cmap))
        cbar_obj.set_label(label)
        state["scat"] = ax.scatter(x, y, z, c=colors, s=sizes, depthshade=True, alpha=0.88)

        q_txt = f"Q slice ≈ {state['q_floor']:.2f}" if state.get("q_mode") == "slice" else f"Q ≥ {state['q_floor']:.2f}"
        mode_txt = "cumulative" if (state["cumulative"] and include_cumulative and view == "count") else "incremental"
        pc1_lab = str((payload.get("params") or {}).get("pc1_axis_label") or (payload.get("params") or {}).get("pc1_match_axis") or "pc1_axis_value")
        ax.set_title(f"Analyze Match Field — {view_labels.get(view, view)} ({q_txt}; {mode_txt}; Y={pc1_lab})")
        ax.set_xlabel("Δ direction cosine")
        ax.set_ylabel(f"PC1 concordance\n{pc1_lab}")
        ax.set_zlabel("Destination lexical overlap coefficient" if view == "lexical_z_quality" else "Semantic quality Q")
        ax.set_xlim(1.0, 0.0)
        ax.set_ylim(1.0, 0.0)
        ax.set_zlim(0.0, 1.0)

        handles = [Patch(facecolor=cmap(norm(0.8 if view != "count" else max(1.0, float(np.max(vals)) if len(vals) else 1.0))), edgecolor='k', label=label)]
        if view in ("acute_candidate", "anchor_edge"):
            handles.append(Patch(facecolor=(0.85, 0.85, 0.85, 1), edgecolor='k', label="top retained candidates"))
        elif view == "lexical_z_quality":
            handles.append(Patch(facecolor=(0.85, 0.85, 0.85, 1), edgecolor='k', label="full retained-match bins"))
        if len(vals) >= max_plot_points:
            handles.append(Patch(facecolor=(0.75, 0.75, 0.75, 1), edgecolor='k', label=f"display capped at {max_plot_points:,} pts"))
        state["legend"] = ax.legend(handles=handles, loc="upper right")
        fig.canvas.draw_idle()

    _refresh()

    ax_q = fig.add_axes([0.76, 0.92, 0.22, 0.03]); ax_q.set_in_layout(False)
    slider_q = Slider(ax_q, "Q floor/slice", 0.0, 1.0, valinit=state["q_floor"], valstep=step)
    def _on_q_change(val):
        state["q_floor"] = float(val)
        _refresh()
    slider_q.on_changed(_on_q_change)

    ax_view = fig.add_axes([0.76, 0.87, 0.22, 0.04]); ax_view.set_in_layout(False)
    btn_view = _Btn(ax_view, "View: count")
    def _cycle_view(_evt):
        cur = view_order.index(state["view"]) if state["view"] in view_order else 0
        state["view"] = view_order[(cur + 1) % len(view_order)]
        btn_view.label.set_text("View: " + state["view"].replace("_", " ")[:18])
        _refresh()
    btn_view.on_clicked(_cycle_view)

    ax_qmode = fig.add_axes([0.76, 0.82, 0.22, 0.04]); ax_qmode.set_in_layout(False)
    btn_qmode = _Btn(ax_qmode, "Q slice")
    def _toggle_qmode(_evt):
        state["q_mode"] = "slice" if state.get("q_mode") != "slice" else "floor"
        btn_qmode.label.set_text("Q floor" if state["q_mode"] == "slice" else "Q slice")
        _refresh()
    btn_qmode.on_clicked(_toggle_qmode)

    ax_mode = fig.add_axes([0.76, 0.77, 0.22, 0.04]); ax_mode.set_in_layout(False)
    if include_cumulative:
        btn_mode = _Btn(ax_mode, "Cumulative")
        def _toggle_cum(_evt):
            state["cumulative"] = not state["cumulative"]
            btn_mode.label.set_text("Incremental" if state["cumulative"] else "Cumulative")
            _refresh()
        btn_mode.on_clicked(_toggle_cum)
    else:
        ax_mode.axis("off")
        ax_mode.text(0.5, 0.5, "Incremental only", ha="center", va="center", fontsize=9)

    ax_reset = fig.add_axes([0.76, 0.72, 0.22, 0.04]); ax_reset.set_in_layout(False)
    btn_reset = _Btn(ax_reset, "Reset view")
    init_elev, init_azim = ax.elev, ax.azim
    def _reset(_evt):
        state["view"] = "count"
        btn_view.label.set_text("View: count")
        state["q_mode"] = "floor"
        btn_qmode.label.set_text("Q slice")
        slider_q.set_val(float(np.clip(initial_quality_floor, 0.0, 1.0)))
        ax.view_init(elev=init_elev, azim=init_azim)
        _refresh()
    btn_reset.on_clicked(_reset)

    plt.tight_layout()
    attach_matplotlib_save_button(fig, default_name="morphism_match_acuity_views_compact.pkl", parent=globals().get("root"))
    fig._compact_morphism_widgets = {
        "slider_q": slider_q,
        "btn_view": btn_view,
        "btn_qmode": btn_qmode,
        "btn_reset": btn_reset,
    }
    plt.show()


def plot_morphism_match_field_3d(
    res: dict,
    step: float = 0.01,
    figsize=(12, 9),
    cmap_name: str = "YlOrRd",
    doc_id: str | None = None,
    log_colors=True,
    initial_quality_floor: float = 0.0,
    cumulative: bool = False,
    include_cumulative: bool = False,
    pc1_axis_mode: str | None = None,
    max_plot_points: int = 150000,
    top_candidate_points: int = 2000,
) -> None:
    """
    Interactive Analyze endpoint graph.

    Base axes:
        X = Δ direction cosine
        Y = selected PC1 concordance axis (default destination PC1)
        Z = semantic quality Q

    Toggleable views:
        • match count field
        • mean lexical overlap field
        • mean lexical divergence field
        • peak acuity field
        • acute candidate scatter
        • lexical-overlap Z / peak-acuity color scatter
        • anchor-edge contribution view

    Acuity is defined per match as:
        harmonic_mean(Δ, PC1_axis, Q) × (1 - destination lexical overlap coefficient)

    The field views aggregate accepted Analyze matches into incremental bins.  The
    acute candidate scatter and anchor-edge view preserve individual high-acuity
    matches so sharp candidates are not averaged into large bins.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.widgets import Slider, Button as _Btn

    compact_payload = _morphism_comparison_payload_from_result(res)
    if compact_payload is not None and (compact_payload.get("plot_cache") or not isinstance(res, dict) or not res.get("index")):
        return _plot_morphism_match_field_3d_compact(
            compact_payload,
            step=step,
            figsize=figsize,
            cmap_name=cmap_name,
            doc_id=doc_id,
            log_colors=log_colors,
            initial_quality_floor=initial_quality_floor,
            cumulative=cumulative,
            include_cumulative=include_cumulative,
            max_plot_points=max_plot_points,
            top_candidate_points=top_candidate_points,
        )

    if not res or "index" not in res or not res["index"]:
        print("[plot_morphism_match_field_3d] Empty or invalid 'res'.")
        return

    def _normalize_pc1_axis(mode):
        raw = str(mode or "dst").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "dst": "dst", "dest": "dst", "destination": "dst", "destination_only": "dst", "dst_only": "dst",
            "src": "src", "source": "src", "source_only": "src", "src_only": "src",
            "both": "both", "srcdst": "both", "src_dst": "both", "min": "both", "composite": "both",
        }
        return aliases.get(raw, "dst")

    def _pc1_axis_label(mode):
        mode = _normalize_pc1_axis(mode)
        return {"dst": "dst_pc1", "src": "src_pc1", "both": "min(src_pc1,dst_pc1)"}.get(mode, "dst_pc1")

    if pc1_axis_mode is None:
        pc1_axis_mode = (res.get("params", {}) or {}).get("pc1_match_axis", "dst")
    pc1_axis_mode = _normalize_pc1_axis(pc1_axis_mode)

    step = max(0.001, float(step))
    n_bins = int(round(1.0 / step)) + 1
    thr = np.round(np.linspace(1.0, 0.0, n_bins), 6)

    def _safe_float(x, default=None):
        try:
            if x is None or x == "":
                return default
            v = float(x)
            return v if np.isfinite(v) else default
        except Exception:
            return default

    def _safe_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        return str(x).strip().lower() in ("true", "1", "yes", "y", "on")

    def _bin_idx(v: float) -> int:
        v = float(np.clip(_safe_float(v, 0.0), 0.0, 1.0))
        idx = int(np.round((1.0 - v) / step))
        return int(np.clip(idx, 0, n_bins - 1))

    matches_maps = {
        "aligned": res.get("aligned_matches", {}) or {},
        "pc1only": res.get("pc1_only_matches", {}) or {},
    }
    index_map = res.get("index", {}) or {}

    inc = np.zeros((n_bins, n_bins, n_bins), dtype=np.int32)
    hl_inc = np.zeros_like(inc)
    lex_sum = np.zeros_like(inc, dtype=np.float64)
    lex_n = np.zeros_like(inc, dtype=np.int32)
    div_sum = np.zeros_like(inc, dtype=np.float64)
    acuity_max = np.zeros_like(inc, dtype=np.float64)

    candidates = []
    edge_stats = {}
    seen = set()
    selected_peak = {"key": None, "joint": -1.0}

    def _pc_from_scores(sc):
        sp = _safe_float(sc.get("src_pc1"), None)
        dp = _safe_float(sc.get("dst_pc1"), None)
        pa = _safe_float(sc.get("pc1_axis_value"), None)
        if pa is not None:
            return float(np.clip(pa, 0.0, 1.0))
        if sp is None or dp is None:
            return 1.0
        if pc1_axis_mode == "src":
            return abs(float(sp))
        if pc1_axis_mode == "both":
            return min(abs(float(sp)), abs(float(dp)))
        return abs(float(dp))

    def _ingest(mmap: dict, kind: str):
        nonlocal selected_peak
        for i_src, lst in mmap.items():
            i_src = int(i_src)
            src_meta = index_map.get(i_src, {}) or {}
            src_doc = src_meta.get("doc")
            for m in (lst or []):
                j_tgt = m.get("j")
                if j_tgt is None:
                    continue
                key_pair = (kind, int(i_src), int(j_tgt))
                if key_pair in seen:
                    continue
                seen.add(key_pair)
                sc = m.get("scores", {}) or {}
                d = _safe_float(sc.get("delta_cos"), None)
                if d is None:
                    continue
                pc = _pc_from_scores(sc)
                q = _safe_float(sc.get("semantic_quality"), 1.0)
                q = 1.0 if q is None else float(np.clip(q, 0.0, 1.0))
                i0, j0, k0 = _bin_idx(d), _bin_idx(pc), _bin_idx(q)
                inc[i0, j0, k0] += 1

                tgt_doc = m.get("doc")
                involved = bool(doc_id and ((src_doc == doc_id) or (tgt_doc == doc_id)))
                if involved:
                    hl_inc[i0, j0, k0] += 1

                lex_available = _safe_bool(sc.get("lexical_available", False))
                lex = _safe_float(sc.get("lexical_overlap_coefficient"), None)
                if lex_available and lex is not None:
                    lex = float(np.clip(lex, 0.0, 1.0))
                    div = _safe_float(sc.get("lexical_divergence"), 1.0 - lex)
                    div = float(np.clip(div, 0.0, 1.0))
                    lex_sum[i0, j0, k0] += lex
                    div_sum[i0, j0, k0] += div
                    lex_n[i0, j0, k0] += 1
                else:
                    lex = 0.0
                    div = 0.0

                core = _safe_float(sc.get("alignment_core"), None)
                if core is None:
                    core = _quality_hmean([max(0.0, d), max(0.0, pc), max(0.0, q)])
                acuity = _safe_float(sc.get("acuity_score"), core * div)
                acuity = float(np.clip(acuity if acuity is not None else 0.0, 0.0, 1.0))
                if acuity > acuity_max[i0, j0, k0]:
                    acuity_max[i0, j0, k0] = acuity

                joint = _safe_float(sc.get("joint_min_4d"), min(d, pc, q))
                if (doc_id and involved) or (not doc_id):
                    if joint > selected_peak["joint"]:
                        selected_peak = {"key": (i0, j0, k0), "joint": float(joint)}

                # Preserve individual candidate records only when lexical/acuteness
                # was actually computed.  Count-field views still include every
                # retained match through `inc`, but this avoids storing huge
                # zero-acuity PC1-only candidate lists in high-k runs when
                # compute_acuity_for="aligned_only".
                acuity_computed = _safe_bool(sc.get("acuity_computed", False))
                if acuity_computed or acuity > 0.0:
                    cand = {
                        "kind": kind,
                        "src_entry_index": i_src,
                        "tgt_entry_index": int(j_tgt),
                        "src_doc": src_doc,
                        "src_from": src_meta.get("from"),
                        "src_to": src_meta.get("to"),
                        "tgt_doc": tgt_doc,
                        "tgt_from": m.get("from"),
                        "tgt_to": m.get("to"),
                        "delta": float(d), "pc": float(pc), "q": float(q),
                        "i0": i0, "j0": j0, "k0": k0,
                        "lexical_overlap": float(lex),
                        "lexical_divergence": float(div),
                        "alignment_core": float(core),
                        "acuity": float(acuity),
                        "involved": involved,
                    }
                    candidates.append(cand)

                    es = edge_stats.setdefault(i_src, {
                        "src_entry_index": i_src,
                        "src_doc": src_doc,
                        "src_from": src_meta.get("from"),
                        "src_to": src_meta.get("to"),
                        "best_acuity": -1.0,
                        "high_acuity_count": 0,
                        "match_count": 0,
                        "best": None,
                    })
                    es["match_count"] += 1
                    if acuity >= 0.50:
                        es["high_acuity_count"] += 1
                    if acuity > es["best_acuity"]:
                        es["best_acuity"] = float(acuity)
                        es["best"] = cand

    _ingest(matches_maps["aligned"], "aligned")
    _ingest(matches_maps["pc1only"], "pc1only")

    if inc.sum() <= 0:
        print("[plot_morphism_match_field_3d] No scored matches to plot.")
        return

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_lex = np.divide(lex_sum, np.maximum(lex_n, 1), where=(lex_n > 0))
        mean_div = np.divide(div_sum, np.maximum(lex_n, 1), where=(lex_n > 0))
    mean_lex[lex_n <= 0] = 0.0
    mean_div[lex_n <= 0] = 0.0

    if include_cumulative:
        cum_grid = inc.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
        cum_hl_grid = hl_inc.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    else:
        cum_grid = None
        cum_hl_grid = None

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    cmap = cm.get_cmap(cmap_name)
    base_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm = cm.ScalarMappable(norm=base_norm, cmap=cmap)
    sm.set_array([])
    cbar_obj = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.08)

    view_order = ["count", "mean_lexical_overlap", "lexical_divergence", "peak_acuity", "acute_candidate", "lexical_z_quality", "anchor_edge"]
    view_labels = {
        "count": "Match count field",
        "mean_lexical_overlap": "Mean destination lexical overlap",
        "lexical_divergence": "Mean lexical divergence field",
        "peak_acuity": "Peak acuity field",
        "acute_candidate": "Acute candidate scatter",
        "lexical_z_quality": "Lexical-overlap Z / peak acuity color",
        "anchor_edge": "Anchor-edge contribution",
    }
    state = {
        "view": "count",
        "cumulative": bool(cumulative),
        "q_floor": float(np.clip(initial_quality_floor, 0.0, 1.0)),
        "q_mode": "floor",
        "scat": None,
        "hl": None,
        "legend": None,
    }

    def _clear_artists():
        for key in ("scat", "hl"):
            art = state.get(key)
            if art is not None:
                try: art.remove()
                except Exception: pass
                state[key] = None
        if state.get("legend") is not None:
            try: state["legend"].remove()
            except Exception: pass
            state["legend"] = None

    def _grid_for_count():
        if state["cumulative"] and include_cumulative and cum_grid is not None:
            return cum_grid, cum_hl_grid
        return inc, hl_inc

    def _q_keep(kidx):
        if state.get("q_mode") == "slice":
            return kidx == _bin_idx(state["q_floor"])
        return thr[kidx] >= state["q_floor"]

    def _grid_points(value_grid, support_grid, hgrid=None, support_min=1):
        mask = support_grid >= int(support_min)
        q_mask = np.array([_q_keep(k) for k in range(support_grid.shape[2])], dtype=bool)
        mask[:, :, ~q_mask] = False
        idx = np.where(mask)
        vals = np.asarray(value_grid[idx], dtype=float)
        hvals = np.asarray(hgrid[idx], dtype=float) if hgrid is not None else np.zeros(vals.shape, dtype=float)
        counts = np.asarray(support_grid[idx], dtype=float)
        if vals.size > max_plot_points:
            # Keep highest-value cells first; count view keeps highest-count cells.
            order = np.argsort(-vals if state["view"] != "count" else -counts)[:max_plot_points]
            idx = tuple(a[order] for a in idx)
            vals = vals[order]; hvals = hvals[order]; counts = counts[order]
        return idx, vals, hvals, counts

    def _candidate_points():
        rows = [c for c in candidates if _q_keep(_bin_idx(c["q"]))]
        rows.sort(key=lambda c: c.get("acuity", 0.0), reverse=True)
        rows = rows[:max(1, int(top_candidate_points))]
        if not rows:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []
        return (
            np.asarray([c["delta"] for c in rows], dtype=float),
            np.asarray([c["pc"] for c in rows], dtype=float),
            np.asarray([c["q"] for c in rows], dtype=float),
            np.asarray([c["acuity"] for c in rows], dtype=float),
            np.asarray([1.0 if c.get("involved") else 0.0 for c in rows], dtype=float),
            rows,
        )

    def _lexical_z_quality_points():
        """
        Full retained-match field with lexical overlap on Z and peak acuity as color.

        Unlike the acute-candidate scatter, this view is intentionally built from the
        full retained aligned + PC1-only match population represented by `inc`.  Each
        displayed point is a Δ/PC1/Q bin, not an individual candidate row:

            X     = Δ threshold bin
            Y     = selected PC1 threshold bin
            Z     = mean destination lexical overlap coefficient for the bin
                    (0.0 when no lexical diagnostics were available for that bin)
            color = peak acuity value observed in the bin
            size  = retained match count in the bin

        This keeps the lexical-Z view comparable to the count field while still
        surfacing acute bins through color.
        """
        idx, vals, hvals, counts = _grid_points(acuity_max, inc, hl_inc, support_min=1)
        if vals.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []

        lex_z = np.asarray(mean_lex[idx], dtype=float)
        lex_z = np.clip(np.nan_to_num(lex_z, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

        # hvals carries retained bin counts for sizing in this view.
        return (
            np.asarray(thr[idx[0]], dtype=float),
            np.asarray(thr[idx[1]], dtype=float),
            lex_z,
            np.asarray(vals, dtype=float),
            np.asarray(counts, dtype=float),
            [],
        )

    def _edge_points():
        rows = []
        for es in edge_stats.values():
            b = es.get("best")
            if not b:
                continue
            if not _q_keep(_bin_idx(b["q"])):
                continue
            rows.append(es)
        rows.sort(key=lambda e: e.get("best_acuity", 0.0), reverse=True)
        if not rows:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), rows
        return (
            np.asarray([e["best"]["delta"] for e in rows], dtype=float),
            np.asarray([e["best"]["pc"] for e in rows], dtype=float),
            np.asarray([e["best"]["q"] for e in rows], dtype=float),
            np.asarray([max(0.0, e.get("best_acuity", 0.0)) for e in rows], dtype=float),
            np.asarray([max(1.0, e.get("high_acuity_count", 0)) for e in rows], dtype=float),
            rows,
        )

    def _refresh(_evt=None):
        _clear_artists()
        view = state["view"]

        if view == "count":
            grid, hgrid = _grid_for_count()
            idx, vals, hvals, counts = _grid_points(grid.astype(float), grid, hgrid, support_min=1)
            label = "match count"
            norm = mcolors.LogNorm(vmin=1.0, vmax=max(1.0, float(vals.max()) if vals.size else 1.0)) if log_colors else mcolors.Normalize(vmin=1.0, vmax=max(1.0, float(vals.max()) if vals.size else 1.0))
            colors = cmap(norm(np.maximum(vals, 1.0))) if vals.size else []
            sizes = np.full(vals.size, 12.0)
            x, y, z = thr[idx[0]], thr[idx[1]], thr[idx[2]]
        elif view == "mean_lexical_overlap":
            idx, vals, hvals, counts = _grid_points(mean_lex, lex_n, hl_inc, support_min=1)
            label = "mean destination lexical overlap coefficient"
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            colors = cmap(norm(vals)) if vals.size else []
            sizes = 10.0 + 6.0 * np.sqrt(np.maximum(counts, 1.0))
            x, y, z = thr[idx[0]], thr[idx[1]], thr[idx[2]]
        elif view == "lexical_divergence":
            idx, vals, hvals, counts = _grid_points(mean_div, lex_n, hl_inc, support_min=1)
            label = "mean lexical divergence (1 - overlap coefficient)"
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            colors = cmap(norm(vals)) if vals.size else []
            sizes = 10.0 + 6.0 * np.sqrt(np.maximum(counts, 1.0))
            x, y, z = thr[idx[0]], thr[idx[1]], thr[idx[2]]
        elif view == "peak_acuity":
            idx, vals, hvals, counts = _grid_points(acuity_max, inc, hl_inc, support_min=1)
            label = "peak semantic/lexical acuity per bin"
            norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, float(vals.max()) if vals.size else 1.0))
            colors = cmap(norm(vals)) if vals.size else []
            sizes = 14.0 + 8.0 * np.sqrt(np.maximum(counts, 1.0))
            x, y, z = thr[idx[0]], thr[idx[1]], thr[idx[2]]
        elif view == "acute_candidate":
            x, y, z, vals, hvals, rows = _candidate_points()
            label = "individual candidate acuity score"
            norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, float(vals.max()) if vals.size else 1.0))
            colors = cmap(norm(vals)) if vals.size else []
            sizes = 18.0 + 80.0 * np.maximum(vals, 0.0)
            counts = np.ones_like(vals)
        elif view == "lexical_z_quality":
            x, y, z, vals, hvals, rows = _lexical_z_quality_points()
            label = "peak acuity value per retained-match bin"
            norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, float(vals.max()) if vals.size else 1.0))
            colors = cmap(norm(vals)) if vals.size else []
            # hvals carries retained match counts in this view; point size now
            # reflects full-field support while color reflects peak acuity.
            sizes = 12.0 + 8.0 * np.sqrt(np.maximum(hvals, 1.0))
            counts = hvals
        else:  # anchor_edge
            x, y, z, vals, counts, rows = _edge_points()
            hvals = np.ones_like(vals) if doc_id else np.zeros_like(vals)
            label = "best acuity per anchor/source edge"
            norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, float(vals.max()) if vals.size else 1.0))
            colors = cmap(norm(vals)) if vals.size else []
            sizes = 24.0 + 18.0 * np.sqrt(np.maximum(counts, 1.0))

        cbar_obj.update_normal(cm.ScalarMappable(norm=norm, cmap=cmap))
        cbar_obj.set_label(label)
        state["scat"] = ax.scatter(x, y, z, c=colors, s=sizes, depthshade=True, alpha=0.88)

        # v18: no selected-document overlay is drawn here.
        # The selected doc_id can still scope/filter the analysis upstream, but drawing
        # a second green scatter layer obscured the gradient coloration used by the
        # lexical/acuteness views.

        if state.get("q_mode") == "slice":
            q_txt = f"Q slice ≈ {state['q_floor']:.2f}"
        else:
            q_txt = f"Q ≥ {state['q_floor']:.2f}"
        mode_txt = "cumulative" if (state["cumulative"] and include_cumulative and view == "count") else "incremental"
        ax.set_title(f"Analyze Match Field — {view_labels.get(view, view)} ({q_txt}; {mode_txt}; Y={_pc1_axis_label(pc1_axis_mode)})")
        ax.set_xlabel("Δ direction cosine")
        ax.set_ylabel(f"PC1 concordance\n{_pc1_axis_label(pc1_axis_mode)}")
        if view == "lexical_z_quality":
            ax.set_zlabel("Destination lexical overlap coefficient")
        else:
            ax.set_zlabel("Semantic quality Q")
        ax.set_xlim(1.0, 0.0)
        ax.set_ylim(1.0, 0.0)
        ax.set_zlim(0.0, 1.0)

        handles = [Patch(facecolor=cmap(norm(0.8 if view != "count" else max(1.0, float(vals.max()) if len(vals) else 1.0))), edgecolor='k', label=label)]
        # v18: selected-document involvement is no longer shown as a green overlay/legend item.
        if view in ("acute_candidate", "anchor_edge"):
            handles.append(Patch(facecolor=(0.85, 0.85, 0.85, 1), edgecolor='k', label="not averaged into bins"))
        elif view == "lexical_z_quality":
            handles.append(Patch(facecolor=(0.85, 0.85, 0.85, 1), edgecolor='k', label="full retained aligned + PC1-only bins"))
        if len(vals) >= max_plot_points:
            handles.append(Patch(facecolor=(0.75, 0.75, 0.75, 1), edgecolor='k', label=f"display capped at {max_plot_points:,} pts"))
        state["legend"] = ax.legend(handles=handles, loc="upper right")
        fig.canvas.draw_idle()

    _refresh()

    ax_q = fig.add_axes([0.76, 0.92, 0.22, 0.03]); ax_q.set_in_layout(False)
    slider_q = Slider(ax_q, "Q floor/slice", 0.0, 1.0, valinit=state["q_floor"], valstep=step)
    def _on_q_change(val):
        state["q_floor"] = float(val)
        _refresh()
    slider_q.on_changed(_on_q_change)

    ax_view = fig.add_axes([0.76, 0.87, 0.22, 0.04]); ax_view.set_in_layout(False)
    btn_view = _Btn(ax_view, "View: count")
    def _cycle_view(_evt):
        cur = view_order.index(state["view"]) if state["view"] in view_order else 0
        state["view"] = view_order[(cur + 1) % len(view_order)]
        btn_view.label.set_text("View: " + state["view"].replace("_", " ")[:18])
        _refresh()
    btn_view.on_clicked(_cycle_view)

    ax_qmode = fig.add_axes([0.76, 0.82, 0.22, 0.04]); ax_qmode.set_in_layout(False)
    btn_qmode = _Btn(ax_qmode, "Q slice")
    def _toggle_qmode(_evt):
        state["q_mode"] = "slice" if state.get("q_mode") != "slice" else "floor"
        btn_qmode.label.set_text("Q floor" if state["q_mode"] == "slice" else "Q slice")
        _refresh()
    btn_qmode.on_clicked(_toggle_qmode)

    ax_mode = fig.add_axes([0.76, 0.77, 0.22, 0.04]); ax_mode.set_in_layout(False)
    if include_cumulative:
        btn_mode = _Btn(ax_mode, "Cumulative")
        def _toggle_cum(_evt):
            state["cumulative"] = not state["cumulative"]
            btn_mode.label.set_text("Incremental" if state["cumulative"] else "Cumulative")
            _refresh()
        btn_mode.on_clicked(_toggle_cum)
    else:
        ax_mode.axis("off")
        ax_mode.text(0.5, 0.5, "Incremental only", ha="center", va="center", fontsize=9)

    ax_reset = fig.add_axes([0.76, 0.72, 0.22, 0.04]); ax_reset.set_in_layout(False)
    btn_reset = _Btn(ax_reset, "Reset view")
    init_elev, init_azim = ax.elev, ax.azim
    def _reset(_evt):
        state["view"] = "count"
        btn_view.label.set_text("View: count")
        state["q_mode"] = "floor"
        btn_qmode.label.set_text("Q slice")
        slider_q.set_val(float(np.clip(initial_quality_floor, 0.0, 1.0)))
        ax.view_init(elev=init_elev, azim=init_azim)
        _refresh()
    btn_reset.on_clicked(_reset)

    plt.tight_layout()
    attach_matplotlib_save_button(fig, default_name="morphism_match_acuity_views.pkl", parent=globals().get("root"))
    plt.show()
