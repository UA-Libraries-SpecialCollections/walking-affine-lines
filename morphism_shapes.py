
"""
morphism_shapes.py

"Zig-zag" morphism-shape arrangement
------------------------------------

Treat each intra-document pairwise cluster morphism as a 3-part shape:
   v = unit(delta_ij)                 (transform direction)
   s = unit(PC1(src))                 (source PC1 direction)
   t = unit(PC1(dst))                 (destination PC1 direction)

We canonicalize the signs relative to the flow so comparisons are meaningful
across the collection, and build a compact, rotation-stable feature vector
for each morphism's "shape". Then we cluster these shapes across the entire
corpus and form document-level arrangements by membership over these shape clusters.

Main APIs
---------
- extract_shapes_from_cdm_dict(document_delta_dict, ...)
    -> returns shapes_df (one row per morphism) and feature matrix X
- cluster_shapes(X, k=64, method='kmeans', random_state=0)
    -> returns labels (len = #morphisms), centroids
- doc_membership(shapes_df, labels, weight_mode='weighted', normalize=True)
    -> returns doc-by-shape membership matrix (dict or np.ndarray) and an index map
- build_hierarchy(document_delta_dict, depth=2, shape_k=64, doc_k=8, ...)
    -> recursively refines shape clusters within doc clusters

Notes
-----
- "Rotation-stable": we use pairwise cosines between (v,s,t): [cos(v,s), cos(-v,t), cos(s,t)],
  which are invariant to any global rotation (Gram matrix entries). To retain optional *global*
  directionality, we also support coarse spherical binning of v (add one-hot of v-bin).
- We allow weighting each morphism by flow-consistent alignment and by delta magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
from collections import defaultdict
import numpy as np
import math

# ------------------------
# Helpers
# ------------------------

def _unit(x: np.ndarray, eps: float=1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = float(np.linalg.norm(x))
    if n < eps:
        return np.zeros_like(x, dtype=float)
    return x / n

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a)+1e-12) / (np.linalg.norm(b)+1e-12))

def _align_signs_to_flow(v: np.ndarray, s: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align ambiguous PC1 signs relative to flow direction v.
    Make s point with v, and t point against v.
    """
    s2 = np.copy(s)
    t2 = np.copy(t)
    if np.dot(s2, v) < 0: s2 = -s2
    if np.dot(t2, -v) < 0: t2 = -t2
    return v, s2, t2

def _spherical_bin(v: np.ndarray, n_az: int=12, n_el: int=6) -> int:
    """Coarse bin of a 3D direction. If dim!=3, project to first 3 dims and renormalize.
    Returns an integer bin id in [0, n_az*n_el). Bins: azimuth in [0, 2pi), elevation in [-pi/2, pi/2].
    """
    if v.shape[0] != 3:
        vv = v[:3].copy()
        v = _unit(vv)
    else:
        v = _unit(v)
    # azimuth phi: atan2(y,x) in [0,2pi)
    phi = math.atan2(v[1], v[0])
    if phi < 0: phi += 2*math.pi
    # elevation theta: arcsin(z) in [-pi/2, pi/2]
    theta = math.asin(max(-1.0, min(1.0, v[2])))
    az_bin = min(n_az-1, int((phi / (2*math.pi)) * n_az))
    el_bin = min(n_el-1, int(((theta + math.pi/2) / math.pi) * n_el))
    return el_bin * n_az + az_bin

# ------------------------
# Featureization
# ------------------------

def extract_shapes_from_cdm_tuple(
    doc_id: str,
    cdm_tuple: tuple,
    weight_scale: Optional[float] = None,
    max_edges_per_doc: Optional[int] = None,
    min_weight: float = 0.0,
    include_length: bool = True,
    include_v_bin: bool = True,
    vbin_az: int = 12,
    vbin_el: int = 6,
    align_signs: bool = True,
    dir_weight_beta: float = 0.0,
) -> Tuple[List[dict], np.ndarray]:
    """Extract morphism "shape" records + feature matrix for a single document's 6-tuple.

    Returns
    -------
    rows : list of dicts with keys:
        ['doc_id','src_label','dst_label','w','len_delta','cos_vs','cos_vt_in','cos_st','vbin']
    X : np.ndarray of features (n_edges x d), using columns:
         [cos_vs, cos_vt_in, cos_st, log_len?] + one-hot(vbin)?
    """
    # Accept both legacy 6-tuples and quality-extended 7-tuples.
    (delta_matrix, cluster_order, seg_labels, cluster_topic_distributions,
     cluster_embeddings, cluster_dirs) = cdm_tuple[:6]
    cluster_quality = cdm_tuple[6] if isinstance(cdm_tuple, (tuple, list)) and len(cdm_tuple) >= 7 else None

    Delta = np.asarray(delta_matrix)
    E = np.asarray(cluster_embeddings)
    D = np.asarray(cluster_dirs)
    n, _, dim = Delta.shape

    # weight scale (robust) if not provided
    if weight_scale is None:
        # sample to estimate
        idx = np.arange(n)
        if n > 50:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=50, replace=False)
        norms = []
        for i in idx:
            for j in idx:
                if i==j: continue
                norms.append(float(np.linalg.norm(Delta[i, j])))
        med = np.median(norms) if norms else 1.0
        weight_scale = max(1e-8, float(med))

    # Collect edges
    rows: List[dict] = []
    feats: List[List[float]] = []

    # Optional downsample of edges per doc (avoid quadratic blow-up)
    # We will randomly sample up to K strongest by base weight
    # First create a candidate list then pick topK by base weight.
    candidates = []
    for i in range(n):
        for j in range(n):
            if i==j: continue
            dvec = Delta[i, j]
            ln = float(np.linalg.norm(dvec))
            base_w = math.exp(-ln / weight_scale)  # magnitude-based
            # optional directional weighting (beta=0 => none)
            s = _unit(D[i]); t = _unit(D[j]); v = _unit(dvec)
            if align_signs:
                v, s, t = _align_signs_to_flow(v, s, t)
            a1 = max(0.0, float(np.dot(v, s)))
            a2 = max(0.0, float(np.dot(-v, t)))
            a3 = abs(float(np.dot(s, t)))
            dir_score = (a1 * a2 * math.sqrt(a3))
            w = base_w * ((dir_score + 1e-12) ** float(dir_weight_beta))

            if w >= min_weight:
                candidates.append((w, i, j, ln, v, s, t))

    # If we need to cap edges, keep top-K by weight
    if max_edges_per_doc is not None and len(candidates) > max_edges_per_doc:
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:max_edges_per_doc]

    # Build rows and features
    # Feature core: Gram entries
    #   cos_vs = <v,s>, cos_vt_in = < -v, t >, cos_st = <s,t>
    # Optional: log_len
    # Optional: one-hot v direction bin
    n_az = int(vbin_az); n_el = int(vbin_el)
    nbins = n_az * n_el if include_v_bin else 0

    for (w, i, j, ln, v, s, t) in candidates:
        cos_vs = float(np.dot(v, s))
        cos_vt_in = float(np.dot(-v, t))  # "incoming" alignment at dst
        cos_st = float(np.dot(s, t))

        feat = [cos_vs, cos_vt_in, cos_st]
        if include_length:
            feat.append(math.log(ln + 1e-9))
        vbin = None
        if include_v_bin:
            vbin = _spherical_bin(v, n_az, n_el)
            # one-hot
            oh = [0.0]*nbins
            oh[vbin] = 1.0
            feat.extend(oh)

        src_label = int(cluster_order[i]) if not isinstance(cluster_order, list) else int(cluster_order[i])
        dst_label = int(cluster_order[j]) if not isinstance(cluster_order, list) else int(cluster_order[j])

        def _q_for(label):
            if isinstance(cluster_quality, dict):
                val = cluster_quality.get(label, cluster_quality.get(int(label), 1.0))
                if isinstance(val, dict):
                    val = val.get("quality", 1.0)
                try:
                    return float(max(0.0, min(1.0, val)))
                except Exception:
                    return 1.0
            return 1.0

        q_src = _q_for(src_label)
        q_dst = _q_for(dst_label)
        q_edge = min(q_src, q_dst)

        rows.append({
            "doc_id": str(doc_id),
            "src_label": src_label,
            "dst_label": dst_label,
            "w": float(w),
            "len_delta": float(ln),
            "cos_vs": cos_vs,
            "cos_vt_in": cos_vt_in,
            "cos_st": cos_st,
            "vbin": vbin,
            "src_quality": float(q_src),
            "dst_quality": float(q_dst),
            "edge_quality": float(q_edge),
        })
        feats.append(feat)

    X = np.asarray(feats, dtype=float) if feats else np.zeros((0, 3 + (1 if include_length else 0) + nbins), dtype=float)
    return rows, X


def extract_shapes_from_cdm_dict(
    document_delta_dict: Dict[str, tuple],
    **kwargs
) -> Tuple[List[dict], np.ndarray]:
    """Batch over a dict-of-6-tuples. Returns concatenated rows and X."""
    all_rows: List[dict] = []
    Xs = []
    for doc_id, tup in document_delta_dict.items():
        rows, X = extract_shapes_from_cdm_tuple(doc_id, tup, **kwargs)
        all_rows.extend(rows)
        if X.size:
            Xs.append(X)
    X_all = np.vstack(Xs) if Xs else np.zeros((0, 3), dtype=float)
    return all_rows, X_all


# ------------------------
# Clustering (shapes)
# ------------------------

def _kmeans_np(X: np.ndarray, k: int, n_iter: int = 30, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Simple numpy k-means with k-means++ init (robust to degenerate cases). Returns (labels, centroids)."""
    rng = np.random.default_rng(random_state)
    if X.ndim != 2:
        X = np.asarray(X, dtype=float).reshape(-1, X.shape[-1])
    n, d = X.shape

    # degenerate: no points
    if n == 0:
        return np.zeros((0,), dtype=int), np.zeros((max(1, k), d), dtype=float)

    # ensure k <= n and k >= 1
    k = max(1, min(int(k), n))

    # --- k-means++ initialization ---
    centroids = np.empty((k, d), dtype=float)

    # pick first center uniformly at random
    first_idx = rng.integers(0, n)
    centroids[0] = X[first_idx]

    # squared distances to nearest center so far
    dist_sq = np.sum((X - centroids[0]) ** 2, axis=1)

    for c in range(1, k):
        # guard numeric pathologies
        dist_sq = np.maximum(dist_sq, 0.0)
        total = float(dist_sq.sum())

        if (not np.isfinite(total)) or total <= 0.0:
            # all points identical to an existing center -> sample uniformly
            idx = rng.integers(0, n)
        else:
            probs = dist_sq / total
            s = float(probs.sum())
            if (not np.isfinite(s)) or s <= 0.0:
                idx = rng.integers(0, n)
            else:
                # re-normalize to be extra safe
                probs = probs / s
                # final guard: if still invalid, fallback uniform
                if (not np.isfinite(probs).all()) or probs.max() <= 0.0:
                    idx = rng.integers(0, n)
                else:
                    idx = rng.choice(n, p=probs)

        centroids[c] = X[idx]
        # update min dist squared to any center
        new_dist_sq = np.sum((X - centroids[c]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq)

    # --- Lloyd iterations ---
    labels = np.zeros(n, dtype=int)
    for it in range(max(1, int(n_iter))):
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)  # (n, k)
        new_labels = np.argmin(dists, axis=1)

        if it > 0 and np.all(new_labels == labels):
            break
        labels = new_labels

        # update centers; if a cluster is empty, re-seed it uniformly
        for c in range(k):
            mask = (labels == c)
            if np.any(mask):
                centroids[c] = X[mask].mean(axis=0)
            else:
                centroids[c] = X[rng.integers(0, n)]
    return labels, centroids


def cluster_shapes(
    X: np.ndarray,
    k: int = 64,
    method: str = "kmeans",
    random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster morphism shapes into k groups. Returns (labels, centroids)."""
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return np.zeros((0,), dtype=int), np.zeros((max(1, k), 0), dtype=float)

    n = X.shape[0]
    k_eff = max(1, min(int(k), n))

    if method == "kmeans":
        return _kmeans_np(X, k=k_eff, n_iter=50, random_state=random_state)

    # fallback
    return _kmeans_np(X, k=k_eff, n_iter=50, random_state=random_state)


# ------------------------
# Document membership over shape clusters
# ------------------------

def doc_membership(
    rows: List[dict],
    shape_labels: np.ndarray,
    weight_mode: str = "weighted",   # "weighted" | "count"
    normalize: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """Aggregate per-document memberships over the shape clusters.

    Returns
    -------
    M : np.ndarray shape (n_docs, k) document-by-shape matrix
    doc_ids : list[str]
    """
    # gather docs and K
    docs = sorted({r["doc_id"] for r in rows})
    doc_index = {d:i for i,d in enumerate(docs)}
    k = int(shape_labels.max()) + 1 if shape_labels.size else 0
    M = np.zeros((len(docs), k), dtype=float)

    for r, lab in zip(rows, shape_labels):
        i = doc_index[r["doc_id"]]
        if weight_mode == "weighted":
            w = float(r.get("w", 1.0))
            M[i, lab] += w
        else:
            M[i, lab] += 1.0

    if normalize and M.size:
        row_sums = M.sum(axis=1, keepdims=True) + 1e-12
        M = M / row_sums
    return M, docs


# ------------------------
# Hierarchical refinement
# ------------------------


def _normalized_entropy(P: np.ndarray, eps: float = 1e-12) -> float:
    """Mean per-doc normalized entropy over rows of P (each row should sum to 1)."""
    if P.size == 0 or P.shape[1] <= 1:
        return 0.0
    P = np.clip(P, eps, 1.0)
    H = -np.sum(P * np.log(P), axis=1) / np.log(P.shape[1])
    return float(np.mean(H))

def _kmeans_docs_np(M: np.ndarray, k: int, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper to run k-means on document membership matrix M (rows=docs)."""
    if M.shape[0] == 0:
        return np.zeros((0,), dtype=int), np.zeros((k, M.shape[1] if M.ndim == 2 else 0))
    from morphism_shapes import _kmeans_np  # reuse k-means
    labels, cents = _kmeans_np(M, k=min(k, max(1, M.shape[0])), n_iter=40, random_state=random_state)
    return labels, cents

def build_hierarchy_all(
    document_delta_dict: Dict[str, tuple],
    depth: int = 3,
    shape_k: int = 64,
    doc_k: int = 8,
    max_edges_per_doc: Optional[int] = 2000,
    min_weight: float = 0.0,
    include_length: bool = True,
    include_v_bin: bool = True,
    vbin_az: int = 12,
    vbin_el: int = 6,
    align_signs: bool = True,
    dir_weight_beta: float = 1.0,
    hetero_threshold: float = 0.45,   # split clusters with mean normalized entropy above this
    min_docs_to_split: int = 10,
    random_state: int = 0
) -> Dict[str, Any]:
    """
    Multi-branch hierarchy: at each level, every cluster that is sufficiently heterogeneous
    AND large enough will be split further, not just the largest cluster.

    Returns a tree-like dict with nodes and parent/children links.
    """
    rng = np.random.default_rng(random_state)

    all_docs = sorted(document_delta_dict.keys())
    nodes: List[Dict[str, Any]] = []
    node_id_counter = 0

    def new_node_id() -> int:
        nonlocal node_id_counter
        nid = node_id_counter
        node_id_counter += 1
        return nid

    def make_node(docs_subset: List[str], level: int, parent: Optional[int]) -> int:
        """Create a node for a subset of docs; cluster shapes then docs; decide children."""
        subset = {d: document_delta_dict[d] for d in docs_subset}
        from morphism_shapes import extract_shapes_from_cdm_dict, cluster_shapes, doc_membership

        rows, X = extract_shapes_from_cdm_dict(
            subset,
            max_edges_per_doc=max_edges_per_doc,
            min_weight=min_weight,
            include_length=include_length,
            include_v_bin=include_v_bin,
            vbin_az=vbin_az, vbin_el=vbin_el,
            align_signs=align_signs,
            dir_weight_beta=dir_weight_beta
        )

        if X.size == 0:
            # Degenerate: no edges
            nid = new_node_id()
            node = {
                "id": nid, "level": level, "parent": parent,
                "docs": docs_subset, "doc_ids": docs_subset,
                "doc_membership": np.zeros((len(docs_subset), 0)),
                "doc_labels": np.zeros((len(docs_subset),), dtype=int),
                "shape_centroids": np.zeros((0,0)),
                "children": [],
                "hetero": 0.0
            }
            print(f"[L{level}] docs={len(doc_ids)}  shapes={X.shape[0]}  shape_k={shape_k}")
            nodes.append(node)
            return nid
        
        shape_labels, centroids = cluster_shapes(X, k=shape_k, method="kmeans", random_state=random_state)
        M, doc_ids = doc_membership(rows, shape_labels, weight_mode="weighted", normalize=True)

        # Cluster docs at this node
        k_eff = min(doc_k, max(1, M.shape[0]))
        doc_labels, doc_cents = _kmeans_docs_np(M, k=k_eff, random_state=random_state)

        # Heterogeneity of this node (mean normalized entropy over its docs)
        hetero = _normalized_entropy(M)

        nid = new_node_id()
        node = {
            "id": nid, "level": level, "parent": parent,
            "docs": doc_ids, "doc_ids": doc_ids,
            "doc_membership": M, "doc_labels": doc_labels,
            "shape_centroids": centroids,
            "children": [],
            "hetero": hetero
        }
        nodes.append(node)

        # Decide splits for children (for next level)
        if level < depth - 1:
            K = int(doc_labels.max()) + 1 if doc_labels.size else 0
            for c in range(K):
                docs_c = [d for d, lab in zip(doc_ids, doc_labels) if lab == c]
                if len(docs_c) < min_docs_to_split:
                    continue
                # Heterogeneity within child: recompute entropy on this subset's M
                mask = np.array([lab == c for lab in doc_labels], dtype=bool)
                M_c = M[mask] if M.size and mask.any() else np.zeros((0, M.shape[1]))
                hetero_c = _normalized_entropy(M_c)

                if hetero_c >= hetero_threshold:
                    child_id = make_node(docs_c, level+1, nid)
                    node["children"].append(child_id)
                # else: do not split; cluster c becomes a leaf
        return nid

    root_id = make_node(all_docs, level=0, parent=None)

    # Build quick doc->path map (sequence of node IDs from root to leaf)
    doc_to_path: Dict[str, List[int]] = {}
    # index nodes by id
    by_id = {n["id"]: n for n in nodes}

    def walk(nid: int, path_prefix: List[int]):
        node = by_id[nid]
        path = path_prefix + [nid]
        if not node["children"]:
            for d in node["doc_ids"]:
                doc_to_path.setdefault(d, path)
        else:
            # propagate to children
            for child_id in node["children"]:
                walk(child_id, path)

    walk(root_id, [])

    return {"nodes": nodes, "root_id": root_id, "doc_to_path": doc_to_path}