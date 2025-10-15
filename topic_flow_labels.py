#!/usr/bin/python
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# ------------------------------------------------
#topic_flow_labels.py
#
#Derive human-readable labels for hierarchy nodes based on *morphism-driven topic flow*.
#
#For each node (subset of docs), we:
#  - Recompute intra-document morphism weights (magnitude + optional directional agreement)
#  - For each morphism i->j, retrieve topic distributions (T_i, T_j)
#  - Accumulate a weighted topic flow matrix F[u,v] ≈ sum_morphisms w * T_i[u] * T_j[v]
#  - Score pairs by LIFT or PMI against marginals to surface characteristic transitions
#  - Render labels using interpretive topic text (lda_int_topics_list) when available
#
#Outputs:
#  - node_labels: dict[node_id] -> short label string (e.g., "Labor → Strike; Immigration → Policy")
#  - node_details: dict[node_id] -> dict with full top pairs and stats
#  - tree updated in place: each node gets 'label' and 'topic_flow' (top pairs) fields
#
#Works with:
#  - document_delta_dict: dict[doc_id] -> (delta_matrix, cluster_order, labels,
#                                         cluster_topic_distributions, cluster_embeddings, cluster_dirs)
#  - tree: result of build_hierarchy_all(...)

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import math
from collections import defaultdict


# -----------------------------------------------------------------------------
# def _unit(x: np.ndarray, eps: float=1e-12) -> np.ndarray:
# Summary: 
#   Safely L2‑normalize a vector (returns zeros if near‑zero length).
# Effect: All angle/cosine computations in morphism weighting depend on unit
#      directions; this guard prevents numerical blow‑ups when a vector has
#      negligible norm.
# -----------------------------------------------------------------------------
def _unit(x: np.ndarray, eps: float=1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = float(np.linalg.norm(x))
    if n < eps:
        return np.zeros_like(x, dtype=float)
    return x / n

# ----------------------------------------------------------------------------
# def _align_signs_to_flow(v: np.ndarray, s: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# Summary: 
#   Canonicalize the ambiguous signs of per‑cluster PC1 directions
#   relative to the morphism’s flow direction v.
#   Given (v, s, t) where s=PC1(src), t=PC1(dst), it flips s to
#   point WITH v and flips t to point AGAINST v when needed.
# Effect: 
#   PCA sign is arbitrary. Aligning s/t to the edge direction
#   makes source/destination alignment comparable across docs,
#   stabilizing directional weights and the topic‑flow signal.
# -----------------------------------------------------------------------------
def _align_signs_to_flow(v: np.ndarray, s: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s2 = np.copy(s)
    t2 = np.copy(t)
    if np.dot(s2, v) < 0: s2 = -s2
    if np.dot(t2, -v) < 0: t2 = -t2
    return v, s2, t2

# ----------------------------------------------------------------------------
# def _topic_name(idx: int, topic_labels: Optional[List[str]]) -> str:
# Summary: 
#   Map a topic index to a human‑readable label if provided; else fall
#   back to 'topic_<idx>'. Used only for rendering labels.
# -----------------------------------------------------------------------------
def _topic_name(idx: int, topic_labels: Optional[List[str]]) -> str:
    if topic_labels is not None and 0 <= idx < len(topic_labels):
        return str(topic_labels[idx])
    return f"topic_{idx}"

# -----------------------------------------------------------------------------
# def _compute_morphism_edges_for_docs
# Summary: 
#   Build a flat, weighted list of intra‑document morphism edges across
#   a set of documents. Each edge is i→j within a document and carries
#   a weight 'w' that encodes both delta magnitude and (optionally)
#   directional agreement.
# Effect:
#   • For each doc, iterate all ordered cluster pairs (i≠j).
#   • Base weight = exp(−‖Δ[i,j]‖ / scale). If 'weight_scale' is not given,
#     a robust per‑doc scale is estimated from the median of a sample of delta
#     norms so that lengths are comparable without being dominated by outliers.
#   • Directional weighting (when cluster principal directions are present):
#       v = unit(Δ[i,j]); s = unit(PC1_i); t = unit(PC1_j); optionally align
#       signs so s points with v and t against v. The agreement terms are:
#         a1 = max(0, ⟨v, s⟩)              (how src points along the flow)
#         a2 = max(0, ⟨−v, t⟩)             (how dst receives the flow)
#         a3 = |⟨s, t⟩|                     (how consistent src/dst axes are)
#       dir_score = a1 * a2 * sqrt(a3); final weight = base * dir_score^β
#     'dir_weight_beta' controls how strongly directionality shapes the weights.
#   • Optionally cap to the top‑K edges per doc by weight to keep computation
#     tractable.
# Output: A list of dicts like
#     {'doc_id','i','j','w','src_label','dst_label'}
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   These weights quantify which within‑doc transformations
#   (morphisms) are most semantically coherent and therefore most informative
#   for cross‑doc structure. They are the measure that drives topic‑flow and
#   ultimately the labels shown on hierarchy nodes.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# -----------------------------------------------------------------------------
def _compute_morphism_edges_for_docs(
    document_delta_dict: Dict[str, tuple],
    doc_ids: List[str],
    weight_scale: Optional[float] = None,
    dir_weight_beta: float = 1.0,
    align_signs: bool = True,
    max_edges_per_doc: Optional[int] = None
) -> List[dict]:
    """
    Build a flat list of morphism edges across the given docs.
    Each edge row: {'doc_id','i','j','w','src_label','dst_label'}
    """
    edges: List[dict] = []
    for doc_id in doc_ids:
        (Delta, cluster_order, seg_labels, topic_dists, E, Dirs) = document_delta_dict[doc_id]
        Delta = np.asarray(Delta); E = np.asarray(E); D = np.asarray(Dirs)
        n = Delta.shape[0]
        # weight scale per doc (median of norms) if not provided
        if weight_scale is None:
            norms = []
            idx = np.arange(n)
            if n > 50:
                rng = np.random.default_rng(0)
                idx = rng.choice(n, size=50, replace=False)
            for a in idx:
                for b in idx:
                    if a==b: continue
                    norms.append(float(np.linalg.norm(Delta[a,b])))
            med = np.median(norms) if norms else 1.0
            sc = max(1e-8, float(med))
        else:
            sc = float(weight_scale)

        # collect candidates
        cands = []
        for i in range(n):
            for j in range(n):
                if i==j: continue
                dvec = Delta[i,j]; ln = float(np.linalg.norm(dvec))
                base_w = math.exp(-ln / sc)
                w = base_w
                # directional weighting
                if D is not None and D.size:
                    v = _unit(dvec); s = _unit(D[i]); t = _unit(D[j])
                    if align_signs: v,s,t = _align_signs_to_flow(v,s,t)
                    a1 = max(0.0, float(np.dot(v, s)))
                    a2 = max(0.0, float(np.dot(-v, t)))
                    a3 = abs(float(np.dot(s, t)))
                    dir_score = (a1 * a2 * math.sqrt(a3))
                    w *= (dir_score + 1e-12)**float(dir_weight_beta)
                cands.append((w, i, j, ln))
        # cap edges per doc by weight if requested
        if max_edges_per_doc is not None and len(cands) > max_edges_per_doc:
            cands.sort(key=lambda x: x[0], reverse=True)
            cands = cands[:max_edges_per_doc]
        # append rows
        for w,i,j,ln in cands:
            edges.append({
                "doc_id": doc_id,
                "i": i, "j": j,
                "w": float(w),
                "src_label": int(cluster_order[i]) if not isinstance(cluster_order, list) else int(cluster_order[i]),
                "dst_label": int(cluster_order[j]) if not isinstance(cluster_order, list) else int(cluster_order[j])
            })
    return edges


# -----------------------------------------------------------------------------
# def _topic_flow_from_edges
# Summary: 
#   Aggregate a global topic‑transition matrix F[u,v] from the weighted
#   morphism edges.
# Effect:
#   • For each edge i→j with weight w, fetch the topic distributions T_i and T_j
#     for the source/destination clusters (per‑doc).
#   • To keep the signal sparse and readable, restrict to the top‑k topics on
#     each side; renormalize those slices; then add w * (T_i⊗T_j) into F using
#     only those indices.
#
#   F encodes how topics tend to move from source → destination
#   across the morphisms that define your manifold. It turns many local edges
#   into a compact, interpretable summary of “topic flow” for any subset of docs.
# -----------------------------------------------------------------------------
def _topic_flow_from_edges(
    edges: List[dict],
    topic_dists_by_label: Dict[int, np.ndarray],
    num_topics: Optional[int] = None,
    top_topic_k: int = 3,
) -> np.ndarray:
    """
    Build a topic flow matrix F[u,v] by summing w * T_i[u] * T_j[v] over edges.
    To keep it sparse and interpretable, use only top-k topics of source and dest.
    """
    # infer num_topics from first dist if not given
    if num_topics is None:
        any_d = next(iter(topic_dists_by_label.values()))
        num_topics = int(np.asarray(any_d).shape[0])
    F = np.zeros((num_topics, num_topics), dtype=float)
    for e in edges:
        Ti = np.asarray(topic_dists_by_label.get(e["src_label"]))
        Tj = np.asarray(topic_dists_by_label.get(e["dst_label"]))
        if Ti is None or Tj is None: continue
        # top-k indices
        src_idx = np.argsort(Ti)[-top_topic_k:]
        dst_idx = np.argsort(Tj)[-top_topic_k:]
        # normalized sub-dists (softmax on top-k slice)
        s = Ti[src_idx] + 1e-12; s = s / s.sum()
        t = Tj[dst_idx] + 1e-12; t = t / t.sum()
        # outer product scaled by morphism weight
        F[np.ix_(src_idx, dst_idx)] += e["w"] * (s[:,None] * t[None,:])
    return F


# ----------------------------------------------------------------------------
# def _top_pairs_from_flow
# Summary: 
#   Score and extract the most characteristic topic transitions (u→v)
#   from a flow matrix F.
# Effect:
#   • Compute P = F / sum(F). Then select a scoring scheme:
#       - 'mass' : just P[u,v] (frequency of the pair).
#       - 'lift' : P[u,v] / (P[u,*]·P[*,v]) → highlights transitions that occur
#                  more often than expected from marginals (good for labels).
#       - 'pmi'  : log P[u,v] − log P[u,*] − log P[*,v] (pointwise mutual info).
#   • Return the top 'top_pairs' (u,v,score) entries (diagonals allowed).
#
#     Lift/PMI favor distinctive transitions over merely common
#     ones, producing concise labels that capture *what’s special* about a node,
#     not just what’s frequent.
# -----------------------------------------------------------------------------
def _top_pairs_from_flow(
    F: np.ndarray,
    top_pairs: int = 4,
    score: str = "lift"   # 'lift' | 'pmi' | 'mass'
) -> List[Tuple[int,int,float]]:
    """Return a list of (u,v,score) for the top topic transitions."""
    eps = 1e-9
    mass = F.sum() + eps
    if score == "mass":
        S = F / mass
    else:
        row = F.sum(axis=1, keepdims=True) + eps
        col = F.sum(axis=0, keepdims=True) + eps
        P = F / mass
        if score == "lift":
            S = P / (row/ mass * col / mass + eps)
        elif score == "pmi":
            S = np.log(P + eps) - np.log(row / mass + eps) - np.log(col / mass + eps)
        else:
            S = P
    # get top pairs (avoid diagonal if you prefer; here we allow u==v)
    flat_idx = np.argsort(S.ravel())[::-1]
    pairs = []
    for idx in flat_idx:
        if len(pairs) >= top_pairs: break
        u = idx // S.shape[1]
        v = idx %  S.shape[1]
        if S[u,v] <= 0: continue
        pairs.append((int(u), int(v), float(S[u,v])))
    return pairs


# ----------------------------------------------------------------------------
# def _render_label
# Summary: 
#   Turn the selected (topic_u, topic_v) pairs into a short, human‑readable
#   label like “Labor → Strike; Immigration → Policy”, optionally using
#   supplied topic names.
# Effect: 
#   Formats up to three pairs and truncates to keep labels legible on
#   plots. These labels make the morphism‑driven structure immediately
#   interpretable to humans.
# -----------------------------------------------------------------------------
def _render_label(
    pairs: List[Tuple[int,int,float]],
    topic_labels: Optional[List[str]] = None,
    max_len: int = 42
) -> str:
    if not pairs: return ""
    parts = []
    for (u,v,sc) in pairs:
        su = _topic_name(u, topic_labels)
        sv = _topic_name(v, topic_labels)
        parts.append(f"{su} → {sv}")
    s = "; ".join(parts[:3])
    if len(s) > max_len:
        s = s[:max_len-1] + "…"
    return s


# ----------------------------------------------------------------------------
# def annotate_tree_with_topic_flow 
# Summary: 
#   Given the hierarchy produced by morphism‑shape clustering, compute a
#   topic‑flow label for every node and attach both the short label and
#   the underlying data to the node.
# Effect:
#   1) For each node, collect its member docs and build the weighted edge set
#      via _compute_morphism_edges_for_docs (using the same directionality
#      conventions as elsewhere).
#   2) Merge the per‑doc cluster topic distributions into a local lookup table
#      (labels are doc‑local, but edges reference the correct ones).
#   3) Build F with _topic_flow_from_edges and extract top (u→v) pairs using
#      _top_pairs_from_flow (default score='lift').
#   4) Render a short string with _render_label and write it to node['label'];
#      also attach a compact 'topic_flow' payload (pairs, num_topics, shape).
#
#   This is the final interpretability bridge. It transforms
#   low‑level morphism geometry into semantically meaningful, human‑readable
#   labels that explain *why* a cluster of documents hangs together at each
#   level of the hierarchy.
#
#   'dir_weight_beta' lets you keep the labeling consistent with whatever
#     morphism weighting you used to form the hierarchy.
#   'top_topic_k' sparsifies F for clarity; 'top_pairs' controls label size.
#   'max_edges_per_doc' keeps large corpora tractable by pruning candidates.
# -----------------------------------------------------------------------------
def annotate_tree_with_topic_flow(
    document_delta_dict: Dict[str, tuple],
    tree: Dict[str, Any],
    lda_topic_labels: Optional[List[str]] = None,
    dir_weight_beta: float = 1.0,
    weight_scale: Optional[float] = None,
    top_topic_k: int = 3,
    top_pairs: int = 4,
    max_edges_per_doc: Optional[int] = 2000
) -> Dict[int, Dict[str, Any]]:
    """
    Compute and attach topic-flow labels for each node in the tree.
    Returns node_details: node_id -> {'label': str, 'pairs': [(u,v,score)], 'flow': ndarray}
    """
    by_id = {n["id"]: n for n in tree.get("nodes", [])}
    node_details: Dict[int, Dict[str, Any]] = {}

    # Pre-extract topic distributions per doc's label -> array (they are per-doc; but we can reuse dict per doc)
    # Each document has its own cluster labels; we will query per-edge using that doc's dict.
    for nid, node in by_id.items():
        docs = list(node.get("doc_ids", []))
        if not docs:
            node["label"] = ""
            node_details[nid] = {"label": "", "pairs": [], "flow": None}
            continue

        # Collect edges in this node (weighted with directional agreement if desired)
        edges = _compute_morphism_edges_for_docs(
            document_delta_dict,
            docs,
            weight_scale=weight_scale,
            dir_weight_beta=dir_weight_beta,
            align_signs=True,
            max_edges_per_doc=max_edges_per_doc
        )

        # Build topic_dists_by_label for *this* node: union over docs (latest wins; labels are doc-local)
        # In your tuples, cluster_topic_distributions is per-doc dict[label]->array; labels are indices local to doc clusters.
        topic_dists_by_label = {}
        # Choose one doc to source each label's dist when encountered; since labels are local per doc,
        # we only use them when paired with edges from same doc (safe).
        for d in docs:
            tup = document_delta_dict[d]
            tdict = tup[3]
            for k,v in tdict.items():
                if k not in topic_dists_by_label:
                    topic_dists_by_label[int(k)] = np.asarray(v, dtype=float)

        if not edges:
            node["label"] = ""
            node_details[nid] = {"label": "", "pairs": [], "flow": None}
            continue

        # Infer num_topics and compute flow matrix
        any_td = next(iter(topic_dists_by_label.values()))
        num_topics = int(np.asarray(any_td).shape[0])
        F = _topic_flow_from_edges(edges, topic_dists_by_label, num_topics=num_topics, top_topic_k=top_topic_k)

        pairs = _top_pairs_from_flow(F, top_pairs=top_pairs, score="lift")
        label = _render_label(pairs, topic_labels=lda_topic_labels, max_len=56)

        node["label"] = label
        node["topic_flow"] = {
            "pairs": pairs,
            "num_topics": num_topics,
            "flow_matrix_shape": (int(F.shape[0]), int(F.shape[1]))
        }
        node_details[nid] = {"label": label, "pairs": pairs, "flow": F}

    return node_details
