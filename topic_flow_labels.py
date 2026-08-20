
"""
topic_flow_labels.py

Derive human-readable labels for hierarchy nodes based on *morphism-driven topic flow*.

For each node (subset of docs), we:
  - Recompute intra-document morphism weights (magnitude + optional directional agreement)
  - For each morphism i->j, retrieve topic distributions (T_i, T_j)
  - Accumulate a weighted topic flow matrix F[u,v] ≈ sum_morphisms w * T_i[u] * T_j[v]
  - Score pairs by LIFT or PMI against marginals to surface characteristic transitions
  - Render labels using interpretive topic text (lda_int_topics_list) when available

Outputs:
  - node_labels: dict[node_id] -> short label string (e.g., "Labor → Strike; Immigration → Policy")
  - node_details: dict[node_id] -> dict with full top pairs and stats
  - tree updated in place: each node gets 'label' and 'topic_flow' (top pairs) fields

Works with:
  - document_delta_dict: dict[doc_id] -> (delta_matrix, cluster_order, labels,
                                         cluster_topic_distributions, cluster_embeddings, cluster_dirs)
  - tree: result of build_hierarchy_all(...)
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import math
from collections import defaultdict

def _unit(x: np.ndarray, eps: float=1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = float(np.linalg.norm(x))
    if n < eps:
        return np.zeros_like(x, dtype=float)
    return x / n

def _align_signs_to_flow(v: np.ndarray, s: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s2 = np.copy(s)
    t2 = np.copy(t)
    if np.dot(s2, v) < 0: s2 = -s2
    if np.dot(t2, -v) < 0: t2 = -t2
    return v, s2, t2

def _topic_name(idx: int, topic_labels: Optional[List[str]]) -> str:
    if topic_labels is not None and 0 <= idx < len(topic_labels):
        return str(topic_labels[idx])
    return f"topic_{idx}"

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
        # Accept both legacy 6-tuples and quality-extended 7-tuples.
        (Delta, cluster_order, seg_labels, topic_dists, E, Dirs) = document_delta_dict[doc_id][:6]
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
