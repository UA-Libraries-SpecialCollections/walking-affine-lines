
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional, Iterable
import numpy as np

from global_categorical_themes import Cluster, Morphism, Transform

def _as_np(a):
    if a is None: return None
    if isinstance(a, np.ndarray): return a
    return np.asarray(a)

def _weight_from_delta(delta: np.ndarray, scale: float = 1.0) -> float:
    n = float(np.linalg.norm(delta)); return float(np.exp(-n / max(1e-8, scale)))

def morphisms_from_cluster_delta_matrix_result(result: Any, clusters: Dict[str, Cluster]) -> List[Morphism]:
    morphisms: List[Morphism] = []
    # Tuple variant: (delta, cluster_order, labels, topic_dists, embeddings, dirs)
    if isinstance(result, (list, tuple)) and len(result)>=2:
        delta = result[0]
        if isinstance(delta, np.ndarray) and delta.ndim==3:
            n = delta.shape[0]; W = None
            ids = list(clusters.keys())
            for i in range(n):
                for j in range(n):
                    if i==j: continue
                    src = ids[i]; dst = ids[j]
                    dvec = delta[i,j]
                    # weight will be derived later with a scale
                    morphisms.append(Morphism(src=src, dst=dst, weight=1.0, transform=Transform(delta=dvec),
                                              meta={'source':'cluster_delta_matrix'}))
            return morphisms
    # DataFrame or dict cases omitted for brevity in this project-specific adapter.
    return morphisms


def adapt_from_cdm_dict(
    document_delta_dict: Dict[str, tuple],
    topic_labels: Optional[Iterable[str]] = None,
    weight_scale: Optional[float] = None,   # if given, passed through to per-doc calls
    cluster_id_prefix: str = "cl_",
) -> Tuple[Dict[str, Cluster], List[Morphism]]:
    """
    Accept a dict mapping doc_id -> 6-tuple:
      (delta_matrix, cluster_order, labels, cluster_topic_distributions, cluster_embeddings, cluster_dirs)

    Builds a global (clusters, morphisms) by concatenating all documents.
    Cluster IDs are prefixed with the doc_id to avoid collisions.
    """
    clusters_all: Dict[str, Cluster] = {}
    morphisms_all: List[Morphism] = []

    # normalize topic labels once
    topic_labels = list(topic_labels) if topic_labels is not None else None

    for doc_id, six_tuple in document_delta_dict.items():
        # map every cluster label in this doc to this doc_id
        # (so we don't need segment_doc_ids here)
        # We also prefix the cluster_id with the doc_id to keep them unique.
        # NOTE: pass weight_scale through; if None, per-doc robust median is used.
        # Build a per-doc mapping (label -> doc_id) after we peek cluster_order
        try:
            _, cluster_order, _, _, _, _ = six_tuple
        except Exception as e:
            raise ValueError(f"Document entry for {doc_id} is not a 6-tuple") from e

        doc_map = {int(lbl): str(doc_id) for lbl in (list(cluster_order) if not isinstance(cluster_order, list) else cluster_order)}

        sub_clusters, sub_morphisms = adapt_from_cdm_tuple(
            six_tuple,
            doc_ids_by_label=doc_map,
            topic_labels=topic_labels,
            weight_scale=weight_scale,
            cluster_id_prefix=f"{doc_id}|{cluster_id_prefix}",
        )
        # merge
        clusters_all.update(sub_clusters)
        morphisms_all.extend(sub_morphisms)

    return clusters_all, morphisms_all


def adapt_from_cdm(
    document_delta_dict: Any,
    **kwargs
) -> Tuple[Dict[str, Cluster], List[Morphism]]:
    """
    Flexible entry:
      - if a dict -> assumes dict[doc_id] = 6-tuple and delegates to adapt_from_cdm_dict
      - if a tuple/list -> delegates to adapt_from_cdm_tuple
    """
    if isinstance(document_delta_dict, dict):
        return adapt_from_cdm_dict(document_delta_dict, **kwargs)
    elif isinstance(document_delta_dict, (list, tuple)):
        return adapt_from_cdm_tuple(document_delta_dict, **kwargs)
    else:
        raise TypeError("document_delta_dict must be dict (doc_id -> 6-tuple) or a 6-tuple")


def adapt_from_cdm_tuple(
    document_delta_dict,
    segment_doc_ids: Optional[Iterable[str]] = None,
    doc_ids_by_label: Optional[Dict[int, str]] = None,
    topic_labels: Optional[Iterable[str]] = None,
    weight_scale: Optional[float] = None,
    cluster_id_prefix: str = "cl_",
) -> Tuple[Dict[str, Cluster], List[Morphism]]:
    (delta_matrix, cluster_order, seg_labels, cluster_topic_distributions,
     cluster_embeddings, cluster_dirs) = document_delta_dict
    Delta = _as_np(delta_matrix); E = _as_np(cluster_embeddings); Dirs = _as_np(cluster_dirs)
    n = E.shape[0]
    if Delta.shape[0] != n or Delta.shape[1] != n:
        raise ValueError(f"delta_matrix shape {Delta.shape} incompatible with cluster_embeddings shape {E.shape}")
    # Build doc id per cluster label
    label_to_doc = {}
    if doc_ids_by_label is not None:
        label_to_doc = {int(k): str(v) for k, v in doc_ids_by_label.items()}
    elif segment_doc_ids is not None:
        seg_labels = list(seg_labels); segment_doc_ids = list(segment_doc_ids)
        if len(segment_doc_ids) != len(seg_labels):
            raise ValueError("segment_doc_ids length must match labels length when provided.")
        from collections import Counter, defaultdict
        buckets = defaultdict(list)
        for seg_idx, lab in enumerate(seg_labels):
            buckets[int(lab)].append(segment_doc_ids[seg_idx])
        for lab, docs in buckets.items():
            doc = Counter(docs).most_common(1)[0][0]
            label_to_doc[int(lab)] = str(doc)
    else:
        for lab in cluster_order:
            label_to_doc[int(lab)] = f"doc_{int(lab)}"
    # Build clusters with meta
    clusters: Dict[str, Cluster] = {}
    # If topic labels provided, convert cluster top topics into 'top_terms' using provided names
    topic_labels = list(topic_labels) if topic_labels is not None else None
    for idx, lab in enumerate(cluster_order):
        cid = f"{cluster_id_prefix}{int(lab)}"
        doc_id = label_to_doc.get(int(lab), f"doc_{int(lab)}")
        centroid = E[idx]
        meta = {'label': int(lab)}
        tw = cluster_topic_distributions.get(int(lab))
        if tw is not None:
            tw = _as_np(tw)
            meta['topic_weights'] = tw
            topk = min(6, len(tw)) if tw.ndim == 1 else 6
            top_idx = np.argsort(tw)[::-1][:topk].tolist() if tw.ndim == 1 else []
            meta['top_topics'] = top_idx
            if topic_labels is not None:
                meta['top_terms'] = [str(topic_labels[i]) for i in top_idx if i < len(topic_labels)]
        if Dirs is not None and idx < len(Dirs):
            meta['dir'] = Dirs[idx]
        clusters[cid] = Cluster(id=cid, doc_id=str(doc_id), centroid=centroid, size=1, meta=meta)
    # Build morphisms from Delta with robust weight scaling
    if weight_scale is None:
        nmax = min(n, 200)
        rng = np.random.RandomState(0)
        idxs = rng.choice(n, size=nmax, replace=False)
        norms = []
        for i in idxs:
            for j in idxs:
                if i==j: continue
                norms.append(float(np.linalg.norm(Delta[i, j])))
        med = np.median(norms) if norms else 1.0
        weight_scale = max(1e-8, med)
    morphisms: List[Morphism] = []
    ids = [f"{cluster_id_prefix}{int(lab)}" for lab in cluster_order]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            src = ids[i]; dst = ids[j]
            dvec = Delta[i, j]
            w = float(np.exp(-float(np.linalg.norm(dvec)) / weight_scale))
            morphisms.append(Morphism(src=src, dst=dst, weight=w, transform=Transform(delta=dvec),
                                      meta={'source':'cdm_tuple'}))
    return clusters, morphisms


# -------- mk_delta_manifold adapter (best-effort, uses common patterns) --------

def _choose_doc_id_from_cluster_id(cluster_id: str) -> str:
    for delim in ('|','::','__',':','/'):
        if delim in cluster_id: return cluster_id.split(delim)[0]
    return cluster_id

def adapt_from_mkdelta_object(dm) -> Tuple[Dict[str, Cluster], List[Morphism]]:
    """
    Attempt to adapt a mk_delta_manifold(...) return object into (clusters, morphisms).
    Supports likely shapes:
      - dm.clusters (dict or list) with fields id, doc_id, centroid[, size, meta]
      - dm.cluster_embeddings + dm.cluster_ids (or dm.cluster_index) [+ dm.cluster_meta]
      - dm.morphisms / dm.edges (records with src,dst, delta|A,b [,weight])
      - dm.cluster_delta_matrix -> will fall back to all pair deltas if needed
    """
    clusters: Dict[str, Cluster] = {}

    # 1) clusters mapping or list
    mapping = getattr(dm, 'clusters', None) or (dm.get('clusters') if isinstance(dm, dict) else None)
    if mapping is not None:
        if isinstance(mapping, dict):
            it = mapping.items()
        else:
            it = [(getattr(c,'id', None) or c.get('id'), c) for c in mapping]
        for cid, obj in it:
            cid_s = str(cid)
            if isinstance(obj, dict):
                doc_id = obj.get('doc_id') or _choose_doc_id_from_cluster_id(cid_s)
                centroid = _as_np(obj.get('centroid'))
                size = int(obj.get('size', 1)); meta = obj.get('meta', {})
            else:
                doc_id = getattr(obj, 'doc_id', None) or _choose_doc_id_from_cluster_id(cid_s)
                centroid = _as_np(getattr(obj, 'centroid', None))
                size = int(getattr(obj, 'size', 1)); meta = getattr(obj, 'meta', {})
            if centroid is None:
                raise ValueError(f"Cluster {cid_s} missing centroid.")
            clusters[cid_s] = Cluster(id=cid_s, doc_id=str(doc_id), centroid=centroid, size=size, meta=meta)
    else:
        # 2) embeddings + ids
        E = getattr(dm, 'cluster_embeddings', None)
        if E is None: raise ValueError("mk_delta_manifold object lacks clusters.")
        ids = getattr(dm, 'cluster_ids', None)
        if ids is None:
            ci = getattr(dm, 'cluster_index', None)
            if isinstance(ci, dict):
                # assume id->idx
                ids = [None]*len(E)
                for k,v in ci.items():
                    ids[int(v)] = str(k)
        if ids is None:
            ids = [f"c{i}" for i in range(len(E))]
        meta_list = getattr(dm, 'cluster_meta', None)
        for i, cid in enumerate(ids):
            doc_id = _choose_doc_id_from_cluster_id(str(cid))
            centroid = _as_np(E[i])
            meta = meta_list[i] if isinstance(meta_list, (list,tuple)) and i<len(meta_list) else {}
            clusters[str(cid)] = Cluster(id=str(cid), doc_id=str(doc_id), centroid=centroid, size=int(meta.get('size',1)), meta=meta)

    # Morphisms
    morphisms: List[Morphism] = []
    edges = getattr(dm, 'morphisms', None) or getattr(dm, 'edges', None) or getattr(dm, 'cluster_morphisms', None)
    if edges is None and isinstance(dm, dict):
        edges = dm.get('morphisms') or dm.get('edges') or dm.get('cluster_morphisms')
    if edges is not None:
        for rec in edges:
            if isinstance(rec, dict):
                src = rec.get('src') or rec.get('source'); dst = rec.get('dst') or rec.get('target')
                delta = rec.get('delta'); A = rec.get('A'); b = rec.get('b'); w = rec.get('weight') or rec.get('score')
            else:
                src = getattr(rec, 'src', None) or getattr(rec, 'source', None)
                dst = getattr(rec, 'dst', None) or getattr(rec, 'target', None)
                delta = getattr(rec, 'delta', None); A = getattr(rec, 'A', None); b = getattr(rec, 'b', None)
                w = getattr(rec, 'weight', None) or getattr(rec, 'score', None)
            if src is None or dst is None: continue
            T = Transform(delta=_as_np(delta)) if delta is not None else Transform(A=_as_np(A) if A is not None else None,
                                                                                   b=_as_np(b) if b is not None else None)
            if w is None and T.delta is not None:
                w = _weight_from_delta(T.delta, scale=1.0)
            morphisms.append(Morphism(src=str(src), dst=str(dst), weight=float(w or 1.0), transform=T,
                                      meta=rec if isinstance(rec, dict) else {}))
    else:
        # derive from cluster_delta_matrix if present
        cdm = getattr(dm, 'cluster_delta_matrix', None)
        if cdm is not None:
            morphisms = morphisms_from_cluster_delta_matrix_result(cdm, clusters)
        else:
            # all-pairs from centroids (last resort)
            ids = list(clusters.keys())
            E = np.stack([clusters[cid].centroid for cid in ids], axis=0)
            for i, src in enumerate(ids):
                for j, dst in enumerate(ids):
                    if i==j: continue
                    delta = E[j]-E[i]
                    w = _weight_from_delta(delta)
                    morphisms.append(Morphism(src=src, dst=dst, weight=w, transform=Transform(delta=delta)))
    return clusters, morphisms
