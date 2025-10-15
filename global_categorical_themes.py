#!/usr/bin/python
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# =============================================================================
# global_categorical_themes.py
#
# the code organizes *cluster morphisms* (semantic
# transformations between cluster centroids) into dataset‑level
# categorical themes. These comments explain how each function works
# and why its features support "Embedding Manifolds as Semantic
# Morphisms"—turning local directional relations in text embeddings
# into global, interpretable categories.
# =============================================================================

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Callable, Any, Set
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Cluster:
    id: str
    doc_id: str
    centroid: np.ndarray
    size: int = 1
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Transform:
    A: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    delta: Optional[np.ndarray] = None
    @staticmethod
    
    # -----------------------------------------------------------------------------
    # Transform.identity(dim: int) -> Transform
    # Summary:
    #   Returns an identity affine transform (A = I_dim, b = 0). Used as a neutral
    #   map when no transformation is learned, and as a safe default in composition
    #   or inversion logic.
    # Effect:
    #   Provides a well‑defined “do nothing” morphism so pipelines can uniformly
    #   apply/compose transforms while preserving semantics and numerical stability.
    # -----------------------------------------------------------------------------
    def identity(dim: int) -> "Transform":
        return Transform(A=np.eye(dim), b=np.zeros(dim))
        
    # -----------------------------------------------------------------------------
    # Transform.apply_forward(self, x: np.ndarray) -> np.ndarray
    # Summary:
    #   Applies this Transform to vector x. If `delta` is set, treat the transform
    #   as a pure translation (x + delta), which matches the “edge direction”
    #   semantics of morphisms. Otherwise, apply the affine map A @ x + b, with
    #   missing components defaulting to identity/zeros.
    # Effect:
    #   Lets us *move* centroids/embeddings along learned morphisms so clusters from
    #   different docs can be compared within a common local frame, enabling
    #   category construction from directed relations.
    # -----------------------------------------------------------------------------
    def apply_forward(self, x: np.ndarray) -> np.ndarray:
        if self.delta is not None:
            return x + self.delta
        if self.A is None and self.b is None:
            return x
        A = self.A if self.A is not None else np.eye(len(x))
        b = self.b if self.b is not None else np.zeros_like(x)
        return A @ x + b
    
    # -----------------------------------------------------------------------------
    # Transform.inverse(self) -> Transform
    # Summary:
    #   Computes an inverse transform. For translations, just negates `delta`.
    #   For affine forms, uses the Moore–Penrose pseudoinverse of A so the method
    #   remains robust even when A is singular/ill‑conditioned; b is inverted as
    #   b_inv = −A_inv @ b (or handled as pure translation when A is None).
    # Effect:
    #   Inverting morphisms is essential for *pulling back* points (e.g. target
    #   centroids) into the seed’s coordinate frame to build theme prototypes that
    #   are consistent and comparable across documents.
    # -----------------------------------------------------------------------------
    def inverse(self) -> "Transform":
        if self.delta is not None:
            return Transform(delta=-self.delta)
        if self.A is None and self.b is None:
            return Transform()
        A = self.A if self.A is not None else None
        b = self.b if self.b is not None else None
        if A is None:
            return Transform(A=np.eye(len(b)), b=-b)
        A_inv = np.linalg.pinv(A)
        b_inv = -A_inv @ (b if b is not None else np.zeros(A.shape[0]))
        return Transform(A=A_inv, b=b_inv)
        
    # -----------------------------------------------------------------------------
    # Transform.compose(self, other: Transform) -> Transform
    # Summary:
    #   Returns the composition T = self ∘ other as a single affine map. The method
    #   normalizes both operands to (A, b) form (respecting pure deltas), infers
    #   dimensionality, then computes: A = A1 @ A0 and b = A1 @ b0 + b1.
    # Effect:
    #   Enables multi‑step semantic transitions (e.g., two‑hop morphisms across a
    #   document’s cluster graph) to be collapsed into a single, analyzable map,
    #   capturing *transitive* structure in the manifold.
    # -----------------------------------------------------------------------------
    def compose(self, other: "Transform") -> "Transform":
        def as_affine(T: "Transform", dim: int):
            if T.delta is not None: return np.eye(dim), T.delta
            A = T.A if T.A is not None else np.eye(dim)
            b = T.b if T.b is not None else np.zeros(dim)
            return A, b
        dim = None
        for T in (self, other):
            if T.A is not None: dim = T.A.shape[0]; break
            if T.b is not None: dim = T.b.shape[0]; break
            if T.delta is not None: dim = T.delta.shape[0]; break
        if dim is None: raise ValueError("Cannot infer dimension for composition")
        A1,b1 = as_affine(self, dim); A0,b0 = as_affine(other, dim)
        A = A1 @ A0; b = A1 @ b0 + b1
        return Transform(A=A, b=b)

@dataclass
class Morphism:
    src: str
    dst: str
    weight: float
    transform: Transform
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Theme:
    id: str
    seed_cluster_id: str
    prototype: np.ndarray
    member_scores: Dict[str, float]
    contributing_edges: List[Morphism] = field(default_factory=list)
    label: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

class ClusterCategory:
    # -----------------------------------------------------------------------------
    # ClusterCategory.__init__(self, clusters, morphisms)
    # Summary:
    #   Stores clusters and builds fast adjacency maps: out_edges[src] and
    #   in_edges[dst] for all Morphism edges.
    # Effect:
    #   Treats per‑cluster relations as a graph so we can efficiently enumerate
    #   outgoing, incoming, and composed (two‑hop) morphisms during theme building.
    # -----------------------------------------------------------------------------
    def __init__(self, clusters: Dict[str, Cluster], morphisms: List[Morphism]) -> None:
        self.clusters = clusters
        self.out_edges: Dict[str, List[Morphism]] = {}
        self.in_edges: Dict[str, List[Morphism]] = {}
        for m in morphisms:
            self.out_edges.setdefault(m.src, []).append(m)
            self.in_edges.setdefault(m.dst, []).append(m)
            
    # -----------------------------------------------------------------------------
    # ClusterCategory.outgoing(self, cid: str) -> List[Morphism]
    # ClusterCategory.incoming(self, cid: str) -> List[Morphism]
    # Summary:
    #   Lightweight accessors returning outgoing/incoming edges for a cluster.
    # Effect:
    #   Small utilities used throughout theme construction and seed selection.
    # -----------------------------------------------------------------------------
    def outgoing(self, cid: str) -> List[Morphism]:
        return self.out_edges.get(cid, [])
    def incoming(self, cid: str) -> List[Morphism]:
        return self.in_edges.get(cid, [])
        
    # -----------------------------------------------------------------------------
    # ClusterCategory.compose_2hop(self, cid: str, weight_min: float = 0.0,
    #                              topk_per_src: int = 0) -> List[Morphism]
    # Summary:
    #   For a given start cluster `cid`, composes all two‑step paths cid→m1.dst→m2.dst
    #   into single Morphism objects. The composed edge’s weight is the product
    #   m1.weight * m2.weight; the transform is m2.transform ∘ m1.transform.
    #   Optional pruning keeps only top‑k mid edges per source and drops weak paths
    #   via `weight_min`.
    # Effect:
    #   Captures *transitive* semantics across clusters, letting themes aggregate
    #   not just direct but also mediated relations—crucial for surfacing stable
    #   cross‑document categories from local structure.
    # -----------------------------------------------------------------------------
    def compose_2hop(self, cid: str, weight_min: float = 0.0, topk_per_src: int = 0) -> List[Morphism]:
        composed: List[Morphism] = []
        for m1 in self.outgoing(cid):
            if m1.weight < weight_min: continue
            mids = self.outgoing(m1.dst)
            if topk_per_src > 0 and len(mids) > topk_per_src:
                mids = sorted(mids, key=lambda m: m.weight, reverse=True)[:topk_per_src]
            for m2 in mids:
                w = m1.weight * m2.weight
                if w < weight_min: continue
                T = m2.transform.compose(m1.transform)
                composed.append(Morphism(src=cid, dst=m2.dst, weight=w, transform=T, meta={'via':(m1.src, m1.dst)}))
        return composed

# -----------------------------------------------------------------------------
# def _pullback_centroid(seed: Cluster, target: Cluster, T_src_to_tgt: Transform) -> np.ndarray
# Summary:
#   Maps the target cluster’s centroid back into the seed cluster’s frame by
#   applying the inverse of the seed→target transform.
# Effect:
#   “Pullback” aligns heterogeneous clusters into a single coordinate system,
#   making it meaningful to average them when forming a theme prototype.
# -----------------------------------------------------------------------------
def _pullback_centroid(seed: Cluster, target: Cluster, T_src_to_tgt: Transform) -> np.ndarray:
    invT = T_src_to_tgt.inverse()
    return invT.apply_forward(target.centroid)

# -----------------------------------------------------------------------------
# def make_colimit_theme(
#     C: ClusterCategory,
#     seed_cluster_id: str,
#     weight_min: float = 0.0,
#     include_identity: bool = True,
#     two_hop: bool = False,
#     topk_per_src: int = 0,
#     doc_agg: str = "topk",
#     doc_topk: int = 3,
# ) -> Theme
# Summary:
#   Builds a dataset‑level *theme* anchored at `seed_cluster_id` by aggregating
#   (pulling back + averaging) all reachable target centroids from the seed via
#   outgoing morphisms, optionally including composed two‑hop paths.
#   Steps:
#     1) Collect edges from seed (and optionally two‑hop composed edges), keeping
#        those with weight ≥ weight_min.
#     2) For each edge, pull the target centroid back into the seed frame via the
#        inverse transform; accumulate {X_i, w_i}.
#     3) Optionally add the seed’s own centroid (include_identity) to anchor the
#        prototype.
#     4) Normalize weights by the max and compute a weighted mean prototype
#        proto = Σ_i (ŵ_i · X_i) / Σ_i ŵ_i.
#     5) Score document membership: bucket edge weights by target doc and reduce
#        with `doc_agg` (sum | max | mean | topk mean), where `doc_topk` limits
#        how many strong edges contribute per doc (mitigates doc length effects).
#     6) Return Theme(prototype, member_scores, contributing_edges, ...); handle
#        the degenerate no‑edge case gracefully.
# Effect:
#   Implements a practical *colimit* over morphisms: a canonical representative
#   vector (prototype) summarizing how the seed’s semantics project into the
#   rest of the corpus. Document scores quantify where that projection is most
#   strongly supported—turning manifold geometry into global categories.
# -----------------------------------------------------------------------------
def make_colimit_theme(
    C: ClusterCategory,
    seed_cluster_id: str,
    weight_min: float = 0.0,
    include_identity: bool = True,
    two_hop: bool = False,
    topk_per_src: int = 0,
    doc_agg: str = "topk",
    doc_topk: int = 3,
) -> Theme:
    seed = C.clusters[seed_cluster_id]
    dim = seed.centroid.shape[0]
    edges = [m for m in C.outgoing(seed_cluster_id) if m.weight >= weight_min]
    if two_hop:
        edges += C.compose_2hop(seed_cluster_id, weight_min=weight_min, topk_per_src=topk_per_src)
    X,W = [],[]
    for m in edges:
        tgt = C.clusters.get(m.dst)
        if tgt is None: continue
        x = _pullback_centroid(seed, tgt, m.transform)
        if x.shape[0] != dim: continue
        X.append(x); W.append(float(m.weight))
    if include_identity:
        X.append(seed.centroid.copy()); W.append(1.0)
    if not X:
        return Theme(id=f"T::{seed_cluster_id}", seed_cluster_id=seed_cluster_id,
                     prototype=seed.centroid.copy(), member_scores={seed.doc_id:1.0},
                     contributing_edges=[], meta={'degenerate':True})
    X = np.stack(X, axis=0)
    W_arr = np.asarray(W, dtype=float)
    if W_arr.max() > 0: W_arr = W_arr/(W_arr.max()+1e-12)
    proto = (W_arr[:,None]*X).sum(axis=0)/(W_arr.sum()+1e-12)
    doc_buckets = {}
    for m in edges:
        tgt = C.clusters.get(m.dst)
        if tgt is None: continue
        doc_buckets.setdefault(tgt.doc_id, []).append(float(m.weight))
    def agg(ws):
        if not ws: return 0.0
        if doc_agg=="sum": return float(np.sum(ws))
        if doc_agg=="max": return float(np.max(ws))
        if doc_agg=="mean": return float(np.mean(ws))
        if doc_agg=="topk": return float(np.mean(sorted(ws, reverse=True)[:max(1, doc_topk)]))
        return float(np.mean(ws))
    member_scores = {d: agg(ws) for d,ws in doc_buckets.items()}
    member_scores.setdefault(seed.doc_id, max(member_scores.get(seed.doc_id, 0.0), 1e-9))
    return Theme(id=f"T::{seed_cluster_id}", seed_cluster_id=seed_cluster_id,
                 prototype=proto, member_scores=member_scores,
                 contributing_edges=edges, meta={'two_hop':two_hop, 'weight_min':weight_min})

# -----------------------------------------------------------------------------
# def _cosine(a: np.ndarray, b: np.ndarray) -> float
# Summary:
#   Safe cosine similarity (unit‑normed dot with small ε), used for prototype
#   comparisons.
# Effect: compact, stable similarity for merging nearly identical themes.
# -----------------------------------------------------------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)+1e-12; nb = np.linalg.norm(b)+1e-12
    return float(np.dot(a,b)/(na*nb))

# -----------------------------------------------------------------------------
# def _jaccard(a: set, b: set) -> float
# Summary:
#   Jaccard index over sets with empty‑set guard (1.0 if both empty).
# Effect: measures overlap of theme membership to merge or analyze themes.
# -----------------------------------------------------------------------------
def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    return len(a & b)/max(1, len(a | b))

# -----------------------------------------------------------------------------
# def label_propagation_communities(
#     C: ClusterCategory, weight_min: float = 0.0, max_iter: int = 20
# ) -> Dict[str, int]
# Summary:
#   Undirected, weighted label‑propagation over the cluster graph. Builds a
#   symmetric neighbor map (taking per‑pair max weight) and repeatedly assigns
#   each node the label with the largest total incident weight until convergence
#   or `max_iter`.
# Effect:
#   Finds coarse *communities* of clusters connected by strong morphisms. Those
#   communities provide balanced, coverage‑oriented seeds for global themes and
#   reduce bias toward high‑degree regions.
# -----------------------------------------------------------------------------
def label_propagation_communities(C: ClusterCategory, weight_min: float=0.0, max_iter:int=20) -> Dict[str,int]:
    nbrs = {}
    for src, edges in C.out_edges.items():
        for m in edges:
            if m.weight < weight_min: continue
            nbrs.setdefault(src, {}).setdefault(m.dst, 0.0)
            nbrs.setdefault(m.dst, {}).setdefault(src, 0.0)
            nbrs[src][m.dst] = max(nbrs[src][m.dst], m.weight)
            nbrs[m.dst][src] = max(nbrs[m.dst][src], m.weight)
    labels = {cid:i for i,cid in enumerate(C.clusters.keys())}
    for _ in range(max_iter):
        moved = 0
        for cid in C.clusters.keys():
            ws = nbrs.get(cid, {})
            if not ws: continue
            votes = {}
            for nb,w in ws.items():
                votes[labels[nb]] = votes.get(labels[nb], 0.0) + w
            new_label = max(votes.items(), key=lambda kv: kv[1])[0]
            if new_label != labels[cid]: labels[cid]=new_label; moved+=1
        if moved==0: break
    uniq = {lab for lab in labels.values()}
    remap = {lab:i for i,lab in enumerate(sorted(uniq))}
    return {cid: remap[lab] for cid,lab in labels.items()}

# -----------------------------------------------------------------------------
# def select_seed_clusters_by_community(C: ClusterCategory, weight_min: float = 0.0) -> List[str]
# Summary:
#   Picks one seed per community: the cluster with highest outgoing strength
#   (sum of edge weights ≥ threshold), returning seeds sorted by strength.
# Effect:
#   Ensures seed selection is representative yet compact—each seed stands for a
#   community’s strongest “voice” for theme construction.
# -----------------------------------------------------------------------------
def select_seed_clusters_by_community(C: ClusterCategory, weight_min: float=0.0) -> List[str]:
    comm = label_propagation_communities(C, weight_min=weight_min)
    out_strength = {cid: sum(m.weight for m in C.outgoing(cid) if m.weight >= weight_min) for cid in C.clusters.keys()}
    best = {}
    for cid,k in comm.items():
        s = out_strength.get(cid, 0.0)
        if k not in best or s > best[k][1]: best[k]=(cid,s)
    seeds = [cid for cid,_ in sorted(best.values(), key=lambda t:-t[1])]
    return seeds

@dataclass
class ThemeSet:
    themes: List[Theme]
    
    # -----------------------------------------------------------------------------
    # ThemeSet.to_frames(self, theme_label_func: Optional[Callable[[Theme], str]] = None,
    #                    membership_threshold: float = 0.0)
    # Summary:
    #   Materializes three tidy DataFrames:
    #     • themes_df: id, seed, label, #edges, meta
    #     • membership_df: (theme_id, doc_id, score) filtered by membership_threshold
    #     • overlaps_df: pairwise theme overlaps via Jaccard(doc sets ≥ threshold)
    #                   (only non‑zero entries are emitted).
    # Effect:
    #   Provides interoperable outputs for analysis/visualization—e.g., labeling,
    #   Sankey diagrams, or auditing theme overlaps across documents.
    # -----------------------------------------------------------------------------
    def to_frames(self, theme_label_func: Optional[Callable[[Theme], str]]=None, membership_threshold: float=0.0):
        rows = []
        for th in self.themes:
            label = th.label
            if theme_label_func is not None:
                try: label = theme_label_func(th)
                except Exception: pass
            rows.append({"theme_id": th.id, "seed_cluster_id": th.seed_cluster_id,
                         "label": label, "num_edges": len(th.contributing_edges), "meta": th.meta})
        themes_df = pd.DataFrame(rows)
        mem_rows = []
        for th in self.themes:
            for doc_id, score in th.member_scores.items():
                if score >= membership_threshold:
                    mem_rows.append({"theme_id": th.id, "doc_id": doc_id, "score": float(score)})
        membership_df = pd.DataFrame(mem_rows)
        ov_rows = []
        for i in range(len(self.themes)):
            docs_i = {d for d,s in self.themes[i].member_scores.items() if s >= membership_threshold}
            for j in range(i+1, len(self.themes)):
                docs_j = {d for d,s in self.themes[j].member_scores.items() if s >= membership_threshold}
                jacc = _jaccard(docs_i, docs_j)
                if jacc > 0.0:
                    ov_rows.append({"theme_i": self.themes[i].id, "theme_j": self.themes[j].id, "jaccard": jacc})
        overlaps_df = pd.DataFrame(ov_rows)
        return themes_df, membership_df, overlaps_df

# -----------------------------------------------------------------------------
# def merge_similar_themes(
#     themes: List[Theme],
#     proto_cos_threshold: float = 0.97,
#     doc_jacc_threshold: float = 0.8,
#     membership_threshold: float = 0.0
# ) -> List[Theme]
# Summary:
#   De‑duplicates near‑identical themes by grouping any pair whose prototype
#   cosine ≥ proto_cos_threshold *and* membership Jaccard ≥ doc_jacc_threshold
#   (using docs with score ≥ membership_threshold). Within each group, chooses a
#   representative (most edges then largest total membership) and merges
#   membership by taking the max score per doc; records provenance in meta.
# Effect:
#   Reduces fragmentation of categories and stabilizes the global organization,
#   yielding fewer, stronger themes with cleaner boundaries.
# -----------------------------------------------------------------------------
def merge_similar_themes(themes: List[Theme], proto_cos_threshold: float=0.97, doc_jacc_threshold: float=0.8, membership_threshold: float=0.0) -> List[Theme]:
    kept = []; used=set()
    for i,ti in enumerate(themes):
        if i in used: continue
        group=[i]
        for j in range(i+1,len(themes)):
            if j in used: continue
            tj = themes[j]
            cos = _cosine(ti.prototype, tj.prototype)
            docs_i = {d for d,s in ti.member_scores.items() if s >= membership_threshold}
            docs_j = {d for d,s in tj.member_scores.items() if s >= membership_threshold}
            jacc = _jaccard(docs_i, docs_j)
            if cos >= proto_cos_threshold and jacc >= doc_jacc_threshold:
                group.append(j); used.add(j)
        rep = ti
        if len(group) > 1:
            candidates = [themes[k] for k in group]
            rep = max(candidates, key=lambda th: (len(th.contributing_edges), sum(th.member_scores.values())))
            merged_members = {}
            for th in candidates:
                for d,s in th.member_scores.items():
                    merged_members[d] = max(merged_members.get(d,0.0), s)
            rep = Theme(id=rep.id, seed_cluster_id=rep.seed_cluster_id, prototype=rep.prototype,
                        member_scores=merged_members, contributing_edges=rep.contributing_edges,
                        label=rep.label, meta=dict(rep.meta, merged_from=[themes[k].id for k in group]))
        kept.append(rep); used.add(i)
    return kept

# -----------------------------------------------------------------------------
# def build_global_categorical_organization(
#     clusters: Dict[str, Cluster],
#     morphisms: List[Morphism],
#     weight_min: float = 0.0,
#     include_identity: bool = True,
#     two_hop: bool = False,
#     topk_per_src: int = 0,
#     doc_agg: str = "topk",
#     doc_topk: int = 3,
#     seed_strategy: str = "community",
#     explicit_seeds: Optional[List[str]] = None,
#     merge_protos: bool = True,
#     proto_cos_threshold: float = 0.97,
#     doc_jacc_threshold: float = 0.8,
#     membership_threshold: float = 0.0,
#     theme_label_func: Optional[Callable[[Theme], str]] = None,
# )
# Summary:
#   Orchestrates the full pipeline to produce a global categorical arrangement:
#     1) Build ClusterCategory(C) from clusters + morphisms.
#     2) Choose seed clusters via `seed_strategy` (community | all | explicit |
#        out‑degree ranking).
#     3) For each seed, construct a colimit theme (make_colimit_theme) with the
#        chosen morphism options (two_hop, thresholds, aggregation policy).
#     4) Optionally merge near‑duplicate themes (merge_similar_themes).
#     5) Package results in ThemeSet + tidy frames (to_frames) with optional
#        human labels via `theme_label_func`.
# Effect:
#   Turns local, directional semantics between clusters into a coherent, corpus‑
#   level taxonomy with interpretable prototypes and document memberships—
#   the core outcome of the “semantic morphisms” approach.
# -----------------------------------------------------------------------------
def build_global_categorical_organization(
    clusters: Dict[str, Cluster],
    morphisms: List[Morphism],
    weight_min: float = 0.0,
    include_identity: bool = True,
    two_hop: bool = False,
    topk_per_src: int = 0,
    doc_agg: str = "topk",
    doc_topk: int = 3,
    seed_strategy: str = "community",
    explicit_seeds: Optional[List[str]] = None,
    merge_protos: bool = True,
    proto_cos_threshold: float = 0.97,
    doc_jacc_threshold: float = 0.8,
    membership_threshold: float = 0.0,
    theme_label_func: Optional[Callable[[Theme], str]] = None,
):
    C = ClusterCategory(clusters, morphisms)
    if seed_strategy=="explicit" and explicit_seeds is not None:
        seeds = explicit_seeds
    elif seed_strategy=="community":
        seeds = select_seed_clusters_by_community(C, weight_min=weight_min)
    elif seed_strategy=="all":
        seeds = list(clusters.keys())
    else:
        seeds = sorted(clusters.keys(), key=lambda cid: -sum(m.weight for m in C.outgoing(cid)))
    themes = []
    for sid in seeds:
        th = make_colimit_theme(C, sid, weight_min=weight_min, include_identity=include_identity,
                                two_hop=two_hop, topk_per_src=topk_per_src,
                                doc_agg=doc_agg, doc_topk=doc_topk)
        themes.append(th)
    if merge_protos:
        themes = merge_similar_themes(themes, proto_cos_threshold=proto_cos_threshold,
                                      doc_jacc_threshold=doc_jacc_threshold,
                                      membership_threshold=membership_threshold)
    theme_set = ThemeSet(themes)
    themes_df, membership_df, overlaps_df = theme_set.to_frames(theme_label_func=theme_label_func,
                                                                membership_threshold=membership_threshold)
    return theme_set, themes_df, membership_df, overlaps_df

# -----------------------------------------------------------------------------
# def make_seed_terms_labeler(clusters: Dict[str, Cluster], top_k: int = 6) -> Callable[[Theme], str]
# Summary:
#   Returns a function that converts a Theme into a short human label by looking
#   up its seed cluster’s metadata: prefers meta['top_terms'] (strings), falling
#   back to meta['top_topics'] (renders as 'topic_i').
# Effect:
#   Supplies compact, interpretable names for themes without re‑computing text
#   features—useful for UI tooltips, tables, and exports.
# -----------------------------------------------------------------------------
def make_seed_terms_labeler(clusters: Dict[str, Cluster], top_k: int = 6):
    def _label(theme: Theme) -> str:
        c = clusters.get(theme.seed_cluster_id)
        if c is None: return ""
        # prefer explicit 'top_terms' if present
        terms = []
        if isinstance(c.meta, dict):
            tt = c.meta.get("top_terms")
            if tt: terms = [str(t) for t in tt][:top_k]
            elif 'top_topics' in c.meta:
                # fall back to topic ids => "topic_#" labels
                ids = c.meta['top_topics']
                terms = [f"topic_{i}" for i in ids][:top_k]
        return ", ".join(terms)
    return _label
