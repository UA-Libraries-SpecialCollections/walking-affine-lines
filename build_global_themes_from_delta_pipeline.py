#!/usr/bin/python
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# =============================================================================
# build_global_themes_from_delta_pipeline.py 
# this script is a thin CLI wrapper that turns a stored
# *Cluster‑Delta Manifold* (CDM) — per‑document cluster centroids and
# pairwise deltas — into a *global categorical arrangement* of themes.
# It loads a 6‑tuple/dict from disk, adapts it to {clusters, morphisms},
# and calls the arrangement engine to compute theme prototypes and
# document memberships, exporting tidy CSVs for analysis.
# =============================================================================
# The CLI here wires into `build_categorical_arrangement_from_cdm_tuple(...)`
# which adapts the CDM, seeds themes, composes morphisms (optionally two‑hop),
# ggregates by document, merges near‑duplicates, and emits DataFrames.
# See arrangement_endpoint.py for those details.

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


import argparse, json
from arrangement_endpoint import build_categorical_arrangement_from_cdm_tuple


# -----------------------------------------------------------------------------
# def main() -> None
# Summary:
#   Orchestrates end‑to‑end theme building from a serialized Cluster‑Delta
#   Manifold (CDM) on disk, then writes three CSVs:
#     • *_themes.csv       — one row per theme (seed id, label, edge count, meta)
#     • *_membership.csv   — theme‑by‑document scores
#     • *_overlaps.csv     — pairwise Jaccard of doc sets (above threshold)
#   Steps:
#     1) Parse CLI flags that control morphism composition, pruning, and
#        membership aggregation (e.g., --two-hop, --weight-min, --doc-agg).
#     2) Load the CDM input:
#        - If .npz: expects keys [delta_matrix, cluster_order, labels,
#          cluster_topic_distributions, cluster_embeddings, cluster_dirs] and
#          packs them into a 6‑tuple.
#        - Else: unpickle whatever object was previously saved (either a single
#          6‑tuple or a dict of {doc_id -> 6‑tuple}).
#     3) Decode --adapter-kwargs (JSON) for the CDM→project adapter; e.g. map
#        label ids to doc ids, attach topic labels, or set a cluster id prefix.
#     4) Call build_categorical_arrangement_from_cdm_tuple(...), which:
#        - adapts CDM to (clusters, morphisms),
#        - constructs *colimit* themes from each seed (optionally with two‑hop
#          composed morphisms and fan‑out pruning),
#        - aggregates theme support per document via 'topk'/'sum'/'max'/'mean',
#        - merges near‑duplicate themes by prototype cosine & doc‑set Jaccard,
#        - returns a ThemeSet and three tidy DataFrames.
#     5) Write the DataFrames to CSV with the chosen --out-prefix and print a
#        short summary of how many themes were produced.
#
# Effect:
#   • Bridges *local* geometric relations (delta vectors between cluster
#     centroids) and *global* categorical semantics: the wrapper exposes knobs
#     that materially change the induced taxonomy — allowing you to include
#     transitive structure (two‑hop), dampen noisy edges (weight_min/top‑k),
#     and choose how strongly to credit prolific documents (doc_agg/doc_topk).
#   • Keeps provenance: the arrangement engine carries meta flags (e.g., whether
#     two‑hop was enabled), and the exported tables are stable inputs to downstream
#     visualization/audit pipelines.
#   • Label quality: --label-topk controls the seed‑terms labeler used inside
#     the arrangement endpoint, producing compact human labels from seed clusters'
#     top terms/topics without recomputation.  (See arrangement_endpoint.py.)
#
# Important knobs (conceptual effects):
#   --two-hop            Enable composition of two directed morphisms into one;
#                        captures *transitive* semantics across clusters.
#   --weight-min         Drops very weak morphisms; increases prototype stability.
#   --topk-per-src       Limits fan‑out during two‑hop composition; controls hub bias.
#   --doc-agg, --doc-topk
#                        Choose how per‑doc evidence is aggregated; 'topk' mitigates
#                        long‑doc advantage by averaging only the strongest K edges.
#   --seed-strategy      'community' picks one representative per graph community;
#                        'all' uses every cluster as a seed; 'explicit' uses your list.
#   --merge-protos (+ thresholds)
#                        Consolidates near‑duplicates by prototype cosine and
#                        membership overlap; yields fewer, cleaner categories.
#   --membership-th      Threshold used when materializing membership/overlap tables.
#
# I/O contract:
#   Input:  --cdm-path points at a .npz or .pkl containing either a single CDM
#           6‑tuple or a dict mapping doc_id → 6‑tuple.
#   Output: three CSVs prefixed by --out-prefix (default 'global_themes').
#
# Implementation notes:
#   - Uses np.load(..., allow_pickle=True) for .npz to support dict‑typed
#     cluster topic distributions; when loading .pkl, the object is trusted as‑is.
#   - Adapter kwargs are parsed from JSON to keep the CLI simple but flexible.
#   - All heavy lifting (adapting, seeding, composing, merging) is delegated to
#     build_categorical_arrangement_from_cdm_tuple; this file remains a lean CLI.
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdm-path", type=str, required=True,
                    help="Path to a numpy .npz or .pkl containing the 6-tuple from build_cluster_delta_matrix().")
    ap.add_argument("--adapter-kwargs", type=str, default="{}",
                    help="JSON for adapter kwargs (e.g., {\"doc_ids_by_label\": {0:\"docA\"}, \"topic_labels\": [\"t0\",...]}).")
    ap.add_argument("--out-prefix", type=str, default="global_themes")
    ap.add_argument("--label-topk", type=int, default=6)
    ap.add_argument("--two-hop", action="store_true")
    ap.add_argument("--weight-min", type=float, default=0.0)
    ap.add_argument("--topk-per-src", type=int, default=3)
    ap.add_argument("--doc-agg", choices=["topk","sum","max","mean"], default="topk")
    ap.add_argument("--doc-topk", type=int, default=3)
    ap.add_argument("--seed-strategy", choices=["community","all","explicit"], default="community")
    ap.add_argument("--merge-protos", action="store_true")
    ap.add_argument("--proto-cos-th", type=float, default=0.97)
    ap.add_argument("--doc-jacc-th", type=float, default=0.8)
    ap.add_argument("--membership-th", type=float, default=0.0)
    args = ap.parse_args()

    # Load the tuple from disk (supports .npz or .pkl)
    if args.cdm_path.endswith(".npz"):
        import numpy as np
        data = np.load(args.cdm_path, allow_pickle=True)
        # expect keys: delta_matrix, cluster_order, labels, cluster_topic_distributions, cluster_embeddings, cluster_dirs
        document_delta_dict = (
            data["delta_matrix"],
            data["cluster_order"].tolist(),
            data["labels"].tolist(),
            data["cluster_topic_distributions"].item(),  # dict
            data["cluster_embeddings"],
            data["cluster_dirs"],
        )
    else:
        import pickle
        with open(args.cdm_path, "rb") as f:
            document_delta_dict = pickle.load(f)

    adapter_kwargs = json.loads(args.adapter_kwargs)

    theme_set, themes_df, membership_df, overlaps_df = build_categorical_arrangement_from_cdm_tuple(
        document_delta_dict,
        adapter_kwargs=adapter_kwargs,
        label_topk=args.label_topk,
        weight_min=args.weight_min,
        two_hop=args.two_hop,
        topk_per_src=args.topk_per_src,
        doc_agg=args.doc_agg,
        doc_topk=args.doc_topk,
        seed_strategy=args.seed_strategy,
        merge_protos=args.merge_protos,
        proto_cos_th=args.proto_cos_th,
        doc_jacc_th=args.doc_jacc_th,
        membership_th=args.membership_th,
    )

    themes_df.to_csv(f"{args.out_prefix}_themes.csv", index=False)
    membership_df.to_csv(f"{args.out_prefix}_membership.csv", index=False)
    overlaps_df.to_csv(f"{args.out_prefix}_overlaps.csv", index=False)
    print(f"[OK] wrote {len(theme_set.themes)} themes.")
    print(f"Files: {args.out_prefix}_themes.csv, {args.out_prefix}_membership.csv, {args.out_prefix}_overlaps.csv")

if __name__ == "__main__":
    main()
