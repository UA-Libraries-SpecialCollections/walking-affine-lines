# Embedding Manifolds as Semantic Morphisms: Semantic Morphism Graph Pipeline

Research software for building document level **semantic morphism graphs** from OCR text, saving reusable pickle artifacts, and computing compact cross document **semantic transition graphlet** payloads for downstream analysis. The project models each document as a directed graph of semantic displacements in a document centered residual SBERT space, then compares those directed graphlets across documents using displacement alignment, endpoint PC1 concordance, semantic quality, lexical diagnostics, and document embedding baselines.

This repository supports the **Embedding Manifolds as Semantic Morphisms** research program developed in the University of Alabama Libraries Digital Services context. The current codebase is a research prototype intended for exploratory computational archives, and evidence generation. It is not a packaged library or preservation data format by itself.

**Developer:** Jeremiah Colonna-Romano, University of Alabama Libraries Digital Services, 2025-2026 

## Reviewed source basis

This README is based on the current pipeline files that build the three primary artifacts:

- `generate_document_delta_manifold_PP.py` — interactive Tk-driven workflow for building/loading/combining document manifolds, saving `document_delta_dict.pkl` and `segments_by_doc.pkl`, and invoking visualization, analysis, null-comparison, and arrangement endpoints.
- `topic_modeling.py` — core manifold construction, semantic-quality scoring, edge extraction, parallel morphism comparison, compact `morphism_comparison.pkl` serialization, lexical/acuteness enrichment, and plot-cache generation.

The companion analysis application, `morphism_analysis_platform.py`, is documented separately. It opens the artifacts produced by this build/compare pipeline and provides schema inspection, match querying, evidence browsing, Edge Match 3D visualization, arrangement experiments, Collection ROC baselines, and Shape Bin Field workspaces.

## What the pipeline produces

The build and analysis workflow produces three primary pickle artifacts.

| Artifact | Produced by | Purpose |
|---|---|---|
| `document_delta_dict.pkl` | Build mode in `generate_document_delta_manifold_PP.py` | Dictionary mapping each `doc_id` to a cluster-delta manifold tuple containing the directed displacement tensor, cluster labels/order, cluster topic distributions, cluster centroids, endpoint PC1 vectors, semantic quality payloads, and document embedding baselines. |
| `segments_by_doc.pkl` | Build mode in `generate_document_delta_manifold_PP.py` | Dictionary mapping each `doc_id` to its sentence-level segment strings. This is the text companion to the geometry object and is required for cluster text inspection, lexical diagnostics, and on-demand re-embedding in the analysis platform. |
| `morphism_comparison.pkl` | Analyze endpoint in `generate_document_delta_manifold_PP.py`, saved by `save_morphism_comparison_pickle()` in `topic_modeling.py` | Compact/enriched array-backed retained-match payload comparing directed cluster morphisms across documents. Stores edge metadata, structured match records, diagnostics, document-cosine baselines, and plot-cache grids when enrichment is enabled. |

Pickle files are convenient research artifacts, but they are not safe interchange formats. Open only pickle files created by this project or trusted collaborators. For dissemination, review, or preservation, export CSV/JSON/Markdown/HTML/PNG derivatives from the analysis platform.

## Conceptual method summary

The current method can be described as a two-stage pipeline:

1. **Document manifold build:** each document becomes a document-centered residual SBERT cluster-delta manifold.
2. **Morphism comparison:** directed cluster-to-cluster displacement morphisms are compared across documents and saved in a compact retained-match payload.

A concise method sequence:

```text
OCR text
→ sentence segmentation
→ normalized raw SBERT segment embeddings
→ raw SBERT document embedding baseline
→ document mean-centering + row-renormalized residual segment directions
→ size-aware clustering of residual segment vectors
→ cluster centroids, endpoint PC1 vectors, topic distributions, semantic quality Q
→ complete directed cluster graph Δ[i,j] = centroid_j - centroid_i
→ cross-document morphism comparison over directed edges
→ compact morphism_comparison.pkl with diagnostics and plot-cache views
```

### Raw SBERT vs residual manifold geometry

The pipeline intentionally stores two document-level embedding baselines:

- `raw_sbert_document_embedding` — normalized mean of raw normalized SBERT segment embeddings before document centering. This supports global SBERT document proximity and raw anchor visualizations.
- `manifold_residual_document_embedding` — normalized mean of the document-centered / row-renormalized segment embeddings used to build the local cluster-delta manifold. This supports residual-manifold document comparison.

The directed cluster morphisms are built in the **document-centered residual space**, not raw SBERT document space. This distinction is central to interpretation: raw SBERT cosine measures broad/global document proximity; residual/morphism geometry measures local document-relative semantic structure and transitions.

### Row-renormalized residual segment directions

The pipeline embeds segments with normalized SBERT vectors, then applies document-centering and row-renormalization. Conceptually:

```python
X = E - E.mean(axis=0, keepdims=True)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
```

Each segment becomes a unit direction of semantic deviation from its document center. Cluster centroids are means of those unit residual directions, and their norms reflect directional concentration inside a cluster.

### Directed cluster morphisms

For each document, the pipeline computes a complete directed graph over cluster centroids:

```text
Δ[i,j] = cluster_centroid[j] - cluster_centroid[i]
```

Each non-self ordered cluster pair is a directed morphism edge. A document with `k` clusters contributes `k × (k - 1)` directed morphisms.

### Endpoint PC1 directions

Each cluster also receives a PC1 direction computed from the segment embeddings assigned to that cluster. PC1 is sign-oriented away from the document mean toward the cluster mean, then normalized. This supplies an endpoint-orientation axis for morphism comparison and shape analysis.

### Semantic quality Q

Each cluster receives a semantic/discursive quality payload. Q combines diagnostics such as segment quality, usable support, fragment burden, non-template/repetition behavior, calibrated embedding spread, and optional language-model fluency. Edge-level Q values are derived conservatively from source and destination cluster quality.

### Morphism match axes

The parallel analyzer retains candidate matches using:

- signed Δ direction cosine,
- source and destination PC1 concordance,
- selected PC1-axis value, currently destination PC1 by default in the Analyze driver,
- harmonic semantic quality Q,
- optional PC1-only match logic,
- top-k retention per source edge.

The compact match records store fields such as:

```text
match_type
src_edge
tgt_edge
delta_cos
src_pc1
dst_pc1
pc1_axis_value
semantic_quality
semantic_quality_min
joint_min
joint_min_4d
detected_delta_thr
detected_pc1_thr
detected_quality_thr
pc1_only_thr
delta_max_for_pc1_only
pc1_only_quality_thr
```

### Compact enrichment

When compact diagnostics are enabled, the comparison payload is enriched with:

- destination lexical overlap coefficient,
- lexical divergence,
- harmonic alignment core,
- acuity score,
- count-cosine acuity variant,
- raw SBERT document cosine,
- manifold-residual document cosine,
- semantic quality arrays,
- top candidate rows,
- binned plot-cache arrays for match count, lexical overlap/divergence, peak acuity, document-cosine summaries, high-acuity counts, and per-source-edge support.

Acuity is intended to surface matches with strong morphism alignment but low lexical overlap:

```text
acuity = harmonic_mean(Δ alignment, PC1-axis concordance, Q) × lexical_divergence
```

## Repository file roles

| File | Role |
|---|---|
| `generate_document_delta_manifold_PP.py` | Main Tk workflow script. Builds new document manifolds, loads existing artifacts, combines manifold sets, launches visualization/analyze/null/arrangement endpoints, saves `document_delta_dict.pkl`, `segments_by_doc.pkl`, and `morphism_comparison.pkl`. |
| `topic_modeling.py` | Core methods: segmentation, SBERT embedding, residual transform, size-aware clustering, CDM tuple construction, cluster semantic quality, morphism edge extraction, parallel comparison, compact serialization, enrichment, plot-cache creation, and legacy CSV/output helpers. |
| `morphism_shapes.py` | Shape-bin and arrangement support: extracts edge-local zig-zag shape records, clusters shape features, and builds document × shape-category membership matrices. |
| `arrangement_endpoint.py` | Collection arrangement and Sankey/topic-flow visualization support. |
| `topic_flow_labels.py` | Topic-flow labeling utilities for interpreting arrangement nodes and morphism transitions. |
| `morphism_analysis_platform.py` | Standalone analysis/evidence application for opening and inspecting saved artifacts. Documented separately. |

Additional project-local modules used by the build script include `backrooms.py` and `figure_pickle.py`. The reviewed source imports these modules, so a working checkout must include them or replace those calls with local equivalents.

## Installation

Python 3.10 is recommended. The code has been developed and tested primarily as desktop research software with Tkinter dialogs and Windows-style paths.

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows PowerShell / cmd
# source .venv/bin/activate # Linux/macOS
```

Install dependencies. A repository requirements file should include at least:

```text
numpy
scipy
scikit-learn
matplotlib
mplcursors
umap-learn
nltk
gensim
sentence-transformers
torch
transformers
threadpoolctl
plotly
```

Tkinter is used for GUI dialogs. It is normally bundled with Python.org Windows installers. On Linux, install the OS Tk package if needed, for example `python3-tk`.

The pipeline uses NLTK sentence tokenization. If the Punkt tokenizer is not already installed, run:

```python
import nltk
nltk.download("punkt")
```

For CUDA acceleration, install a PyTorch build compatible with the workstation GPU/CUDA environment before installing `sentence-transformers`, or follow PyTorch’s current installation instructions for the local CUDA version.

## Required inputs

A typical build/analyze run requires:

1. OCR/plain-text item data available through the project-local loader `load_document_texts_by_prefix(...)` in `backrooms.py`.
2. A trained Gensim `LdaModel` file.
3. A matching Gensim `Dictionary` file.
4. A topic-label file used by the visualization and interpretation tools.
5. Enough local RAM/disk for pickle artifacts and temporary array files.

The current build script calls:

```python
item_text_dict = load_document_texts_by_prefix(21)
```

This is a local project assumption. Public reuse should document or replace the `backrooms` loader so new users can point the script at their own text directory or metadata manifest.

## Running the pipeline

Run the main script:

```bash
python generate_document_delta_manifold_PP.py
```

The script opens a Tkinter configuration dialog. It supports three high-level modes:

| Mode | Purpose |
|---|---|
| `build` | Build a new `document_delta_dict` and `segments_by_doc` from source texts. |
| `load` | Load an existing manifold pickle and segment pickle for visualization or analysis. |
| `compare` | Load two manifold/segment pickle sets and combine them into one in-memory dataset. |

The workflow then executes one endpoint:

| Endpoint | Purpose |
|---|---|
| `visualize` | Render collection-level document/cluster/morphism visualizations. |
| `analyze` | Run morphism comparison and optionally save `morphism_comparison.pkl`. |
| `null_compare` | Run anchor/null comparison diagnostics. |
| `arrange` | Run arrangement/Sankey-style document grouping support from CDM data. |

## Building `document_delta_dict.pkl` and `segments_by_doc.pkl`

Choose:

```text
mode = build
```

The script:

1. Loads source OCR text through `load_document_texts_by_prefix(...)`.
2. Starts a `ProcessPoolExecutor` with worker initialization for LDA, dictionary, SBERT, and optional fluency settings.
3. Runs `manifoldit(item_id, item_text)` for each item.
4. Calls `mk_delta_manifold(..., return_raw_embeddings=True)` to build sentence segments, raw SBERT embeddings, residual embeddings, labels, and initial delta data.
5. Calls `build_cluster_delta_matrix(...)` to construct the canonical CDM tuple, including raw and residual document embedding payloads.
6. Stores successful items in:

```python
document_delta_dict[item_id] = CDM
segments_by_doc[item_id] = segments
```

7. Prompts to save both dictionaries as `.pkl` files.

The current worker skips documents with fewer than three usable clusters.

## CDM tuple schema

Each value in `document_delta_dict` is an 8-part tuple:

```python
(
    delta_matrix,                 # n_clusters × n_clusters × embedding_dim
    cluster_order,                # cluster labels in matrix order
    labels,                       # raw label for each segment
    cluster_topic_distributions,  # dict[label] -> topic distribution vector
    cluster_embeddings,           # n_clusters × embedding_dim centroid array
    cluster_dirs,                 # n_clusters × embedding_dim PC1 array
    cluster_semantic_quality,     # dict[label] -> quality/components payload
    document_embedding_payload,   # raw and residual document baselines
)
```

The document embedding payload stores explicit raw/residual fields, including:

```text
raw_sbert_document_embedding
raw_sbert_document_embedding_available
raw_sbert_document_embedding_method
raw_sbert_document_embedding_source
raw_sbert_document_embedding_norm_before_unit
manifold_residual_document_embedding
manifold_residual_document_embedding_available
manifold_residual_document_embedding_method
manifold_residual_document_embedding_source
manifold_residual_document_embedding_norm_before_unit
```

A deprecated compatibility alias named `document_embedding` may point to the manifold-residual baseline for older helper functions.

## Building `morphism_comparison.pkl`

Choose or load a manifold dataset, then run:

```text
endpoint = analyze
analyze_engine = parallel
save_morphism_comparison_pkl = true
```

The current default analysis path uses the parallel compact backend rather than the older legacy dict/CSV path. The driver calls:

```python
analyze_morphism_match_field_parallel(...)
enrich_morphism_comparison_diagnostics(...)
save_morphism_comparison_pickle(...)
```

Important default analyze settings in the current script include:

```text
analyze_engine = parallel
analyze_parallel_workers = auto
analyze_source_chunk_size = 384
analyze_target_block_size = 8192
save_morphism_comparison_pkl = true
analyze_build_legacy_result = false
analyze_print_summaries = false
analyze_compact_diagnostics = plot_cache
analyze_compact_top_candidates = 5000
analyze_plot_cache_step = 0.01
analyze_top_k_per_delta = 100
analyze_pc1_only_threshold = 0.60
analyze_delta_max_for_pc1_only = 0.60
analyze_pc1_only_quality_threshold = 0.0
compute_acuity_for = aligned_only
acuity_csv_mode = none
```

The current Analyze driver compares source edges against target edges with:

```text
Δ thresholds: 0.99 downward by 0.03
PC1 thresholds: 0.99 downward by 0.03
Q thresholds: 0.99 downward by 0.01
PC1 match axis: destination PC1
require_cross_doc = true
```

The comparison can be run over the full dataset or, in anchor mode, over a selected document’s source edges against the collection.

## Compact comparison payload schema

A saved compact comparison has the top-level structure:

```python
{
    "kind": "morphism_comparison",
    "version": 2,
    "match_type_codes": {"aligned": 0, "pc1_only": 1},
    "edge_index": {...},
    "edge_vectors": {...},             # optional, depending on settings
    "matches": structured_numpy_array,
    "match_diagnostics": {...},        # when enrichment is enabled
    "plot_cache": {...},               # when enrichment is enabled
    "document_embeddings": {...},      # when enrichment is enabled
    "params": {...},
    "summary": {...},
    "diagnostics": {...}
}
```

The `edge_index` stores aligned arrays such as:

```text
doc_ids
doc_code
src_label
dst_label
src_cluster_quality
dst_cluster_quality
edge_quality
edge_quality_min
delta_norm
```

The `matches` array stores retained source-edge / target-edge relationships and their alignment scores. `match_diagnostics` stores row-aligned lexical/acuteness/document-cosine values. `plot_cache` stores binned fields for fast analysis-platform rendering.

## Parallel comparison architecture

The compact comparison backend is designed to avoid materializing the full edge-by-edge matrix in memory. It:

1. extracts every directed cluster morphism edge into compact arrays,
2. writes NumPy arrays to temporary `.npy` files,
3. memory-maps those arrays in worker processes,
4. compares source-edge chunks against target blocks,
5. retains top-k matches per source edge,
6. concatenates structured records into the compact payload,
7. enriches the payload without rebuilding legacy nested dictionaries.

The driver caps BLAS/OpenMP thread counts per worker to avoid oversubscription.

## Interpreting the main scores

| Field | Interpretation |
|---|---|
| `delta_cos` | Signed cosine alignment between source and target displacement directions. |
| `src_pc1` | Absolute source endpoint PC1 concordance. |
| `dst_pc1` | Absolute destination endpoint PC1 concordance. |
| `pc1_axis_value` | Selected PC1 value used for thresholding/ranking; current Analyze driver uses destination PC1 by default. |
| `semantic_quality` | Harmonic pair quality derived from participating edge/cluster quality values. |
| `semantic_quality_min` | Conservative minimum quality among participating endpoints/edges. |
| `lexical_overlap_coefficient` | Destination-cluster lexical overlap coefficient for retained matches when diagnostics are enabled. |
| `lexical_divergence` | `1 - lexical_overlap_coefficient`. |
| `alignment_core` | Harmonic mean of Δ, selected PC1-axis concordance, and Q. |
| `acuity_score` | `alignment_core × lexical_divergence`. Highlights geometrically/qualitatively aligned but lexically divergent matches. |
| `raw_sbert_doc_cosine` | Global raw SBERT document cosine baseline between source and target documents. |
| `manifold_residual_doc_cosine` | Cosine baseline between document-centered residual document embeddings. |

## Optional legacy outputs

The older Analyze path can still build legacy nested result dictionaries and text/CSV-like outputs, but the current defaults avoid them because they are slow and memory-expensive for large retained match sets. To enable them, use:

```text
analyze_build_legacy_result = true
analyze_csv_mode = selected or full
analyze_print_summaries = true
```

For most current research workflows, use the compact `.pkl` plus the analysis platform instead.

## Arrangement and shape-bin analysis

Although the three primary artifacts are built by the files above, the repository also includes shape-membership arrangement support. Shape bins are learned morphism-action categories derived from edge-local features such as:

```text
cos(v, source PC1)
cos(-v, destination PC1)
cos(source PC1, destination PC1)
optional displacement length
optional coarse v-bin direction encoding
```

The current strongest collection-arrangement experiments have used:

```text
dir_weight_beta = 0.0
include_length = true
include_v_bin = false
```

This emphasizes edge-local transition geometry plus displacement scale, while leaving absolute residual-space displacement direction out of the shape definition. The analysis platform can build arrangement experiments from `document_delta_dict.pkl` and export shape summaries, document × shape-membership matrices, collection ROC baselines, shape-neighbor tables, and Shape Bin Field diagnostics.

## Reproducibility notes

Record these values for every run intended for review or publication:

```text
source text corpus and selection rule
LDA model and dictionary files
LDA topic-label file
SentenceTransformer model name and revision, if available
Python version
NumPy / BLAS configuration
PyTorch / CUDA configuration
worker counts and chunk/block sizes
semantic fluency settings
Analyze thresholds and top_k_per_delta
compact diagnostics mode and plot-cache step
shape arrangement settings, if used
```

The raw/residual document embedding distinction should be preserved in any exported methods text.

## Known limitations and assumptions

- The current build script relies on project-local modules, including `backrooms.py`, `figure_pickle.py`, and several analysis helpers. Public reuse requires including these modules or replacing their calls.
- Source text loading is currently tied to `load_document_texts_by_prefix(21)`, which is a local collection-loader convention rather than a generic command-line input interface.
- The GUI workflow is designed for research operation rather than headless batch deployment.
- Pickle artifacts are Python-specific and unsafe to load from untrusted sources.
- Shape bins learned from one batch are batch-relative unless a fixed atlas is introduced.
- PCA/3D graph views are visual projections; full-dimensional scores are computed before visualization and should be treated as the analytic values.

## Disclaimer

This software is provided as a research prototype without warranty. It may load local machine-learning models and third-party libraries. Users are responsible for dependency management, compute costs, API/model-license review, secure handling of source data, and safe handling of pickle files.
