#!/usr/bin/python
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# --------------------------------------------------------
# arrangement_endpoint.py — module overview
# Summary:
# Provides the “arrangement” endpoints that turn morphism‑based document
# manifolds into interpretable outputs along two complementary paths:
# • Interactive flow view — sankey_docs_left(tree, ...) renders a
# document‑on‑left Sankey of a multi‑branch hierarchy (e.g., from
# build_hierarchy_all), wiring doc→L0 and parent→child links, tinting
# colors by depth, and injecting custom JS so hovering any document or
# cluster highlights its entire path; if the root has no explicit L0,
# it synthesizes virtual L0 clusters from doc_labels so the diagram
# always has meaningful links.
# • Categorical theme view — build_categorical_arrangement_from_cdm_tuple(...)
# and build_categorical_arrangement_from_mkdelta(...) adapt a document‑delta
# manifold into (clusters, morphisms), then run the global categorical
# organizer to produce: a set of labeled themes, a document‑by‑theme
# membership table, and inter‑theme overlaps (with knobs for two‑hop
# composition, edge pruning, and prototype merging).
# The Sankey accepts the tree produced upstream (e.g., morphism_shapes.build_hierarchy_all)
# and optionally displays topic‑flow labels/hover text computed by
# topic_flow_labels.annotate_tree_with_topic_flow, linking geometry back to
# interpretable topic transitions.
# These endpoints are used from the interactive pipeline (see the “arrange”
# branch in generate_document_delta_manifold.py) and the CLI that exports
# global themes CSVs.
# Effect:
# Bridges dense, intra‑document morphisms to human‑navigable structure:
# • The Sankey makes each document’s sequence through the refinement
# hierarchy visible (not just its destination), so analysts can inspect
# how semantic transformations compose along a path; virtual L0 fallback,
# depth‑based tinting, and doc “Other” bucketing preserve legibility for
# large collections without distorting counts.
# • The categorical organizer lifts manifold‑level relations into stable,
# named themes with tunable composition (two‑hop), label seeding, and
# prototype merging—yielding compact artifacts (themes, memberships,
# overlaps) that can be compared across runs and consumed by reports or
# UIs. Together, these views translate vector dynamics into flow narratives
# and categories that advance the project’s goal of *Embedding Manifolds
# as Semantic Morphisms*.
# --------------------------------------------------------

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


from typing import Dict, Any, Tuple, Optional
from global_categorical_themes import build_global_categorical_organization, make_seed_terms_labeler
from project_adapter_mkdelta import adapt_from_cdm_tuple, adapt_from_cdm

import math
from collections import defaultdict

import json
import webbrowser

import plotly.graph_objects as go
import plotly.io as pio


# --------------------------------------------------------
# def _mix_with_white(hex_color, alpha)
# Summary:
#   Lightens a HEX color by linearly interpolating each RGB channel toward 255
#   (white) with the given alpha, then returns the tinted HEX string. This keeps
#   hue relationships intact while softening saturation.
# Effect:
#   Provides depth-aware, readable node coloring (e.g., lighter tints for deeper
#   levels in the hierarchy) so viewers can perceive structural nuance in
#   semantic arrangements without altering any underlying counts or topology.
# --------------------------------------------------------
def _mix_with_white(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16); g = int(hex_color[2:4], 16); b = int(hex_color[4:6], 16)
    r = int(r + (255 - r) * alpha); g = int(g + (255 - g) * alpha); b = int(b + (255 - b) * alpha)
    return f'#{r:02x}{g:02x}{b:02x}'


# --------------------------------------------------------
# def _palette()
# Summary:
#   Returns a stable 10‑color base palette used to seed node/link colors
#   deterministically across runs and views.
# Effect:
#   Ensures visual consistency when mapping clusters/themes to colors, helping
#   readers track semantic groupings across different renders and exports.
# --------------------------------------------------------
def _palette():
    return ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
            '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
         

# --------------------------------------------------------
# def _rgba_from_hex(hex_str, alpha)
# Summary:
#   Converts a HEX color into a CSS 'rgba(r,g,b,a)' string with a specified
#   transparency value.
# Effect:
#   Enables subtle, layered link styling (e.g., slightly transparent flows) that
#   emphasizes relative magnitude and path focus without changing data values.
# --------------------------------------------------------    
def _rgba_from_hex(hex_str: str, alpha: float) -> str:
    h = hex_str.lstrip('#')
    r = int(h[0:2],16); g = int(h[2:4],16); b = int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'



# --------------------------------------------------------
# def sankey_docs_left(tree, title, max_docs_left, other_label, save_html, open_browser, div_id, height_vh, responsive)
# Summary:
#   Builds an interactive Sankey diagram that lists documents on the left and
#   shows their flow through a multi‑level arrangement on the right. The function:
#   • Collects all docs at the root and optionally caps left‑hand doc nodes,
#   bucketing the remainder under a single “Other” node for legibility.
#   • If the root has real L0 children, it creates cluster nodes for every
#   non‑root node (ordered by level/id) and connects doc→L0 and parent→child
#   links. Colors are lightened by level for depth cueing.
#   • If there are no L0 children, it synthesizes virtual L0 clusters from
#   the root’s doc_labels so the diagram still renders meaningful doc→cluster
#   links.
#   • Builds rich hover content: document labels and, when available, each
#   cluster’s top topic‑to‑topic transitions (lift‑scored) from prior topic‑flow
#   annotation.
#   • Serializes a mapping of doc indices and doc→path (root→…→leaf) into the
#   HTML and injects custom JavaScript so that hovering a document or cluster
#   highlights its entire path (dims non‑participants), and adjusts height
#   responsively with an optional scrollable wrapper.
#   Returns a Plotly Figure; if save_html is provided, writes a self‑contained,
#   responsive HTML file and optionally opens it in the browser.
# Effect:
#   Turns the project’s morphism‑derived hierarchy into an explorable flow
#   narrative where each document’s trajectory through clusters becomes visible:
#   • The doc→cluster→cluster links visualize how a document’s internal
#   transformation patterns (zig‑zag morphisms clustered into document groups)
#   place it within the collection’s arrangement—leveraging the tree produced
#   by hierarchical refinement over morphism shapes.
#   • Showing topic‑flow pairs on node hovers grounds the geometry in language,
#   surfacing characteristic transitions (e.g., “Labor → Strike”) used to
#   label nodes, which ties vector‑space morphisms back to interpretable text
#   semantics.
#   • The “virtual L0” fallback guarantees a readable representation even when
#   the root has no explicit children, preserving conceptual continuity
#   between document membership and emergent categories.
#   • Path highlighting makes the sequence of semantic morphisms legible,
#   supporting the project’s goal of tracing how text segments move through
#   thematic structure rather than only where they end up.
# --------------------------------------------------------
def sankey_docs_left(
    tree: dict,
    title: str = None,
    max_docs_left: Optional[int] = None,
    other_label: str = "Other docs",
    save_html: str = None,
    open_browser: bool = True,
    div_id: str = "sankeyDocsLeft",
    height_vh: int = 92,          # NEW: graph height as % of viewport height
    responsive: bool = True       # NEW: enable Plotly responsive mode
):
    """
    Build a Sankey where the left-hand side lists individual docs (optionally capped).
    If the root has no explicit children, we synthesize virtual L0 clusters
    from root doc_labels, so the diagram always has links to render.

    Adds interactivity: on hover, highlights the full path(s) that pass through
    the hovered node (documents or clusters).
    """
    nodes = tree["nodes"]
    root_id = tree["root_id"]
    by_id = {n["id"]: n for n in nodes}
    root = by_id[root_id]
    node_hover = []

    # All docs present at root
    all_docs = list(root.get("doc_ids", []))

    # Optional cap on doc nodes
    shown_docs = all_docs
    doc_bucketed = False
    if max_docs_left is not None and len(all_docs) > max_docs_left:
        shown_docs = all_docs[:max_docs_left]
        doc_bucketed = True
    other_docs = [d for d in all_docs if d not in set(shown_docs)]

    # Build node list: doc nodes first
    node_labels = []
    node_colors = []
    node_index = {}  # ("doc", docId) or ("doc_bucket", other_label) or ("cluster", treeNodeId) or ("virtL0", label) -> idx
    idx_to_kind = {} # idx -> "doc" | "cluster" | "bucket" | "virtL0"
    idx_to_doc  = {} # idx -> docId (for doc nodes)

    base = _palette()
    DOC_COLOR = '#cccccc'

    # Add left-side doc nodes (or a bucket)
    for d in shown_docs:
        idx = len(node_labels)
        node_index[("doc", d)] = idx
        idx_to_kind[idx] = "doc"
        idx_to_doc[idx]  = d
        node_labels.append(d)
        node_colors.append(DOC_COLOR)

    if doc_bucketed:
        idx = len(node_labels)
        node_index[("doc_bucket", other_label)] = idx
        idx_to_kind[idx] = "bucket"
        node_labels.append(f"{other_label} (n={len(other_docs)})")
        node_colors.append('#bdbdbd')

    # Determine whether we have explicit L0 children (multi-branch hierarchy)
    L0_children = list(root.get("children", []) or [])

    srcs = []; tgts = []; vals = []; link_labels = []; link_colors = []

    # Build a reverse map for doc->cluster link resolution
    link_index_map = {}  # "srcIdx-tgtIdx" -> linkIdx (string key)
    def _remember_link(src_idx, tgt_idx, link_idx):
        link_index_map[f"{src_idx}-{tgt_idx}"] = int(link_idx)

    # CASE A: real L0 children -> use them
    if L0_children:
        # Add cluster nodes (all nodes except root), ordered by level then id
        cluster_nodes = [n for n in nodes if n["id"] != root_id]
        cluster_nodes.sort(key=lambda n: (n["level"], n["id"]))

        for n in cluster_nodes:
            L = n["level"]
            color_idx = (n["id"] * 1315423911) % len(base)
            idx = len(node_labels)
            node_index[("cluster", n["id"])] = idx
            idx_to_kind[idx] = "cluster"
            count = len(n.get("doc_ids", []))
            label_text = n.get("label") or ""
            disp = (f"L{L}:{n['id']} · {label_text}" if label_text else f"L{L}:{n['id']}")
            node_labels.append(f"{disp} (n={count})")
            node_colors.append(_mix_with_white(base[color_idx], 0.12 + 0.12*L))
            tf = n.get("topic_flow", {})
            pair_str = ""
            for (u,v,sc) in tf.get("pairs", [])[:4]:
                # Render with interpretive names if available
                # You can precompute a map idx->name or reuse _topic_name if you import it.
                pair_str += f"{u} → {v} (lift={sc:.2f})<br>"
            node_hover.append(pair_str or f"{count} docs")

        # Map doc -> L0 child id
        doc_to_child = {}
        for cid in L0_children:
            child = by_id[cid]
            for d in child.get("doc_ids", []):
                doc_to_child[d] = cid

        # Doc -> L0 edges
        for d in shown_docs:
            c_id = doc_to_child.get(d)
            if c_id is None:
                continue
            s = node_index[("doc", d)]
            t = node_index[("cluster", c_id)]
            link_idx = len(srcs)
            srcs.append(s); tgts.append(t); vals.append(1)
            link_labels.append(d); link_colors.append(_rgba_from_hex(node_colors[t], 0.55))
            _remember_link(s, t, link_idx)

        if doc_bucketed:
            counts = defaultdict(int)
            for d in other_docs:
                c_id = doc_to_child.get(d)
                if c_id is not None:
                    counts[c_id] += 1
            s = node_index[("doc_bucket", other_label)]
            for c_id, cnt in counts.items():
                t = node_index[("cluster", c_id)]
                link_idx = len(srcs)
                srcs.append(s); tgts.append(t); vals.append(cnt)
                link_labels.append(f"{cnt} docs"); link_colors.append(_rgba_from_hex(node_colors[t], 0.45))
                _remember_link(s, t, link_idx)

        # Deeper cluster -> cluster edges
        for parent in cluster_nodes:
            for cid in (parent.get("children") or []):
                child = by_id[cid]
                s = node_index[("cluster", parent["id"])]
                t = node_index[("cluster", child["id"])]
                val = len(child.get("doc_ids", []))
                if val <= 0: 
                    continue
                link_idx = len(srcs)
                srcs.append(s); tgts.append(t); vals.append(val)
                link_labels.append(f"{val} docs"); link_colors.append(_rgba_from_hex(node_colors[t], 0.65))
                _remember_link(s, t, link_idx)

    # CASE B: no L0 children -> synthesize virtual L0 clusters from root doc_labels
    else:
        docs = list(root.get("doc_ids", []))
        labs = list(root.get("doc_labels", []))
        if not docs or not labs:
            # Nothing to show; render empty skeleton (title only)
            fig = go.Figure()
            fig.update_layout(title=title, font_size=12)
            if save_html:
                pio.write_html(fig, file=save_html, include_plotlyjs="cdn", full_html=True)
                if open_browser:
                    webbrowser.open(save_html)
            return fig

        # build buckets for virtual L0 clusters
        K = int(max(labs)) + 1 if len(labs) else 0
        buckets = [ [] for _ in range(K) ]
        for d, lab in zip(docs, labs):
            buckets[int(lab)].append(d)

        # add virtual L0 cluster nodes
        for c in range(K):
            idx = len(node_labels)
            node_index[("virtL0", c)] = idx
            idx_to_kind[idx] = "virtL0"
            count = len(buckets[c])
            node_labels.append(f"L0:{c} (n={count})")
            node_colors.append(_mix_with_white(base[c % len(base)], 0.12))

        # doc -> virtual L0 edges
        for d in shown_docs:
            try:
                lab = labs[docs.index(d)]
            except ValueError:
                continue
            s = node_index[("doc", d)]
            t = node_index[("virtL0", int(lab))]
            link_idx = len(srcs)
            srcs.append(s); tgts.append(t); vals.append(1)
            link_labels.append(d); link_colors.append('#bbbbbb')
            _remember_link(s, t, link_idx)

        if doc_bucketed:
            counts = defaultdict(int)
            for d in other_docs:
                try:
                    lab = labs[docs.index(d)]
                except ValueError:
                    continue
                counts[int(lab)] += 1
            s = node_index[("doc_bucket", other_label)]
            for c, cnt in counts.items():
                t = node_index[("virtL0", c)]
                link_idx = len(srcs)
                srcs.append(s); tgts.append(t); vals.append(cnt)
                link_labels.append(f"{cnt} docs"); link_colors.append('#bdbdbd')
                _remember_link(s, t, link_idx)

    node_custom = []

    # 1) Doc nodes first
    for d in shown_docs:
        node_custom.append(f"Document: {d}")

    if doc_bucketed:
        node_custom.append(f"{other_label}: {len(other_docs)} docs")

    # 2) Cluster nodes (for the branch with real L0 children)
    if L0_children:
        for n in cluster_nodes:
            # Use the topic-flow label we attached earlier (if you called annotate_tree_with_topic_flow)
            lbl = n.get("label") or ""
            tf  = n.get("topic_flow", {})
            # Optional: show the top topic→topic pairs (ids or already humanized label text from annotate step)
            pair_lines = []
            for (u, v, sc) in (tf.get("pairs") or [])[:4]:
                # If annotate_tree_with_topic_flow used your named topics, 'label' already carries human text.
                # Otherwise, you'll see numeric IDs here (safe fallback).
                pair_lines.append(f"{u} → {v} (lift={sc:.2f})")
            detail = "<br>".join(pair_lines) if pair_lines else ""
            node_custom.append((lbl + ("<br>"+detail if detail else "")) or f"{len(n.get('doc_ids', []))} docs")

    # 3) Virtual L0 case (if there were no L0 children)
    else:
        # You created K virtual L0 nodes earlier; add simple detail
        for c in range(K):
            node_custom.append(f"Virtual L0 cluster {c}")

    # Build the Sankey figure
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            label=node_labels,
            color=node_colors,
            pad=16,
            thickness=16,
            customdata=node_custom,   # per-node strings
            hovertemplate="%{label}<br>%{customdata}<extra></extra>"
        ),
        link=dict(
            source=srcs, target=tgts, value=vals,
            label=link_labels,
            color=link_colors,
            hovertemplate="Flow: %{value} docs<extra></extra>"
        )
    )])
    levels = sorted(set(n["level"] for n in nodes))
    fig.update_layout(
        title=title,
        font_size=12,
        autosize=True,                      # let Plotly size to container
        margin=dict(l=10, r=10, t=54, b=10) # optional: tighter margins
    )

    # ---------- Build path index for interactivity ----------
    # doc -> path of cluster-node indices (Sankey node indices)
    doc_paths_indices = {}
    doc_to_path_tree = tree.get("doc_to_path", {})
    for d, path_ids in doc_to_path_tree.items():
        # exclude root, map tree node id -> sankey node index ("cluster", id)
        path_idx = []
        for nid in path_ids:
            if nid == root_id:
                continue
            key = ("cluster", nid)
            if key in node_index:
                path_idx.append(node_index[key])
        doc_paths_indices[d] = path_idx

    # doc -> sankey doc-node index (only for visible docs)
    doc_idx_map = {d: node_index[("doc", d)] for d in shown_docs if ("doc", d) in node_index}

    # JS metadata mapping (serialized into the HTML)
    mapping = {
        "doc_idx_map": doc_idx_map,                                      # docId -> doc-node index
        "doc_paths": doc_paths_indices,                                   # docId -> [cluster-node indices...]
        "node_kind_by_idx": idx_to_kind,                                  # idx -> kind
        "idx_to_doc": {str(idx): d for idx, d in idx_to_doc.items()},     # idx -> docId (only for doc nodes)
        "link_index": link_index_map                                      # "srcIdx-tgtIdx" -> linkIndex
    }
    mapping_json = json.dumps(mapping)

    # ---------- Emit interactive HTML with hover highlighting ----------
    if save_html:
        # Build the post_script to attach hover handlers
        # NOTE: 'gd' is the graph div; we compute highlight arrays and restyle on hover/unhover.
        post_script = f"""
(function(){{
  var gd = document.getElementById("{div_id}");
  var mapping = {mapping_json};

  // ------- utils -------
  function cloneArr(a) {{ return (a || []).slice(); }}
  function fill(n, val) {{ var x=new Array(n); for (var i=0;i<n;i++) x[i]=val; return x; }}

  // dim palette + highlight accents
  var DIM_NODE = 'rgba(200,200,200,0.10)';
  var DIM_LINK = 'rgba(200,200,200,0.04)';
  var H_NODE  = '#111111';               // node accent when doc node is hovered (optional)
  var H_LINK  = 'rgba(8,81,156,0.95)';   // strong blue for highlighted links

  // cache originals (immutable)
  var baseNode = cloneArr(gd.data[0].node.color);
  var baseLink = cloneArr(gd.data[0].link.color);

  function restyle(nodeColors, linkColors) {{
    Plotly.restyle(gd, {{'node.color':[nodeColors], 'link.color':[linkColors]}}, [0]);
  }}

  // Build a map from (srcIdx,tgtIdx) -> linkIndex for quick lookup
  var linkIndex = mapping.link_index || {{}};

  function highlightDoc(docId) {{
    var Nn = baseNode.length, Nl = baseLink.length;
    var ncol = fill(Nn, DIM_NODE), lcol = fill(Nl, DIM_LINK);

    var docIdx = mapping.doc_idx_map[docId];
    var path = mapping.doc_paths[docId] || [];

    // doc node (if visible)
    if (docIdx !== undefined) ncol[docIdx] = baseNode[docIdx] || H_NODE;

    // traverse full path: doc -> c1 -> c2 -> ...
    var prev = (docIdx !== undefined) ? docIdx : undefined;
    for (var i=0;i<path.length;i++) {{
      var ni = path[i];
      ncol[ni] = baseNode[ni] || H_NODE;

      if (prev !== undefined) {{
        var k = prev + '-' + ni;
        var li = linkIndex[k];
        if (li !== undefined) lcol[li] = H_LINK;
      }}
      prev = ni;
    }}
    restyle(ncol, lcol);
  }}

  function highlightCluster(clusterIdx) {{
    var Nn = baseNode.length, Nl = baseLink.length;
    var ncol = fill(Nn, DIM_NODE), lcol = fill(Nl, DIM_LINK);

    // always highlight the hovered cluster node
    ncol[clusterIdx] = baseNode[clusterIdx] || H_NODE;

    // union of all doc paths that include this cluster
    var docPaths = mapping.doc_paths || {{}};
    for (var docId in docPaths) {{
      var path = docPaths[docId] || [];
      var has = false;
      for (var i=0;i<path.length;i++) if (path[i] === clusterIdx) {{ has = true; break; }}
      if (!has) continue;

      var docIdx = mapping.doc_idx_map[docId];
      if (docIdx !== undefined) ncol[docIdx] = baseNode[docIdx] || H_NODE;

      var prev = (docIdx !== undefined) ? docIdx : undefined;
      for (var i=0;i<path.length;i++) {{
        var ni = path[i];
        ncol[ni] = baseNode[ni] || H_NODE;
        if (prev !== undefined) {{
          var k = prev + '-' + ni;
          var li = linkIndex[k];
          if (li !== undefined) lcol[li] = H_LINK;
        }}
        prev = ni;
      }}
    }}
    restyle(ncol, lcol);
  }}

  gd.on('plotly_hover', function(evt){{
    if (!evt || !evt.points || !evt.points.length) return;
    var p = evt.points[0];
    if (p.curveNumber !== 0 || typeof p.pointNumber !== 'number') return;

    // node index hovered
    var idx = p.pointNumber;
    // mapping keys may be strings; check both
    var kind = mapping.node_kind_by_idx[String(idx)] || mapping.node_kind_by_idx[idx];

    if (kind === 'doc') {{
      var docId = mapping.idx_to_doc[String(idx)];
      if (docId !== undefined) highlightDoc(docId);
    }} else if (kind === 'cluster' || kind === 'virtL0') {{
      highlightCluster(idx);
    }} else {{
      // bucket or others: ignore
    }}
  }});

  gd.on('plotly_unhover', function(evt){{
    // Restore the exact original colors
    restyle(baseNode.slice(), baseLink.slice());
  }});

  // --------- Responsive height & scroll support ----------
  function _calcNeededHeightPx(mult) {{
    var wrap = document.getElementById("{div_id}-wrap");
    var base = wrap ? wrap.clientHeight : Math.floor(window.innerHeight * {locals().get("height_vh", 92)}/100);
    // Try to estimate needed height from level counts if provided; else heuristic
    var lvl = (mapping.level_counts || []);
    var levels = lvl.length || 3;
    var maxLevelCount = 0;
    for (var i=0;i<lvl.length;i++) if (lvl[i] > maxLevelCount) maxLevelCount = lvl[i];
    if (!maxLevelCount) {{
      var nn = (gd.data && gd.data[0] && gd.data[0].node && (gd.data[0].node.label||[]) ).length || 0;
      maxLevelCount = Math.max(8, Math.floor(nn / Math.max(1, levels)));
    }}
    var needed = 220 + levels*120 + maxLevelCount*28; // base + per-level + per-node spacing
    if (!isFinite(needed) || needed < base) needed = base;
    if (mult && mult > 1) needed = Math.floor(needed * mult);
    return needed;
  }}

  function ensureScrollable(mult) {{
    var wrap = document.getElementById("{div_id}-wrap");
    var div  = document.getElementById("{div_id}");
    if (!div) return;
    var need = _calcNeededHeightPx(mult);
    if (wrap) {{
      // If needed height exceeds wrapper height, enlarge inner plot div -> triggers scrollbars
      var target = Math.max(need, wrap.clientHeight);
      if (div.clientHeight !== target) {{
        div.style.height = target + "px";
        Plotly.Plots.resize(gd);
      }}
    }} else {{
      // No wrapper: set explicit pixel height on the plot div
      if (div.clientHeight !== need) {{
        div.style.height = need + "px";
        Plotly.Plots.resize(gd);
      }}
    }}
  }}

  // Initial pass (after current call stack)
  setTimeout(function() {{ ensureScrollable(1.0); }}, 0);

  // On window resize, refit to wrapper
  window.addEventListener('resize', function() {{ ensureScrollable(1.0); }});

  // When user clicks a node (often triggers spreading), give extra headroom to force scroll
  gd.on('plotly_click', function(evt) {{
    ensureScrollable(1.25);   // modest growth; adjust multiplier if you want more room
  }});
}})();
"""
        # --- Append a resize handler so the graph reflows with window height changes ---
        post_script = post_script + f"""
(function(){{
  var gd = document.getElementById("{div_id}");
  if (gd) {{
    window.addEventListener('resize', function() {{ Plotly.Plots.resize(gd); }});
  }}
}})();
"""

        # --- Viewport-based height CSS (optionally scrollable wrapper for very tall graphs) ---
        _height_vh = locals().get("height_vh", 92)                       # % of viewport height
        _use_scroll_wrapper = locals().get("use_scroll_wrapper", True)   # set True to enable vertical scrolling

        if _use_scroll_wrapper:
            style_tag = f"""
<style>
  html, body {{ height: 100%; margin: 0; padding: 0; }}
  #{div_id}-wrap {{ height: {_height_vh}vh; width: 100%; overflow-y: auto; overscroll-behavior: contain; }}
  #{div_id} {{ height: 100%; width: 100%; }}
</style>
"""
        else:
            style_tag = f"""
<style>
  html, body {{ height: 100%; margin: 0; padding: 0; }}
  #{div_id} {{ height: {_height_vh}vh; width: 100%; }}
</style>
"""

        # --- Generate responsive HTML; default to 100% width and viewport-based height --- ensureScrollable
        html = pio.to_html(
            fig,
            include_plotlyjs="cdn",
            full_html=True,
            div_id=div_id,
            post_script=post_script,
            config={"responsive": True},
            default_width="100%",
            default_height=f"{_height_vh}vh"
        )

        # Inject CSS <style> into <head>
        if "</head>" in html:
            html = html.replace("</head>", style_tag + "</head>")
        else:
            html = style_tag + html

        # Optional: wrap the plot div in a scrollable container if requested
        if _use_scroll_wrapper:
            html = html.replace(
                f'<div id="{div_id}"',
                f'<div id="{div_id}-wrap"><div id="{div_id}"'
            ).replace(
                f'</div>\n</body>',
                f'</div></div>\n</body>'
            )

        with open(save_html, "w", encoding="utf-8") as f:
            f.write(html)
        if open_browser:
            webbrowser.open(save_html)
        return fig

    # If save_html not given, just return the figure (no custom JS attached)
    return fig

# --------------------------------------------------------
# def build_categorical_arrangement_from_cdm_tuple(document_delta_dict, adapter_kwargs, label_topk, weight_min, two_hop, topk_per_src, doc_agg, doc_topk, seed_strategy, merge_protos, proto_cos_th, doc_jacc_th, membership_th)
# Summary:
#   Adapts a document‑delta manifold (dict or 6‑tuple per doc) into a global
#   categorical “themes” organization:
#   • Uses project adapters to convert cluster/delta data into a graph of
#   (clusters, morphisms).
#   • Creates a labeler from seed terms (top‑k per cluster) for human‑readable
#   theme names.
#   • Calls the global organization routine to construct themes and compute
#   (a) theme metadata, (b) document‑by‑theme memberships, and
#   (c) theme‑to‑theme overlaps—respecting thresholds and pruning knobs:
#   – weight_min: drop weak edges
#   – two_hop + topk_per_src: allow/prune composed morphisms
#   – doc_agg + doc_topk: control doc membership aggregation
#   – seed_strategy, merge_protos, proto_cos_th, doc_jacc_th, membership_th:
#   guide labeling, prototype merging, and noise suppression.
# Effect:
#   Lifts low‑level morphism geometry into stable, interpretable themes that
#   summarize how texts transform:
#   • Composed (two‑hop) morphisms capture multi‑step semantic shifts observed
#   in documents, aligning with the project’s view of semantic morphisms
#   as composable transformations.
#   • Seed‑term labeling and prototype merging translate dense vector relations
#   into compact, named categories suitable for analysis, reporting, and
#   downstream visualization (e.g., Sankey).
#   • The returned themes, membership, and overlap tables provide a bridge from
#   per‑document manifold structure to collection‑level thematic organization,
#   enabling comparative study and arrangement at scale.
# --------------------------------------------------------
def build_categorical_arrangement_from_cdm_tuple(   # keep the public name to avoid refactors
    document_delta_dict: Any,                       # now supports dict or 6-tuple
    adapter_kwargs: Dict[str, Any] = None,
    label_topk: int = 6,
    weight_min: float = 0.0,
    two_hop: bool = False,
    topk_per_src: int = 3,
    doc_agg: str = "topk",
    doc_topk: int = 3,
    seed_strategy: str = "community",
    merge_protos: bool = True,
    proto_cos_th: float = 0.97,
    doc_jacc_th: float = 0.8,
    membership_th: float = 0.0,
):
    if adapter_kwargs is None:
        adapter_kwargs = {}

    clusters, morphisms = adapt_from_cdm(document_delta_dict, **adapter_kwargs)

    labeler = make_seed_terms_labeler(clusters, top_k=label_topk)
    theme_set, themes_df, membership_df, overlaps_df = build_global_categorical_organization(
        clusters, morphisms,
        weight_min=weight_min,
        include_identity=True,
        two_hop=two_hop,
        topk_per_src=topk_per_src,
        doc_agg=doc_agg,
        doc_topk=doc_topk,
        seed_strategy=seed_strategy,
        merge_protos=merge_protos,
        proto_cos_threshold=proto_cos_th,
        doc_jacc_threshold=doc_jacc_th,
        membership_threshold=membership_th,
        theme_label_func=labeler,
    )
    return theme_set, themes_df, membership_df, overlaps_df


# --------------------------------------------------------
# def build_categorical_arrangement_from_mkdelta(mk_delta_kwargs, label_topk, weight_min, two_hop, topk_per_src, doc_agg, doc_topk, seed_strategy, merge_protos, proto_cos_th, doc_jacc_th, membership_th)
# Summary:
#   Convenience wrapper that (1) builds a delta manifold via mk_delta_manifold,
#   (2) adapts it into (clusters, morphisms), and (3) runs the same global
#   categorical organization pipeline with the labeling/threshold knobs described
#   above—returning themes, doc‑membership, and overlaps.
# Effect:
#   Connects the text‑to‑manifold construction step directly to the theme
#   arrangement step in one call:
#   • Ensures that freshly computed morphisms (from sentence clusters, deltas,
#   and principal directions) can be immediately summarized as categories
#   usable in UI/analysis, preserving parameter parity and reproducibility.
#   • Facilitates rapid iteration on preprocessing or segmentation choices by
#   collapsing pipeline hand‑offs into a single, auditable entry point.
# --------------------------------------------------------
from typing import Dict, Any, Tuple
def build_categorical_arrangement_from_mkdelta(
    mk_delta_kwargs: Dict[str, Any] = None,
    label_topk: int = 6,
    weight_min: float = 0.0,
    two_hop: bool = False,
    topk_per_src: int = 3,
    doc_agg: str = "topk",
    doc_topk: int = 3,
    seed_strategy: str = "community",
    merge_protos: bool = True,
    proto_cos_th: float = 0.97,
    doc_jacc_th: float = 0.8,
    membership_th: float = 0.0,
):
    """
    Calls generate_document_delta_manifold.mk_delta_manifold(**mk_delta_kwargs)
    and adapts the result into the global categorical organization.
    """
    if mk_delta_kwargs is None: mk_delta_kwargs = {}
    from generate_document_delta_manifold import mk_delta_manifold
    dm = mk_delta_manifold(**mk_delta_kwargs)

    from project_adapter_mkdelta import adapt_from_mkdelta_object
    clusters, morphisms = adapt_from_mkdelta_object(dm)

    from global_categorical_themes import build_global_categorical_organization, make_seed_terms_labeler
    labeler = make_seed_terms_labeler(clusters, top_k=label_topk)
    theme_set, themes_df, membership_df, overlaps_df = build_global_categorical_organization(
        clusters, morphisms,
        weight_min=weight_min,
        include_identity=True,
        two_hop=two_hop,
        topk_per_src=topk_per_src,
        doc_agg=doc_agg,
        doc_topk=doc_topk,
        seed_strategy=seed_strategy,
        merge_protos=merge_protos,
        proto_cos_threshold=proto_cos_th,
        doc_jacc_threshold=doc_jacc_th,
        membership_threshold=membership_th,
        theme_label_func=labeler,
    )
    return theme_set, themes_df, membership_df, overlaps_df
