import pickle
from pathlib import Path

# utility script that analyzes the segment pkl file and creates segment count metrics 

segments_path = Path("C:\datasets\ISandT_published_test_sets\DPQ_SC_many_k5-8\DPQ_SC_many_k8-12_segments.pkl")

with segments_path.open("rb") as f:
    segments_by_doc = pickle.load(f)

if not isinstance(segments_by_doc, dict) or not segments_by_doc:
    raise ValueError("segments_by_doc.pkl did not contain a non-empty dictionary.")

counts = {
    str(doc_id): len(segments)
    for doc_id, segments in segments_by_doc.items()
    if segments is not None
}

min_doc = min(counts, key=counts.get)
max_doc = max(counts, key=counts.get)

print(f"Documents counted: {len(counts):,}")
print(f"Fewest segments: {min_doc} = {counts[min_doc]:,}")
print(f"Most segments:   {max_doc} = {counts[max_doc]:,}")

values = list(counts.values())
print(f"Mean segments/document: {sum(values) / len(values):,.2f}")
print(f"Total segments: {sum(values):,}")