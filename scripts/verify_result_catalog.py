"""Check preserved experiment files against the recorded catalog."""
import csv
import hashlib
import math
from pathlib import Path


root = Path(__file__).resolve().parents[1]
with (root / "results/catalog.csv").open(newline="") as handle:
    records = list(csv.DictReader(handle))
paths = set()
hashes = set()
for record in records:
    path = root / record["path"]
    assert path not in paths, f"Duplicate catalog path: {path}"
    paths.add(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == record["sha256"], f"Changed raw data: {path}"
    hashes.add(digest)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["round", "loss", "accuracy"], path
        rows = list(reader)
    assert len(rows) == int(record["rounds"]) == 10, path
    assert [int(row["round"]) for row in rows] == list(range(1, 11)), path
    for row in rows:
        assert math.isfinite(float(row["loss"])) and float(row["loss"]) >= 0, path
        assert 0 <= float(row["accuracy"]) <= 1, path
    assert rows[-1]["accuracy"] == record["final_accuracy"], path
    assert rows[-1]["loss"] == record["final_loss"], path
actual = set((root / "results").rglob("*.csv")) - {root / "results/catalog.csv"}
assert actual == paths, "CSV inventory differs from catalog"
print(f"Verified {len(paths)} files, {len(hashes)} unique byte contents; duplicates are not repeats.")
