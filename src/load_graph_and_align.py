import argparse
import glob
import json
import os
import re
from datetime import datetime
from typing import Dict, Tuple

import numpy as np

BASE = r"C:\Users\Vermi\Documents\MyStanford\CS221\Project\Data"
COLLECT_PATTERN = "worldcities_1_days_{sats}_sats_collects_sc_*.json"
GRAPH_PATTERN = "worldcities_1_days_{sats}_sats_metis_graph.metis"


def _to_epoch_seconds(ts):
    """Convert ISO8601 timestamp strings (or numeric) to POSIX seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).timestamp()


def load_collects(base: str, sats: int) -> Dict[str, np.ndarray]:
    """Load and concatenate all collect JSONs for the given constellation."""
    pattern = os.path.join(base, COLLECT_PATTERN.format(sats=sats))
    files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"_sc_(\d+)\.json$", p).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No collect files match {pattern}")

    recs = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            recs.extend(json.load(f))
    if not recs:
        raise ValueError(f"Collect files for {sats} sats were empty")

    ids = np.array([r["id"] for r in recs], dtype=object)
    t_start = np.array([_to_epoch_seconds(r["t_start"]) for r in recs], dtype=np.float64)
    t_end = np.array([_to_epoch_seconds(r["t_end"]) for r in recs], dtype=np.float64)
    dwell = np.array(
        [float(r.get("t_duration") or (te - ts)) for r, ts, te in zip(recs, t_start, t_end)],
        dtype=np.float64,
    )
    sat = np.array([int(r["spacecraft_id"]) for r in recs], dtype=np.int32)
    weights = np.array([float(r.get("reward", 1.0)) for r in recs], dtype=np.float64)

    return {
        "records": recs,
        "ids": ids,
        "t_start": t_start,
        "t_end": t_end,
        "dwell": dwell,
        "sat": sat,
        "w": weights,
        "sources": files,
    }


def load_metis_graph(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse an (optionally weighted) METIS graph into CSR arrays."""
    indptr = [0]
    indices = []
    with open(path, "r", encoding="utf-8") as f:
        # read header (skip blanks/comments)
        def _next_line():
            line = f.readline()
            while line and (line.strip() == "" or line.lstrip().startswith("%")):
                line = f.readline()
            return line

        hdr = _next_line()
        if not hdr:
            raise ValueError("Empty METIS file")
        parts = hdr.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Bad METIS header: {hdr}")
        n = int(parts[0])
        # m (edge count) is parts[1], may not be exact if weighted; we don't rely on it
        fmt = parts[2] if len(parts) >= 3 else "0"
        ncon = int(parts[3]) if len(parts) >= 4 else 1

        # fmt is a string like '011' where last bit=edge weights, second-last=node weights
        has_node_w = len(fmt) >= 2 and fmt[-2] == "1"
        has_edge_w = len(fmt) >= 1 and fmt[-1] == "1"

        for row in range(n):
            line = _next_line()
            if not line:
                raise ValueError("Unexpected EOF in adjacency section")
            toks = line.strip().split()
            idx = 0
            if has_node_w:
                idx += ncon  # skip node weight(s)
            if has_edge_w:
                # neighbor, weight, neighbor, weight, ...
                t = toks[idx:]
                neigh = [int(t[j]) - 1 for j in range(0, len(t), 2)]
            else:
                neigh = [int(x) - 1 for x in toks[idx:]] if len(toks) > idx else []
            if any(v < 0 or v >= n for v in neigh):
                raise ValueError(f"Invalid neighbor index on line {row+2}: {neigh}")
            indices.extend(neigh)
            indptr.append(len(indices))

    return np.asarray(indptr, dtype=np.int64), np.asarray(indices, dtype=np.int32)


def align_graph_and_features(base: str, sats: int):
    collects = load_collects(base, sats)
    graph_path = os.path.join(base, GRAPH_PATTERN.format(sats=sats))
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Missing METIS graph at {graph_path}")
    indptr, indices = load_metis_graph(graph_path)
    n_graph = len(indptr) - 1
    n_collects = len(collects["records"])
    if n_graph != n_collects:
        raise ValueError(
            f"Node count mismatch: graph has {n_graph} vertices but collects supplied {n_collects}"
        )

    features = {
        "ids": collects["ids"],
        "w": np.ascontiguousarray(collects["w"], dtype=np.float64),
        "t_start": np.ascontiguousarray(collects["t_start"], dtype=np.float64),
        "t_end": np.ascontiguousarray(collects["t_end"], dtype=np.float64),
        "dwell": np.ascontiguousarray(collects["dwell"], dtype=np.float64),
        "sat": np.ascontiguousarray(collects["sat"], dtype=np.int32),
    }
    return {"graph": {"indptr": indptr, "indices": indices}, "features": features}


def main():
    parser = argparse.ArgumentParser(description="Load METIS graph and align collect features.")
    parser.add_argument("--base", default=BASE, help="Root data directory")
    parser.add_argument("--sats", type=int, default=4, help="Number of satellites (4,12,24,36)")
    args = parser.parse_args()

    aligned = align_graph_and_features(args.base, args.sats)
    graph = aligned["graph"]
    feats = aligned["features"]

    n = len(graph["indptr"]) - 1
    deg = np.diff(graph["indptr"])
    edges = graph["indices"].size
    avg_degree = float(deg.mean()) if deg.size else 0.0
    max_degree = int(deg.max()) if deg.size else 0
    w_min, w_max = float(feats["w"].min()), float(feats["w"].max())
    t_min = float(np.nanmin(feats["t_end"])) if feats["t_end"].size else float("nan")
    t_max = float(np.nanmax(feats["t_end"])) if feats["t_end"].size else float("nan")

    print(f"{args.sats}-sat graph located. Node/collect count: {n}")
    print(f"degree avg={avg_degree:.2f} max={max_degree}, edges={edges}")
    print(f"weights min={w_min:.3f} max={w_max:.3f} | t_end {t_min:.1f} -> {t_max:.1f}")


if __name__ == "__main__":
    main()
