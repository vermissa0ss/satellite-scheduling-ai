import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from load_graph_and_align import (
    BASE as DEFAULT_BASE,
    GRAPH_PATTERN,
    load_collects,
    load_metis_graph,
)


ANGLE_LIMIT_DEFAULT = 55.0
FROZEN_DIR = Path("frozen")


def angle_max_deg(record):
    ap = record.get("access_properties") or {}
    for key in (
        "look_angle_max",
        "look_angle",
        "look_angle_deg",
        "off_nadir_deg",
        "off_nadir_angle_deg",
        "sensor_view_angle_deg",
        "view_angle_deg",
        "incidence_angle_deg",
        "lookAngleDeg",
        "look_angle_min",
    ):
        if key in ap and ap[key] is not None:
            try:
                return float(ap[key])
            except (TypeError, ValueError):
                return np.nan
    for key in ("look_angle_rad", "off_nadir_rad"):
        if key in ap and ap[key] is not None:
            try:
                return float(ap[key]) * 180.0 / np.pi
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def angle_filter_mask(records, limit_deg):
    angles = np.array([angle_max_deg(r) for r in records], dtype=np.float64)
    return np.isfinite(angles) & (angles <= limit_deg + 1e-9)


def greedy_mwis(order, indptr, indices, weights):
    n = len(weights)
    chosen = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    total = 0.0
    for v in order:
        if not forbidden[v]:
            chosen[v] = True
            total += float(weights[v])
            start, end = indptr[v], indptr[v + 1]
            forbidden[indices[start:end]] = True
            forbidden[v] = True
    return chosen, total


def connected_components(indptr, indices):
    n = len(indptr) - 1
    comp_id = np.full(n, -1, dtype=np.int32)
    comp_order = np.empty(n, dtype=np.int64)
    comp_ptr = [0]
    write = 0
    comp_idx = 0

    for v in range(n):
        if comp_id[v] != -1:
            continue
        stack = [v]
        comp_vertices = []
        comp_id[v] = comp_idx
        while stack:
            u = stack.pop()
            comp_vertices.append(u)
            nbrs = indices[indptr[u] : indptr[u + 1]]
            for nbr in nbrs:
                if comp_id[nbr] == -1:
                    comp_id[nbr] = comp_idx
                    stack.append(nbr)
        comp_vertices.sort()
        size = len(comp_vertices)
        comp_order[write : write + size] = comp_vertices
        write += size
        comp_ptr.append(write)
        comp_idx += 1

    return comp_id, comp_order, np.asarray(comp_ptr, dtype=np.int64)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset(base_dir, sats, angle_max):
    collects = load_collects(base_dir, sats)
    records = collects["records"]
    mask = angle_filter_mask(records, angle_max)
    selected = int(mask.sum())

    graph_path = os.path.join(base_dir, GRAPH_PATTERN.format(sats=sats))
    indptr, indices = load_metis_graph(graph_path)
    n_graph = len(indptr) - 1

    if selected != n_graph:
        raise ValueError(
            f"Filtered collects ({selected}) do not match METIS vertices ({n_graph}) for {sats} sats"
        )

    metis_to_rec = np.flatnonzero(mask).astype(np.int64)
    w = np.ascontiguousarray(collects["w"][mask], dtype=np.float64)
    t_end = np.ascontiguousarray(collects["t_end"][mask], dtype=np.float64)
    dwell = np.ascontiguousarray(collects["dwell"][mask], dtype=np.float64)
    sat = np.ascontiguousarray(collects["sat"][mask], dtype=np.int32)

    deg = np.diff(indptr).astype(np.int32, copy=False)
    safe_dwell = np.maximum(dwell, 1.0)

    order_heaviest = np.argsort(-w, kind="mergesort")
    order_degree = np.argsort(-deg, kind="mergesort")
    order_deadline = np.argsort(t_end, kind="mergesort")
    order_hybrid = np.lexsort((t_end, -deg.astype(np.int64), -w))

    comp_id, comp_order, comp_ptr = connected_components(indptr, indices)

    baselines = {}
    for name, order in (
        ("edf", order_deadline),
        ("density", np.argsort(-(w / safe_dwell), kind="mergesort")),
        ("heaviest", order_heaviest),
    ):
        chosen, total = greedy_mwis(order, indptr, indices, w)
        baselines[name] = {"picks": int(chosen.sum()), "total": float(total)}

    dataset_id = f"worldcities_1d_{sats}sats"
    bundle_dir = FROZEN_DIR / dataset_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    npz_path = bundle_dir / "data.npz"
    meta_path = bundle_dir / "meta.json"

    np.savez_compressed(
        npz_path,
        indptr=indptr.astype(np.int64, copy=False),
        indices=indices.astype(np.int32, copy=False),
        w=w,
        t_end=t_end,
        dwell=dwell,
        sat=sat,
        metis_to_rec=metis_to_rec,
        degree=deg,
        order_heaviest=order_heaviest.astype(np.int64),
        order_degree=order_degree.astype(np.int64),
        order_deadline=order_deadline.astype(np.int64),
        order_hybrid=order_hybrid.astype(np.int64),
        comp_id=comp_id,
        comp_order=comp_order,
        comp_ptr=comp_ptr,
    )

    rel_graph = os.path.relpath(graph_path)
    sources = [os.path.relpath(path) for path in collects.get("sources", [])]
    stats = {
        "degree_avg": float(deg.mean()),
        "degree_max": int(deg.max(initial=0)),
        "w_min": float(w.min(initial=0.0)),
        "w_max": float(w.max(initial=0.0)),
        "t_end_min": float(np.nanmin(t_end)),
        "t_end_max": float(np.nanmax(t_end)),
    }
    meta = {
        "dataset_id": dataset_id,
        "sats": sats,
        "days": 1,
        "filter": {"look_angle_max_deg": angle_max, "look_direction": "both"},
        "counts": {
            "n_vertices": n_graph,
            "n_edges_directed": int(indices.size),
            "n_edges_undirected": int(indices.size // 2),
            "n_components": int(comp_ptr.size - 1),
        },
        "baselines": baselines,
        "stats": stats,
        "dtypes": {
            name: str(arr.dtype)
            for name, arr in {
                "indptr": indptr,
                "indices": indices,
                "w": w,
                "t_end": t_end,
                "dwell": dwell,
                "sat": sat,
                "metis_to_rec": metis_to_rec,
                "degree": deg,
                "order_heaviest": order_heaviest,
                "order_degree": order_degree,
                "order_deadline": order_deadline,
                "order_hybrid": order_hybrid,
                "comp_id": comp_id,
                "comp_order": comp_order,
                "comp_ptr": comp_ptr,
            }.items()
        },
        "shapes": {
            name: list(arr.shape)
            for name, arr in {
                "indptr": indptr,
                "indices": indices,
                "w": w,
                "t_end": t_end,
                "dwell": dwell,
                "sat": sat,
                "metis_to_rec": metis_to_rec,
                "degree": deg,
                "order_heaviest": order_heaviest,
                "order_degree": order_degree,
                "order_deadline": order_deadline,
                "order_hybrid": order_hybrid,
                "comp_id": comp_id,
                "comp_order": comp_order,
                "comp_ptr": comp_ptr,
            }.items()
        },
        "graph_file": rel_graph,
        "graph_sha256": sha256_file(graph_path),
        "source_collect_files": sources,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    verify_bundle(npz_path, meta)
    return npz_path, meta_path


def verify_bundle(npz_path, meta):
    data = np.load(npz_path)
    for name, shape in meta["shapes"].items():
        if name not in data:
            raise ValueError(f"Missing array '{name}' in {npz_path}")
        arr = data[name]
        expected_shape = tuple(shape)
        if arr.shape != expected_shape:
            raise ValueError(f"Shape mismatch for {name}: {arr.shape} != {expected_shape}")
        expected_dtype = meta["dtypes"][name]
        if str(arr.dtype) != expected_dtype:
            raise ValueError(f"Dtype mismatch for {name}: {arr.dtype} != {expected_dtype}")

    indptr = data["indptr"]
    n = indptr.size - 1
    for key in ("w", "t_end", "dwell", "sat"):
        if data[key].shape[0] != n:
            raise ValueError(f"{key} length {data[key].shape[0]} != {n}")

    degree = np.diff(indptr)
    stats = {
        "degree_avg": float(degree.mean()),
        "degree_max": int(degree.max(initial=0)),
        "w_min": float(data["w"].min(initial=0.0)),
        "w_max": float(data["w"].max(initial=0.0)),
        "t_end_min": float(np.nanmin(data["t_end"])),
        "t_end_max": float(np.nanmax(data["t_end"])),
    }
    for key, val in stats.items():
        exp = meta["stats"][key]
        if not np.isclose(val, exp, atol=1e-6):
            raise ValueError(f"Stat mismatch {key}: {val} != {exp}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build frozen datasets for satellite MIS scenarios.")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Root data directory")
    parser.add_argument(
        "--sats",
        nargs="+",
        type=int,
        default=[4],
        help="Constellation sizes to process (e.g., 4 12 24 36)",
    )
    parser.add_argument("--angle-max", type=float, default=ANGLE_LIMIT_DEFAULT, help="Look angle cutoff")
    return parser.parse_args()


def main():
    args = parse_args()
    FROZEN_DIR.mkdir(exist_ok=True)

    for sats in args.sats:
        npz_path, meta_path = build_dataset(args.base, sats, args.angle_max)
        print(f"Wrote dataset for {sats} sats -> {npz_path}, {meta_path}")


if __name__ == "__main__":
    main()
