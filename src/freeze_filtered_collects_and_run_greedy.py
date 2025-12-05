import argparse
import os

import numpy as np

from load_graph_and_align import (
    BASE as DEFAULT_BASE,
    GRAPH_PATTERN,
    load_collects,
    load_metis_graph,
)

ANGLE_KEYS = (
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
)


def extract_angle_deg(record):
    ap = record.get("access_properties") or {}
    for key in ANGLE_KEYS:
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
    angles = np.array([extract_angle_deg(r) for r in records], dtype=np.float64)
    mask = np.isfinite(angles) & (angles <= limit_deg + 1e-9)
    return mask


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


def main():
    parser = argparse.ArgumentParser(
        description="Freeze <=55° collects to METIS order and run greedy MWIS heuristics."
    )
    parser.add_argument("--base", default=DEFAULT_BASE, help="Root data directory")
    parser.add_argument("--sats", type=int, default=4, help="Constellation size (4/12/24/36)")
    parser.add_argument("--angle-max", type=float, default=55.0, help="Look angle cutoff (deg)")
    args = parser.parse_args()

    collects = load_collects(args.base, args.sats)
    recs = collects["records"]
    mask = angle_filter_mask(recs, args.angle_max)
    selected = int(mask.sum())

    graph_path = os.path.join(args.base, GRAPH_PATTERN.format(sats=args.sats))
    indptr, indices = load_metis_graph(graph_path)
    n_graph = len(indptr) - 1

    print(f"Angle-filtered records: {selected} / {len(recs)} (graph expects {n_graph})")
    if selected != n_graph:
        raise ValueError(f"Filtered collects ({selected}) do not match graph nodes ({n_graph})")

    w = np.ascontiguousarray(collects["w"][mask], dtype=np.float64)
    t_end = np.ascontiguousarray(collects["t_end"][mask], dtype=np.float64)
    dwell = np.ascontiguousarray(collects["dwell"][mask], dtype=np.float64)

    safe_dwell = np.maximum(dwell, 1.0)
    order_edf = np.argsort(t_end, kind="mergesort")
    order_density = np.argsort(-(w / safe_dwell), kind="mergesort")
    order_heavy = np.argsort(-w, kind="mergesort")

    for name, order in (
        ("edf", order_edf),
        ("density", order_density),
        ("heaviest", order_heavy),
    ):
        chosen, total = greedy_mwis(order, indptr, indices, w)
        print(f"{name:8s} picks={int(chosen.sum()):6d} total_value={total:.2f}")


if __name__ == "__main__":
    main()
