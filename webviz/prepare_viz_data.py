"""Build a lightweight JSON payload for the web-based globe visualizer.

This script trims the bulky solver outputs into a browser-friendly bundle:
- Samples a subset of collect requests (with lat/lon/time/reward/spacecraft id).
- Extracts short orbit trails per spacecraft for animated arcs.
- Slims the evaluation summary so we can show scores without loading best masks.

Example (uses the latest 12-sat run + 4 collect files by default):
    python webviz/prepare_viz_data.py

You can point it at any other summary/collect files:
    python webviz/prepare_viz_data.py \
      --summary results/worldcities_1d_4sats/<run>/summary.json \
      --collect-pattern \"Data/worldcities_1_days_4_sats_collects_sc_*.json\" \
      --sample-size 800 --track-points 240
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


@dataclass
class RequestPoint:
    lon: float
    lat: float
    reward: float
    spacecraft_id: int
    t_start: str
    t_mid: str
    t_end: str
    look_angle_min: float | None
    look_angle_max: float | None
    request_id: str
    tile_id: str | None

    @property
    def timestamp(self) -> float:
        ref = self.t_mid or self.t_start or self.t_end
        return iso_to_ts(ref) if ref else 0.0


def iso_to_ts(value: str) -> float:
    """Convert ISO time to epoch seconds."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def ts_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(ISO_FMT)


def even_sample(items: List, target: int) -> List:
    """Deterministically sample items by walking them at a fixed stride."""
    if target <= 0 or target >= len(items):
        return list(items)
    stride = max(1, len(items) // target)
    sampled = [items[i] for i in range(0, len(items), stride)]
    return sampled[:target]


def slim_components(raw_components: List[dict]) -> List[dict]:
    slimmed = []
    for comp in raw_components:
        greedy = comp.get("greedy", {}) or {}
        greedy_best_weight = -math.inf
        greedy_best_name = None
        greedy_weights: Dict[str, float] = {}
        for name, data in greedy.items():
            w = data.get("weight")
            greedy_weights[name] = w
            if w is not None and w > greedy_best_weight:
                greedy_best_weight = w
                greedy_best_name = name

        search_entries = []
        best_search = {"name": None, "weight": -math.inf}
        for name in ("bnb", "astar", "wastar", "ls"):
            if name not in comp:
                continue
            info = comp[name] or {}
            entry = {
                "name": name,
                "weight": info.get("best_weight") or info.get("weight"),
                "time": info.get("time"),
                "nodes": info.get("nodes"),
                "interrupted": info.get("interrupted", False),
            }
            search_entries.append(entry)
            if entry["weight"] is not None and entry["weight"] > best_search["weight"]:
                best_search = entry

        slimmed.append(
            {
                "id": comp.get("component"),
                "size": comp.get("size"),
                "greedy_best": {
                    "name": greedy_best_name,
                    "weight": greedy_best_weight if greedy_best_name else None,
                },
                "search_best": best_search if best_search["name"] else None,
                "greedy_weights": greedy_weights,
                "search": search_entries,
            }
        )
    return slimmed


def build_tracks(requests: Dict[int, List[RequestPoint]], max_points: int) -> List[dict]:
    tracks = []
    for sc_id, reqs in requests.items():
        if not reqs:
            continue
        reqs_sorted = sorted(reqs, key=lambda r: r.timestamp)
        stride = max(1, len(reqs_sorted) // max_points)
        trimmed = reqs_sorted[::stride][:max_points]
        tracks.append(
            {
                "spacecraft_id": sc_id,
                "points": [
                    {"lon": r.lon, "lat": r.lat, "time": r.timestamp} for r in trimmed
                ],
            }
        )
    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default="results/worldcities_1d_12sats/20251130_222032/summary.json",
        help="Path to an evaluation summary.json file.",
    )
    parser.add_argument(
        "--collect-pattern",
        default="Data/worldcities_1_days_12_sats_collects_sc_*.json",
        help="Glob pattern for collect files (multiple spacecraft files allowed).",
    )
    parser.add_argument(
        "--max-collect-files",
        type=int,
        default=4,
        help="Limit how many collect files to load (sorted). Use 0 or negative for all.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=900,
        help="Number of requests to keep for point rendering.",
    )
    parser.add_argument(
        "--track-points",
        type=int,
        default=260,
        help="Max orbit points per spacecraft (controls arc smoothness).",
    )
    parser.add_argument(
        "--out",
        default="webviz/data/viz_payload.json",
        help="Where to write the web-ready JSON bundle.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    def sc_sort_key(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return 0

    collect_paths = sorted(
        (Path(p) for p in glob.glob(args.collect_pattern) if Path(p).is_file()),
        key=sc_sort_key,
    )
    if args.max_collect_files > 0:
        collect_paths = collect_paths[: args.max_collect_files]

    if not collect_paths:
        raise SystemExit("No collect files found for pattern: " + args.collect_pattern)
    if not summary_path.exists():
        raise SystemExit(f"Summary file not found: {summary_path}")

    # Load and merge collect files.
    collects_raw: List[dict] = []
    for path in collect_paths:
        print(f"[collects] loading {path}")
        collects_raw.extend(json.loads(path.read_text()))
    print(f"[collects] merged {len(collects_raw):,} requests from {len(collect_paths)} files")

    # Group by spacecraft to preserve coverage balance.
    track_source: Dict[int, List[RequestPoint]] = defaultdict(list)
    for rec in collects_raw:
        lon, lat = rec.get("center", [0, 0])
        ap = rec.get("access_properties", {}) or {}
        rp = RequestPoint(
            lon=lon,
            lat=lat,
            reward=float(rec.get("reward", 0.0)),
            spacecraft_id=int(rec.get("spacecraft_id", -1)),
            t_start=rec.get("t_start"),
            t_mid=rec.get("t_mid"),
            t_end=rec.get("t_end"),
            look_angle_min=ap.get("look_angle_min"),
            look_angle_max=ap.get("look_angle_max"),
            request_id=rec.get("id"),
            tile_id=rec.get("tile_id"),
        )
        track_source[rp.spacecraft_id].append(rp)

    # Determine per-spacecraft sample counts proportional to availability.
    total_requests = len(collects_raw)
    sampled_requests: List[RequestPoint] = []
    rng = random.Random(42)
    for sc_id, reqs in track_source.items():
        target = max(1, round(args.sample_size * (len(reqs) / total_requests)))
        picks = even_sample(reqs, target)
        # Add a bit of randomness so repeated runs aren't too uniform.
        rng.shuffle(picks)
        picks = picks[:target]
        sampled_requests.extend(picks)
    # Keep global sample_size cap.
    rng.shuffle(sampled_requests)
    sampled_requests = sampled_requests[: args.sample_size]

    # Build orbit paths.
    tracks = build_tracks(track_source, args.track_points)

    times = [r.timestamp for r in sampled_requests if r.timestamp]
    time_window = {
        "start": ts_to_iso(min(times)) if times else None,
        "end": ts_to_iso(max(times)) if times else None,
    }

    # Slim solver summary.
    summary_raw = json.loads(summary_path.read_text())
    payload = {
        "dataset": summary_raw.get("dataset"),
        "search_mode": summary_raw.get("search_mode"),
        "time_limit": summary_raw.get("time_limit"),
        "node_limit": summary_raw.get("node_limit"),
        "sample_subgraphs": summary_raw.get("sample_subgraphs"),
        "subset_size": summary_raw.get("subset_size"),
        "totals": summary_raw.get("totals"),
        "component_stats": slim_components(summary_raw.get("components", [])),
        "request_file": collect_paths[0].as_posix(),
        "collect_files_loaded": [p.as_posix() for p in collect_paths],
        "total_requests_in_files": total_requests,
        "sampled_requests_count": len(sampled_requests),
        "unique_spacecraft": sorted(int(k) for k in track_source.keys()),
        "time_window": time_window,
        "generated_at": datetime.now(tz=timezone.utc).strftime(ISO_FMT),
    }

    payload["requests"] = [
        {
            "id": r.request_id,
            "lon": r.lon,
            "lat": r.lat,
            "reward": r.reward,
            "spacecraft_id": r.spacecraft_id,
            "t_start": r.t_start,
            "t_mid": r.t_mid,
            "t_end": r.t_end,
            "look_angle_min": r.look_angle_min,
            "look_angle_max": r.look_angle_max,
            "tile_id": r.tile_id,
            "timestamp": r.timestamp,
        }
        for r in sampled_requests
    ]
    payload["tracks"] = tracks

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[done] wrote {out_path} ({len(payload['requests'])} points, {len(tracks)} tracks)")


if __name__ == "__main__":
    main()
