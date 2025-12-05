import os, re, glob, json
from datetime import datetime
import numpy as np

BASE = r"C:\Users\Vermi\Documents\MyStanford\CS221\Project\Data"

def _to_epoch_seconds(ts):
    """Convert ISO8601 timestamps (with trailing Z) to unix seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).timestamp()


def load_collects(base, sats):
    pattern = os.path.join(base, f"worldcities_1_days_{sats}_sats_collects_sc_*.json")
    files = sorted(glob.glob(pattern), key=lambda p:int(re.search(r"_sc_(\d+)\.json$", p).group(1)))
    recs = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            recs.extend(json.load(f))
    n = len(recs)
    ids     = np.array([r["id"] for r in recs], dtype=object)
    t_start = np.fromiter((_to_epoch_seconds(r["t_start"]) for r in recs), dtype=np.float64, count=n)
    t_end   = np.fromiter((_to_epoch_seconds(r["t_end"]) for r in recs), dtype=np.float64, count=n)
    dwell   = np.fromiter((float(r.get("t_duration") or (_to_epoch_seconds(r["t_end"]) - _to_epoch_seconds(r["t_start"]))) for r in recs), dtype=np.float64, count=n)
    sat_id  = np.fromiter((r["spacecraft_id"] for r in recs), dtype=np.int32, count=n)
    weights = np.fromiter((r.get("reward", 1.0) for r in recs), dtype=np.float64, count=n)
    return {"records": recs, "ids": ids, "t_start": t_start, "t_end": t_end, "dwell": dwell, "sat": sat_id, "w": weights}

C = load_collects(BASE, sats=4)
print("collects:", len(C["ids"]), "min_id", C["ids"].min(), "max_id", C["ids"].max())
