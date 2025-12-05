# Satellite Scheduling AI - WorldCities MIS Toolkit

Tools for the AIAA WorldCities one-day satellite scheduling problem:
- Build frozen, reproducible MIS graph bundles from the raw collect files.
- Run greedy baselines plus branch-and-bound/A* search with anytime logging.
- Export slimmed solver results and collect samples into a browser-based 3D globe visualizer.

Data is **not** included. Only code lives here. Bring your own WorldCities collect files and METIS graph.

## Repo layout
- `src/` - end-to-end pipeline and solvers:
  - `build_frozen_datasets.py`: parse/filter collects, align with the METIS graph, and emit `frozen/<dataset>/data.npz` + `meta.json`.
  - `freeze_filtered_collects_and_run_greedy.py`: quick sanity run of greedy heuristics directly on raw JSON.
  - `load_graph_and_align.py`: loader/aligner utilities used by the freezer and solvers.
  - `mis_solver.py`: greedy suite plus branch-and-bound and A* search (anytime traces, per-component evaluation harness).
  - `neural_priority.py`, `gibbs_priority.py`: placeholder priority/heuristic experiments.
  - `merge_collect_build_feature_array.py`: helper for fusing collect JSONs into feature arrays.
- `webviz/` - static visualization (globe.gl + Chart.js):
  - `prepare_viz_data.py`: trims solver output and collect samples into `webviz/data/viz_payload.json`.
  - `index.html`, `styles.css`, `main.js`: interactive globe, time scrubber, and solver scorecards.

## Setup
Requirements: Python 3.10+, `numpy`; `matplotlib` is optional (for anytime plots).

```bash
pip install numpy matplotlib
```

## Bring your own data (not committed)
Place the raw WorldCities files in `Data/` (not tracked):
- `worldcities_1_days_12_sats_collects_sc_*.json`
- `worldcities_1_days_12_sats_metis_graph.metis`
- Spacecraft specs JSONs as provided upstream.

## Freeze a dataset
Create reproducible bundles under `frozen/worldcities_1d_12sats/`:
```bash
python src/build_frozen_datasets.py --sats 12
```
This writes `data.npz` + `meta.json` (filtered to look_angle_max <= 55, aligned with the METIS node ordering).

Quick sanity (greedy only, no frozen files needed):
```bash
python src/freeze_filtered_collects_and_run_greedy.py --sats 12
```

## Solve mode (ad-hoc components)
```bash
python src/mis_solver.py --mode solve \
  --dataset worldcities_1d_12sats \
  --algo both \
  --components 6 7 \
  --time-limit 30
```
- `--algo bnb|astar|both` selects search flavor.
- `--components` chooses connected components (see `meta.json` for sizes).
- Outputs best weights, nodes explored, and anytime traces.

## Evaluation mode (batch metrics)
```bash
python src/mis_solver.py --mode eval \
  --dataset worldcities_1d_12sats \
  --eval-max-components 3 \
  --eval-subgraphs 2 \
  --time-limit 60 \
  --eval-search both
```
Writes `results/<dataset>/<timestamp>/summary.json` (plus optional anytime PNGs if matplotlib is installed).

## Visualization (3D globe)
1) Generate a browser payload from a solver run and collect files:
   ```bash
   python webviz/prepare_viz_data.py \
     --summary results/worldcities_1d_12sats/<run>/summary.json \
     --collect-pattern "Data/worldcities_1_days_12_sats_collects_sc_*.json" \
     --max-collect-files 6 \
     --sample-size 900 \
     --track-points 260 \
     --out webviz/data/viz_payload.json
   ```
   - Points = sampled requests (graph vertices) colored by spacecraft ID (SC-1, SC-2, etc. from the collect files).
   - Arcs = ground tracks for those spacecraft over time; the time scrubber filters to the current window.
2) Serve the static app:
   ```bash
   cd webviz
   python -m http.server 8000
   # open http://localhost:8000
   ```
   Click "Refresh data" after regenerating the payload.

## What not to commit
`Data/`, `frozen/`, `results/`, generated `.npz`/`.json` payloads, and any zips/pdfs/media. Keep the repo code-only; the `.gitignore` already excludes these.
