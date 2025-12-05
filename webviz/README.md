# WorldCities MIS Web Visualizer

Globe-first, browser-only view of the satellite scheduling runs. It loads a compact JSON bundle (generated from your solver outputs plus collect files) and renders:

- Glowing request points on an interactive 3D Earth.
- Animated arcs following each spacecraft ground track.
- A scorecard comparing greedy heuristics vs. bounded search per connected component.

No Python changes needed; just serve the static files.

## Quick start

1) Generate a payload (see below).
2) Serve the static files:
   ```bash
   cd webviz
   python -m http.server 8000
   # open http://localhost:8000 in your browser
   ```

## Refresh the data bundle

Use the helper script to trim solver output and collect files into a browser-friendly payload:

```bash
python webviz/prepare_viz_data.py \
  --summary results/worldcities_1d_12sats/20251130_222032/summary.json \
  --collect-pattern "Data/worldcities_1_days_12_sats_collects_sc_*.json" \
  --max-collect-files 6 \
  --sample-size 1000 \
  --track-points 260 \
  --out webviz/data/viz_payload.json
```

Notes:
- `--max-collect-files` lets you cap how many spacecraft files to load (defaults to 4 for quick runs).
- `--sample-size` controls how many requests become points on the globe.
- `--track-points` caps the per-spacecraft orbit trail density (arcs in the visualization).
- The script drops heavyweight fields (e.g., best masks) so the page stays snappy; generated JSON is gitignored.

After regenerating the payload, refresh the browser or hit the "Refresh data" button in the UI.
