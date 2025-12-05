const palette = [
  "#7cf3e4",
  "#ffb457",
  "#ff7eb6",
  "#7ad7f0",
  "#c3f27b",
  "#f0a3ff",
  "#5bf5b8",
  "#ffa27b",
  "#9be3ff",
  "#ffd76f",
  "#9d7bff",
  "#72ffc2",
];

let globe;
let chart;
let points = [];
let arcs = [];
let timeRange = [0, 0];
let resizeHandler;

const fmt = new Intl.NumberFormat("en-US");

const colorForSat = (id) => palette[(id - 1) % palette.length];

function numberLabel(value) {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 10_000) return `${(value / 1000).toFixed(1)}k`;
  return fmt.format(Number(value));
}

function formatTime(ms) {
  if (!ms) return "All times";
  const d = new Date(ms);
  return `${d.toUTCString().replace(/ GMT$/, "")} UTC`;
}

async function fetchData() {
  const res = await fetch(`data/viz_payload.json?ts=${Date.now()}`);
  if (!res.ok) throw new Error("Unable to load viz_payload.json");
  return res.json();
}

function buildStatCards(data) {
  const root = document.getElementById("stat-cards");
  if (!root) return;
  const bestTotal =
    data.totals && Object.values(data.totals).length
      ? Math.max(...Object.values(data.totals))
      : null;
  const cards = [
    {
      label: "Dataset",
      value: data.dataset || "unknown",
      hint: `${(data.unique_spacecraft || []).length || "?"} spacecraft`,
    },
    {
      label: "Points on globe",
      value: numberLabel(data.sampled_requests_count),
      hint: `from ${numberLabel(data.total_requests_in_files)} requests`,
    },
    {
      label: "Best solver total",
      value: bestTotal ? `${numberLabel(bestTotal)} picks` : "—",
      hint: data.search_mode ? `search: ${data.search_mode}` : "",
    },
    {
      label: "Time limit",
      value: data.time_limit ? `${data.time_limit}s` : "none",
      hint: data.node_limit ? `${data.node_limit} node cap` : "no cap",
    },
  ];
  root.innerHTML = cards
    .map(
      (c) => `
      <div class="stat">
        <div class="label">${c.label}</div>
        <div class="value">${c.value}</div>
        <div class="hint">${c.hint || ""}</div>
      </div>`
    )
    .join("");
}

function buildLegend(data) {
  const legend = document.getElementById("legend");
  if (!legend) return;
  const ids = data.unique_spacecraft || [];
  legend.innerHTML = ids
    .slice(0, 8)
    .map(
      (id) => `
      <div class="legend__item">
        <span class="legend__swatch" style="background:${colorForSat(id)}"></span>
        SC-${id}
      </div>`
    )
    .join("");
}

function buildComponentChart(data) {
  const ctx = document.getElementById("component-chart");
  if (!ctx) return;
  const comps = (data.component_stats || []).slice(0, 8);
  const labels = comps.map((c) => `C${c.id}`);
  const greedy = comps.map((c) => c.greedy_best?.weight ?? 0);
  const search = comps.map(
    (c) => c.search_best?.weight ?? c.greedy_best?.weight ?? 0
  );
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Greedy best",
          data: greedy,
          backgroundColor: "rgba(124, 243, 228, 0.6)",
          borderRadius: 6,
        },
        {
          label: "Search best",
          data: search,
          backgroundColor: "rgba(255, 126, 182, 0.5)",
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          labels: { color: "#dfe6f5", boxWidth: 12 },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${numberLabel(ctx.raw)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#a6b3c8" },
          grid: { color: "rgba(255,255,255,0.06)" },
        },
        y: {
          ticks: { color: "#a6b3c8" },
          grid: { color: "rgba(255,255,255,0.06)" },
        },
      },
    },
  });
  const caption = document.getElementById("chart-caption");
  if (caption) {
    caption.textContent = `Showing ${comps.length} of ${
      data.component_stats?.length || 0
    } components · weights = scheduled requests per component.`;
  }
  const meta = document.getElementById("summary-meta");
  if (meta) {
    meta.textContent = data.generated_at
      ? `built ${data.generated_at}`
      : "static bundle";
  }
}

function buildComponentList(data) {
  const container = document.getElementById("component-list");
  if (!container) return;
  const comps = data.component_stats || [];
  container.innerHTML = comps
    .slice(0, 6)
    .map((c) => {
      const greedy = c.greedy_best?.weight ?? 0;
      const search = c.search_best?.weight ?? greedy;
      const fill =
        c.size && c.size > 0 ? Math.min((search / c.size) * 100, 100) : 0;
      return `
        <div class="component">
          <div class="row">
            <div class="label">Component ${c.id}</div>
            <span class="pill pill--ghost">${numberLabel(c.size)} nodes</span>
          </div>
          <div class="row muted">
            <span>Greedy ${numberLabel(greedy)}</span>
            <span>Search ${numberLabel(search)}</span>
          </div>
          <div class="progress" aria-label="solution quality">
            <div class="progress__fill" style="width:${fill}%"></div>
          </div>
        </div>`;
    })
    .join("");
}

function prepareGlobeData(data) {
  points =
    (data.requests || []).map((r) => ({
      ...r,
      timestampMs: (r.timestamp || 0) * 1000,
      color: colorForSat(r.spacecraft_id || 1),
      active: true,
      altitude: 0.28,
      radius: 0.28,
    })) || [];

  arcs = [];
  (data.tracks || []).forEach((track, idx) => {
    const c = colorForSat(track.spacecraft_id || idx + 1);
    const pts = track.points || [];
    for (let i = 1; i < pts.length; i++) {
      arcs.push({
        startLat: pts[i - 1].lat,
        startLng: pts[i - 1].lon,
        endLat: pts[i].lat,
        endLng: pts[i].lon,
        color: c,
        colors: [c, c],
        time: (pts[i].time || 0) * 1000,
      });
    }
  });
}

function initGlobe() {
  const container = document.getElementById("globe-container");
  if (!container) return;
  container.innerHTML = "";
  const { offsetWidth, offsetHeight } = container.getBoundingClientRect();
  globe = Globe()(container)
    .globeImageUrl(
      "https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
    )
    .bumpImageUrl("https://unpkg.com/three-globe/example/img/earth-topology.png")
    .backgroundImageUrl("https://unpkg.com/three-globe/example/img/night-sky.png")
    .pointLat("lat")
    .pointLng("lon")
    .pointColor((d) => (d.active ? d.color : "rgba(255,255,255,0.25)"))
    .pointAltitude((d) => (d.active ? d.altitude || 0.26 : 0.09))
    .pointRadius((d) => (d.active ? d.radius || 0.24 : 0.12))
    .pointLabel(
      (d) => `<div style="max-width:220px">
        <strong>Request ${d.id || ""}</strong><br>
        SC-${d.spacecraft_id} · reward ${d.reward}<br>
        ${d.t_mid || d.t_start || ""} UTC
      </div>`
    )
    .arcsData(arcs)
    .arcColor("colors")
    .arcDashLength(0.35)
    .arcDashGap(0.8)
    .arcDashAnimateTime(3600)
    .arcAltitude((d) => 0.2)
    .width(offsetWidth)
    .height(offsetHeight);

  globe.controls().autoRotate = true;
  globe.controls().autoRotateSpeed = 0.4;
  globe.controls().enableZoom = true;
  globe.atmosphereColor("#7cf3e4");
  globe.atmosphereAltitude(0.15);
  globe.pointOfView({ lat: 15, lng: 0, altitude: 2.6 }, 0);

  globe.pointsData(points);

  if (resizeHandler) {
    window.removeEventListener("resize", resizeHandler);
  }
  resizeHandler = () => {
    const { offsetWidth: w, offsetHeight: h } = container.getBoundingClientRect();
    globe.width(w);
    globe.height(h);
  };
  window.addEventListener("resize", resizeHandler);
}

function setupTimeSlider(data) {
  const slider = document.getElementById("time-slider");
  if (!slider) return;
  const startIso = data.time_window?.start;
  const endIso = data.time_window?.end;
  const start = startIso ? Date.parse(startIso) : null;
  const end = endIso ? Date.parse(endIso) : null;
  if (!start || !end || start === end) {
    slider.disabled = true;
    return;
  }
  slider.disabled = false;
  slider.value = "100";
  timeRange = [start, end];
  document.getElementById("time-start").textContent = formatTime(start);
  document.getElementById("time-end").textContent = formatTime(end);

  slider.oninput = (e) => {
    const pct = Number(e.target.value) / 100;
    const current = start + (end - start) * pct;
    applyTimeFilter(current);
  };
  applyTimeFilter(end);
}

function applyTimeFilter(currentMs) {
  if (!globe) return;
  const span = timeRange[1] - timeRange[0];
  const windowMs = Math.max(span / 10, 20 * 60 * 1000); // at least 20 minutes
  points.forEach((p) => {
    p.active = Math.abs((p.timestampMs || 0) - currentMs) <= windowMs;
  });
  globe
    .pointColor((d) => (d.active ? d.color : "rgba(255,255,255,0.2)"))
    .pointAltitude((d) => (d.active ? d.altitude || 0.28 : 0.07))
    .pointRadius((d) => (d.active ? d.radius || 0.26 : 0.1))
    .pointsData(points);
  globe.arcsData(arcs.filter((a) => a.time <= currentMs));
  const label = document.getElementById("time-current");
  if (label) label.textContent = formatTime(currentMs);
}

async function bootstrap() {
  try {
    const data = await fetchData();
    buildStatCards(data);
    buildLegend(data);
    buildComponentChart(data);
    buildComponentList(data);
    prepareGlobeData(data);
    initGlobe();
    setupTimeSlider(data);
  } catch (err) {
    console.error(err);
    const container = document.getElementById("globe-container");
    if (container) container.innerHTML = "<p>Could not load viz data.</p>";
  }
}

document.getElementById("reload-btn")?.addEventListener("click", bootstrap);
document.addEventListener("DOMContentLoaded", bootstrap);
