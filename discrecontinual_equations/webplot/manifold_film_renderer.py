"""Render a :class:`~.manifold_film.ManifoldFilm` as a playable page.

The surface is drawn as triangles sorted far to near on every redraw - the
painter's algorithm, which is exact for one surface that does not pierce
itself and is all a single grown manifold needs, and which keeps the page a
single self-contained file with no WebGL. The triangles go on a 2-D canvas
and everything drawn over them (the current ring, curves, markers, the axes)
in SVG: as SVG paths the Lorenz manifold's 20,402 triangles took 250 ms a
redraw in headless Chrome, too slow to turn by hand. Each triangle is coloured by the
level set it closes, on a ramp from the equilibrium outward with alternate
rings a shade apart, so the growth rings read as rings; a small Lambert term
from the triangle's normal gives the surface its shape when it is turned.

The scrub bar is geodesic distance. Because every triangle belongs to the band
between two rings, the triangles are sorted once by the distance of their
outer ring and the surface up to ``r`` is a prefix of that order: scrubbing is
a binary search, not a filter. Dragging turns the surface; the crochet
pattern - stitches, increases and decreases per round - is in a table beneath
it and can be saved as CSV.
"""

import json

from discrecontinual_equations.webplot.latex import latex_to_svg
from discrecontinual_equations.webplot.manifold_film import ManifoldFilm, film_payload
from discrecontinual_equations.webplot.renderer import HtmlRenderer
from discrecontinual_equations.webplot.stage_renderer import (
    _BODY_OPEN,
    _FONTS,
    _STYLE,
    _escape,
)


class ManifoldFilmRenderer(HtmlRenderer):
    """Draw a grown manifold whose scrub bar is geodesic distance."""

    __slots__: list[str] = []

    def render(self, scene: ManifoldFilm) -> str:  # type: ignore[override]
        equation = latex_to_svg(scene.equation) if scene.equation else ""
        return (
            _TEMPLATE.replace("__TITLE__", _escape(scene.title))
            .replace("__SUBTITLE__", _escape(scene.subtitle))
            .replace("__NOTE__", _escape(scene.note))
            .replace("__EQUATION__", equation)
            .replace("__FILM_JSON__", json.dumps(film_payload(scene)))
        )


_FILM_STYLE = """
  .surface{background:var(--panel);border:1px solid var(--line);border-radius:14px;
    position:relative;overflow:hidden;box-shadow:var(--shadow)}
  .surface .frame{position:relative;width:100%;aspect-ratio:1.45/1}
  .surface canvas,.surface svg{position:absolute;inset:0;width:100%;height:100%}
  .surface svg{cursor:grab;touch-action:none}
  .scrub{display:flex;align-items:center;gap:14px;margin-top:14px;
    color:var(--muted);font-size:13.5px;flex-wrap:wrap}
  .scrub input[type=range]{flex:1;min-width:200px;accent-color:var(--curve)}
  .scrub .mono{font:500 14px "IBM Plex Mono",monospace;color:var(--ink);
    font-variant-numeric:tabular-nums;min-width:22ch}
  details.pattern{margin-top:16px;background:var(--panel2);
    border:1px solid var(--line);border-radius:14px;padding:10px 16px}
  details.pattern summary{cursor:pointer;color:var(--muted);font-size:13.5px}
  .pattern .table{max-height:300px;overflow:auto;margin-top:10px}
  .pattern table{border-collapse:collapse;font:400 12.5px "IBM Plex Mono",
    monospace;font-variant-numeric:tabular-nums}
  .pattern td,.pattern th{padding:2px 14px 2px 0;text-align:right}
  .pattern th{color:var(--faint);font-weight:500}
  .pattern tr.now td{color:var(--curve)}
  .foot{color:var(--muted);font-size:13px;max-width:92ch;margin-top:18px}
  .legend i{display:inline-block;width:18px;height:0;border-top:2.5px solid;
    vertical-align:3px;margin-right:6px}
  .ramp{display:inline-block;width:90px;height:9px;border-radius:3px;
    vertical-align:0;margin-right:6px}
"""

_BODY_FILM = """  <div class="mast">
    <div>
      <div class="kicker">Manifold film &middot; geodesic level sets</div>
      <h1>__TITLE__</h1>
      <div class="eqn">__EQUATION__</div>
    </div>
    <p class="lede">__SUBTITLE__</p>
  </div>

  <section class="surface" aria-label="manifold in three dimensions">
    <div class="panel-head">
      <span class="kicker">Surface &middot; drag to turn, double-click to reset</span>
      <span class="mono" id="counts"></span>
    </div>
    <div class="frame"><canvas id="mesh" aria-hidden="true"></canvas><svg
      id="surface" viewBox="0 0 900 620" role="img"
      aria-label="the grown manifold"></svg></div>
    <div class="legend" id="legend">
      <span><span class="ramp" id="ramp"></span>geodesic distance, ring by
        ring</span>
      <span><i style="border-color:var(--ink)"></i>current level set</span>
    </div>
  </section>

  <div class="scrub">
    <button class="btn play" id="play" aria-pressed="false">Grow
      <kbd>space</kbd></button>
    <span class="group"><kbd>&larr;</kbd><kbd>&rarr;</kbd> a ring</span>
    <input id="r" type="range" min="0" step="1" aria-label="geodesic distance">
    <span class="mono" id="readout"></span>
    <label class="toggle"><input id="l-front" type="checkbox" checked>
      <span class="sw"></span>level set</label>
    <label class="toggle"><input id="l-edges" type="checkbox">
      <span class="sw"></span>mesh</label>
    <label class="toggle"><input id="l-curves" type="checkbox" checked>
      <span class="sw"></span>curves</label>
  </div>

  <details class="pattern">
    <summary id="pattern-summary">Crochet pattern</summary>
    <p class="lede">One round per level set: the stitches on it, and how many
      were added (where the surface flares) and taken away against the round
      before. <button class="btn" id="csv">Save as CSV</button></p>
    <div class="table"><table id="pattern"></table></div>
  </details>
"""

_BODY_FOOT = """  <p class="foot">__NOTE__</p>
</div>
"""

_SCRIPT = r"""
(function(){
const D = window.__FILM__;
const css = k =>
  getComputedStyle(document.documentElement).getPropertyValue(k).trim();

// ---------- theme (shared key with the atlas pages) ----------
function applyTheme(name){
  document.documentElement.setAttribute("data-theme", name);
  document.querySelectorAll(".theme-btn").forEach(b =>
    b.setAttribute("aria-pressed", b.dataset.theme===name ? "true" : "false"));
}
let stored = null;
try { stored = localStorage.getItem("dce-theme"); } catch (e) { stored = null; }
applyTheme(stored ||
  (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"));
document.querySelectorAll(".theme-btn").forEach(b =>
  b.addEventListener("click", () => {
    try { localStorage.setItem("dce-theme", b.dataset.theme); } catch (e) {}
    applyTheme(b.dataset.theme); draw();
  }));

// ---------- the mesh ----------
const V = D.vertices, T = D.triangles, RD = D.ringDistances, RINGS = D.rings;
const NV = V.length / 3, NT = T.length / 3, NR = RD.length;
const ringOf = new Int32Array(NV);          // the equilibrium is ring -1
ringOf[0] = -1;
RINGS.forEach((ring, k) => ring.forEach(i => { ringOf[i] = k; }));
const triRing = new Int32Array(NT);
for (let t = 0; t < NT; t++){
  triRing[t] = Math.max(ringOf[T[3*t]], ringOf[T[3*t+1]], ringOf[T[3*t+2]]);
}
// Sorted once by the ring each triangle closes: the surface up to ring k
// is then the first `upto[k]` triangles of this order.
const order = Uint32Array.from({length: NT}, (_, t) => t)
  .sort((a, b) => triRing[a] - triRing[b]);
const upto = new Int32Array(NR);
{ let n = 0; for (let k = 0; k < NR; k++){
    while (n < NT && triRing[order[n]] <= k) n++;
    upto[k] = n; } }

let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
const grow = pts => {
  for (let i = 0; i < pts.length; i += 3) for (let a = 0; a < 3; a++){
    lo[a] = Math.min(lo[a], pts[i+a]); hi[a] = Math.max(hi[a], pts[i+a]); } };
grow(V);
D.curves.forEach(c => grow(c.points.flat()));
const C = [0, 1, 2].map(a => 0.5 * (lo[a] + hi[a]));
const R = Math.max(1e-9, 0.5 * Math.hypot(hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]));

// ---------- colour: a ramp out from the equilibrium, rings a shade apart ----
const STOPS = [[40,30,112],[37,99,171],[22,160,160],[94,201,98],[234,215,64],
  [247,140,52]];
function ramp(f){
  const x = Math.max(0, Math.min(1, f)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(x)), u = x - i;
  return STOPS[i].map((v, a) => v + u * (STOPS[i+1][a] - v));
}
document.getElementById("ramp").style.background =
  `linear-gradient(90deg,${STOPS.map(s => `rgb(${s})`).join(",")})`;
const BAND = 0.16;
const BASE = RD.map((r, k) =>
  ramp(r / RD[NR-1]).map(v => v * (k % 2 ? 1 - BAND : 1)));

// ---------- view ----------
const W = 900, H = 620, CX = W / 2, CY = H / 2 + 6, SCALE = 0.58 * H / R;
const START = {az: -0.75, el: 0.32};
let az = START.az, el = START.el;
const SX = new Float64Array(NV), SY = new Float64Array(NV);
const DEPTH = new Float64Array(NV);
function view(x, y, z){
  x -= C[0]; y -= C[1]; z -= C[2];
  const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
  const xr = x * ca - y * sa, yr = x * sa + y * ca;
  const up = z * ce + yr * se, depth = yr * ce - z * se;
  return [CX + SCALE * xr, CY - SCALE * up, depth];
}
function project(){
  for (let i = 0; i < NV; i++){
    const [sx, sy, d] = view(V[3*i], V[3*i+1], V[3*i+2]);
    SX[i] = sx; SY[i] = sy; DEPTH[i] = d;
  }
}

// ---------- drawing ----------
const svg = document.getElementById("surface");
const canvas = document.getElementById("mesh");
const ctx = canvas.getContext("2d");
const PIXELS = Math.min(2, window.devicePixelRatio || 1);
canvas.width = W * PIXELS; canvas.height = H * PIXELS;
const range = document.getElementById("r");
range.max = String(NR - 1);
let ring = NR - 1;
range.value = String(ring);
const f1 = v => v.toFixed(1);
const colour = {stable: "--stable", unstable: "--unstable", orbit: "--hopf"};

function draw(){
  project();
  const n = upto[ring];
  const picked = order.subarray(0, n);
  const depth = new Float64Array(n);
  for (let k = 0; k < n; k++){
    const t = picked[k];
    depth[k] = DEPTH[T[3*t]] + DEPTH[T[3*t+1]] + DEPTH[T[3*t+2]];
  }
  const back = Uint32Array.from({length: n}, (_, k) => k)
    .sort((a, b) => depth[b] - depth[a]);
  const edges = document.getElementById("l-edges").checked;
  ctx.setTransform(PIXELS, 0, 0, PIXELS, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.lineJoin = "round";
  ctx.lineWidth = edges ? 0.5 : 0.6;
  const line = css("--panel");
  for (let k = 0; k < n; k++){
    const t = picked[back[k]];
    const a = T[3*t], b = T[3*t+1], c = T[3*t+2];
    // Lambert shading from the normal in screen space, two-sided.
    const ux = SX[b]-SX[a], uy = SY[b]-SY[a], ud = DEPTH[b]-DEPTH[a];
    const vx = SX[c]-SX[a], vy = SY[c]-SY[a], vd = DEPTH[c]-DEPTH[a];
    const nx = (uy*vd - ud*vy) * SCALE, ny = (ud*vx - ux*vd) * SCALE;
    const nz = ux*vy - uy*vx;
    const shade = 0.55 + 0.45 * Math.abs(nz) / (Math.hypot(nx, ny, nz) || 1);
    const base = BASE[Math.max(0, triRing[t])];
    const fill = `rgb(${Math.round(base[0]*shade)},${Math.round(base[1]*shade)},${
      Math.round(base[2]*shade)})`;
    ctx.beginPath();
    ctx.moveTo(SX[a], SY[a]); ctx.lineTo(SX[b], SY[b]); ctx.lineTo(SX[c], SY[c]);
    ctx.closePath();
    ctx.fillStyle = fill; ctx.fill();
    // Stroked in its own colour, so neighbours meet without a hairline seam.
    ctx.strokeStyle = edges ? line : fill; ctx.stroke();
  }
  const parts = [];
  if (document.getElementById("l-front").checked){
    const front = RINGS[ring];
    const d = front.map((i, k) => `${k ? "L" : "M"}${SX[i].toFixed(1)},${
      SY[i].toFixed(1)}`).join("") + "Z";
    parts.push(`<path d="${d}" fill="none" stroke="${css("--ink")}"
      stroke-width="1.8" stroke-linejoin="round" opacity="0.9"/>`);
  }
  if (document.getElementById("l-curves").checked){
    D.curves.forEach(c => {
      const d = c.points.map((p, k) => { const [sx, sy] = view(p[0], p[1], p[2]);
        return `${k ? "L" : "M"}${sx.toFixed(1)},${sy.toFixed(1)}`; }).join("");
      parts.push(`<path d="${d}" fill="none" stroke="${css(colour[c.kind] ||
        "--curve")}" stroke-width="2.6" stroke-linecap="round"/>`);
    });
  }
  D.markers.forEach(m => {
    const [sx, sy] = view(m.point[0], m.point[1], m.point[2]);
    parts.push(`<circle cx="${sx.toFixed(1)}" cy="${sy.toFixed(1)}" r="5"
      fill="${css("--saddle")}" stroke="${css("--panel")}" stroke-width="1.5"/>`);
    parts.push(`<text x="${(sx+9).toFixed(1)}" y="${(sy-7).toFixed(1)}"
      fill="${css("--ink")}" font-size="13">${m.label}</text>`);
  });
  parts.push(triad());
  svg.innerHTML = parts.join("");
  const row = D.pattern[ring];
  document.getElementById("readout").textContent =
    `r = ${RD[ring].toFixed(2)} · round ${ring} · ${row[2]} stitches`;
  document.getElementById("counts").textContent =
    `${n.toLocaleString()} of ${NT.toLocaleString()} triangles`;
  document.querySelectorAll("#pattern tr.now").forEach(r => r.classList.remove("now"));
  const now = document.getElementById(`round-${ring}`);
  if (now) now.classList.add("now");
}

function triad(){
  // The axes, turned with the surface, in the corner.
  const ox = 58, oy = H - 52, len = 34, out = [];
  D.axes.forEach((name, a) => {
    const e = [0, 0, 0]; e[a] = 1;
    const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
    const xr = e[0]*ca - e[1]*sa, yr = e[0]*sa + e[1]*ca, up = e[2]*ce + yr*se;
    const x = ox + len * xr, y = oy - len * up;
    out.push(`<line x1="${ox}" y1="${oy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}"
      stroke="${css("--muted")}" stroke-width="1.4"/>`);
    out.push(`<text x="${(ox + 1.3*len*xr).toFixed(1)}" y="${(oy - 1.3*len*up + 4)
      .toFixed(1)}" fill="${css("--muted")}" font-size="12"
      text-anchor="middle">${name}</text>`);
  });
  return out.join("");
}

// ---------- pattern table and its CSV ----------
const table = document.getElementById("pattern");
table.innerHTML = "<tr><th>round</th><th>distance</th><th>stitches</th>" +
  "<th>increases</th><th>decreases</th></tr>" + D.pattern.map(p =>
  `<tr id="round-${p[0]}"><td>${p[0]}</td><td>${p[1].toFixed(3)}</td><td>${
  p[2]}</td><td>${p[3]}</td><td>${p[4]}</td></tr>`).join("");
document.getElementById("pattern-summary").textContent =
  `Crochet pattern · ${NR} rounds, ${NV - 1} stitches`;
document.getElementById("csv").addEventListener("click", () => {
  const text = "round,distance,stitches,increases,decreases\n" +
    D.pattern.map(p => p.join(",")).join("\n") + "\n";
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], {type: "text/csv"}));
  link.download = (D.title || "manifold").replace(/\W+/g, "_") + "_pattern.csv";
  link.click();
});

// ---------- interaction ----------
let pending = false;
const redraw = () => { if (!pending){ pending = true;
  requestAnimationFrame(() => { pending = false; draw(); }); } };
range.addEventListener("input", () => { ring = +range.value; stop(); redraw(); });
["l-front", "l-edges", "l-curves"].forEach(id =>
  document.getElementById(id).addEventListener("change", redraw));
let dragging = false, lastX = 0, lastY = 0;
svg.addEventListener("pointerdown", ev => { dragging = true; lastX = ev.clientX;
  lastY = ev.clientY; svg.setPointerCapture(ev.pointerId);
  svg.style.cursor = "grabbing"; });
svg.addEventListener("pointermove", ev => { if (!dragging) return;
  az += (ev.clientX - lastX) * 0.008;
  el = Math.max(-1.5, Math.min(1.5, el + (ev.clientY - lastY) * 0.006));
  lastX = ev.clientX; lastY = ev.clientY; redraw(); });
const release = () => { dragging = false; svg.style.cursor = "grab"; };
svg.addEventListener("pointerup", release);
svg.addEventListener("pointercancel", release);
svg.addEventListener("dblclick", () => { az = START.az; el = START.el; redraw(); });

const play = document.getElementById("play");
let timer = null;
function stop(){ if (timer){ clearInterval(timer); timer = null; }
  play.setAttribute("aria-pressed", "false"); }
function start(){
  if (ring >= NR - 1) ring = 0;
  play.setAttribute("aria-pressed", "true");
  // About eight seconds for the whole growth, whatever the ring count.
  timer = setInterval(() => {
    ring = Math.min(NR - 1, ring + 1); range.value = String(ring); redraw();
    if (ring >= NR - 1) stop();
  }, Math.max(16, 8000 / NR));
}
play.addEventListener("click", () => (timer ? stop() : start()));
// As on the stage: Space plays and pauses unless a control has the focus
// (Space is how a checkbox or a button is pressed), and the arrows step one
// ring - except on the scrub bar itself, which steps on its own.
document.addEventListener("keydown", ev => {
  const control = ev.target.tagName === "INPUT" || ev.target.tagName === "BUTTON";
  if (ev.code === "Space" && !control){
    ev.preventDefault(); timer ? stop() : start(); }
  if (ev.target === range) return;
  if (ev.key === "ArrowRight" || ev.key === "ArrowLeft"){
    stop();
    ring = Math.max(0, Math.min(NR - 1, ring + (ev.key === "ArrowRight" ? 1 : -1)));
    range.value = String(ring); redraw();
  }
});
draw();
})();
"""

_TEMPLATE = (
    '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    "<title>__TITLE__</title>\n"
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    f'<link rel="stylesheet" href="{_FONTS}">\n'
    f"<style>{_STYLE}{_FILM_STYLE}</style>\n</head>\n"
    f"<body>{_BODY_OPEN}{_BODY_FILM}{_BODY_FOOT}"
    "<script>window.__FILM__ = __FILM_JSON__;</script>\n"
    f"<script>{_SCRIPT}</script>\n</body>\n</html>\n"
)
