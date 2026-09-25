"""Render a :class:`~.stage.StageScene` to a self-contained playable page.

The page has three synchronised panels driven by one parameter: a *stage*,
where particles are advected through the sampled field on a canvas with the
equilibria, saddle manifolds and cycle drawn over them; a *spectral clock*,
the complex plane with the Hopf line and the unit circle, showing the
eigenvalues of every equilibrium and the Floquet multipliers of the cycle; and
the bifurcation diagram as a *timeline* with a playhead. D3 is inlined as in
:class:`~.renderer.D3Renderer`, so the page renders offline; the typefaces are
fetched from Google Fonts when a network is available and fall back to the
system stacks when it is not.
"""

import json
from pathlib import Path

from discrecontinual_equations.webplot.latex import latex_to_svg
from discrecontinual_equations.webplot.renderer import HtmlRenderer
from discrecontinual_equations.webplot.stage import StageScene, stage_payload

_ASSET = Path(__file__).parent / "assets" / "d3.min.js"


class StageRenderer(HtmlRenderer):
    """Draw a stage as a self-contained HTML page with an inlined D3."""

    __slots__ = ["_library"]

    def __init__(self, library: str | None = None) -> None:
        self._library = library if library is not None else _ASSET.read_text()

    def render(self, scene: StageScene) -> str:
        system = scene.system
        equation = latex_to_svg(system.equation) if system.equation else ""
        return (
            _TEMPLATE.replace("__TITLE__", _escape(system.title))
            .replace("__SUBTITLE__", _escape(system.subtitle))
            .replace("__NOTE__", _escape(system.note or ""))
            .replace("__EQUATION__", equation)
            .replace("__D3_SOURCE__", self._library)
            .replace("__STAGE_JSON__", json.dumps(stage_payload(scene)))
        )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_STYLE = """
  :root{
    --ground:#f3f5fa; --panel:#ffffff; --panel2:#eef1f7; --line:#d9dfea;
    --grid:#e4e8f0; --ink:#0f1626; --muted:#5b6579; --faint:#8b94a8;
    --stable:#0f9f6e; --unstable:#e0405f; --saddle:#7c5cd6; --hopf:#c98a00;
    --fold:#d6339a; --curve:#0e8fc7;
    --particle:30,58,138; --particle-alpha:.55; --fade:.14;
    --shadow:0 24px 60px -34px rgba(15,22,38,.35);
  }
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){
      --ground:#0b0e17; --panel:#10131d; --panel2:#141826; --line:#232a3c;
      --grid:#1a2030; --ink:#eef1f7; --muted:#9aa3b8; --faint:#5f6a82;
      --stable:#34d399; --unstable:#fb7185; --saddle:#c4b5fd; --hopf:#fbbf24;
      --fold:#f472b6; --curve:#38bdf8;
      --particle:188,211,255; --particle-alpha:.62; --fade:.11;
      --shadow:0 30px 80px -40px rgba(0,0,0,.9);
    }
  }
  :root[data-theme="dark"]{
    --ground:#0b0e17; --panel:#10131d; --panel2:#141826; --line:#232a3c;
    --grid:#1a2030; --ink:#eef1f7; --muted:#9aa3b8; --faint:#5f6a82;
    --stable:#34d399; --unstable:#fb7185; --saddle:#c4b5fd; --hopf:#fbbf24;
    --fold:#f472b6; --curve:#38bdf8;
    --particle:188,211,255; --particle-alpha:.62; --fade:.11;
    --shadow:0 30px 80px -40px rgba(0,0,0,.9);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--ink);
    font:15px/1.55 "IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
    -webkit-font-smoothing:antialiased}
  .wrap{max-width:1180px;margin:0 auto;padding-block:26px 60px;padding-inline:20px}
  .topbar{display:flex;justify-content:space-between;align-items:center;gap:14px;
    flex-wrap:wrap;margin-bottom:22px}
  a.back{color:var(--muted);text-decoration:none;font-size:14px}
  a.back:hover{color:var(--ink)}
  .themes{display:inline-flex;gap:4px;background:var(--panel2);
    border:1px solid var(--line);border-radius:11px;padding:4px}
  .theme-btn{appearance:none;border:0;background:transparent;color:var(--muted);
    font:600 13px/1 "IBM Plex Sans",sans-serif;padding:7px 13px;border-radius:8px;
    cursor:pointer}
  .theme-btn[aria-pressed="true"]{background:var(--curve);color:#fff}
  .mast{display:flex;flex-wrap:wrap;align-items:flex-end;
    justify-content:space-between;gap:14px 28px;margin-bottom:18px}
  h1{font:300 46px/1.02 "Fraunces","Iowan Old Style",Georgia,serif;
    font-variation-settings:"opsz" 96,"SOFT" 40;letter-spacing:-.015em;
    margin:0 0 8px;text-wrap:balance}
  .eqn{color:var(--ink);overflow-x:auto}
  .eqn svg{height:1.35em;width:auto;max-width:100%;vertical-align:middle}
  .lede{max-width:62ch;color:var(--muted);margin:0;font-size:14.5px}
  .kicker{font:500 11.5px "IBM Plex Sans",sans-serif;letter-spacing:.14em;
    text-transform:uppercase;color:var(--faint)}
  .instrument{display:grid;grid-template-columns:minmax(0,1fr) minmax(280px,340px);
    gap:16px;align-items:stretch}
  .stage,.clock,.timeline{background:var(--panel);border:1px solid var(--line);
    border-radius:14px;position:relative;overflow:hidden}
  .stage{box-shadow:var(--shadow)}
  .stage .frame{position:relative;width:100%;aspect-ratio:1.36/1;max-width:100%}
  .stage canvas,.stage svg{position:absolute;inset:0;width:100%;height:100%}
  .stage svg{pointer-events:none}
  .panel-head{display:flex;justify-content:space-between;align-items:baseline;
    padding:12px 16px 0;gap:10px}
  .panel-head .mono{font:400 12.5px "IBM Plex Mono",monospace;color:var(--muted);
    font-variant-numeric:tabular-nums}
  .clock{display:flex;flex-direction:column}
  .clock .frame{position:relative;width:100%;flex:1;min-height:280px}
  .clock svg{position:absolute;inset:0;width:100%;height:100%}
  .readouts{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
    gap:0;margin-top:16px;border:1px solid var(--line);border-radius:14px;
    background:var(--panel2);overflow:hidden}
  .ro{padding:12px 16px 13px;border-right:1px solid var(--line);min-width:0}
  .ro:last-child{border-right:0}
  .ro .kicker{display:block;margin-bottom:5px}
  /* mu and lambda are not abbreviations: the label is uppercased, they
     are not */
  .kicker .sym{text-transform:none;letter-spacing:.02em;font-size:12.5px}
  .ro .val{font:500 20px/1.15 "IBM Plex Mono",monospace;
    font-variant-numeric:tabular-nums;letter-spacing:-.01em}
  .ro .sub{font:400 12.5px/1.45 "IBM Plex Mono",monospace;color:var(--muted);
    font-variant-numeric:tabular-nums}
  .ro.regime .val{font-family:"IBM Plex Sans",sans-serif;font-weight:600;
    font-size:17px;line-height:1.25}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;
    vertical-align:1px;margin-right:6px}
  .timeline{margin-top:16px}
  .timeline .frame{position:relative;width:100%;height:230px}
  .timeline svg{position:absolute;inset:0;width:100%;height:100%;
    cursor:ew-resize;touch-action:none}
  .controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px 18px;
    margin-top:14px;color:var(--muted);font-size:13.5px}
  .btn{appearance:none;border:1px solid var(--line);background:var(--panel);
    color:var(--ink);border-radius:9px;padding:8px 14px;
    font:500 13.5px "IBM Plex Sans",sans-serif;cursor:pointer;display:inline-flex;
    align-items:center;gap:8px;min-width:96px;justify-content:center}
  .btn:hover{border-color:var(--curve)}
  .btn:focus-visible,.toggle input:focus-visible+span,.range:focus-visible,
  .theme-btn:focus-visible{outline:2px solid var(--curve);outline-offset:2px}
  .btn kbd{font:500 11px "IBM Plex Mono",monospace;color:var(--faint);
    border:1px solid var(--line);border-radius:4px;padding:1px 5px}
  .group{display:inline-flex;align-items:center;gap:8px}
  .range{width:130px;accent-color:var(--curve)}
  .sel{font:inherit;color:var(--ink);background:var(--panel);
    border:1px solid var(--faint);border-radius:5px;padding:2px 6px;
    cursor:pointer}
  .toggle{display:inline-flex;align-items:center;gap:6px;cursor:pointer;
    user-select:none}
  .toggle input{position:absolute;opacity:0;width:1px;height:1px}
  .toggle span.sw{width:12px;height:12px;border-radius:3px;
    border:1.5px solid var(--faint);display:inline-block}
  .toggle input:checked+span.sw{background:var(--curve);border-color:var(--curve)}
  .legend{display:flex;flex-wrap:wrap;gap:6px 16px;padding:0 16px 12px;
    font-size:12.5px;color:var(--muted)}
  .legend i{display:inline-block;width:18px;height:0;border-top:2px solid;
    vertical-align:middle;margin-right:6px}
  .legend i.dash{border-top-style:dashed}
  .foot{margin-top:22px;color:var(--faint);font-size:12.5px;max-width:78ch}
  .foot:empty{display:none}
  @media (max-width:820px){
    .instrument{grid-template-columns:1fr}
    h1{font-size:36px}
    .clock .frame{flex:none;height:300px;min-height:0}
  }
  @media (prefers-reduced-motion: reduce){ .btn.play{display:none} }
"""

_SCRIPT = r"""
(function(){
const D = window.__STAGE__;
const css = k =>
  getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const F = D.frames, NX = D.grid.nx, NY = D.grid.ny, V = D.view;
let [x0, x1] = D.box.x, [y0, y1] = D.box.y;
let xName = D.system.x_label, yName = D.system.y_label;
const PARAM = D.system.parameter;
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const mono = "IBM Plex Mono", sans = "IBM Plex Sans";
function setKicker(node, words, symbol){
  node.replaceChildren();
  if (words) node.append(`${words} ${DOT} `);
  const span = document.createElement("span");
  span.className = "sym";
  span.textContent = symbol;
  node.append(span);
}
const colourOf = s =>
  css(s==="stable" ? "--stable" : s==="saddle" ? "--saddle" : "--unstable");
const NAMES = {hopf:"Hopf", fold:"fold", branch_point:"branch point",
  transcritical:"transcritical", pitchfork:"pitchfork"};
const pretty = k => NAMES[k] || k.replace(/_/g," ");
const isUnstable = c => c.multipliers.some(([a,b]) => Math.hypot(a,b) > 1.02);
const MINUS = "\u2212", LAMBDA = "\u03bb", MU = "\u03bc", DOT = "\u00b7";
const PLUSMINUS = "\u00b1", DASH = "\u2014";

// ---------- theme (shared key with the atlas pages) ----------
function applyTheme(name){
  document.documentElement.setAttribute("data-theme", name);
  document.querySelectorAll(".theme-btn").forEach(b =>
    b.setAttribute("aria-pressed", b.dataset.theme===name ? "true" : "false"));
}
let stored = null;
try { stored = localStorage.getItem("dce-theme"); } catch (e) { stored = null; }
if (stored === "print") stored = "light";
if (stored === "dark" || stored === "light") applyTheme(stored);
document.querySelectorAll(".theme-btn").forEach(b =>
  b.addEventListener("click", () => {
    try { localStorage.setItem("dce-theme", b.dataset.theme); } catch (e) {}
    applyTheme(b.dataset.theme);
  }));

// ---------- state ----------
let t = 0, playing = false, speed = 1, dir = 1;
const LAST = F.length - 1;
const hash = parseFloat((location.hash.match(/p=(-?[\d.]+)/)||[])[1]);
function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
function toFrameIndex(p){ return (p - F[0].p) / (F[LAST].p - F[0].p) * LAST; }
if (!isNaN(hash)) t = clamp(toFrameIndex(hash), 0, LAST);
function frameAt(tt){
  const i = Math.floor(tt), f = tt - i;
  return {i, j: Math.min(i+1, LAST), f};
}
function paramAt(tt){ const {i,j,f} = frameAt(tt); return F[i].p*(1-f) + F[j].p*f; }
function nearestFrame(){ return F[Math.round(t)]; }

// ---------- a view between frames: the skeleton blended, not snapped ----------
const lerp = (a, b, f) => a + (b - a) * f;
function blendPairs(A, B, f){
  const n = Math.min(A.length, B.length), out = new Array(n);
  for (let k=0;k<n;k++) out[k] = [lerp(A[k][0],B[k][0],f), lerp(A[k][1],B[k][1],f)];
  return out;
}
function blendNearest(A, B, f){
  const used = new Set();
  return A.map(a => {
    let best = -1, dist = Infinity;
    B.forEach((b, k) => {
      const d = Math.hypot(a[0]-b[0], a[1]-b[1]);
      if (!used.has(k) && d < dist){ dist = d; best = k; }
    });
    if (best < 0) return a;
    used.add(best);
    return [lerp(a[0],B[best][0],f), lerp(a[1],B[best][1],f)];
  });
}
const cyclesOf = frame => (frame.cycles || []).map(i => D.cycles[i]);
// Cycles are matched between frames by amplitude, which is what separates a
// stable outer cycle from an unstable inner one; a cycle with no partner in
// the other frame is appearing or disappearing and fades rather than pops.
function blendCycles(A, B, f){
  const before = cyclesOf(A), after = cyclesOf(B), out = [], used = new Set();
  for (const a of before){
    let best = -1, gap = Infinity;
    after.forEach((b, k) => {
      const d = Math.abs(b.amplitude - a.amplitude);
      if (!used.has(k) && d < gap){ gap = d; best = k; }
    });
    if (best < 0){ out.push({...a, alpha: 1 - f}); continue; }
    used.add(best);
    out.push(blendCycle(a, after[best], f));
  }
  after.forEach((b, k) => { if (!used.has(k)) out.push({...b, alpha: f}); });
  return out;
}
// A manifold branch that winds onto a cycle samples many turns into its points,
// so two such branches from neighbouring frames sit at different phases at the
// same index and a point-wise morph invents loops that cross the cycle. Morph only
// branches that stay close; cross-fade the real curves otherwise.
const CLOSE = 0.06 * Math.hypot(x1 - x0, y1 - y0);
function farApart(a, b){
  const n = Math.min(a.length, b.length); let far = 0;
  for (let q = 0; q < n; q += 6){
    const d = Math.hypot(a[q][0]-b[q][0], a[q][1]-b[q][1]); if (d > far) far = d;
  }
  return far > CLOSE;
}
function blendManifolds(A, B, f){
  const out = [];
  const fade = (list, alpha) => list.forEach(m => out.push({...m, alpha}));
  if (A.manifolds.length !== B.manifolds.length){
    fade(A.manifolds, 1 - f); fade(B.manifolds, f); return out;
  }
  A.manifolds.forEach((a, k) => {
    const b = B.manifolds[k];
    if (farApart(a.points, b.points)){ fade([a], 1 - f); fade([b], f); }
    else out.push({kind: a.kind, points: blendPairs(a.points, b.points, f), alpha: 1});
  });
  return out;
}
function blendEquilibria(A, B, f, near){
  const out = [];
  for (const e of A.equilibria){
    const saddle = e.stability === "saddle";
    const twins = B.equilibria.filter(x => (x.stability === "saddle") === saddle);
    if (!twins.length) continue;
    const gap = x => Math.hypot(x.x - e.x, x.y - e.y);
    const m = twins.reduce((best, x) => gap(x) < gap(best) ? x : best);
    out.push({x: lerp(e.x, m.x, f), y: lerp(e.y, m.y, f),
      stability: (near === A ? e : m).stability, eig: blendNearest(e.eig, m.eig, f)});
  }
  return out.length ? out : near.equilibria;
}
function blendCycle(cA, cB, f){
  return {p: lerp(cA.p, cB.p, f), period: lerp(cA.period, cB.period, f),
    amplitude: lerp(cA.amplitude, cB.amplitude, f), alpha: 1,
    error: lerp(cA.error || 0, cB.error || 0, f),
    multipliers: blendNearest(cA.multipliers, cB.multipliers, f),
    states: blendPairs(cA.states, cB.states, f)};
}
function viewAt(tt){
  const {i, j, f} = frameAt(tt); const A = F[i], B = F[j];
  if (i === j || f < 1e-6){
    return {p: A.p, equilibria: A.equilibria, manifolds: A.manifolds,
      cycles: cyclesOf(A).map(c => ({...c, alpha: 1})), label: A.label};
  }
  const near = f < 0.5 ? A : B;
  return {p: paramAt(tt), equilibria: blendEquilibria(A, B, f, near),
    manifolds: blendManifolds(A, B, f),
    cycles: blendCycles(A, B, f), label: near.label};
}

// ---------- field: bilinear in space, linear between frames ----------
function fieldAt(x, y, out){
  const {i, j, f} = frameAt(t);
  const gx = (x - x0) / (x1 - x0) * (NX - 1), gy = (y - y0) / (y1 - y0) * (NY - 1);
  const ix = clamp(Math.floor(gx), 0, NX-2), iy = clamp(Math.floor(gy), 0, NY-2);
  const fx = clamp(gx - ix, 0, 1), fy = clamp(gy - iy, 0, 1);
  const A = F[i].field, B = F[j].field;
  const k00 = 2*(iy*NX+ix), k10 = k00+2, k01 = 2*((iy+1)*NX+ix), k11 = k01+2;
  const w00=(1-fx)*(1-fy), w10=fx*(1-fy), w01=(1-fx)*fy, w11=fx*fy;
  const ua = A[k00]*w00 + A[k10]*w10 + A[k01]*w01 + A[k11]*w11;
  const va = A[k00+1]*w00 + A[k10+1]*w10 + A[k01+1]*w01 + A[k11+1]*w11;
  const ub = B[k00]*w00 + B[k10]*w10 + B[k01]*w01 + B[k11]*w11;
  const vb = B[k00+1]*w00 + B[k10+1]*w10 + B[k01+1]*w01 + B[k11+1]*w11;
  out[0] = ua*(1-f) + ub*f; out[1] = va*(1-f) + vb*f;
}

// ---------- stage: particles on canvas ----------
// A planar system's particles ride the sampled grid. A system seen through a
// view is integrated in its own dimension from the polynomial terms - exact in
// the parameter - and only drawn projected, so what flows is the real flow.
const canvas = document.getElementById("flow"), ctx = canvas.getContext("2d");
const skel = d3.select("#skeleton");
let W = 0, H = 0, dpr = 1;
const sx = v => (v - x0) / (x1 - x0) * W, sy = v => H - (v - y0) / (y1 - y0) * H;
const N = V ? V.matrix[0].length : 2;
let P = [];
// No single plane shows a system of three or more dimensions, so a view may
// carry several and the viewer picks. The payload gives every equilibrium its
// full state and every cycle its full orbit, so switching is a matter of
// projecting them again here rather than asking for a different page.
const PROJS = V
  ? (V.projections || [{name:"", matrix:V.matrix, x_label:"", y_label:"", box:null}])
  : [];
let PROJ = 0;
function projectWith(M, x){
  let u = 0, v = 0;
  for (let j=0;j<N;j++){ u += M[0][j]*x[j]; v += M[1][j]*x[j]; }
  return [u, v];
}
function project(x){ return projectWith(PROJS[PROJ].matrix, x); }
function applyProjection(k){
  if (!V) return;
  PROJ = k;
  const p = PROJS[k], M = p.matrix;
  const box = p.box || D.box;
  [x0, x1] = box.x; [y0, y1] = box.y;
  xName = p.x_label || D.system.x_label;
  yName = p.y_label || D.system.y_label;
  for (const f of D.frames){
    for (const e of f.equilibria){
      if (e.state && e.state.length){
        const q = projectWith(M, e.state); e.x = q[0]; e.y = q[1];
      }
    }
  }
  for (const c of D.cycles){
    if (c.orbit && c.orbit.length) c.states = c.orbit.map(st => projectWith(M, st));
  }
  for (const q of P){ const r = projectWith(M, q.x); q.u = r[0]; q.v = r[1]; }
}
function spawn(p){
  if (V){
    for (let j=0;j<N;j++){
      const [lo,hi] = V.bounds[j]; p.x[j] = lo + Math.random()*(hi-lo);
    }
  } else { p.x[0] = x0 + Math.random()*(x1-x0); p.x[1] = y0 + Math.random()*(y1-y0); }
  const q = V ? project(p.x) : p.x; p.u = q[0]; p.v = q[1];
  p.age = 40 + Math.random()*140;
}
function setDensity(n){
  P = [];
  for (let k=0;k<n;k++){
    const p = {x: new Array(N).fill(0), u: 0, v: 0, age: 0};
    spawn(p); p.age = Math.random()*180; P.push(p);
  }
}
function resize(){
  const r = canvas.getBoundingClientRect();
  dpr = Math.min(2, window.devicePixelRatio||1);
  W = r.width; H = r.height;
  canvas.width = Math.round(W*dpr); canvas.height = Math.round(H*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.fillStyle = css("--panel"); ctx.fillRect(0,0,W,H);
  skel.attr("viewBox", `0 0 ${W} ${H}`);
  drawSkeleton(); drawClock(); drawTimeline();
}
const tmp = new Array(N).fill(0), mid = new Array(N).fill(0);
function polyField(x, param, out){
  for (let i=0;i<N;i++){
    let total = 0;
    for (const term of V.field[i]){
      let m = term.c; if (term.q) m *= Math.pow(param, term.q);
      for (let j=0;j<N;j++){
        const e = term.e[j]; if (e === 1) m *= x[j]; else if (e) m *= Math.pow(x[j], e);
      }
      total += m;
    }
    out[i] = total;
  }
}
function velocity(x, out){
  if (V) polyField(x, paramAt(t), out); else fieldAt(x[0], x[1], out);
}
function stepParticles(){
  ctx.fillStyle = css("--panel"); ctx.globalAlpha = parseFloat(css("--fade"));
  ctx.fillRect(0,0,W,H); ctx.globalAlpha = 1;
  ctx.strokeStyle = `rgba(${css("--particle")},${css("--particle-alpha")})`;
  ctx.lineWidth = 1.1; ctx.lineCap = "round";
  const h = 0.022;
  ctx.beginPath();
  for (const p of P){
    const pu = p.u, pv = p.v;
    velocity(p.x, tmp);
    for (let j=0;j<N;j++) mid[j] = p.x[j] + 0.5*h*tmp[j];
    velocity(mid, tmp);
    let out = false;
    for (let j=0;j<N;j++){
      p.x[j] += h*tmp[j];
      if (V){ const [lo,hi] = V.bounds[j]; if (p.x[j] < lo || p.x[j] > hi) out = true; }
    }
    const q = V ? project(p.x) : p.x; p.u = q[0]; p.v = q[1]; p.age -= 1;
    const moved = Math.hypot(p.u-pu, p.v-pv);
    const gone = out || p.u < x0 || p.u > x1 || p.v < y0 || p.v > y1;
    if (gone || p.age <= 0 || moved < 1e-5 || moved > 0.25){ spawn(p); continue; }
    ctx.moveTo(sx(pu), sy(pv)); ctx.lineTo(sx(p.u), sy(p.v));
  }
  ctx.stroke();
}

// ---------- stage: skeleton on svg ----------
function label(sel, x, y, text, fill, anchor){
  return sel.append("text").attr("x",x).attr("y",y).attr("fill",fill)
    .attr("font-size",11).attr("font-family",mono)
    .attr("text-anchor", anchor || "start").text(text);
}
// The stage carries a scale, not just a pair of axis names: ticks at round
// values with their numbers, and the zero lines picked out where they fall
// inside the box. d3.ticks chooses the round values; how many is set by the
// pixel width so a narrow phone does not crowd them together.
const TICK_PX = 68;
function ticksFor(lo, hi, extent){
  return d3.ticks(lo, hi, Math.max(2, Math.min(8, Math.round(extent / TICK_PX))));
}
function tickText(v, step){
  // Enough decimals for the step to be visible, and no more.
  const places = Math.max(0, Math.min(4, -Math.floor(Math.log10(step)) ));
  const text = v.toFixed(places);
  return text === "-0" ? "0" : text;
}
function drawAxes(){
  const xs = ticksFor(x0, x1, W), ys = ticksFor(y0, y1, H);
  const xStep = xs.length > 1 ? xs[1]-xs[0] : (x1-x0);
  const yStep = ys.length > 1 ? ys[1]-ys[0] : (y1-y0);
  const grid = css("--grid"), faint = css("--faint");
  for (const v of xs){
    const px = sx(v);
    skel.append("line").attr("x1",px).attr("x2",px).attr("y1",0).attr("y2",H)
      .attr("stroke",grid).attr("stroke-width",1).attr("opacity",.45);
    skel.append("line").attr("x1",px).attr("x2",px).attr("y1",H-6).attr("y2",H)
      .attr("stroke",faint).attr("stroke-width",1).attr("opacity",.7);
    label(skel, px, H-10, tickText(v, xStep), faint, "middle")
      .attr("font-size",10).attr("opacity",.85);
  }
  for (const v of ys){
    const py = sy(v);
    skel.append("line").attr("y1",py).attr("y2",py).attr("x1",0).attr("x2",W)
      .attr("stroke",grid).attr("stroke-width",1).attr("opacity",.45);
    skel.append("line").attr("y1",py).attr("y2",py).attr("x1",0).attr("x2",6)
      .attr("stroke",faint).attr("stroke-width",1).attr("opacity",.7);
    label(skel, 9, py-4, tickText(v, yStep), faint)
      .attr("font-size",10).attr("opacity",.85);
  }
  // Zero is worth more than a gridline, but only when it is on screen.
  if (x0 < 0 && x1 > 0){
    skel.append("line").attr("x1",sx(0)).attr("x2",sx(0)).attr("y1",0).attr("y2",H)
      .attr("stroke",faint).attr("stroke-width",1).attr("opacity",.55);
  }
  if (y0 < 0 && y1 > 0){
    skel.append("line").attr("y1",sy(0)).attr("y2",sy(0)).attr("x1",0).attr("x2",W)
      .attr("stroke",faint).attr("stroke-width",1).attr("opacity",.55);
  }
}
function drawSkeleton(){
  const fr = viewAt(t); skel.selectAll("*").remove();
  const line = d3.line().x(d=>sx(d[0])).y(d=>sy(d[1]));
  const glow = skel.append("defs").append("filter").attr("id","g")
    .attr("x","-50%").attr("y","-50%").attr("width","200%").attr("height","200%");
  glow.append("feGaussianBlur").attr("stdDeviation",2.2).attr("result","b");
  const merge = glow.append("feMerge");
  merge.append("feMergeNode").attr("in","b");
  merge.append("feMergeNode").attr("in","SourceGraphic");
  drawAxes();
  if (document.getElementById("l-manifolds").checked){
    for (const mf of fr.manifolds){
      skel.append("path").attr("d", line(mf.points)).attr("fill","none")
        .attr("stroke", css(mf.kind==="stable" ? "--stable" : "--unstable"))
        .attr("stroke-width",1.6).attr("opacity", .85 * (mf.alpha ?? 1));
    }
  }
  if (document.getElementById("l-cycle").checked){
    for (const c of fr.cycles){
      const unstable = isUnstable(c);
      skel.append("path").attr("d", line(c.states)+"Z").attr("fill","none")
        .attr("stroke",css(unstable ? "--unstable" : "--stable"))
        .attr("stroke-width",2.2).attr("stroke-dasharray", unstable ? "7 5" : "none")
        .attr("opacity", c.alpha).attr("filter","url(#g)");
    }
  }
  for (const e of fr.equilibria){
    const col = colourOf(e.stability);
    skel.append("circle").attr("cx",sx(e.x)).attr("cy",sy(e.y)).attr("r",11)
      .attr("fill",col).attr("opacity",.18);
    skel.append("circle").attr("cx",sx(e.x)).attr("cy",sy(e.y)).attr("r",5)
      .attr("fill",col).attr("stroke",css("--panel")).attr("stroke-width",1.5);
  }
  label(skel, W-12, H-24, xName, css("--faint"), "end")
    .attr("font-size",12);
  label(skel, 10, 16, yName, css("--faint")).attr("font-size",12);
  document.getElementById("stage-p").textContent =
    `${PARAM} = ${paramAt(t).toFixed(4)}`;
}

// ---------- spectral clock ----------
const clock = d3.select("#clock");
// Eigenvalue tracks: each equilibrium chained across frames by class and
// position, each of its eigenvalues chained by nearest neighbour in the plane
// (their order from the solver is arbitrary). Built once, so a trail is a
// fixed curve through the parameter and scrubbing only moves along it.
const TRAIL = 40;
const NEAR_EQ = 0.1 * Math.hypot(x1 - x0, y1 - y0);
function newTrack(){ return {pts: new Array(F.length).fill(null), cls: []}; }
function buildTracks(){
  const tracks = []; let prevEq = [], prevMap = [];
  F.forEach((fr, k) => {
    const map = [], used = new Set();
    fr.equilibria.forEach(e => {
      let best = -1, dist = Infinity;
      prevEq.forEach((q, m) => {
        const other = (q.stability==="saddle") !== (e.stability==="saddle");
        if (used.has(m) || other) return;
        const d = Math.hypot(q.x - e.x, q.y - e.y);
        if (d < dist){ dist = d; best = m; }
      });
      let own;
      if (best >= 0 && dist < NEAR_EQ){
        used.add(best); const prev = prevMap[best], taken = new Set();
        own = e.eig.map(ev => {
          let bt = -1, bd = Infinity;
          prev.forEach((tr, i) => {
            const last = tr.pts[k-1]; if (taken.has(i) || !last) return;
            const d = Math.hypot(last[0]-ev[0], last[1]-ev[1]);
            if (d < bd){ bd = d; bt = i; }
          });
          if (bt >= 0){ taken.add(bt); return prev[bt]; }
          const tr = newTrack(); tracks.push(tr); return tr;
        });
      } else {
        own = e.eig.map(() => { const tr = newTrack(); tracks.push(tr); return tr; });
      }
      own.forEach((tr, i) => { tr.pts[k] = e.eig[i]; tr.cls[k] = e.stability; });
      map.push(own);
    });
    prevEq = fr.equilibria; prevMap = map;
  });
  return tracks;
}
const TRACKS = buildTracks();
function drawClock(){
  const fr = viewAt(t); const r = clock.node().getBoundingClientRect();
  const cw = r.width, ch = r.height;
  clock.attr("viewBox",`0 0 ${cw} ${ch}`).selectAll("*").remove();
  const cx = cw/2, cy = ch/2 + 4;
  const S = Math.min((cw/2 - 26)/2.3, (ch/2 - 34)/1.6);
  const X = v => cx + v*S, Y = v => cy - v*S;
  clock.append("line").attr("x1",X(-2.3)).attr("x2",X(2.3)).attr("y1",Y(0))
    .attr("y2",Y(0)).attr("stroke",css("--grid"));
  clock.append("line").attr("x1",X(0)).attr("x2",X(0)).attr("y1",Y(-1.55))
    .attr("y2",Y(1.55)).attr("stroke",css("--hopf")).attr("stroke-width",1.2)
    .attr("opacity",.7);
  clock.append("circle").attr("cx",X(0)).attr("cy",Y(0)).attr("r",S)
    .attr("fill","none").attr("stroke",css("--line")).attr("stroke-dasharray","2 4");
  for (const v of [-2,-1,1,2]){
    label(clock, X(v), Y(0)+15, (v<0 ? MINUS : "") + Math.abs(v), css("--faint"),
      "middle");
  }
  label(clock, X(0)+7, Y(1.38), `Re ${LAMBDA} = 0 ${DOT} Hopf`, css("--hopf"));
  label(clock, X(0.72), Y(-0.78), `|${MU}| = 1`, css("--faint"));
  if (document.getElementById("l-trails").checked){
    const here = Math.round(t), from = Math.max(1, here - TRAIL);
    for (const tr of TRACKS){
      for (let k = from; k <= here; k++){
        const A = tr.pts[k-1], B = tr.pts[k]; if (!A || !B) continue;
        clock.append("line").attr("x1",X(A[0])).attr("y1",Y(A[1]))
          .attr("x2",X(B[0])).attr("y2",Y(B[1]))
          .attr("stroke", css(tr.cls[k]==="saddle" ? "--saddle" : "--curve"))
          .attr("stroke-width",1.4)
          .attr("opacity", 0.08 + 0.5*(k - from + 1)/(here - from + 1));
      }
    }
  }
  for (const e of fr.equilibria){
    const col = colourOf(e.stability);
    for (const [re,im] of e.eig){
      clock.append("circle").attr("cx",X(re)).attr("cy",Y(im)).attr("r",9)
        .attr("fill",col).attr("opacity",.2);
      clock.append("circle").attr("cx",X(re)).attr("cy",Y(im)).attr("r",4.5)
        .attr("fill",col).attr("stroke",css("--panel")).attr("stroke-width",1.2);
    }
  }
  for (const cycle of fr.cycles){
    const trivial = cycle.multipliers.reduce((best, m) =>
      Math.hypot(m[0]-1, m[1]) < Math.hypot(best[0]-1, best[1]) ? m : best);
    for (const [re0,im0] of cycle.multipliers){
      const mod = Math.hypot(re0, im0), off = mod > 2.2, s = off ? 2.2/mod : 1;
      const re = re0*s, im = im0*s;
      const isTrivial = re0 === trivial[0] && im0 === trivial[1];
      clock.append("rect").attr("x",X(re)-4.5).attr("y",Y(im)-4.5)
        .attr("width",9).attr("height",9)
        .attr("opacity", (off ? .55 : 1) * cycle.alpha)
        .attr("fill",css("--panel"))
        .attr("stroke",css(isTrivial ? "--faint" : "--unstable"))
        .attr("stroke-width",1.8).attr("transform",`rotate(45 ${X(re)} ${Y(im)})`);
      if (off) label(clock, X(re)+(re>=0?-8:8), Y(im)-9,
        `|${MU}| = ${mod.toFixed(0)} →`, css("--unstable"), re>=0?"end":"start");
    }
  }
}

// ---------- timeline ----------
const tl = d3.select("#tl");
function drawTimeline(){
  const r = tl.node().getBoundingClientRect(); const tw = r.width, th = r.height;
  tl.attr("viewBox",`0 0 ${tw} ${th}`).selectAll("*").remove();
  const m = {l:56, r:22, t:18, b:34}; const iw = tw-m.l-m.r, ih = th-m.t-m.b;
  const arcs = D.branch.arcs;
  const px = d3.scaleLinear().domain([F[0].p, F[LAST].p]).range([m.l, m.l+iw]);
  const xlo = c => d3.min(c.states, s=>s[0]), xhi = c => d3.max(c.states, s=>s[0]);
  const xs = arcs.flat().map(d=>d.x).concat(D.cycles.flatMap(c => [xhi(c), xlo(c)]));
  const lo = d3.min(xs), hi = d3.max(xs), pad = Math.max(1e-6, (hi-lo)*0.08);
  const py = d3.scaleLinear().domain([lo-pad, hi+pad]).range([m.t+ih, m.t]);
  const g = tl.append("g");
  const axisStyle = ax => {
    ax.select(".domain").attr("stroke",css("--line"));
    ax.selectAll("line").attr("stroke",css("--line"));
    ax.selectAll("text").attr("fill",css("--muted")).attr("font-family",mono)
      .attr("font-size",11);
  };
  g.append("g").selectAll("line").data(py.ticks(4)).join("line")
    .attr("x1",m.l).attr("x2",m.l+iw).attr("y1",d=>py(d)).attr("y2",d=>py(d))
    .attr("stroke",css("--grid"));
  g.append("g").attr("transform",`translate(0,${m.t+ih})`)
    .call(d3.axisBottom(px).ticks(8).tickSize(4)).call(axisStyle);
  g.append("g").attr("transform",`translate(${m.l},0)`)
    .call(d3.axisLeft(py).ticks(4).tickSize(4))
    .call(ax => { axisStyle(ax); ax.select(".domain").remove(); });
  label(g, m.l-40, m.t+10, D.system.x_label, css("--faint"));
  label(g, m.l+iw, th-4, PARAM, css("--faint"), "end");
  const line = d3.line().x(d=>px(d.p)).y(d=>py(d.x))
    .curve(d3.curveCatmullRom.alpha(0.5));
  const dashes = {stable:"none", saddle:"2 5", unstable:"8 5"};
  for (const pts of arcs){
    let run = [];
    const flush = () => {
      if (run.length < 2) return;
      const s = run[0].stability;
      g.append("path").attr("d", line(run)).attr("fill","none")
        .attr("stroke", colourOf(s)).attr("stroke-width",2.4)
        .attr("stroke-dasharray", dashes[s] || "none");
    };
    for (const pt of pts){
      if (run.length && run[run.length-1].stability !== pt.stability){
        run.push(pt); flush(); run = [pt];
      } else run.push(pt);
    }
    flush();
  }
  if (D.cycles.length){
    const amp = d3.line().x(c=>px(c.p)).curve(d3.curveCatmullRom.alpha(0.5));
    // One path only where the branch is continuous and keeps its stability. A
    // folded branch comes back as the other kind of cycle, and joining the two
    // would draw a line across the diagram that no cycle follows.
    const JUMP = 0.08 * Math.hypot(iw, ih);
    let run = [];
    const flushCycles = () => {
      if (run.length > 1){
        const unstable = isUnstable(run[0]);
        for (const edge of [xhi, xlo]){
          g.append("path").attr("d", amp.y(c=>py(edge(c)))(run)).attr("fill","none")
            .attr("stroke", css(unstable ? "--unstable" : "--stable"))
            .attr("stroke-width",1.6)
            .attr("stroke-dasharray", unstable ? "5 4" : "none").attr("opacity",.9);
        }
      }
      run = [];
    };
    for (const c of D.cycles){
      const prev = run[run.length-1];
      const gap = prev
        ? Math.hypot(px(c.p)-px(prev.p), py(xhi(c))-py(xhi(prev))) : 0;
      if (prev && gap > JUMP){ flushCycles(); }
      else if (prev && isUnstable(c) !== isUnstable(prev)){
        run.push(c); flushCycles();
      }
      run.push(c);
    }
    flushCycles();
    // The branch ends where the parameter stops: a homoclinic, a fold, an
    // onset. That is the extreme the terminus names, whichever arc reaches it.
    const c = D.cycles.reduce((a,b) => b.p < a.p ? b : a);
    const col = css(isUnstable(c) ? "--unstable" : "--stable");
    for (const yy of [xhi(c), xlo(c)]){
      g.append("circle").attr("cx",px(c.p)).attr("cy",py(yy)).attr("r",3.5)
        .attr("fill",css("--panel")).attr("stroke",col).attr("stroke-width",1.6);
    }
    if (D.system.cycle_terminus){
      label(g, px(c.p)-9, py(xhi(c))+4, D.system.cycle_terminus, col, "end")
        .attr("font-size",12).attr("font-family",sans).attr("font-weight",500);
    }
  }
  for (const s of D.branch.special){
    const col = css(s.kind==="hopf"?"--hopf":"--fold"), left = s.kind==="hopf";
    g.append("circle").attr("cx",px(s.p)).attr("cy",py(s.x)).attr("r",8)
      .attr("fill",col).attr("opacity",.22);
    g.append("circle").attr("cx",px(s.p)).attr("cy",py(s.x)).attr("r",4.5)
      .attr("fill",col).attr("stroke",css("--panel")).attr("stroke-width",1.5);
    label(g, px(s.p)+(left?-9:9), py(s.x)+(left?18:-10), s.label || pretty(s.kind),
      col, left?"end":"start")
      .attr("font-size",12).attr("font-family",sans).attr("font-weight",500);
  }
  const ph = g.append("g");
  ph.append("line").attr("y1",m.t-6).attr("y2",m.t+ih).attr("stroke",css("--ink"))
    .attr("stroke-width",1.4);
  ph.append("polygon").attr("points","-6,-6 6,-6 0,2").attr("fill",css("--ink"))
    .attr("transform",`translate(0,${m.t-6})`);
  const toT = ev => {
    const [mx] = d3.pointer(ev, tl.node());
    return toFrameIndex(px.invert(clamp(mx, m.l, m.l+iw)));
  };
  let down = false;
  tl.on("pointerdown", ev => {
      down = true; tl.node().setPointerCapture(ev.pointerId); setT(toT(ev)); })
    .on("pointermove", ev => { if (down) setT(toT(ev)); })
    .on("pointerup pointercancel", () => { down = false; });
  drawTimeline.updatePlayhead = () =>
    ph.attr("transform",`translate(${px(paramAt(t))},0)`);
  drawTimeline.updatePlayhead();
}

// ---------- readouts ----------
const fmtC = ([re,im]) => `${re>=0?"+":MINUS}${Math.abs(re).toFixed(3)} `
  + `${im>=0?"+":MINUS} ${Math.abs(im).toFixed(3)}i`;
function genericRegime(fr){
  const n = fr.equilibria.length; if (!n) return "No equilibria";
  const count = {};
  for (const e of fr.equilibria) count[e.stability] = (count[e.stability]||0)+1;
  const parts = ["stable","unstable","saddle"].filter(k => count[k])
    .map(k => `${count[k]} ${k}`);
  let s = `${n} equilibri${n===1?"um":"a"}: ${parts.join(", ")}`;
  if (fr.cycles.length){
    const kinds = fr.cycles.map(c => isUnstable(c) ? "unstable" : "stable");
    s += ` ${DOT} ${kinds.join(" and ")} cycle${fr.cycles.length > 1 ? "s" : ""}`;
  }
  return s;
}
function readouts(){
  const fr = viewAt(t);
  const focus = fr.equilibria.find(e=>e.stability!=="saddle");
  const saddle = fr.equilibria.find(e=>e.stability==="saddle");
  document.getElementById("ro-p").textContent = paramAt(t).toFixed(4);
  document.getElementById("ro-regime-sub").textContent =
    `frame ${Math.round(t)+1} / ${F.length}`;
  const setEq = (id, e, fallback) => {
    const k = document.getElementById(id+"-kicker"), v = document.getElementById(id);
    const s = document.getElementById(id+"-sub");
    if (!e){
      setKicker(k, fallback, LAMBDA);
      v.textContent = "\u2014"; s.textContent = "absent";
      return;
    }
    setKicker(k, e.stability, LAMBDA);
    v.innerHTML = `<span class="dot" style="background:${colourOf(e.stability)}">`
      + `</span>${fmtC(e.eig[0])}`;
    s.textContent =
      `${fmtC(e.eig[1])} ${DOT} at ${D.system.x_label} = ${e.x.toFixed(3)}`;
  };
  setEq("ro-focus", focus, "equilibrium");
  setEq("ro-saddle", saddle, "saddle");
  const cv = document.getElementById("ro-cycle");
  const cs = document.getElementById("ro-cycle-sub");
  if (fr.cycles.length){
    const worst = Math.max(...fr.cycles.map(c => c.error || 0));
    const digits = worst < 0.01 ? 2 : 1;
    cv.textContent = `T = ${fr.cycles.map(c => c.period.toFixed(2)).join(", ")}`;
    cv.style.color = worst > 0.02 ? css("--hopf") : "";
    const each = fr.cycles.map(c => {
      const mods = c.multipliers.map(([a,b]) => Math.hypot(a,b));
      const kind = isUnstable(c) ? "unstable" : "stable";
      return `${kind} a=${c.amplitude.toFixed(3)} `
        + `|${MU}|=${mods.map(m => m.toFixed(3)).join(",")}`;
    });
    cs.textContent = `${each.join(` ${DOT} `)} ${DOT} Floquet `
      + `${PLUSMINUS}${(100*worst).toFixed(digits)}%`;
  } else {
    cv.textContent = DASH; cv.style.color = "";
    cs.textContent = `no cycle at this ${PARAM}`;
  }
  document.getElementById("ro-regime").textContent = fr.label || genericRegime(fr);
}

// ---------- orchestration ----------
let lastFrame = -1;
function setT(v){ t = clamp(v, 0, LAST); onFrame(); }
function onFrame(){
  const k = Math.round(t);
  lastFrame = k;
  drawSkeleton(); drawClock(); readouts();
  if (drawTimeline.updatePlayhead) drawTimeline.updatePlayhead();
}
let last = performance.now();
function loop(now){
  const dt = Math.min(0.05, (now-last)/1000); last = now;
  if (playing){
    t += dir * speed * dt * 4.2;
    if (t >= LAST){ t = LAST; dir = -1; } if (t <= 0){ t = 0; dir = 1; }
    onFrame();
  }
  if (!reduced) stepParticles();
  requestAnimationFrame(loop);
}
const playBtn = document.getElementById("play");
function setPlaying(v){
  playing = v; playBtn.setAttribute("aria-pressed", String(v));
  playBtn.firstChild.textContent = (v ? "Pause " : "Play ");
}
playBtn.addEventListener("click", () => setPlaying(!playing));
window.addEventListener("keydown", e => {
  const typing = e.target.tagName==="INPUT" || e.target.tagName==="BUTTON";
  if (e.code==="Space" && !typing){ e.preventDefault(); setPlaying(!playing); }
  if (e.key==="ArrowRight") setT(t+1); if (e.key==="ArrowLeft") setT(t-1);
});
document.getElementById("speed").addEventListener("input",
  e => speed = +e.target.value);
document.getElementById("density").addEventListener("input",
  e => setDensity(+e.target.value));
for (const id of ["l-manifolds","l-cycle","l-trails"]){
  document.getElementById(id).addEventListener("change",
    () => { drawSkeleton(); drawClock(); });
}
// The plane picker only appears where there is a choice to make, so a planar
// system's controls are unchanged.
if (PROJS.length > 1){
  const sel = document.getElementById("plane");
  PROJS.forEach((p, k) => {
    const o = document.createElement("option");
    o.value = String(k);
    o.textContent = p.name || `${p.x_label || "?"} - ${p.y_label || "?"}`;
    sel.appendChild(o);
  });
  sel.value = "0";
  document.getElementById("plane-group").hidden = false;
  sel.addEventListener("change", e => {
    applyProjection(+e.target.value);
    setDensity(+document.getElementById("density").value);
    drawSkeleton();
  });
}
setKicker(document.getElementById("ro-p-kicker"), "", PARAM);
document.getElementById("tl-hint").textContent =
  `drag the playhead ${DOT} ${D.system.x_label} against ${PARAM}`;
if (PROJS.length) applyProjection(0);
window.addEventListener("resize", resize);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", resize);
new MutationObserver(resize).observe(document.documentElement,
  {attributes:true, attributeFilter:["data-theme"]});

setDensity(+document.getElementById("density").value);
resize(); readouts(); onFrame();
if (reduced){ for (let k=0;k<40;k++) stepParticles(); }
requestAnimationFrame(loop);
})();
"""

_FONTS = (
    "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,300..600,"
    "0..100&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500"
    "&display=swap"
)

_BODY = """
<div class="wrap">
  <div class="topbar">
    <a class="back" href="index.html">&larr; atlas</a>
    <div class="themes" role="group" aria-label="colour theme">
      <button class="theme-btn" data-theme="dark">Dark</button>
      <button class="theme-btn" data-theme="light">Light</button>
    </div>
  </div>
  <div class="mast">
    <div>
      <div class="kicker">Stage &amp; timeline &middot; continuation as a film</div>
      <h1>__TITLE__</h1>
      <div class="eqn">__EQUATION__</div>
    </div>
    <p class="lede">__SUBTITLE__</p>
  </div>

  <div class="instrument">
    <section class="stage" aria-label="phase portrait">
      <div class="panel-head">
        <span class="kicker">Stage &middot; phase plane</span>
        <span class="mono" id="stage-p"></span>
      </div>
      <div class="frame"><canvas id="flow"></canvas><svg id="skeleton"></svg></div>
      <div class="legend">
        <span><i style="border-color:var(--stable)"></i>stable manifold</span>
        <span><i style="border-color:var(--unstable)"></i>unstable manifold</span>
        <span><i class="dash" style="border-color:var(--unstable)"></i>cycle
          (dashed if unstable)</span>
        <span><span class="dot" style="background:var(--stable)"></span>stable</span>
        <span><span class="dot" style="background:var(--unstable)"></span>unstable
          </span>
        <span><span class="dot" style="background:var(--saddle)"></span>saddle</span>
      </div>
    </section>
    <section class="clock" aria-label="spectral clock">
      <div class="panel-head">
        <span class="kicker">Spectral clock</span><span class="mono">&#8450;</span>
      </div>
      <div class="frame"><svg id="clock"></svg></div>
      <div class="legend">
        <span><span class="dot" style="background:var(--ink)"></span>eigenvalue
          &lambda;</span>
        <span><span class="dot" style="background:transparent;
          border:1.5px solid var(--unstable);border-radius:1px"></span>Floquet
          &mu;</span>
        <span><span class="dot" style="background:transparent;
          border:1.5px solid var(--faint);border-radius:1px"></span>trivial &mu;
          (should be 1; its drift is the error)</span>
      </div>
    </section>
  </div>

  <div class="readouts" id="readouts">
    <div class="ro"><span class="kicker" id="ro-p-kicker"></span>
      <div class="val" id="ro-p">&mdash;</div>
      <div class="sub" id="ro-regime-sub"></div></div>
    <div class="ro"><span class="kicker" id="ro-focus-kicker"></span>
      <div class="val" id="ro-focus">&mdash;</div>
      <div class="sub" id="ro-focus-sub"></div></div>
    <div class="ro"><span class="kicker" id="ro-saddle-kicker"></span>
      <div class="val" id="ro-saddle">&mdash;</div>
      <div class="sub" id="ro-saddle-sub"></div></div>
    <div class="ro"><span class="kicker">Cycle &middot; period, Floquet</span>
      <div class="val" id="ro-cycle">&mdash;</div>
      <div class="sub" id="ro-cycle-sub"></div></div>
    <div class="ro regime"><span class="kicker">Regime</span>
      <div class="val" id="ro-regime">&mdash;</div></div>
  </div>

  <section class="timeline" aria-label="bifurcation diagram timeline">
    <div class="panel-head">
      <span class="kicker">Timeline &middot; bifurcation diagram</span>
      <span class="mono" id="tl-hint"></span>
    </div>
    <div class="frame"><svg id="tl"></svg></div>
  </section>

  <div class="controls">
    <button class="btn play" id="play" aria-pressed="false">Play
      <kbd>space</kbd></button>
    <label class="group">speed <input class="range" id="speed" type="range"
      min="0.2" max="3" step="0.1" value="1"></label>
    <label class="group">particles <input class="range" id="density" type="range"
      min="400" max="6000" step="200" value="2600"></label>
    <label class="group" id="plane-group" hidden>plane
      <select class="sel" id="plane"></select></label>
    <label class="toggle"><input id="l-manifolds" type="checkbox" checked>
      <span class="sw"></span>manifolds</label>
    <label class="toggle"><input id="l-cycle" type="checkbox" checked>
      <span class="sw"></span>cycle</label>
    <label class="toggle"><input id="l-trails" type="checkbox" checked>
      <span class="sw"></span>eigen-trails</label>
  </div>

  <p class="foot">__NOTE__</p>
</div>
"""

_TEMPLATE = (
    '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    "<title>__TITLE__</title>\n"
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    f'<link rel="stylesheet" href="{_FONTS}">\n'
    f"<style>{_STYLE}</style>\n</head>\n<body>{_BODY}"
    "<script>__D3_SOURCE__</script>\n"
    "<script>window.__STAGE__ = __STAGE_JSON__;</script>\n"
    f"<script>{_SCRIPT}</script>\n</body>\n</html>\n"
)
