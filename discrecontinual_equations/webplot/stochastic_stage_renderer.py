"""Render a stochastic :class:`~.stage.StageScene` as a playable page.

The deterministic stage has three panels: the flow, a spectral clock, and the
timeline. A stochastic system keeps the first and the last and changes what
each shows, and replaces the clock outright, because three of its instruments
measure the wrong thing once there is noise:

* The *stage* no longer advects particles along a flow; it integrates sample
  paths of ``dx = f dt + G dW`` by Euler-Maruyama, with the drift and the
  noise read from polynomial terms so they are exact in the parameter, and
  with every random number drawn from a generator seeded from the payload.
  The seed is what makes the film the same on every build and every reload;
  paths are integrated in the browser rather than shipped precomputed because
  a replayed path cannot follow the parameter as the viewer scrubs it, and
  because a page of precomputed paths would be several times the size of the
  page of everything else.
* A *density panel* is new: the stationary density at the current parameter
  as a heatmap, with the interior maxima marked, so the peak leaving the
  reference state - the phenomenological bifurcation - is watched happening.
* The *clock* showed a spectrum of eigenvalues. A stochastic system has no
  spectrum to show; what the dynamical bifurcation changes is the sign of one
  real number, the top Lyapunov exponent. The panel in the clock's place
  draws that exponent as a function of the parameter, with its zero marked
  and the closed form beside it where one is known.
* The *timeline* draws both thresholds as vertical rules at their own
  parameter values, across the whole diagram, because the fact that they do
  not coincide is the film's one claim and must be impossible to miss.

Everything else - the mast, theme, controls, skeleton drawing, the blending
of frames - is the deterministic page's own script, reused as segments; a
deterministic scene handed to this renderer comes out byte for byte as
:class:`~.stage_renderer.StageRenderer` renders it.
"""

import json

from discrecontinual_equations.webplot.latex import latex_to_svg
from discrecontinual_equations.webplot.stage import StageScene, stage_payload
from discrecontinual_equations.webplot.stage_renderer import (
    _BODY_FOOT,
    _BODY_OPEN,
    _BODY_STAGE,
    _BODY_TIMELINE,
    _FONTS,
    _SCRIPT_HEAD,
    _SCRIPT_SKELETON,
    _STYLE,
    StageRenderer,
    _escape,
)


class StochasticStageRenderer(StageRenderer):
    """Draw a stochastic stage; a deterministic one renders exactly as before."""

    __slots__ = []

    def render(self, scene: StageScene) -> str:
        if scene.stochastic is None:
            return super().render(scene)
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


_STOCHASTIC_STYLE = """
  .side{display:flex;flex-direction:column;gap:16px;min-width:0}
  .density,.exponent{background:var(--panel);border:1px solid var(--line);
    border-radius:14px;position:relative;overflow:hidden}
  .density .frame{position:relative;width:100%;aspect-ratio:1/1}
  .density canvas,.density svg{position:absolute;inset:0;width:100%;height:100%}
  .density svg{pointer-events:none}
  .exponent{display:flex;flex-direction:column;flex:1}
  .exponent .frame{position:relative;width:100%;flex:1;min-height:190px}
  .exponent svg{position:absolute;inset:0;width:100%;height:100%}
  .legend i.dot-ring{width:10px;height:10px;border:1.5px dashed;border-radius:50%;
    vertical-align:middle}
  @media (max-width:820px){ .exponent .frame{flex:none;height:220px} }
"""

_STOCHASTIC_MAST = """  <div class="mast">
    <div>
      <div class="kicker">Stage, density &amp; timeline &middot; a stochastic
        bifurcation as a film</div>
      <h1>__TITLE__</h1>
      <div class="eqn">__EQUATION__</div>
    </div>
    <p class="lede">__SUBTITLE__</p>
  </div>

"""

_STOCHASTIC_PANELS = """    <div class="side">
      <section class="density" aria-label="stationary density">
        <div class="panel-head">
          <span class="kicker">Density &middot; stationary p</span>
          <span class="mono" id="density-note"></span>
        </div>
        <div class="frame"><canvas id="heat"></canvas><svg id="heat-skel"></svg></div>
        <div class="legend">
          <span><span class="dot" style="background:var(--fold)"></span>interior
            maxima of p</span>
          <span><i class="dot-ring" style="border-color:var(--fold)"></i>crest</span>
        </div>
      </section>
      <section class="exponent" aria-label="top Lyapunov exponent">
        <div class="panel-head">
          <span class="kicker">Top Lyapunov exponent &middot; &lambda;(parameter)</span>
          <span class="mono" id="exp-value"></span>
        </div>
        <div class="frame"><svg id="exp"></svg></div>
        <div class="legend">
          <span><i style="border-color:var(--hopf)"></i>estimate (Benettin, common
            random numbers)</span>
          <span><i class="dash" style="border-color:var(--faint)"></i>closed form</span>
        </div>
      </section>
    </div>
  </div>

"""

_STOCHASTIC_READOUTS_BODY = """  <div class="readouts" id="readouts">
    <div class="ro"><span class="kicker" id="ro-p-kicker"></span>
      <div class="val" id="ro-p">&mdash;</div>
      <div class="sub" id="ro-regime-sub"></div></div>
    <div class="ro"><span class="kicker" id="ro-exp-kicker"></span>
      <div class="val" id="ro-exp">&mdash;</div>
      <div class="sub" id="ro-exp-sub"></div></div>
    <div class="ro"><span class="kicker">Density &middot; crest</span>
      <div class="val" id="ro-crest">&mdash;</div>
      <div class="sub" id="ro-crest-sub"></div></div>
    <div class="ro regime"><span class="kicker">Regime</span>
      <div class="val" id="ro-regime">&mdash;</div></div>
  </div>

"""

_STOCHASTIC_CONTROLS = """  <div class="controls">
    <button class="btn play" id="play" aria-pressed="false">Play
      <kbd>space</kbd></button>
    <label class="group">speed <input class="range" id="speed" type="range"
      min="0.2" max="3" step="0.1" value="1"></label>
    <label class="group">paths <input class="range" id="paths" type="range"
      min="200" max="4000" step="200" value="1600"></label>
    <label class="toggle" hidden><input id="l-manifolds" type="checkbox">
      <span class="sw"></span>manifolds</label>
    <label class="toggle"><input id="l-cycle" type="checkbox" checked>
      <span class="sw"></span>deterministic cycle</label>
    <label class="toggle"><input id="l-crest" type="checkbox" checked>
      <span class="sw"></span>crest</label>
    <label class="toggle"><input id="l-maxima" type="checkbox" checked>
      <span class="sw"></span>maxima</label>
    <label class="toggle"><input id="l-exact" type="checkbox" checked>
      <span class="sw"></span>closed forms</label>
  </div>

"""

_STOCHASTIC_PARTICLES = r"""// ---------- stage: sample paths on canvas ----------
// The particles are sample paths of dx = f dt + G dW, stepped by Euler-Maruyama
// from the drift terms the view carries and the noise terms the stochastic
// block adds; both exact in the parameter. Every random number - where a path
// starts, how long it lives, each Brownian increment - comes from one generator
// seeded from the payload, so the film is the same on every reload and every
// build. (srk2-srk5 in the library are not used: see DEQ-25.)
const S = D.stochastic;
const canvas = document.getElementById("flow"), ctx = canvas.getContext("2d");
const skel = d3.select("#skeleton");
let W = 0, H = 0, dpr = 1;
const sx = v => (v - x0) / (x1 - x0) * W, sy = v => H - (v - y0) / (y1 - y0) * H;
const N = V.matrix[0].length, K = S.noise.length;
const [CX, CY] = S.centre;
function mulberry32(a){
  return function(){
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
let uniform = mulberry32(S.seed), spare = null;
function gaussian(){
  if (spare !== null){ const g = spare; spare = null; return g; }
  let u = 0; while (u <= 1e-12) u = uniform();
  const v = uniform(), r = Math.sqrt(-2 * Math.log(u));
  spare = r * Math.sin(2 * Math.PI * v);
  return r * Math.cos(2 * Math.PI * v);
}
let P = [];
function projectWith(M, x){
  let u = 0, v = 0;
  for (let j=0;j<N;j++){ u += M[0][j]*x[j]; v += M[1][j]*x[j]; }
  return [u, v];
}
function project(x){ return projectWith(V.matrix, x); }
function evalTerms(terms, x, param, out){
  for (let i=0;i<terms.length;i++){
    let total = 0;
    for (const term of terms[i]){
      let m = term.c; if (term.q) m *= Math.pow(param, term.q);
      for (let j=0;j<N;j++){
        const e = term.e[j]; if (e === 1) m *= x[j]; else if (e) m *= Math.pow(x[j], e);
      }
      total += m;
    }
    out[i] = total;
  }
}
function spawn(p){
  for (let j=0;j<N;j++){
    const [lo,hi] = V.bounds[j]; p.x[j] = lo + uniform()*(hi-lo);
  }
  const q = project(p.x); p.u = q[0]; p.v = q[1];
  p.age = 40 + uniform()*140;
}
function setDensity(n){
  // Reseeded, so the ensemble after changing the count is as reproducible as
  // the one the page opened with.
  uniform = mulberry32(S.seed); spare = null; P = [];
  for (let k=0;k<n;k++){
    const p = {x: new Array(N).fill(0), u: 0, v: 0, age: 0};
    spawn(p); p.age = uniform()*180; P.push(p);
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
  drawSkeleton(); drawCrest(); drawDensity(); drawExponent(); drawTimeline();
}
const drift = new Array(N).fill(0), kick = new Array(N).fill(0);
function stepParticles(){
  ctx.fillStyle = css("--panel"); ctx.globalAlpha = parseFloat(css("--fade"));
  ctx.fillRect(0,0,W,H); ctx.globalAlpha = 1;
  ctx.strokeStyle = `rgba(${css("--particle")},${css("--particle-alpha")})`;
  ctx.lineWidth = 1.1; ctx.lineCap = "round";
  const h = S.step, root = Math.sqrt(h), param = paramAt(t);
  ctx.beginPath();
  for (const p of P){
    const pu = p.u, pv = p.v;
    // Drift and every noise column at the current state, before any update:
    // that is what makes the step Euler-Maruyama rather than something else.
    evalTerms(V.field, p.x, param, drift);
    for (let j=0;j<N;j++) kick[j] = h*drift[j];
    for (let k=0;k<K;k++){
      const xi = root*gaussian();
      evalTerms(S.noise[k], p.x, param, drift);
      for (let j=0;j<N;j++) kick[j] += xi*drift[j];
    }
    let out = false;
    for (let j=0;j<N;j++){
      p.x[j] += kick[j];
      const [lo,hi] = V.bounds[j]; if (p.x[j] < lo || p.x[j] > hi) out = true;
    }
    const q = project(p.x); p.u = q[0]; p.v = q[1]; p.age -= 1;
    const moved = Math.hypot(p.u-pu, p.v-pv);
    const gone = out || p.u < x0 || p.u > x1 || p.v < y0 || p.v > y1;
    // A path that has stopped moving has fallen into an invariant point
    // where the noise vanishes with it; it is reborn elsewhere.
    if (gone || p.age <= 0 || moved < 1e-5){ spawn(p); continue; }
    ctx.moveTo(sx(pu), sy(pv)); ctx.lineTo(sx(p.u), sy(p.v));
  }
  ctx.stroke();
}

"""

_STOCHASTIC_PANELS_SCRIPT = r"""// ---------- density between frames ----------
function densityPair(tt){
  const {i, j, f} = frameAt(tt);
  return {A: F[i].density || null, B: F[j].density || null, f, near: f < 0.5 ? i : j};
}
// Densities are blended linearly between frames, as the field is, so the
// heatmap moves with the playhead rather than stepping; the crest is blended
// where both frames have one and taken from the nearer frame otherwise.
function densityAt(tt){
  const {A, B, f, near} = densityPair(tt);
  const nearest = F[near].density || null;
  if (!A || !B || !A.values.length || !B.values.length || f < 1e-6) return nearest;
  const n = A.values.length, values = new Float64Array(n);
  for (let k=0;k<n;k++) values[k] = A.values[k]*(1-f) + B.values[k]*f;
  const crest = (A.crest != null && B.crest != null) ? lerp(A.crest, B.crest, f)
    : nearest.crest;
  return {values, crest, maxima: nearest.maxima, mode: nearest.mode,
    peak: lerp(A.peak, B.peak, f)};
}
function drawCrest(){
  if (!document.getElementById("l-crest").checked) return;
  const d = densityAt(t); if (!d || d.crest == null) return;
  skel.append("ellipse").attr("cx", sx(CX)).attr("cy", sy(CY))
    .attr("rx", d.crest/(x1-x0)*W).attr("ry", d.crest/(y1-y0)*H)
    .attr("fill","none").attr("stroke", css("--fold")).attr("stroke-width", 2)
    .attr("stroke-dasharray","3 4").attr("opacity",.9);
}

// ---------- density panel ----------
const heat = document.getElementById("heat"), hctx = heat.getContext("2d");
const hskel = d3.select("#heat-skel");
const DB = S.density, HNX = DB.nx, HNY = DB.ny;
const [hx0, hx1] = DB.box.x, [hy0, hy1] = DB.box.y;
// The colour ramp is rebuilt from the theme on every draw, and sampled into
// a small table so the per-cell work is a lookup.
function rampTable(){
  const ramp = d3.scaleLinear().domain([0, 0.55, 1])
    .range([css("--panel"), css("--curve"), css("--hopf")])
    .interpolate(d3.interpolateRgb);
  const table = new Uint8ClampedArray(256*3);
  for (let k=0;k<256;k++){
    const c = d3.rgb(ramp(k/255));
    table[3*k] = c.r; table[3*k+1] = c.g; table[3*k+2] = c.b;
  }
  return table;
}
function styleAxis(ax){
  ax.select(".domain").attr("stroke",css("--line"));
  ax.selectAll("line").attr("stroke",css("--line"));
  ax.selectAll("text").attr("fill",css("--muted")).attr("font-family",mono)
    .attr("font-size",11);
}
function drawDensity(){
  const r = heat.getBoundingClientRect(); const w = r.width, hh = r.height;
  hskel.attr("viewBox", `0 0 ${w} ${hh}`).selectAll("*").remove();
  const hsx = v => (v - hx0) / (hx1 - hx0) * w;
  const hsy = v => hh - (v - hy0) / (hy1 - hy0) * hh;
  const note = document.getElementById("density-note");
  const d = densityAt(t);
  // Painted at grid resolution and left to the browser to scale smoothly.
  heat.width = HNX; heat.height = HNY;
  if (!d || !d.values.length){
    hctx.fillStyle = css("--panel"); hctx.fillRect(0, 0, HNX, HNY);
    hskel.append("circle").attr("cx", hsx(CX)).attr("cy", hsy(CY)).attr("r", 14)
      .attr("fill", css("--curve")).attr("opacity", .18);
    hskel.append("circle").attr("cx", hsx(CX)).attr("cy", hsy(CY)).attr("r", 5)
      .attr("fill", css("--curve"));
    label(hskel, w/2, hh/2 + 34, "p is a point mass at the origin", css("--muted"),
      "middle").attr("font-size", 12).attr("font-family", sans);
    label(hskel, w/2, hh/2 + 52, "every path falls in; nothing to spread",
      css("--faint"), "middle").attr("font-size", 11).attr("font-family", sans);
    note.textContent = "no density on the plane";
  } else {
    const table = rampTable();
    // Clipped at a high quantile rather than the peak, so a spike at the
    // reference state does not black out the rest of the plane.
    const sorted = Float64Array.from(d.values).sort();
    const clip = Math.max(1e-9, sorted[Math.floor(0.985*(sorted.length-1))]);
    const img = hctx.createImageData(HNX, HNY);
    for (let iy=0; iy<HNY; iy++){
      for (let ix=0; ix<HNX; ix++){
        const v = Math.min(1, d.values[iy*HNX + ix]/clip);
        const c = 3*Math.round(255*v), k = 4*((HNY-1-iy)*HNX + ix);
        img.data[k] = table[c]; img.data[k+1] = table[c+1]; img.data[k+2] = table[c+2];
        img.data[k+3] = 255;
      }
    }
    hctx.putImageData(img, 0, 0);
    if (document.getElementById("l-maxima").checked){
      for (const [mx, my] of d.maxima){
        hskel.append("circle").attr("cx", hsx(mx)).attr("cy", hsy(my)).attr("r", 2.6)
          .attr("fill", css("--fold")).attr("stroke", css("--panel"))
          .attr("stroke-width", .8);
      }
    }
    if (d.crest != null && document.getElementById("l-crest").checked){
      hskel.append("ellipse").attr("cx", hsx(CX)).attr("cy", hsy(CY))
        .attr("rx", d.crest/(hx1-hx0)*w).attr("ry", d.crest/(hy1-hy0)*hh)
        .attr("fill","none").attr("stroke", css("--fold")).attr("stroke-width", 1.6)
        .attr("stroke-dasharray","3 4").attr("opacity",.9);
    }
    note.textContent = d.crest == null
      ? `peak at the origin ${DOT} p max ${d.peak.toFixed(2)}`
      : `crest r = ${d.crest.toFixed(3)} ${DOT} p max ${d.peak.toFixed(2)}`;
  }
  const faint = css("--faint");
  for (const v of ticksFor(hx0, hx1, w)){
    hskel.append("line").attr("x1", hsx(v)).attr("x2", hsx(v)).attr("y1", hh-5)
      .attr("y2", hh).attr("stroke", faint).attr("opacity", .7);
  }
  for (const v of ticksFor(hy0, hy1, hh)){
    hskel.append("line").attr("y1", hsy(v)).attr("y2", hsy(v)).attr("x1", 0)
      .attr("x2", 5).attr("stroke", faint).attr("opacity", .7);
  }
  label(hskel, w-10, hh-8, xName, faint, "end").attr("font-size", 11);
  label(hskel, 8, 14, yName, faint).attr("font-size", 11);
}

// ---------- exponent panel: lambda against the parameter ----------
const exp = d3.select("#exp");
const EXP = S.exponent, EXACT = S.exact_exponent;
function interpolate(pairs, p){
  if (!pairs || !pairs.length) return null;
  if (p <= pairs[0][0]) return pairs[0][1];
  for (let k=1;k<pairs.length;k++){
    if (p <= pairs[k][0]){
      const a = pairs[k-1], b = pairs[k], f = (p-a[0])/((b[0]-a[0]) || 1);
      return a[1] + (b[1]-a[1])*f;
    }
  }
  return pairs[pairs.length-1][1];
}
function exponentAt(p){ return interpolate(EXP, p); }
function exactExponentAt(p){ return interpolate(EXACT, p); }
const THRESHOLDS = D.branch.special.filter(s =>
  s.kind === "d_bifurcation" || s.kind === "p_bifurcation");
const ruleColour = s => css(s.kind === "p_bifurcation" ? "--fold" : "--hopf");
function drawExponent(){
  const r = exp.node().getBoundingClientRect(); const w = r.width, h = r.height;
  exp.attr("viewBox",`0 0 ${w} ${h}`).selectAll("*").remove();
  const m = {l:46, r:16, t:14, b:30}; const iw = w-m.l-m.r, ih = h-m.t-m.b;
  const px = d3.scaleLinear().domain([F[0].p, F[LAST].p]).range([m.l, m.l+iw]);
  const ys = EXP.map(d=>d[1]).concat(EXACT ? EXACT.map(d=>d[1]) : []).concat([0]);
  const lo = d3.min(ys), hi = d3.max(ys), pad = Math.max(1e-6, (hi-lo)*0.1);
  const py = d3.scaleLinear().domain([lo-pad, hi+pad]).range([m.t+ih, m.t]);
  const g = exp.append("g");
  // The sign is the whole content: tint the two half-planes of lambda.
  g.append("rect").attr("x",m.l).attr("width",iw).attr("y",m.t).attr("height",py(0)-m.t)
    .attr("fill",css("--unstable")).attr("opacity",.07);
  g.append("rect").attr("x",m.l).attr("width",iw).attr("y",py(0))
    .attr("height",m.t+ih-py(0)).attr("fill",css("--stable")).attr("opacity",.07);
  g.append("line").attr("x1",m.l).attr("x2",m.l+iw).attr("y1",py(0)).attr("y2",py(0))
    .attr("stroke",css("--faint")).attr("stroke-width",1.2);
  g.append("g").attr("transform",`translate(0,${m.t+ih})`)
    .call(d3.axisBottom(px).ticks(5).tickSize(4)).call(styleAxis);
  g.append("g").attr("transform",`translate(${m.l},0)`)
    .call(d3.axisLeft(py).ticks(4).tickSize(4))
    .call(ax => { styleAxis(ax); ax.select(".domain").remove(); });
  label(g, m.l-36, m.t+10, LAMBDA, css("--faint"));
  label(g, m.l+iw, h-4, PARAM, css("--faint"), "end");
  label(g, m.l+6, py(0)-6, `${LAMBDA} > 0 ${DOT} unstable`, css("--unstable"))
    .attr("opacity",.8);
  label(g, m.l+6, py(0)+13, `${LAMBDA} < 0 ${DOT} stable`, css("--stable"))
    .attr("opacity",.8);
  const line = d3.line().x(d=>px(d[0])).y(d=>py(d[1]));
  if (EXACT && document.getElementById("l-exact").checked){
    g.append("path").attr("d", line(EXACT)).attr("fill","none")
      .attr("stroke",css("--faint")).attr("stroke-width",1.3)
      .attr("stroke-dasharray","5 4").attr("opacity",.9);
  }
  g.append("path").attr("d", line(EXP)).attr("fill","none")
    .attr("stroke",css("--hopf")).attr("stroke-width",2.2);
  for (const [p, v] of EXP){
    g.append("circle").attr("cx",px(p)).attr("cy",py(v)).attr("r",2.6)
      .attr("fill",css("--hopf")).attr("stroke",css("--panel")).attr("stroke-width",1);
  }
  for (const s of THRESHOLDS){
    if (s.kind !== "d_bifurcation") continue;
    g.append("line").attr("x1",px(s.p)).attr("x2",px(s.p)).attr("y1",m.t)
      .attr("y2",m.t+ih).attr("stroke",css("--hopf")).attr("stroke-dasharray","2 4")
      .attr("opacity",.8);
    g.append("circle").attr("cx",px(s.p)).attr("cy",py(0)).attr("r",5)
      .attr("fill",css("--panel")).attr("stroke",css("--hopf")).attr("stroke-width",2);
    label(g, px(s.p)+7, m.t+11, `D ${DOT} ${PARAM} = ${s.p.toFixed(3)}`, css("--hopf"))
      .attr("font-family",sans).attr("font-weight",500).attr("font-size",12);
  }
  const p = paramAt(t), v = exponentAt(p);
  g.append("line").attr("x1",px(p)).attr("x2",px(p)).attr("y1",m.t).attr("y2",m.t+ih)
    .attr("stroke",css("--ink")).attr("stroke-width",1).attr("opacity",.35);
  g.append("circle").attr("cx",px(p)).attr("cy",py(v)).attr("r",6)
    .attr("fill",css(v > 0 ? "--unstable" : "--stable")).attr("stroke",css("--panel"))
    .attr("stroke-width",1.5);
  document.getElementById("exp-value").textContent =
    `${LAMBDA} = ${v >= 0 ? "+" : MINUS}${Math.abs(v).toFixed(3)}`;
}

"""

_STOCHASTIC_TIMELINE = r"""// ---------- timeline: branch, crest, thresholds ----------
const tl = d3.select("#tl");
const CREST = F.filter(f => f.density && f.density.crest != null)
  .map(f => [f.p, f.density.crest]);
const EXACT_CREST = S.exact_crest;
function drawTimeline(){
  const r = tl.node().getBoundingClientRect(); const tw = r.width, th = r.height;
  tl.attr("viewBox",`0 0 ${tw} ${th}`).selectAll("*").remove();
  const m = {l:56, r:22, t:30, b:34}; const iw = tw-m.l-m.r, ih = th-m.t-m.b;
  const arcs = D.branch.arcs;
  const px = d3.scaleLinear().domain([F[0].p, F[LAST].p]).range([m.l, m.l+iw]);
  const xlo = c => d3.min(c.states, s=>s[0]), xhi = c => d3.max(c.states, s=>s[0]);
  const xs = arcs.flat().map(d=>d.x)
    .concat(D.cycles.flatMap(c => [xhi(c), xlo(c)]))
    .concat(CREST.flatMap(([_p, c]) => [CX + c, CX - c]));
  const lo = d3.min(xs), hi = d3.max(xs), pad = Math.max(1e-6, (hi-lo)*0.08);
  const py = d3.scaleLinear().domain([lo-pad, hi+pad]).range([m.t+ih, m.t]);
  const g = tl.append("g");
  g.append("g").selectAll("line").data(py.ticks(4)).join("line")
    .attr("x1",m.l).attr("x2",m.l+iw).attr("y1",d=>py(d)).attr("y2",d=>py(d))
    .attr("stroke",css("--grid"));
  g.append("g").attr("transform",`translate(0,${m.t+ih})`)
    .call(d3.axisBottom(px).ticks(8).tickSize(4)).call(styleAxis);
  g.append("g").attr("transform",`translate(${m.l},0)`)
    .call(d3.axisLeft(py).ticks(4).tickSize(4))
    .call(ax => { styleAxis(ax); ax.select(".domain").remove(); });
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
  // The deterministic cycle is a reference here, drawn faint: what the drift
  // alone would do, so the eye can read the gap the noise opens below it.
  if (D.cycles.length){
    const ordered = D.cycles.slice().sort((a, b) => a.p - b.p);
    const amp = d3.line().x(c=>px(c.p)).curve(d3.curveCatmullRom.alpha(0.5));
    for (const edge of [xhi, xlo]){
      g.append("path").attr("d", amp.y(c=>py(edge(c)))(ordered)).attr("fill","none")
        .attr("stroke", css("--stable")).attr("stroke-width",1.4).attr("opacity",.45);
    }
    label(g, px(ordered[ordered.length-1].p)-6, py(xhi(ordered[ordered.length-1]))-6,
      "deterministic cycle", css("--stable"), "end")
      .attr("font-size",11).attr("font-family",sans).attr("opacity",.8);
  }
  const showExact = document.getElementById("l-exact").checked;
  if (EXACT_CREST && showExact){
    const exact = d3.line().x(d=>px(d[0]));
    for (const sign of [1, -1]){
      g.append("path").attr("d", exact.y(d=>py(CX + sign*d[1]))(EXACT_CREST))
        .attr("fill","none").attr("stroke",css("--faint")).attr("stroke-width",1.3)
        .attr("stroke-dasharray","5 4");
    }
  }
  if (CREST.length){
    const crest = d3.line().x(d=>px(d[0]));
    for (const sign of [1, -1]){
      g.append("path").attr("d", crest.y(d=>py(CX + sign*d[1]))(CREST))
        .attr("fill","none").attr("stroke",css("--fold")).attr("stroke-width",2.2);
    }
    label(g, px(CREST[CREST.length-1][0])-6, py(CX + CREST[CREST.length-1][1])+14,
      "crest of p", css("--fold"), "end")
      .attr("font-size",11).attr("font-family",sans).attr("font-weight",500);
  }
  // Each threshold is a vertical rule across the whole diagram. The two are
  // at different parameter values, and that gap is what the film is for.
  const rules = THRESHOLDS.slice().sort((a, b) => a.p - b.p);
  rules.forEach((s, k) => {
    const col = ruleColour(s), left = k === 0 && rules.length > 1;
    g.append("line").attr("x1",px(s.p)).attr("x2",px(s.p)).attr("y1",m.t-14)
      .attr("y2",m.t+ih).attr("stroke",col).attr("stroke-width",1.6)
      .attr("stroke-dasharray","6 4").attr("opacity",.9);
    g.append("circle").attr("cx",px(s.p)).attr("cy",py(s.x)).attr("r",5)
      .attr("fill",col).attr("stroke",css("--panel")).attr("stroke-width",1.5);
    label(g, px(s.p)+(left?-8:8), m.t-4, s.label || pretty(s.kind), col,
      left?"end":"start")
      .attr("font-size",12).attr("font-family",sans).attr("font-weight",500);
  });
  for (const s of D.branch.special){
    if (THRESHOLDS.includes(s)) continue;
    const col = css(s.kind==="hopf"?"--hopf":"--fold");
    g.append("circle").attr("cx",px(s.p)).attr("cy",py(s.x)).attr("r",4)
      .attr("fill",css("--panel")).attr("stroke",col).attr("stroke-width",1.5);
    label(g, px(s.p)-9, py(s.x)+18, `deterministic ${pretty(s.kind)}`, col, "end")
      .attr("font-size",11).attr("font-family",sans).attr("opacity",.85);
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

"""

_STOCHASTIC_READOUTS = r"""// ---------- readouts ----------
const fmt = v => `${v >= 0 ? "+" : MINUS}${Math.abs(v).toFixed(3)}`;
function exactCrestAt(p){
  if (!EXACT_CREST || !EXACT_CREST.length || p < EXACT_CREST[0][0]) return null;
  return interpolate(EXACT_CREST, p);
}
function genericRegime(fr, d, v){
  const parts = [v > 0 ? "origin unstable" : "origin stable"];
  if (!d || !d.values.length) parts.push("no density on the plane");
  else parts.push(d.crest == null ? "density peaks at the origin" : "density craters");
  return parts.join(` ${DOT} `);
}
function readouts(){
  const fr = viewAt(t), p = paramAt(t), v = exponentAt(p), d = densityAt(t);
  document.getElementById("ro-p").textContent = p.toFixed(4);
  document.getElementById("ro-regime-sub").textContent =
    `frame ${Math.round(t)+1} / ${F.length}`;
  const ev = document.getElementById("ro-exp");
  const es = document.getElementById("ro-exp-sub");
  const tint = css(v > 0 ? "--unstable" : "--stable");
  ev.innerHTML = `<span class="dot" style="background:${tint}"></span>${fmt(v)}`;
  const exact = exactExponentAt(p);
  es.textContent = exact == null
    ? `estimate ${DOT} ${EXP.length} sampled values`
    : `closed form ${fmt(exact)} ${DOT} gap ${fmt(v - exact)} is the Monte-Carlo `
      + "offset";
  const cv = document.getElementById("ro-crest");
  const cs = document.getElementById("ro-crest-sub");
  const pmax = `p max ${d && d.values.length ? d.peak.toFixed(2) : DASH}`;
  if (!d || !d.values.length){
    cv.textContent = "point mass"; cs.textContent = "all mass at the origin";
  } else if (d.crest == null){
    cv.textContent = "r = 0"; cs.textContent = `peak at the origin ${DOT} ${pmax}`;
  } else {
    const ec = exactCrestAt(p);
    cv.textContent = `r = ${d.crest.toFixed(3)}`;
    cs.textContent = ec == null ? pmax : `closed form ${ec.toFixed(3)} ${DOT} ${pmax}`;
  }
  document.getElementById("ro-regime").textContent =
    fr.label || genericRegime(fr, d, v);
}

"""

_STOCHASTIC_TAIL = r"""// ---------- orchestration ----------
function setT(v){ t = clamp(v, 0, LAST); onFrame(); }
function onFrame(){
  drawSkeleton(); drawCrest(); drawDensity(); drawExponent(); readouts();
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
document.getElementById("paths").addEventListener("input",
  e => setDensity(+e.target.value));
for (const id of ["l-cycle","l-crest"]){
  document.getElementById(id).addEventListener("change",
    () => { drawSkeleton(); drawCrest(); drawDensity(); });
}
document.getElementById("l-maxima").addEventListener("change", drawDensity);
document.getElementById("l-exact").addEventListener("change",
  () => { drawExponent(); drawTimeline(); });
setKicker(document.getElementById("ro-p-kicker"), "", PARAM);
setKicker(document.getElementById("ro-exp-kicker"), "top exponent", LAMBDA);
document.getElementById("tl-hint").textContent =
  `drag the playhead ${DOT} ${D.system.x_label} against ${PARAM} ${DOT} `
  + `${S.convention} reading`;
window.addEventListener("resize", resize);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", resize);
new MutationObserver(resize).observe(document.documentElement,
  {attributes:true, attributeFilter:["data-theme"]});

setDensity(+document.getElementById("paths").value);
resize(); readouts(); onFrame();
if (reduced){ for (let k=0;k<40;k++) stepParticles(); }
requestAnimationFrame(loop);
})();
"""

_SCRIPT = (
    _SCRIPT_HEAD
    + _STOCHASTIC_PARTICLES
    + _SCRIPT_SKELETON
    + _STOCHASTIC_PANELS_SCRIPT
    + _STOCHASTIC_TIMELINE
    + _STOCHASTIC_READOUTS
    + _STOCHASTIC_TAIL
)

_BODY = (
    _BODY_OPEN
    + _STOCHASTIC_MAST
    + _BODY_STAGE
    + _STOCHASTIC_PANELS
    + _STOCHASTIC_READOUTS_BODY
    + _BODY_TIMELINE
    + _STOCHASTIC_CONTROLS
    + _BODY_FOOT
)

_TEMPLATE = (
    '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    "<title>__TITLE__</title>\n"
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    f'<link rel="stylesheet" href="{_FONTS}">\n'
    f"<style>{_STYLE}{_STOCHASTIC_STYLE}</style>\n</head>\n<body>{_BODY}"
    "<script>__D3_SOURCE__</script>\n"
    "<script>window.__STAGE__ = __STAGE_JSON__;</script>\n"
    f"<script>{_SCRIPT}</script>\n</body>\n</html>\n"
)
