"""Render a :class:`~.scene.Scene` to a self-contained HTML plot.

:class:`HtmlRenderer` is the abstraction; :class:`D3Renderer` draws with D3.js.
The D3 library is vendored under ``assets/`` and inlined into every document, so
the output needs no network and renders offline. Curves are drawn as centripetal
Catmull-Rom splines through the continuation points, with luminous gradient strokes
and haloed markers; the drawing script is render-agnostic over the series, so it
handles bifurcation diagrams and parameter-plane curves alike.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from discrecontinual_equations.webplot.latex import latex_to_svg
from discrecontinual_equations.webplot.scene import Scene

_ASSET = Path(__file__).parent / "assets" / "d3.min.js"


class HtmlRenderer(ABC):
    """Turn a scene into a standalone HTML document."""

    @abstractmethod
    def render(self, scene: Scene) -> str:
        """Return a complete HTML document drawing the scene."""
        raise NotImplementedError


class D3Renderer(HtmlRenderer):
    """Draw a scene as self-contained HTML using an inlined D3.js."""

    __slots__ = ["_library"]

    def __init__(self, library: str | None = None) -> None:
        self._library = library if library is not None else _ASSET.read_text()

    def render(self, scene: Scene) -> str:
        payload = json.dumps(_scene_payload(scene))
        equation = latex_to_svg(scene.equation) if scene.equation else ""
        return (
            _TEMPLATE.replace("__TITLE__", _escape(scene.title))
            .replace("__D3_SOURCE__", self._library)
            .replace("__EQUATION__", equation)
            .replace("__SCENE_JSON__", payload)
        )


def _scene_payload(scene: Scene) -> dict:
    return {
        "title": scene.title,
        "subtitle": scene.subtitle,
        "xLabel": scene.x_label,
        "yLabel": scene.y_label,
        "gradient": _gradient_payload(scene.gradient),
        "surface": _surface_payload(scene.surface),
        "tree": _tree_payload(scene.tree),
        "regions": [
            {"label": region.label, "x": float(region.x), "y": float(region.y)}
            for region in scene.regions
        ],
        "series": [
            {
                "role": item.role,
                "kind": item.kind,
                "label": item.label,
                "colour": item.colour,
                "points": [[float(x), float(y)] for x, y in item.points],
            }
            for item in scene.series
        ],
    }


def _gradient_payload(gradient) -> dict | None:
    if gradient is None:
        return None
    return {
        "label": gradient.label,
        "low": float(gradient.low),
        "high": float(gradient.high),
        "lowColour": gradient.low_colour,
        "highColour": gradient.high_colour,
    }


def _tree_payload(tree) -> dict | None:
    if tree is None:
        return None
    return {
        "nodes": [
            {
                "identifier": node.identifier,
                "level": node.level,
                "order": float(node.order),
                "label": node.label,
                "detail": node.detail,
                "kind": node.kind,
            }
            for node in tree.nodes
        ],
        "edges": [[int(a), int(b)] for a, b in tree.edges],
    }


def _surface_payload(surface) -> dict | None:
    if surface is None:
        return None
    return {
        "grid": [
            [[float(x), float(y), float(z)] for x, y, z in row] for row in surface.grid
        ],
        "loci": [
            {
                "kind": locus.kind,
                "label": locus.label,
                "points": [[float(x), float(y), float(z)] for x, y, z in locus.points],
            }
            for locus in surface.loci
        ],
        "markers": [
            {
                "kind": marker.kind,
                "label": marker.label,
                "points": [[float(x), float(y), float(z)] for x, y, z in marker.points],
            }
            for marker in surface.markers
        ],
        "x_label": surface.x_label,
        "y_label": surface.y_label,
        "z_label": surface.z_label,
    }


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_DRAW = """
const scene = window.__SCENE__;
const PALETTE = {
  stable:"#34d399", unstable:"#fb7185", saddle:"#c4b5fd", curve:"#38bdf8",
  fold:"#e5478f", branch_point:"#5c9e00", transcritical:"#5c9e00",
  pitchfork:"#0d9488", hopf:"#d98a00",
  bogdanov_takens:"#c084fc", zero_hopf:"#0891b2", hopf_hopf:"#a855f7",
  cusp:"#e5478f", generalized_hopf:"#d98a00",
  swallowtail:"#059669", degenerate_bautin:"#d946ef",
  degenerate_bogdanov_takens:"#c084fc", degenerate_bogdanov_takens_b:"#7c3aed",
  equilibrium:"#334155", orbit:"#e5478f", zero_crossing:"#d98a00",
  manifold_unstable:"#fb7185", manifold_stable:"#059669", root:"#2563eb"
};
const PALETTE_DARK = {
  stable:"#34d399", unstable:"#fb7185", saddle:"#c4b5fd", curve:"#38bdf8",
  fold:"#f472b6", branch_point:"#a3e635", transcritical:"#a3e635",
  pitchfork:"#2dd4bf", hopf:"#fbbf24",
  bogdanov_takens:"#fb7185", zero_hopf:"#22d3ee", hopf_hopf:"#c084fc",
  cusp:"#fb7185", generalized_hopf:"#fbbf24",
  swallowtail:"#34d399", degenerate_bautin:"#e879f9",
  degenerate_bogdanov_takens:"#c084fc", degenerate_bogdanov_takens_b:"#d8b4fe",
  equilibrium:"#e5e7eb", orbit:"#f472b6", zero_crossing:"#fbbf24",
  manifold_unstable:"#fb7185", manifold_stable:"#34d399", root:"#93c5fd"
};
const DASH_CYCLE = ["none","7 5","2 5","10 5 2 5","6 4 1 4","1 6"];
const THEMES = {
  dark:  {ink:"#eef1f7", muted:"#9aa3b8", axis:"#aeb6c8", grid:"#232a3c",
          bg1:"#151a28", bg2:"#0a0c12", panel:"#10131d", stroke:"#0c0e15",
          palette:PALETTE_DARK, mono:false},
  light: {ink:"#0f1626", muted:"#5b6579", axis:"#3a4252", grid:"#d7dde8",
          bg1:"#ffffff", bg2:"#eef1f6", panel:"#ffffff", stroke:"#ffffff",
          palette:PALETTE, mono:false},
  print: {ink:"#0a0a0a", muted:"#333333", axis:"#0a0a0a", grid:"#c8c8c8",
          bg1:"#ffffff", bg2:"#ffffff", panel:"#ffffff", stroke:"#ffffff",
          palette:PALETTE, mono:true}
};
let THEME_NAME = "dark";
try {
  if (window.localStorage) {
    THEME_NAME = localStorage.getItem("dce-theme") || "dark";
  }
} catch (e) { THEME_NAME = "dark"; }
if (!THEMES[THEME_NAME]) THEME_NAME = "dark";
let T = THEMES[THEME_NAME];
const colour = k => T.mono ? T.ink : (T.palette[k] || (T.mono?T.ink:"#94a3b8"));
const shade = (hex,f) => {
  if (T.mono) return T.ink;
  const c = hex.charAt(0)==="#" ? hex.slice(1) : "808080";
  const n = parseInt(c,16);
  const r = Math.min(255,Math.round(((n>>16)&255)*(1+f)));
  const g = Math.min(255,Math.round(((n>>8)&255)*(1+f)));
  const b = Math.min(255,Math.round((n&255)*(1+f)));
  return `rgb(${r},${g},${b})`;
};

const width = 1000, height = 600;
let sliceMode = false, sliceRow = 0;
function render(){
  T = THEMES[THEME_NAME];
  d3.select("#plot").selectAll("*").remove();
  if (scene.tree) { drawTree(scene.tree); }
  else if (scene.surface) {
    const g = scene.surface.grid;
    if (sliceMode && g && g.length) { drawSurfaceSlice(scene.surface, sliceRow); }
    else { drawSurface(scene.surface); }
    mountSurfaceTools(scene.surface);
  }
  else { draw2D(); }
}
function setTheme(name){
  THEME_NAME = name;
  try { if (window.localStorage) localStorage.setItem("dce-theme", name); } catch(e){}
  document.documentElement.setAttribute("data-theme", name);
  document.querySelectorAll(".theme-btn").forEach(b=>
    b.setAttribute("aria-pressed", b.dataset.theme===name ? "true":"false"));
  render();
}

function draw2D(){
const _kindDash = {}; let _dashN = 0;
const dashFor = kind => {
  if (!T.mono) {
    const base = {stable:"none", unstable:"9 7", saddle:"1.5 6", curve:"none"};
    return base[kind] || "none";
  }
  if (!(kind in _kindDash)) {
    _kindDash[kind] = DASH_CYCLE[_dashN % DASH_CYCLE.length]; _dashN++;
  }
  return _kindDash[kind];
};
const margin = { top:28, right:196, bottom:62, left:82 };
const iw = width - margin.left - margin.right;
const ih = height - margin.top - margin.bottom;

const svg = d3.select("#plot").append("svg")
  .attr("viewBox", `0 0 ${width} ${height}`).attr("width","100%")
  .attr("font-family","inherit");
const defs = svg.append("defs");

// soft glow for luminous strokes and markers
const glow = defs.append("filter").attr("id","glow")
  .attr("x","-60%").attr("y","-60%").attr("width","220%").attr("height","220%");
glow.append("feGaussianBlur").attr("stdDeviation","4").attr("result","b");
const gm = glow.append("feMerge");
gm.append("feMergeNode").attr("in","b");
gm.append("feMergeNode").attr("in","SourceGraphic");

// panel backdrop with a faint vignette
const bg = defs.append("radialGradient").attr("id","bg")
  .attr("cx","50%").attr("cy","36%").attr("r","80%");
bg.append("stop").attr("offset","0%").attr("stop-color",T.bg1);
bg.append("stop").attr("offset","100%").attr("stop-color",T.bg2);
svg.append("rect").attr("width",width).attr("height",height)
  .attr("rx",16).attr("fill","url(#bg)");

const g = svg.append("g").attr("transform",`translate(${margin.left},${margin.top})`);

let xs = [], ys = [];
scene.series.forEach(s => s.points.forEach(p => { xs.push(p[0]); ys.push(p[1]); }));
(scene.regions || []).forEach(r => { xs.push(r.x); ys.push(r.y); });
const pad = (lo,hi) => { const d=(hi-lo)||1; return [lo-0.08*d, hi+0.08*d]; };
const xd = pad(Math.min(...xs), Math.max(...xs));
const yd = pad(Math.min(...ys), Math.max(...ys));
const x = d3.scaleLinear().domain(xd).nice().range([0, iw]);
const y = d3.scaleLinear().domain(yd).nice().range([ih, 0]);

// subtle dotted grid
const grid = g.append("g").attr("stroke",T.grid).attr("stroke-dasharray","1 5")
  .attr("stroke-width",1);
grid.append("g").attr("transform",`translate(0,${ih})`)
  .call(d3.axisBottom(x).ticks(9).tickSize(-ih).tickFormat(""))
  .call(a=>a.select(".domain").remove());
grid.append("g")
  .call(d3.axisLeft(y).ticks(7).tickSize(-iw).tickFormat(""))
  .call(a=>a.select(".domain").remove());

// zero reference lines when in range
const ref = g.append("g").attr("stroke","#38425c").attr("stroke-width",1);
if (xd[0] < 0 && xd[1] > 0)
  ref.append("line").attr("x1",x(0)).attr("x2",x(0)).attr("y1",0).attr("y2",ih);
if (yd[0] < 0 && yd[1] > 0)
  ref.append("line").attr("x1",0).attr("x2",iw).attr("y1",y(0)).attr("y2",y(0));

const axisColour = a => {
  a.selectAll("text").attr("fill",T.muted).attr("font-size",17);
  a.selectAll("line").attr("stroke","#3a4152");
  a.selectAll(".domain").attr("stroke","#3a4152"); };
g.append("g").attr("transform",`translate(0,${ih})`)
  .call(d3.axisBottom(x).ticks(9).tickPadding(8)).call(axisColour);
g.append("g").call(d3.axisLeft(y).ticks(7).tickPadding(8)).call(axisColour);

g.append("text").attr("x",iw/2).attr("y",ih+46).attr("text-anchor","middle")
  .attr("fill",T.axis).attr("font-size",19).attr("letter-spacing",".02em")
  .text(scene.xLabel);
g.append("text").attr("transform","rotate(-90)").attr("x",-ih/2).attr("y",-58)
  .attr("text-anchor","middle").attr("fill",T.axis).attr("font-size",19)
  .attr("letter-spacing",".02em").text(scene.yLabel);

const spline = d3.line().curve(d3.curveCatmullRom.alpha(0.5))
  .x(d=>x(d[0])).y(d=>y(d[1]));

// region annotations: what dynamics live in each part of the plane
(scene.regions || []).forEach(r => {
  const gx = x(r.x), gy = y(r.y);
  const words = String(r.label).split(" + ");
  const lineH = 19, boxW = Math.max(...words.map(w=>w.length))*8.4 + 18;
  const boxH = words.length*lineH + 12;
  const rg = g.append("g").attr("transform",`translate(${gx},${gy})`);
  rg.append("rect").attr("x",-boxW/2).attr("y",-boxH/2).attr("width",boxW)
    .attr("height",boxH).attr("rx",9).attr("fill",T.panel).attr("opacity",0.82)
    .attr("stroke",T.grid).attr("stroke-width",1);
  words.forEach((w,i)=>{
    rg.append("text").attr("x",0).attr("y",-boxH/2 + 12 + i*lineH + 6)
      .attr("text-anchor","middle").attr("fill",T.ink).attr("font-size",15)
      .attr("font-weight", i===0?600:400).text(w);
  });
});

const family = !!scene.gradient;

scene.series.filter(s=>s.role==="line").forEach((s,i)=>{
  const c = s.colour || colour(s.kind);
  const d = spline(s.points);
  let stroke = c;
  if (!family && s.kind==="curve") {
    const gradId = `grad${i}`;
    const lg = defs.append("linearGradient").attr("id",gradId)
      .attr("x1","0%").attr("x2","100%");
    lg.append("stop").attr("offset","0%").attr("stop-color",shade(c,-0.12));
    lg.append("stop").attr("offset","100%").attr("stop-color",shade(c,0.25));
    stroke = `url(#${gradId})`;
  }
  g.append("path").attr("fill","none").attr("stroke",c)
    .attr("stroke-width",family?4:7).attr("stroke-linecap","round")
    .attr("opacity",family?0.08:0.16).attr("filter","url(#glow)").attr("d",d);
  const path = g.append("path").attr("fill","none").attr("stroke",stroke)
    .attr("stroke-width",family?1.7:2.8).attr("stroke-linecap","round")
    .attr("stroke-linejoin","round").attr("opacity",family?0.92:1)
    .attr("stroke-dasharray",dashFor(s.kind)).attr("d",d);
  if (!family && dashFor(s.kind)==="none") {
    const len = path.node().getTotalLength();
    path.attr("stroke-dasharray",len).attr("stroke-dashoffset",len)
      .transition().duration(900).ease(d3.easeCubicInOut)
      .attr("stroke-dashoffset",0);
  }
});

const tip = d3.select("body").append("div").attr("class","tip");
const markers = scene.series.filter(s=>s.role==="marker");
markers.forEach(s=>{
  const c = s.colour || colour(s.kind);
  s.points.forEach(p=>{
    const cx=x(p[0]), cy=y(p[1]);
    g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",family?9:13)
      .attr("fill",c).attr("opacity",0.22).attr("filter","url(#glow)");
    const dot = g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",family?4.5:0)
      .attr("fill",c).attr("stroke",T.bg2).attr("stroke-width",2.5)
      .style("cursor","pointer");
    if (!family)
      dot.transition().delay(650).duration(360).ease(d3.easeBackOut).attr("r",6.5);
    dot.on("mousemove",(ev)=>tip.style("opacity",1)
        .style("left",(ev.pageX+12)+"px").style("top",(ev.pageY-10)+"px")
        .html(`<b>${s.label}</b><br>${scene.xLabel} ${p[0].toFixed(4)}<br>`+
              `${scene.yLabel} ${p[1].toFixed(4)}`))
      .on("mouseleave",()=>tip.style("opacity",0));
  });
  const f = s.points[0];
  if (f && !family) {
    const tx=x(f[0])+12, ty=y(f[1])-12;
    const w=s.label.length*6.6+16;
    const pill=g.append("g").attr("opacity",0);
    pill.append("rect").attr("x",tx-6).attr("y",ty-13).attr("rx",7)
      .attr("width",w).attr("height",19).attr("fill","#171b28")
      .attr("stroke",c).attr("stroke-opacity",0.5);
    pill.append("text").attr("x",tx+2).attr("y",ty+1).attr("fill","#e8eaf0")
      .attr("font-size",17).text(s.label);
    pill.transition().delay(760).duration(300).attr("opacity",1);
  }
});

const legend = g.append("g").attr("transform",`translate(${iw+26},2)`);
let rowStart = 0;
if (family) {
  const barId = "fambar", barH = 120, barW = 12;
  const lg = defs.append("linearGradient").attr("id",barId)
    .attr("x1","0%").attr("y1","100%").attr("x2","0%").attr("y2","0%");
  lg.append("stop").attr("offset","0%").attr("stop-color",scene.gradient.lowColour);
  lg.append("stop").attr("offset","100%").attr("stop-color",scene.gradient.highColour);
  legend.append("text").attr("x",0).attr("y",-2).attr("fill",T.ink)
    .attr("font-size",17.5).text(scene.gradient.label);
  legend.append("rect").attr("x",0).attr("y",8).attr("width",barW).attr("height",barH)
    .attr("rx",3).attr("fill",`url(#${barId})`).attr("stroke",T.grid);
  legend.append("text").attr("x",barW+8).attr("y",16).attr("fill",T.muted)
    .attr("font-size",15.5).text(scene.gradient.high.toFixed(2));
  legend.append("text").attr("x",barW+8).attr("y",8+barH).attr("fill",T.muted)
    .attr("font-size",15.5).text(scene.gradient.low.toFixed(2));
  rowStart = 8 + barH + 24;
}
const lineEntries = [];
if (!family) {
  scene.series.filter(s=>s.role==="line").forEach(s=>{
    if (!lineEntries.some(e=>e.kind===s.kind)) lineEntries.push({kind:s.kind});
  });
}
let row = 0;
lineEntries.forEach(e=>{
  const rr = legend.append("g").attr("transform",`translate(0,${rowStart+row*24})`);
  rr.append("line").attr("x1",0).attr("x2",24).attr("y1",6).attr("y2",6)
    .attr("stroke",colour(e.kind)).attr("stroke-width",2.8)
    .attr("stroke-linecap","round").attr("stroke-dasharray",dashFor(e.kind));
  rr.append("text").attr("x",34).attr("y",10).attr("fill",T.ink)
    .attr("font-size",17.5).text(e.kind.replace(/_/g," "));
  row++;
});
markers.forEach(s=>{
  const rr = legend.append("g").attr("transform",`translate(0,${rowStart+row*24})`);
  rr.append("circle").attr("cx",9).attr("cy",6).attr("r",6.5)
    .attr("fill",s.colour || colour(s.kind))
    .attr("stroke",T.bg2).attr("stroke-width",2);
  rr.append("text").attr("x",34).attr("y",10).attr("fill",T.ink)
    .attr("font-size",17.5).text(s.label);
  row++;
});
}

function drawTree(Tr){
  const svg = d3.select("#plot").append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`).attr("width","100%")
    .attr("font-family","inherit");
  const defs = svg.append("defs");
  const bg = defs.append("radialGradient").attr("id","tbg")
    .attr("cx","50%").attr("cy","30%").attr("r","85%");
  bg.append("stop").attr("offset","0%").attr("stop-color",T.bg1);
  bg.append("stop").attr("offset","100%").attr("stop-color",T.bg2);
  svg.append("rect").attr("width",width).attr("height",height).attr("rx",16)
    .attr("fill","url(#tbg)");
  const kindColour={
    root:"#93c5fd", fold:"#f472b6", hopf:"#fbbf24",
    cusp:"#f9a8d4", generalized_hopf:"#fde68a"
  };
  const nodes=Tr.nodes, edges=Tr.edges;
  const levels=Math.max(...nodes.map(n=>n.level))+1;
  const orders=Math.max(...nodes.map(n=>n.order))+1;
  const marginX=150, marginY=70;
  const spanX=(width-2*marginX)/Math.max(1,levels-1);
  const spanY=(height-2*marginY)/Math.max(1,orders);
  const px=n=>marginX + n.level*spanX;
  const py=n=>marginY + (n.order+0.5)*spanY;
  const byId={}; nodes.forEach(n=>byId[n.identifier]=n);
  // edges first
  edges.forEach(([a,b])=>{
    const na=byId[a], nb=byId[b];
    const x1=px(na)+70, y1=py(na), x2=px(nb)-70, y2=py(nb);
    const mx=(x1+x2)/2;
    svg.append("path").attr("d",`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`)
      .attr("fill","none").attr("stroke",T.grid).attr("stroke-width",1.6);
  });
  // nodes
  nodes.forEach(n=>{
    const x=px(n), y=py(n), c=kindColour[n.kind]||"#cbd5e1";
    const g=svg.append("g");
    g.append("rect").attr("x",x-70).attr("y",y-26).attr("width",140).attr("height",52)
      .attr("rx",10).attr("fill",T.panel).attr("stroke",c).attr("stroke-width",1.6);
    g.append("circle").attr("cx",x-54).attr("cy",y-10).attr("r",4).attr("fill",c);
    g.append("text").attr("x",x-44).attr("y",y-6).attr("fill",T.ink)
      .attr("font-size",17.5).attr("font-weight",600).text(n.label);
    g.append("text").attr("x",x-58).attr("y",y+14).attr("fill",T.muted)
      .attr("font-size",14.5).text(n.detail.length>26?n.detail.slice(0,25)+"\u2026":n.detail);
  });
  // codimension column headers
  for(let l=0;l<levels;l++){
    svg.append("text").attr("x",marginX+l*spanX).attr("y",34)
      .attr("fill",T.muted).attr("font-size",17).attr("text-anchor","middle")
      .text("codim "+(l+1));
  }
}

function drawSurface(S){
  const svg = d3.select("#plot").append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`).attr("width","100%")
    .attr("font-family","inherit").style("cursor","grab").style("touch-action","none");
  const defs = svg.append("defs");
  const bg = defs.append("radialGradient").attr("id","bg")
    .attr("cx","50%").attr("cy","36%").attr("r","80%");
  bg.append("stop").attr("offset","0%").attr("stop-color",T.bg1);
  bg.append("stop").attr("offset","100%").attr("stop-color",T.bg2);
  svg.append("rect").attr("width",width).attr("height",height).attr("rx",16)
    .attr("fill","url(#bg)");
  const glow = defs.append("filter").attr("id","sglow")
    .attr("x","-60%").attr("y","-60%").attr("width","220%").attr("height","220%");
  glow.append("feGaussianBlur").attr("stdDeviation","3.2").attr("result","b");
  const gm = glow.append("feMerge");
  gm.append("feMergeNode").attr("in","b");
  gm.append("feMergeNode").attr("in","SourceGraphic");

  const rows = S.grid.length, cols = rows ? S.grid[0].length : 0;
  let xs=[], ys=[], zs=[];
  S.grid.forEach(r=>r.forEach(p=>{ xs.push(p[0]); ys.push(p[1]); zs.push(p[2]); }));
  (S.loci||[]).forEach(L=>L.points.forEach(p=>{
    xs.push(p[0]); ys.push(p[1]); zs.push(p[2]);
  }));
  const rng = a => [Math.min(...a), Math.max(...a)];
  const [x0,x1]=rng(xs), [y0,y1]=rng(ys), [z0,z1]=rng(zs);
  const norm=(v,lo,hi)=>(hi>lo?(v-lo)/(hi-lo):0.5);
  const vex=0.85;
  const lowC=[56,189,248], highC=[251,191,36];
  const heightColour = t => {
    const mix = lowC.map((c,i)=>Math.round(c+(highC[i]-c)*t));
    return `rgb(${mix.join(",")})`;
  };
  const lociColour={
    fold:"#f472b6", branch_point:"#a3e635", transcritical:"#a3e635",
  pitchfork:"#2dd4bf", hopf:"#fbbf24",
    cusp:"#f9a8d4", bogdanov_takens:"#c084fc", generalized_hopf:"#fde68a",
    zero_hopf:"#67e8f9", swallowtail:"#f472b6", degenerate_bautin:"#fbbf24"
  };

  // static overlays (do not rotate): colour bar, legend, hint
  const barId="hbar", barH=140, barW=12, bx=width-40, by=40;
  const lg=defs.append("linearGradient").attr("id",barId)
    .attr("x1","0%").attr("y1","100%").attr("x2","0%").attr("y2","0%");
  lg.append("stop").attr("offset","0%").attr("stop-color",heightColour(0));
  lg.append("stop").attr("offset","100%").attr("stop-color",heightColour(1));
  svg.append("text").attr("x",bx).attr("y",by-8).attr("fill",T.ink)
    .attr("font-size",17).attr("text-anchor","end").text(S.z_label);
  svg.append("rect").attr("x",bx).attr("y",by).attr("width",barW).attr("height",barH)
    .attr("rx",3).attr("fill",`url(#${barId})`).attr("stroke",T.grid);
  svg.append("text").attr("x",bx-6).attr("y",by+10).attr("fill",T.muted)
    .attr("font-size",15.5).attr("text-anchor","end").text(z1.toFixed(2));
  svg.append("text").attr("x",bx-6).attr("y",by+barH).attr("fill",T.muted)
    .attr("font-size",15.5).attr("text-anchor","end").text(z0.toFixed(2));
  const legend=svg.append("g").attr("transform",`translate(40,${height-30})`);
  let li=0;
  (S.loci||[]).forEach(L=>{
    if(L.points.length<2) return;
    const key=L.kind+":"+L.label;
    if(legend.selectAll("g").nodes().some(n=>n.__k===key)) return;
    const c=lociColour[L.kind]||"#e5e7eb";
    const row=legend.append("g").attr("transform",`translate(${li*180},0)`);
    row.node().__k=key;
    row.append("line").attr("x1",0).attr("x2",22).attr("y1",0).attr("y2",0)
      .attr("stroke",c).attr("stroke-width",3).attr("stroke-linecap","round");
    row.append("text").attr("x",30).attr("y",4).attr("fill",T.ink)
      .attr("font-size",17.5).text(L.label+" curve");
    li++;
  });
  svg.append("text").attr("x",width-40).attr("y",height-20).attr("fill",T.muted)
    .attr("font-size",16).attr("text-anchor","end").text("drag to rotate");

  const content = svg.append("g");
  let az=Math.PI*0.32, el=Math.PI*0.26;

  function draw(){
    content.selectAll("*").remove();
    const ca=Math.cos(az), sa=Math.sin(az), ce=Math.cos(el), se=Math.sin(el);
    const rotate=(x,y,z)=>{
      const u=norm(x,x0,x1)-0.5, v=norm(y,y0,y1)-0.5, w=(norm(z,z0,z1)-0.5)*vex;
      const rx=u*ca - v*sa, ry=u*sa + v*ca;
      return { rx, up: ry*se + w*ce, depth: ry*ce - w*se };
    };
    let rxs=[], ups=[];
    S.grid.forEach(r=>r.forEach(p=>{
      const q=rotate(p[0],p[1],p[2]); rxs.push(q.rx); ups.push(q.up);
    }));
    (S.loci||[]).forEach(L=>L.points.forEach(p=>{
      const q=rotate(p[0],p[1],p[2]); rxs.push(q.rx); ups.push(q.up);
    }));
    const [rxlo,rxhi]=rng(rxs), [uplo,uphi]=rng(ups);
    const plotL=70, plotT=54, plotW=width-240, plotH=height-150;
    const fit=Math.min(plotW/((rxhi-rxlo)||1), plotH/((uphi-uplo)||1));
    const offX=plotL + (plotW-(rxhi-rxlo)*fit)/2 - rxlo*fit;
    const offY=plotT + (plotH-(uphi-uplo)*fit)/2;
    const project=(x,y,z)=>{
      const q=rotate(x,y,z);
      return { sx: offX + q.rx*fit, sy: offY + (uphi-q.up)*fit, d: q.depth };
    };

    // base-plane grid (faint)
    for(let i=0;i<rows;i++) for(let j=0;j<cols-1;j++){
      const a=project(S.grid[i][j][0],S.grid[i][j][1],z0);
      const b=project(S.grid[i][j+1][0],S.grid[i][j+1][1],z0);
      content.append("line").attr("x1",a.sx).attr("y1",a.sy)
        .attr("x2",b.sx).attr("y2",b.sy).attr("stroke",T.grid).attr("stroke-width",0.6);
    }
    // quads, depth-sorted far-first
    const quads=[];
    for(let i=0;i<rows-1;i++) for(let j=0;j<cols-1;j++){
      const a=S.grid[i][j], b=S.grid[i][j+1], c=S.grid[i+1][j+1], e=S.grid[i+1][j];
      const P=[project(a[0],a[1],a[2]),project(b[0],b[1],b[2]),
               project(c[0],c[1],c[2]),project(e[0],e[1],e[2])];
      const meanZ=(a[2]+b[2]+c[2]+e[2])/4;
      const depth=(P[0].d+P[1].d+P[2].d+P[3].d)/4;
      quads.push({P, t:norm(meanZ,z0,z1), depth});
    }
    quads.sort((p,q)=>q.depth-p.depth);
    quads.forEach(q=>{
      content.append("polygon").attr("points",q.P.map(p=>`${p.sx},${p.sy}`).join(" "))
        .attr("fill",heightColour(q.t)).attr("fill-opacity",0.82)
        .attr("stroke",T.bg2).attr("stroke-width",0.4).attr("stroke-opacity",0.5);
    });
    // loci over the surface + base-plane projection
    (S.loci||[]).forEach(L=>{
      if(L.points.length<2) return;
      const c=lociColour[L.kind]||"#e5e7eb";
      const line=L.points.map(p=>project(p[0],p[1],p[2]));
      content.append("path").attr("d","M"+line.map(p=>`${p.sx},${p.sy}`).join("L"))
        .attr("fill","none").attr("stroke",c).attr("stroke-width",3)
        .attr("stroke-linecap","round").attr("filter","url(#sglow)");
      const proj=L.points.map(p=>project(p[0],p[1],z0));
      content.append("path").attr("d","M"+proj.map(p=>`${p.sx},${p.sy}`).join("L"))
        .attr("fill","none").attr("stroke",c).attr("stroke-width",1.2)
        .attr("stroke-dasharray","3 4").attr("opacity",0.55);
    });
    // organizing-centre markers (codim-3 points)
    (S.markers||[]).forEach(M=>{
      const c=lociColour[M.kind]||"#f8fafc";
      M.points.forEach(pt=>{
        const p=project(pt[0],pt[1],pt[2]);
        content.append("circle").attr("cx",p.sx).attr("cy",p.sy).attr("r",6.5)
          .attr("fill",c).attr("stroke",T.bg2).attr("stroke-width",1.5)
          .attr("filter","url(#sglow)");
        content.append("text").attr("x",p.sx+10).attr("y",p.sy+4)
          .attr("fill",T.ink).attr("font-size",17).text(M.label);
      });
    });
    // rotating axis labels
    const lab=(x,y,z,text,dy)=>{ const p=project(x,y,z);
      content.append("text").attr("x",p.sx).attr("y",p.sy+(dy||0)).attr("fill",T.axis)
        .attr("font-size",18).attr("text-anchor","middle").text(text); };
    lab((x0+x1)/2,y0,z0,S.x_label,24);
    lab(x1,(y0+y1)/2,z0,S.y_label,18);
    lab(x0,y0,z1,S.z_label,-10);
  }
  draw();

  let dragging=false, lastX=0, lastY=0;
  svg.on("pointerdown",ev=>{ dragging=true; lastX=ev.clientX; lastY=ev.clientY;
    svg.style("cursor","grabbing"); });
  svg.on("pointermove",ev=>{ if(!dragging) return;
    az += (ev.clientX-lastX)*0.008;
    el = Math.max(0.06, Math.min(1.45, el + (ev.clientY-lastY)*0.006));
    lastX=ev.clientX; lastY=ev.clientY; draw(); });
  svg.on("pointerup",()=>{ dragging=false; svg.style("cursor","grab"); });
  svg.on("pointerleave",()=>{ dragging=false; svg.style("cursor","grab"); });
}

function mountSurfaceTools(S){
  const tools = document.getElementById("tools");
  if (!tools) return;
  tools.innerHTML = "";
  const rows = S.grid ? S.grid.length : 0;
  if (!rows) { tools.style.display = "none"; return; }
  tools.style.display = "flex";
  const toggle = document.createElement("button");
  toggle.className = "theme-btn";
  toggle.style.background = "var(--panel2)";
  toggle.style.border = "1px solid var(--border)";
  toggle.textContent = sliceMode ? "Rotate 3D" : "Slice 2D";
  toggle.addEventListener("click", () => { sliceMode = !sliceMode; render(); });
  tools.appendChild(toggle);
  const box = document.createElement("div");
  box.className = "slider";
  box.style.display = sliceMode ? "inline-flex" : "none";
  const fixed = S.grid[sliceRow] && S.grid[sliceRow][0]
    ? S.grid[sliceRow][0][1] : 0;
  const label = document.createElement("span");
  label.textContent = (S.y_label || "param") + " = " + Number(fixed).toFixed(3);
  const input = document.createElement("input");
  input.type = "range"; input.min = "0"; input.max = String(rows - 1);
  input.step = "1"; input.value = String(sliceRow);
  input.addEventListener("input", () => {
    sliceRow = +input.value;
    const f = S.grid[sliceRow] && S.grid[sliceRow][0] ? S.grid[sliceRow][0][1] : 0;
    label.textContent = (S.y_label || "param") + " = " + Number(f).toFixed(3);
    if (sliceMode) drawSurfaceSlice(S, sliceRow);
  });
  box.appendChild(input); box.appendChild(label);
  tools.appendChild(box);
}

function drawSurfaceSlice(S, row){
  d3.select("#plot").selectAll("*").remove();
  const r = S.grid[row] || [];
  const pts = r.map(p => [p[0], p[2]]);
  const fixed = r[0] ? r[0][1] : 0;
  const svg = d3.select("#plot").append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`).attr("width","100%")
    .attr("font-family","inherit");
  svg.append("rect").attr("width",width).attr("height",height).attr("rx",16)
    .attr("fill",T.bg2);
  const m = {top:34, right:40, bottom:66, left:92};
  const iw = width - m.left - m.right, ih = height - m.top - m.bottom;
  const g = svg.append("g").attr("transform",`translate(${m.left},${m.top})`);
  const xe = d3.extent(pts, p=>p[0]);
  let ye = d3.extent(pts, p=>p[1]);
  if (ye[0]===ye[1]) { ye = [ye[0]-1, ye[1]+1]; }
  const padY = (ye[1]-ye[0])*0.08;
  const sx = d3.scaleLinear().domain(xe).range([0,iw]);
  const sy = d3.scaleLinear().domain([ye[0]-padY, ye[1]+padY]).range([ih,0]);
  const xa = d3.axisBottom(sx).ticks(8), ya = d3.axisLeft(sy).ticks(7);
  const gx = g.append("g").attr("transform",`translate(0,${ih})`).call(xa);
  const gy = g.append("g").call(ya);
  [gx,gy].forEach(ax=>{
    ax.selectAll("path,line").attr("stroke",T.axis);
    ax.selectAll("text").attr("fill",T.muted).attr("font-size",19.5);
  });
  g.append("g").attr("stroke",T.grid).attr("stroke-dasharray","1 5")
    .selectAll("line").data(sy.ticks(7)).join("line")
    .attr("x1",0).attr("x2",iw).attr("y1",d=>sy(d)).attr("y2",d=>sy(d));
  const line = d3.line().x(p=>sx(p[0])).y(p=>sy(p[1])).curve(d3.curveCatmullRom);
  g.append("path").datum(pts).attr("fill","none")
    .attr("stroke", T.mono ? T.ink : colour("curve"))
    .attr("stroke-width", T.mono ? 2.4 : 3).attr("stroke-linecap","round")
    .attr("stroke-linejoin","round").attr("d",line);
  g.append("text").attr("x",iw/2).attr("y",ih+52).attr("text-anchor","middle")
    .attr("fill",T.muted).attr("font-size",22.5).text(S.x_label || "");
  g.append("text").attr("transform","rotate(-90)").attr("x",-ih/2).attr("y",-64)
    .attr("text-anchor","middle").attr("fill",T.muted).attr("font-size",22.5)
    .text(S.z_label || "");
  g.append("text").attr("x",0).attr("y",-12).attr("fill",T.ink)
    .attr("font-size",21).attr("font-weight",600)
    .text("slice at " + (S.y_label || "param") + " = " + Number(fixed).toFixed(3));
}

document.documentElement.setAttribute("data-theme", THEME_NAME);
document.querySelectorAll(".theme-btn").forEach(b=>{
  b.addEventListener("click", ()=>setTheme(b.dataset.theme));
  b.setAttribute("aria-pressed", b.dataset.theme===THEME_NAME ? "true":"false");
});
render();
"""

_TEMPLATE = (
    """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --ink:#eef1f7; --muted:#9aa3b8; --bg1:#151a28; --bg2:#0a0c12;
    --panel1:#141824; --panel2:#10131d; --border:#222838; --accent:#8ab4ff;
  }
  html[data-theme="light"]{
    --ink:#0f1626; --muted:#5b6579; --bg1:#ffffff; --bg2:#eef1f6;
    --panel1:#ffffff; --panel2:#f7f9fc; --border:#dde3ee; --accent:#2563eb;
  }
  html[data-theme="print"]{
    --ink:#0a0a0a; --muted:#333333; --bg1:#ffffff; --bg2:#ffffff;
    --panel1:#ffffff; --panel2:#ffffff; --border:#111111; --accent:#000000;
  }
  *{box-sizing:border-box}
  body{margin:0;color:var(--ink);
    background:radial-gradient(1200px 700px at 50% -8%, var(--bg1) 0%, var(--bg2) 60%),
      var(--bg2);
    font:17px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,
      Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1100px;margin:0 auto;padding:30px 22px 70px}
  .topbar{display:flex;justify-content:space-between;align-items:center;gap:14px;
    flex-wrap:wrap}
  a.back{color:var(--muted);text-decoration:none;font-size:14px;
    display:inline-flex;gap:6px;align-items:center;transition:color .15s}
  a.back:hover{color:var(--ink)}
  .themes{display:inline-flex;gap:4px;background:var(--panel2);
    border:1px solid var(--border);border-radius:11px;padding:4px}
  .theme-btn{appearance:none;border:0;background:transparent;color:var(--muted);
    font:600 13px/1 inherit;padding:7px 13px;border-radius:8px;cursor:pointer;
    transition:background .15s,color .15s}
  .theme-btn:hover{color:var(--ink)}
  .theme-btn[aria-pressed="true"]{background:var(--accent);
    color:#fff;box-shadow:0 2px 8px -3px rgba(0,0,0,.5)}
  html[data-theme="light"] .theme-btn[aria-pressed="true"],
  html[data-theme="print"] .theme-btn[aria-pressed="true"]{color:#fff}
  h1{font-size:32px;margin:20px 0 6px;letter-spacing:-.02em;font-weight:700}
  .sub{color:var(--muted);margin:0 0 14px;max-width:76ch;font-size:16px}
  .eqn{margin:2px 0 22px;overflow-x:auto;color:var(--ink)}
  .eqn svg{height:1.35em;width:auto;max-width:100%;vertical-align:middle}
  .card{background:linear-gradient(180deg,var(--panel1),var(--panel2));
    border:1px solid var(--border);border-radius:18px;padding:12px;
    box-shadow:0 24px 60px -30px rgba(0,0,0,.55),
      inset 0 1px 0 rgba(255,255,255,.03)}
  html[data-theme="print"] .card{box-shadow:none}
  .tools{display:flex;gap:8px;align-items:center;margin:0 0 10px;
    color:var(--muted);font-size:13px;flex-wrap:wrap}
  .slider{display:none;align-items:center;gap:10px}
  .slider input{width:220px}
  .tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .12s;
    background:var(--panel2);border:1px solid var(--border);border-radius:9px;
    padding:7px 10px;font-size:13px;color:var(--ink);
    box-shadow:0 10px 30px -12px rgba(0,0,0,.6);z-index:9}
  .tip b{color:var(--ink)}
</style>
</head>
<body>
<div class="wrap">
<div class="topbar">
<a class="back" href="index.html">&larr; atlas</a>
<div class="themes" role="group" aria-label="colour theme">
<button class="theme-btn" data-theme="dark">Dark</button>
<button class="theme-btn" data-theme="light">Light</button>
<button class="theme-btn" data-theme="print">Print</button>
</div>
</div>
<h1>__TITLE__</h1>
<p class="sub" id="subtitle"></p>
<div class="eqn" id="equation">__EQUATION__</div>
<div class="card"><div class="tools" id="tools"></div><div id="plot"></div></div>
</div>
<script>__D3_SOURCE__</script>
<script>window.__SCENE__ = __SCENE_JSON__;</script>
<script>document.getElementById("subtitle").textContent =
  window.__SCENE__.subtitle;</script>
<script>"""
    + _DRAW
    + """</script>
</body>
</html>
"""
)
