"""Build a browsable site from every example in this directory.

Each example runs in its own process and what it writes is gathered under one
directory, with an index page linking them all. Three kinds are handled:
examples that write self-contained HTML are handed the output directory;
examples that write PNG and HTML beside their own source have that directory
harvested; and examples that only print have their console output made into a
page.

    python examples/build_sites.py                  # everything, into examples_site
    python examples/build_sites.py out --only lorenz thaler
    python examples/build_sites.py --list

Everything is written locally and nothing is uploaded. An example of the
harvested kind rewrites the ``images`` directory beside its own source, and
those files are tracked by git, so expect them to show as modified afterwards.
The atlas is by far the longest to build - ``--list`` shows rough timings, and
``--jobs`` runs examples at the same time, which is worth it when the atlas is
in the set.
"""

import argparse
import html
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = "examples_site"
DEFAULT_TIMEOUT_MINUTES = 90.0
CHROMIUM = "chromium-*/chrome-linux*/chrome"
FIGURES = (".png", ".html", ".svg", ".jpg")
# Slack for a coarse filesystem clock when deciding what a run just wrote.
CLOCK_SLACK = 2.0
SECONDS_PER_MINUTE = 60.0


class Kind(Enum):
    """How an example delivers what it makes."""

    # Takes the output directory as its one argument and writes HTML into it.
    PAGES = "pages"
    # Writes PNG and HTML into an ``images`` directory beside its own source.
    IMAGES = "images"
    # Prints its results and writes nothing.
    CONSOLE = "console"


@dataclass(frozen=True, slots=True)
class Example:
    """One runnable example, and how to collect what it writes."""

    name: str
    script: str
    kind: Kind
    title: str
    blurb: str
    minutes: float


EXAMPLES: tuple[Example, ...] = (
    Example(
        "continuation",
        "examples/continuation/webplot_examples.py",
        Kind.PAGES,
        "Continuation and bifurcation atlas",
        "Self-contained pages for equilibrium branches, codimension-1, -2 and "
        "-3 bifurcations, parameter-family surfaces, exploration trees and "
        "connecting orbits, plus ten scrubbable stage films.",
        33.0,
    ),
    Example(
        "lorenz",
        "examples/lorenz/lorenz_equation.py",
        Kind.IMAGES,
        "Lorenz attractor",
        "The chaotic attractor in three dimensions, with a phase-space "
        "companion and a comparison across solvers.",
        1.0,
    ),
    Example(
        "quartic",
        "examples/quartic/quartic_equation.py",
        Kind.IMAGES,
        "Quartic double well",
        "The energy landscape of a quartic potential, the double well behind "
        "every bistable cell in the library.",
        0.5,
    ),
    Example(
        "ccefs",
        "examples/ccefs/ccefs_example.py",
        Kind.IMAGES,
        "Coupled-cell electric-field sensor",
        "Time series and phase space of the coupled-cell sensor array, the "
        "applied system behind the ferroelectric ring.",
        1.0,
    ),
    Example(
        "complex_ode_system",
        "examples/complex_ode_system/complex_ode_example.py",
        Kind.IMAGES,
        "Coupled nonlinear system",
        "Time evolution, phase portraits and delay embeddings for a coupled "
        "system with absolute values and fractional powers.",
        2.0,
    ),
    Example(
        "brownian_motion",
        "examples/brownian_motion/brownian_motion.py",
        Kind.IMAGES,
        "Brownian motion",
        "Sample paths of the Wiener process, the driving noise the stochastic "
        "solvers integrate against.",
        0.5,
    ),
    Example(
        "geometric_brownian_motion",
        "examples/geometric_brownian_motion/geometric_brownian_motion.py",
        Kind.IMAGES,
        "Geometric Brownian motion",
        "Multiplicative noise against its closed-form solution.",
        0.5,
    ),
    Example(
        "thaler",
        "examples/thaler/thaler_example.py",
        Kind.IMAGES,
        "Thaler method for alpha-stable equations",
        "Heavy-tailed alpha-stable diffusion, and boundary preservation for "
        "equations with a natural boundary.",
        2.0,
    ),
    Example(
        "zhan_duan_li_li",
        "examples/zhan_duan_li_li/zhan_duan_li_li_example.py",
        Kind.IMAGES,
        "Zhan-Duan-Li-Li method",
        "The Zhan-Duan-Li-Li stochastic solver, with a comparison against the "
        "other methods on the same path.",
        1.0,
    ),
    Example(
        "bifurcation_walkthrough",
        "examples/continuation/bifurcation_examples.py",
        Kind.CONSOLE,
        "Continuation walkthrough",
        "Saddle-node, transcritical, pitchfork, cusp, Hopf, van der Pol and "
        "Selkov, each continued and reported at the console.",
        3.0,
    ),
    Example(
        "codim2_walkthrough",
        "examples/continuation/codim2_examples.py",
        Kind.CONSOLE,
        "Two-parameter walkthrough",
        "Bogdanov-Takens, zero-Hopf and Hopf-Hopf located by two-parameter "
        "continuation, reported at the console.",
        3.0,
    ),
)


@dataclass(frozen=True, slots=True)
class Ran:
    """One finished run, before what it wrote has been gathered."""

    example: Example
    target: Path
    stdout: str
    seconds: float
    since: float


@dataclass(frozen=True, slots=True)
class Result:
    """What running one example produced."""

    example: Example
    seconds: float
    detail: str
    href: str | None = None
    thumbnail: str | None = None

    @property
    def ok(self) -> bool:
        """An example built exactly when it left something to link to."""
        return self.href is not None


def find_browser() -> str | None:
    """A Chrome or Chromium for the figure export, if this machine has one.

    Plotly writes its still images by driving a real browser, and reports a
    bare ``plotly_get_chrome`` when it cannot find one. ``BROWSER_PATH`` is
    what its browser layer reads, so an installed Chrome - or the copy
    Playwright keeps in its cache - spares the caller that download.
    """
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    cached = sorted((Path.home() / ".cache" / "ms-playwright").glob(CHROMIUM))
    return str(cached[-1]) if cached else None


def run(example: Example, site: Path, timeout: float, browser: str | None) -> Result:
    """Run one example in its own process and collect what it wrote."""
    target = site / example.name
    command = [sys.executable, str(ROOT / example.script)]
    if example.kind is Kind.PAGES:
        target.mkdir(parents=True, exist_ok=True)
        command.append(str(target))
    started = time.monotonic()
    since = time.time() - CLOCK_SLACK
    try:
        finished = subprocess.run(  # noqa: S603 (a fixed command from EXAMPLES)
            command,
            cwd=ROOT,
            env=_environment(browser),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        limit = timeout / SECONDS_PER_MINUTE
        return Result(example, elapsed, f"timed out after {limit:.0f} min")
    seconds = time.monotonic() - started
    if finished.returncode != 0:
        return Result(example, seconds, _failure(finished))
    return _collect(Ran(example, target, finished.stdout, seconds, since))


def _environment(browser: str | None) -> dict[str, str]:
    """The child's environment: the package importable, no interactive backend."""
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    if browser and "BROWSER_PATH" not in environment:
        environment["BROWSER_PATH"] = browser
    return environment


def _failure(finished: subprocess.CompletedProcess) -> str:
    """The last meaningful line of a failed run, for the report."""
    lines = (finished.stderr or finished.stdout or "").strip().splitlines()
    return lines[-1].strip() if lines else "exited non-zero with no output"


def _collect(ran: Ran) -> Result:
    """Gather what the example wrote into its directory and link it."""
    if ran.example.kind is Kind.PAGES:
        pages = sorted(ran.target.glob("*.html"))
        if not pages:
            return Result(ran.example, ran.seconds, "wrote no pages")
        entry = ran.target / "index.html"
        landing = entry if entry.exists() else pages[0]
        detail = f"{len(pages)} pages"
        return Result(ran.example, ran.seconds, detail, _href(ran.target, landing))
    if ran.example.kind is Kind.CONSOLE:
        page = _write_console(ran.example, ran.target, ran.stdout)
        lines = len(ran.stdout.strip().splitlines())
        detail = f"{lines} lines"
        return Result(ran.example, ran.seconds, detail, _href(ran.target, page))
    return _harvest(ran)


def _written(ran: Ran) -> list[Path]:
    """The figures this run just wrote.

    An example writes either into an ``images`` directory or straight beside
    its own source, so both are searched, and the modification time decides
    what belongs to this run rather than to an earlier one.
    """
    source = ROOT / Path(ran.example.script).parent
    return [
        item
        for folder in (source, source / "images")
        if folder.is_dir()
        for item in sorted(folder.iterdir())
        if item.is_file()
        and item.suffix in FIGURES
        and item.stat().st_mtime >= ran.since
    ]


def _harvest(ran: Ran) -> Result:
    """Copy the figures an example just wrote into the site and index them."""
    written = _written(ran)
    pictures = [item for item in written if item.suffix == ".png"]
    if not pictures:
        return Result(ran.example, ran.seconds, "wrote no figures")
    ran.target.mkdir(parents=True, exist_ok=True)
    for item in written:
        shutil.copy2(item, ran.target / item.name)
    copied = sorted(ran.target / item.name for item in pictures)
    page = _write_gallery(ran.example, ran.target, copied)
    detail = f"{len(copied)} figures"
    return Result(
        ran.example,
        ran.seconds,
        detail,
        _href(ran.target, page),
        _href(ran.target, copied[0]),
    )


def _href(target: Path, page: Path) -> str:
    """A link from the index to a file inside one example's directory."""
    return f"{target.name}/{page.name}"


def _write_console(example: Example, target: Path, stdout: str) -> Path:
    """Make a page of an example's console output."""
    target.mkdir(parents=True, exist_ok=True)
    body = f"<pre>{html.escape(stdout.strip() or '(no output)')}</pre>"
    page = target / "index.html"
    page.write_text(_shell(example.title, example.blurb, body), encoding="utf-8")
    return page


def _write_gallery(example: Example, target: Path, pictures: list[Path]) -> Path:
    """Make a page of an example's figures, each linking its interactive twin."""
    figures = []
    for picture in pictures:
        interactive = picture.with_suffix(".html")
        caption = picture.stem.replace("_", " ")
        link = (
            f'<a href="{interactive.name}">interactive</a>'
            if interactive.exists()
            else "<span>still only</span>"
        )
        figures.append(
            f'<figure><img src="{picture.name}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)} {link}</figcaption></figure>",
        )
    body = f'<div class="figures">{"".join(figures)}</div>'
    page = target / "index.html"
    page.write_text(_shell(example.title, example.blurb, body), encoding="utf-8")
    return page


def write_index(results: list[Result], site: Path) -> Path:
    """Write the index linking every example that built."""
    cards = []
    for result in sorted(results, key=lambda item: item.example.title):
        if not result.ok:
            continue
        picture = f'<img src="{result.thumbnail}" alt="">' if result.thumbnail else ""
        cards.append(
            f'<a class="card" href="{result.href}">{picture}'
            f"<h2>{html.escape(result.example.title)}</h2>"
            f"<p>{html.escape(result.example.blurb)}</p>"
            f'<span class="meta">{result.detail} &middot; '
            f"{result.seconds:.0f}s</span></a>",
        )
    built = sum(1 for result in results if result.ok)
    lead = (
        f"{built} of {len(results)} examples built. Every page is self-contained "
        f"and renders offline."
    )
    body = f'<div class="grid">{"".join(cards)}</div>'
    site.mkdir(parents=True, exist_ok=True)
    page = site / "index.html"
    page.write_text(_shell("Examples", lead, body), encoding="utf-8")
    return page


_STYLE = """
  *{box-sizing:border-box}
  body{margin:0;color:#eef1f7;
    background:radial-gradient(1200px 700px at 50% -8%, #151a28 0%, #0a0c12 60%),
      #0a0c12;
    font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,
      Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1040px;margin:0 auto;padding:44px 22px 72px}
  a.back{color:#9aa3b8;text-decoration:none;font-size:14px}
  a.back:hover{color:#eef1f7}
  h1{font-size:30px;margin:14px 0 8px;letter-spacing:-.02em;font-weight:680}
  .lead{color:#9aa3b8;margin:0 0 30px;max-width:76ch}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(292px,1fr));
    gap:18px}
  a.card{display:block;position:relative;overflow:hidden;
    background:linear-gradient(180deg,#141824,#10131d);
    border:1px solid #222838;border-radius:16px;padding:20px;
    text-decoration:none;color:inherit;transition:transform .15s,
      border-color .15s,box-shadow .15s;
    box-shadow:0 18px 44px -28px rgba(0,0,0,.9)}
  a.card:hover{transform:translateY(-3px);border-color:#38bdf8;
    box-shadow:0 26px 60px -28px rgba(56,189,248,.35)}
  a.card img{width:100%;height:132px;object-fit:cover;border-radius:10px;
    margin:0 0 14px;background:#0a0c12}
  a.card h2{font-size:16.5px;margin:0 0 7px;font-weight:620}
  a.card p{color:#9aa3b8;font-size:13px;margin:0 0 10px}
  a.card .meta{color:#5f6a82;font-size:12px;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .figures{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
    gap:20px}
  figure{margin:0;background:#10131d;border:1px solid #222838;border-radius:14px;
    padding:14px;overflow:hidden}
  figure img{width:100%;height:auto;border-radius:8px;background:#0a0c12}
  figcaption{color:#9aa3b8;font-size:13px;margin-top:10px}
  figcaption a{color:#38bdf8}
  pre{background:#10131d;border:1px solid #222838;border-radius:14px;padding:18px;
    overflow-x:auto;font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;
    color:#cfd6e6}
"""

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>__STYLE__</style>
</head>
<body>
<div class="wrap">
__BACK__
<h1>__TITLE__</h1>
<p class="lead">__LEAD__</p>
__BODY__
</div>
</body>
</html>
"""


def _shell(title: str, lead: str, body: str) -> str:
    """Wrap a page body in the shared shell."""
    back = "" if title == "Examples" else '<a class="back" href="../index.html">'
    back = back + "&larr; examples</a>" if back else ""
    return (
        _TEMPLATE.replace("__STYLE__", _STYLE)
        .replace("__BACK__", back)
        .replace("__LEAD__", html.escape(lead))
        .replace("__BODY__", body)
        .replace("__TITLE__", html.escape(title))
    )


def chosen(only: list[str] | None, skip: list[str] | None) -> list[Example]:
    """The examples to build, in registry order."""
    known = {example.name for example in EXAMPLES}
    for name in (only or []) + (skip or []):
        if name not in known:
            message = f"unknown example {name!r}; choose from {sorted(known)}"
            raise SystemExit(message)
    picked = [e for e in EXAMPLES if not only or e.name in only]
    return [e for e in picked if not skip or e.name not in skip]


def report(results: list[Result]) -> int:
    """Print what built and what did not; return the process exit code."""
    print()
    for result in sorted(results, key=lambda item: -item.seconds):
        mark = "ok  " if result.ok else "FAIL"
        name = result.example.name
        print(f"  {mark} {name:<26} {result.seconds:6.1f}s  {result.detail}")
    failed = [result for result in results if not result.ok]
    if failed:
        names = ", ".join(result.example.name for result in failed)
        print(f"\n{len(failed)} of {len(results)} failed: {names}")
        return 1
    print(f"\nall {len(results)} built")
    return 0


def build(examples: list[Example], site: Path, jobs: int, timeout: float) -> int:
    """Run the examples, write the index, and report."""
    browser = find_browser()
    print(f"building {len(examples)} examples into {site} with {jobs} job(s)")
    print(f"figure export: {browser or 'no browser found - run plotly_get_chrome'}")
    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run, e, site, timeout, browser): e for e in examples}
        for future in futures:
            result = future.result()
            state = "ok" if result.ok else "FAILED"
            print(f"  {result.example.name:<26} {state:>6}  {result.detail}")
            results.append(result)
    page = write_index(results, site)
    print(f"\nindex: {page}")
    return report(results)


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and build."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--only", nargs="+", metavar="NAME")
    parser.add_argument("--skip", nargs="+", metavar="NAME")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_MINUTES)
    parser.add_argument("--list", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.list:
        for example in EXAMPLES:
            print(f"  {example.name:<26} ~{example.minutes:>4.1f} min  {example.title}")
        return 0
    examples = chosen(arguments.only, arguments.skip)
    site = Path(arguments.output).resolve()
    timeout = arguments.timeout * SECONDS_PER_MINUTE
    return build(examples, site, max(1, arguments.jobs), timeout)


if __name__ == "__main__":
    sys.exit(main())
