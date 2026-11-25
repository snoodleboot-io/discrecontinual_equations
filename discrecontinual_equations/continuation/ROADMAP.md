# Invariant manifolds, connecting orbits, and stochastic bifurcation

Status of the three extensions beyond equilibrium continuation. Each is validated
against an analytic oracle before being trusted; the remaining gaps are listed
honestly rather than papered over.

## 1. Invariant manifolds W^s / W^u - IMPLEMENTED

`manifold.py`. The stable and unstable manifolds of an equilibrium are computed by
the parameterization method: a Taylor map ``P(theta)`` from the eigenspace into
state space satisfying ``f(P(theta)) = DP(theta) . Lambda theta``, solved order by
order as a per-monomial linear solve ``(A - (k . lambda) I) P_k = -N_k`` (the same
homological idea as the center manifold, with the selected real spectrum). A
`ManifoldSelection` strategy picks the unstable or stable eigenvalues.

Validation. Flow-invariance: integrating the field from ``P(theta)`` reproduces
``P(e^{Lambda t} theta)`` with error falling spectrally with order (7e-7, 1.7e-10,
3e-14 at orders 3, 5, 7). For the Hamiltonian saddle the unstable manifold is the
homoclinic loop, and the chart lies on the exact level set ``H = 0`` to 1e-6.

Global growth of a two-dimensional manifold beyond the chart is
`geodesic_level_sets.py` (DEQ-30): Krauskopf and Osinga's geodesic level sets,
seeded on the chart, each new point an orbit-segment BVP solved from its landing
end. Exact on a plane for every linear saddle tried (rings are circles of their
radius to 1e-11, stable-rate ratios up to 20, near-parallel eigenvectors),
second order on the tilted homoclinic cylinder, and the Lorenz manifold to
geodesic distance 100 in 1.5 s with no point lost.

Step control is the paper's (DEQ-61): Delta halves where any point turns by
alpha_max or Delta * alpha_max, doubles where every point is below both lower
bounds, and a ring is accepted at the smallest step with a warning
(`GeodesicMesh.warnings`). A landing Newton misses is retried alone by the
paper's continuation in the flight time, so a ring is recomputed only when
retries fail. At the SIADS / IJBC settings the band distances are the papers'
exactly (25, 40, 55 bands at 62.75, 92.75, 122.75) and the last circle has
2,330 points against the published 2,300.

A complex pair (a saddle-focus) is charted in the pair's complex coordinates
(DEQ-62): on a saddle-focus whose stable manifold is an exact degree-9 graph the
order-N chart is off it by O(rho^(N+1)) (fitted slopes 3.13 to 9.00 for N = 2
to 8) and on it to 5e-17 from N = 9, and the level-set grower seeded on it is
31 to 36,000 times closer on its first five rings than seeded on the tangent
plane.

Remaining: the strict Lorenz growth stops at distance 165.25 (step 1.5, spacing 1.5 to 4.5) and
says so in `GeodesicMesh.stopped`: 1 of 1,248 landings fails at the smallest
step, 0.094, and seven rings from 157.09 on were accepted there turning by up
to 0.95. It stopped at 157.4 while the continuation in the flight time gave a
landing up after 200 steps: where the flow runs along the level set the
landings go 18.7 inside before coming round to the outward one. A landing
lost at 200 steps is now continued again with 4,000, once per front (a row
the larger budget failed is not escalated again at a smaller Delta), and a
final Newton solve outside its bracket counts as lost. The growth to 157.4
costs what it did (the same mesh); the rings beyond it take 824 s against
64 s for the 200-step growth to 157.4, side by side. At the papers' settings,
where a few rows never land, the same stops cost 1.7 (SIADS) to 2.2 (IJBC)
times as long, and the lossy growth to 170 2.8 times, losing 1 point not 57. With `loss` it goes on, with holes. At the papers' settings it stops
short of their distances: 150.75 at the SIADS spacing (0.25 to 1.0; published
151.75) and 152.75 at the survey's (0.67 to 2.0; published 154.75), on 7 and 4
landings at the smallest step 0.01.
Krauskopf and Osinga reached 151.75 (SIADS 2003, Sec. 4.5) and 154.75 (IJBC
2005 survey, Sec. 2.3).

## 2. Connecting orbits - ORBIT SOLVER IMPLEMENTED, continuation pending

`connecting_orbit.py`. A homoclinic orbit is solved as a truncated boundary-value
problem: the trajectory is discretised by the trapezoidal rule on ``[-T, T]`` with
projection boundary conditions (departure end in the unstable eigenspace, arrival
end in the stable eigenspace, using the Jacobian's left eigenvectors) and a phase
condition, solved by Gauss-Newton.

Validation. From a deliberately wrong seed it converges to the exact homoclinic
``x(t) = 3/2 sech^2(t/2)`` of the Hamiltonian saddle; the boundary-value residual
falls to 1e-12 and the remaining error is trapezoidal ``O(h^2)`` (4.98e-3 at
h = 0.20, 1.25e-3 at h = 0.10 - a factor of four).

Remaining: continue the orbit in a parameter (the codim-1 global bifurcation) by
wrapping the boundary-value residual as a `ResidualFunction` for the existing
`ContinuerBuilder`, with Bogdanov-Takens as the oracle (the homoclinic curve must
emanate from it); heteroclinic connections between distinct saddles; higher-order
manifold boundary conditions from part 1 for shorter intervals.

## 3. Stochastic bifurcation - IMPLEMENTED (1-D)

`stochastic.py`. Two distinct objects, treated separately, with the Ito/Stratonovich
distinction explicit via `NoiseConvention`:

- **P-bifurcation** (phenomenological): the one-dimensional stationary
  Fokker-Planck density ``p_s(x) = (C / g^2) exp(integral 2 f_ito / g^2 dx)``
  (`StationaryDensity`), whose shape change `DensityModes` reports.
- **D-bifurcation** (dynamical): the top Lyapunov exponent of the linearised flow,
  ``f'(x0) - g'(x0)^2 / 2`` (Ito), from `LyapunovExponent`.

Validation. For the pitchfork with multiplicative noise
``dx = (a x - x^3) dt + s x dW`` (Stratonovich) the dominant density mode leaves the
origin at ``a = s^2 / 2`` (numeric 0.1803 vs oracle 0.1800) and the Lyapunov
exponent equals ``a`` exactly, crossing zero at ``a = 0`` - the known P and D
thresholds.

Remaining: multi-dimensional stationary densities (discretised Fokker-Planck
eigenproblem); Lyapunov exponents by ergodic simulation where no closed form
exists; the coupled noisy SQUID and fluxgate systems (Palacios-Aven) as applied
targets now that the 1-D oracle passes.
