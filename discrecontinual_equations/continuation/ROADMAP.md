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

Remaining: complex (spiral) manifolds need the conjugate-coordinate variant (the
level-set grower seeds a spiral manifold on its tangent plane meanwhile). The
strict Lorenz growth stops at distance 157.2 (step 1.5, spacing 1.5 to 4.5) and
says so in `GeodesicMesh.stopped`: there the flow runs along the level set and
the front is 0.04 to 0.05 off W^s(0) (measured independently), so the radial
steps turn by 0.38 rad at every Delta and four landings fail. With `loss` it
goes on, with holes. Krauskopf and Osinga reached 151.75 (SIADS 2003, Sec. 4.5)
and 154.75 (IJBC 2005 survey, Sec. 2.3). Their step control also bounds
Delta * alpha, which this grower does not.

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
