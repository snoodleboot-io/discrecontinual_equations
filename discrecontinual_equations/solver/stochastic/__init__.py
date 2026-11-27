"""Solvers for stochastic differential equations, and the noise they consume.

An Ito equation ``dX = a(X, t) dt + G(X, t) dW`` driven by ``m`` independent
Wiener processes has no single "next value" to compute: a one-step scheme
approximates the Ito-Taylor expansion of the solution over ``[t, t + h]``, and
each term of that expansion multiplies a functional of the Brownian path over
the step - the increment ``dW_j``, the time integral ``I_(j,0)``, the twofold
integrals ``I_(j,k)`` and so on. A scheme has the order it was derived for only
when it is given those functionals with their exact joint law. The package is
organised around that fact:

* :mod:`.wiener` draws the increments, ``I_(j,0)`` and the Levy areas, and holds
  a Brownian path fixed in advance for convergence studies;
* :mod:`.iterated_integrals` draws the twofold and threefold integrals jointly,
  for the several-channel order 1.5 scheme;
* :mod:`.coefficients` turns an equation into Ito coefficients (adding the
  Stratonovich correction when asked) and estimates their derivatives;
* the solvers - Euler-Maruyama (strong order 0.5), Milstein (1.0, with Levy
  areas for non-commuting noise), ``srk2`` (1.0, derivative-free), ``srk3``
  (1.5), ``srk4`` (weak order 2.0), the Gottwald-Melbourne map for alpha-stable
  noise, and the Zhan-Duan-Li-Li theta step;
* :mod:`.ensemble` steps many paths of one equation at once.

``srk5`` refuses to run: no scheme of the order its name once claimed exists
for this interface. The concept page ``docs/user-guide/stochastic-solvers.md``
explains each method with an example and its limits.

References:
    Kloeden, P. E. and Platen, E. Numerical Solution of Stochastic Differential
    Equations. Springer, 1992. DOI 10.1007/978-3-662-12616-5.

    Kuznetsov, D. F. To numerical modeling with strong orders 1.0, 1.5, and 2.0
    of convergence for multidimensional dynamical systems with random
    disturbances. arXiv:1802.00888v6 (2022); Automation and Remote Control 79(7),
    2018. Eq. (5), the order 1.5 Ito-Taylor scheme for several channels.

    Mrongowius, J. and Rossler, A. On the approximation and simulation of
    iterated stochastic integrals and the corresponding Levy areas in terms of a
    multidimensional Brownian motion. arXiv:2101.09542 (2021).
"""
