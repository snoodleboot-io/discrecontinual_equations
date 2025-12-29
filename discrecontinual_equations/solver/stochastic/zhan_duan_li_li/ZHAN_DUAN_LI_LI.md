# ZHAN_DUAN_LI_LI

## Overview

The Zhan-Duan-Li-Li method is a stochastic theta method based on the trapezoidal rule for solving stochastic differential equations, proposed by Zhan, Duan, Li, and Li in their paper.

The method combines implicit and explicit approaches using a theta parameter to balance stability and accuracy in stochastic systems.

## Architecture

```
ZhanDuanLiLiSolver
├── Config: method_parameter (theta), start_time, end_time, step_size, random_seed
├── Method: Stochastic theta method with trapezoidal rule
├── Order: Improved convergence through predictor-corrector approach
└── Range: General SDEs with smooth coefficients
```

## Executive Summary

**Purpose**: Solve stochastic differential equations using a theta method approach
**Key Features**: Predictor-corrector scheme, theta parameter control, improved stability
**Performance**: Moderate computational cost with better accuracy than Euler-Maruyama
**Use Cases**: General stochastic differential equations requiring higher accuracy

## Core Classes

### ZhanDuanLiLiConfig

```python
class ZhanDuanLiLiConfig(StochasticConfig):
    """Configuration for Zhan-Duan-Li-Li stochastic theta method."""

    method_parameter: float = 0.5  # Theta parameter (0=explicit, 1=implicit, 0.5=trapezoidal)
```

### ZhanDuanLiLiSolver

```python
class ZhanDuanLiLiSolver(Solver):
    """Zhan-Duan-Li-Li stochastic theta method."""

    def __init__(self, solver_config: ZhanDuanLiLiConfig)

    def solve(self, equation: DifferentialEquation, initial_values: list[float]):
        """Solve SDE using stochastic theta method."""

    def _stochastic_theta_step(self, y, t_current, t_next, dt, dW, equation):
        """Perform predictor-corrector step with theta weighting."""
```

## Mathematical Foundation

The Zhan-Duan-Li-Li method implements a stochastic theta method:

```
y_{n+1} = y_n + θ * f(t_{n+1}, y^*) * dt + θ * g(t_{n+1}, y^*) * dW
```

where `y^*` is a predictor step, and `θ` controls the implicit/explicit balance.

### Theta Parameter
- **θ = 0**: Fully explicit (Euler-Maruyama)
- **θ = 1**: Fully implicit (potentially unstable)
- **θ = 0.5**: Trapezoidal rule (Crank-Nicolson for SDEs)

### Predictor-Corrector Scheme

1. **Predictor**: `y^* = y_n + f(t_n, y_n) dt + g(t_n, y_n) dW`
2. **Corrector**: `y_{n+1} = y_n + θ[f(t_{n+1}, y^*) dt + g(t_{n+1}, y^*) dW] + (1-θ)[f(t_n, y_n) dt + g(t_n, y_n) dW]`

## Algorithm Details

### Single Time Step

```python
def _stochastic_theta_step(self, y, t_current, t_next, dt, dW, equation):
    theta = self.theta

    # Current point evaluation
    drift_current = equation.eval(y, t_current)
    diffusion_current = equation.diffusion(y, t_current)

    # Predictor step (explicit)
    y_predictor = y + drift_current * dt + diffusion_current * dW

    # Predicted point evaluation
    drift_predictor = equation.eval(y_predictor, t_next)
    diffusion_predictor = equation.diffusion(y_predictor, t_next)

    # Theta-weighted combination
    drift_combined = theta * drift_predictor + (1 - theta) * drift_current
    diffusion_combined = theta * diffusion_predictor + (1 - theta) * diffusion_current

    # Final update
    y_new = y + drift_combined * dt + diffusion_combined * dW

    return y_new
```

## Performance Characteristics

| Theta Value | Stability | Accuracy | Computational Cost |
|-------------|-----------|----------|-------------------|
| θ = 0.0 | Good | Low | Low |
| θ = 0.5 | Better | Higher | Medium |
| θ = 1.0 | Best | Highest | High (may need iteration) |

## Implementation Notes

- **Convergence**: The method provides better convergence properties than basic Euler-Maruyama
- **Stability**: Theta parameter allows tuning for stability vs accuracy trade-offs
- **Applicability**: Works for SDEs with sufficiently smooth coefficients
- **Extensions**: Can be combined with adaptive time stepping for improved efficiency

## References

- Zhan, Duan, Li, Li. "A stochastic theta method based on the trapezoidal rule" (https://arxiv.org/pdf/2010.07691)

---

**Parent Module:** [STOCHASTIC](../STOCHASTIC.md)

**Related Modules:**
- [EULER_MARUYAMA](../euler_maruyama/EULER_MARUYAMA.md) - Basic explicit method
- [MILSTEIN](../milstein/MILSTEIN.md) - Strong order 1.0 method
