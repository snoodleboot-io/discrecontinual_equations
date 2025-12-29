# Advanced Features

## Stochastic Differential Equations

```python
from discrecontinual_equations.function.stochastic import StochasticFunction

class MySDE(StochasticFunction):
    def eval(self, point, time=None):
        return [-self.alpha * point[0]]  # Drift term

    def diffusion(self, point, time=None):
        return [self.sigma]  # Diffusion term
```

## Custom Solvers

Extend the base `Solver` class to implement custom numerical methods.

## Plotting Results

```python
from discrecontinual_equations.plot.line_plot import LinePlot

plot = LinePlot()
plot.plot(x=solution.time, y=solution.results[0])
