# Functions API

::: discrecontinual_equations.function

## Coloured noise

`ColouredNoise` turns a white-noise `StochasticFunction` into one whose drivers are
Ornstein-Uhlenbeck processes of correlation time `tau`, carried as extra state
variables. The intensity is held fixed as `tau` varies, so `tau -> 0` is the white
equation in the Stratonovich sense; every solver, the ensemble and the continuation
estimators take the augmented equation unchanged (`driver_columns()` hands its noise
to a `NoiseMatrix`).

::: discrecontinual_equations.function.coloured_noise
