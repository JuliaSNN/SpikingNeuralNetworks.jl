# Contributing

The library is organised in packages of the [JuliaSNN](https://github.com/JuliaSNN) organisation:
`SNNModels.jl` contains the models and the simulation engine, `SNNPlots.jl` the plotting functions,
`SNNUtils.jl` stimulation protocols and analysis tools, and `SpikingNeuralNetworks.jl` loads the three
and hosts this documentation. Contributions (new models, bug reports, documentation) are welcome as
issues or pull requests on the corresponding repository.

A model is a `NamedTuple` built by `compose` from populations (`AbstractPopulation`), connections
(`AbstractConnection`) and stimuli (`AbstractStimulus`). It is run with

<!-- norun -->
```julia
sim!(; model, duration = 1s, dt = 0.125ms)     # no plasticity
train!(; model, duration = 1s, dt = 0.125ms)   # with plasticity
```

Both functions also accept the components directly, e.g. `sim!(P::Vector, C::Vector, S::Vector;
duration)`. Every population must implement `integrate!(p, p.param, dt)`, every connection
`forward!(c, c.param, dt, T)` (or `forward!(c, c.param)`) and `plasticity!(c, c.param, dt, T)`, and
every stimulus `stimulate!(s, s.param, T, dt)`. The full list of requirements, with examples, is in
[Model Extensions](models_ext.md); the existing models in the [Model catalogue](catalogue/index.md)
are good starting points.

New models should come with a docstring (equations, parameters with defaults and units, integration
scheme, references) and a test in `SNNModels.jl/test`.
