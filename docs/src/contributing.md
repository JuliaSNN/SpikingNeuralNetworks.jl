# Contributing

The library is split in packages of the [JuliaSNN](https://github.com/JuliaSNN) organisation:
`SNNModels.jl` (models and simulation engine), `SNNPlots.jl` (plots), `SNNUtils.jl` (stimulation
protocols and analysis) and `SpikingNeuralNetworks.jl` (loads the three and hosts this
documentation). Report bugs and propose changes as issues or pull requests on the repository
concerned.

A new model implements the interface of [Model extensions](models_ext.md) and comes with a
docstring (equations, parameters with defaults and units, integration scheme, references) and a
test in `SNNModels.jl/test`.
