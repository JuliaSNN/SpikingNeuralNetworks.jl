# SpikingNeuralNetworks.jl

Simulate networks of spiking neurons in Julia: point and dendritic neuron models, conductance and
receptor synapses, delays, long- and short-term plasticity, Poisson and spike-time inputs, and
recording, analysis and plotting of the results.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.1, μ = 2nS))
drive = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), E, :ge)
model = SNN.compose(; E, EE, drive)
SNN.monitor!(E, [:fire])
SNN.sim!(model, 1s)
SNN.firing_rate(E; interval = 0:10ms:1s, pop_average = true)
```

Install with `] add SpikingNeuralNetworks`, then follow [Getting started](getting_started.md).

## Packages

`SpikingNeuralNetworks` loads and re-exports three packages of the
[JuliaSNN](https://github.com/JuliaSNN) organisation, and defines the alias `SNN`:

- [SNNModels](https://github.com/JuliaSNN/SNNModels.jl): models and simulation engine;
- [SNNPlots](https://github.com/JuliaSNN/SNNPlots.jl): plots of the recordings (Makie);
- [SNNUtils](https://github.com/JuliaSNN/SNNUtils.jl): stimulation protocols and analysis.

## Where to go

| To | Read |
|:---|:-----|
| build and run a first network | [Getting started](getting_started.md) |
| understand populations, connections, stimuli | [Populations](populations.md), [Stimuli](stimuli.md) |
| run, train, change precision or threads, save | [Running simulations](simulation.md) |
| record and plot | [Recordings](recordings.md), [Visualization](visualization.md) |
| add plasticity | [Plasticity](plasticity.md) |
| find a model and its equations | [Model catalogue](catalogue/index.md) |
| write a new model | [Model extensions](models_ext.md) |
| upgrade from 1.x | [Release notes](release_notes.md) |
