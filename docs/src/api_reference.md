# API Reference

```@meta
CurrentModule = SNNModels
```

This page collects the API that is not tied to a specific model: building and simulating a
network, units and helper macros, saving and loading, spatial networks, perturbation
experiments and spike-train analysis. The docstrings of the models themselves (neurons,
synapses, connections, plasticity rules, stimuli) are in the model catalogue, next to the
equations and parameter tables; recording and plotting have their own pages.

| Topic | Page |
|:----- |:---- |
| Integrate-and-fire neurons (IF, ExtendedIF, AdEx) | [Integrate-and-fire models](catalogue/neurons_if.md) |
| Izhikevich, Hodgkin-Huxley, Morris-Lecar | [Other neuron models](catalogue/neurons_other.md) |
| Rate and mean-field models | [Rate models](catalogue/rate_models.md) |
| Poisson and other spike sources | [Sources](catalogue/sources.md) |
| Tripod, BallAndStick, dendrites | [Multicompartment models](catalogue/multicompartment.md) |
| Synapse and receptor models | [Synapses and receptors](catalogue/synapses.md) |
| Connection types and connectivity rules | [Connections](catalogue/connections.md) |
| STDP, iSTDP, vSTDP, short-term plasticity | [Plasticity rules](catalogue/plasticity_rules.md) |
| Normalization, scaling, turnover | [Metaplasticity](catalogue/metaplasticity.md) |
| Stimulus models | [Stimuli](catalogue/stimuli.md) |
| Models shipped with SNNUtils | [SNNUtils models](catalogue/snnutils_models.md) |
| `monitor!`, `record`, `getvariable`, ... | [Recordings](recordings.md) |
| SNNPlots functions | [Visualization](visualization.md) |

All symbols below are exported by `SNNModels`; with `using SpikingNeuralNetworks` they are
available as `SNN.<name>` (most of them also unqualified).

## Index

```@index
```

## Model composition and simulation

A network model is a `NamedTuple` `(pop, syn, stim, name, time)` built with [`compose`](@ref).
[`sim!`](@ref) integrates it without plasticity; [`train!`](@ref) runs the same loop and in
addition calls `update_traces!` and `plasticity!` of every population and connection.

```@autodocs
Modules = [SNNModels]
Pages   = ["utils/structs.jl", "utils/util.jl", "utils/main.jl", "utils/graph.jl", "utils/copying.jl"]
```

## Units and macros

```@autodocs
Modules = [SNNModels]
Pages   = ["utils/unit.jl", "utils/macros.jl"]
```

## Saving and loading

```@autodocs
Modules = [SNNModels]
Pages   = ["utils/io.jl"]
```

## Spatial networks

```@autodocs
Modules = [SNNModels]
Pages   = ["utils/spatial.jl"]
```

## Perturbation analysis

```@autodocs
Modules = [SNNModels]
Pages   = ["utils/perturbation.jl"]
```

## Spike-train analysis

```@autodocs
Modules = [SNNModels]
Pages   = ["analysis/spikes.jl", "analysis/targets.jl", "analysis/populations.jl"]
```

## Module

```@autodocs
Modules = [SNNModels]
Pages   = ["src/SNNModels.jl"]
```

## Umbrella package

`SpikingNeuralNetworks` re-exports `SNNModels`, `SNNPlots` and `SNNUtils`, and every exported
name is defined. (Up to SpikingNeuralNetworks 1.2.1 the export list contained the undefined
`LTPParam`, `STPParam`, `SNNModel`, `make_copy` and `raster!`. `LTPParam` and `STPParam` are
keyword arguments and field names of `SpikingSynapse`; the model copy function is `modelcopy`.)

```@meta
CurrentModule = SpikingNeuralNetworks
```

```@autodocs
Modules = [SpikingNeuralNetworks]
```
