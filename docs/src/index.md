# SpikingNeuralNetworks.jl Documentation

Julia Spiking Neural Networks (JuliaSNN) is a library for simulating biophysical neuronal network models.

This documentation is _work in progress_; please contact me via the GitHub repository if you have any specific questions or want to collaborate!

## Simple and powerful simulation framework

The library's strength points are:
 - Modular, intuitive, and quick instantiation of complex biophysical models;
 - Large pool of standard models already available and easy implementation of custom new models;
 - High performance and native multi-threading support, laptop and cluster-friendly;
 - Access to all network's variables at runtime and save-load-rerun of arbitrarily complex networks;
 - Growing ecosystem for stimulation protocols, network analysis, and visualization ([SNNUtils](https://github.com/JuliaSNN/SNNUtils), [SNNPlots](https://github.com/JuliaSNN/SNNPlots), [SNNGeometry](https://github.com/JuliaSNN/SNNGeometry)).

`SpikingNeuralNetworks.jl` is the umbrella package of the `JuliaSNN` ecosystem: it loads and re-exports
`SNNModels` (models and simulation engine), `SNNPlots` (plots of the recordings, Makie based) and
`SNNUtils` (stimulation protocols and analysis). After `using SpikingNeuralNetworks`, the module is
also available under the short alias `SNN`.

## Models: populations, connections, and stimuli

SpikingNeuralNetworks.jl builds on the idea that a neural network is composed of three classes of
objects: the network _populations_, their recurrent _connections_, and the external _stimuli_ they
receive. A network model is a `NamedTuple` with keys `pop`, `syn`, `stim`, `name` and `time` (the
simulation clock, a `Time` object). The elements of `pop`, `syn` and `stim` are concrete subtypes of
`AbstractPopulation`, `AbstractConnection` and `AbstractStimulus`.

Network models are built with `compose`, which takes any population, connection or stimulus (or a
previously composed model) as keyword arguments, sorts them into `pop`, `syn` and `stim`, and checks
that names are not duplicated. For example:

```julia
using SpikingNeuralNetworks
SNN.@load_units

E = SNN.IF(N = 100)   # integrate-and-fire population with 100 neurons and default parameters
# recurrent spiking synapses onto the excitatory receptors (:ge is an alias of :glu),
# connection probability 0.1 and weight 2 nS
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.1, μ = 2nS))
my_model = SNN.compose(E = E, EE = EE)   # model with the E population and the EE connection
# my_model = SNN.compose(; E, EE)        # equivalent
SNN.monitor!(E, [:fire])
SNN.sim!(; model = my_model, duration = 100ms)
```

The population and synapse are then available as `my_model.pop.E` and `my_model.syn.EE`.

!!! note
    - Users are not expected to use the abstract types, but only their concrete subtypes.
    - Network models must include at least one population. Connections and stimuli always target one population.
    - Because in biophysical network models connections are typically synapses, the two terms are used interchangeably.

### Pre-existing models

For each abstract type, JuliaSNN offers a library of models: in the example above, an
integrate-and-fire population (`IF <: AbstractPopulation`) and a sparse spiking synapse
(`SpikingSynapse <: AbstractConnection`). All available models, with their equations and parameters,
are listed in the [Model catalogue](catalogue/index.md); see also [Populations](populations.md),
[Stimuli](stimuli.md), [Plasticity](plasticity.md) and [Recordings](recordings.md).

Models can be extended by defining new subtypes of `AbstractPopulation`, `AbstractConnection` or
`AbstractStimulus`, and the methods that the simulation loop calls for them. See
[Model Extensions](models_ext.md).

## Simulation

Leveraging Julia's [multiple dispatch](https://docs.julialang.org/en/v1/manual/methods/#Methods),
the simulation loop calls the methods defined for each type of component and parameter. One time
step of `sim!` and `train!` does:

<!-- norun -->
```julia
record_zero!(P, C, S, T)                  # at t = 0 only: record the initial state
update_time!(T, dt)                       # advance the clock
for s in stimuli
    stimulate!(s, s.param, T, dt)
    record!(s, T)
end
for p in populations
    update_traces!(p, p.param, dt, T)     # train! only
    integrate!(p, p.param, dt)
    plasticity!(p, p.param, dt, T)        # train! only
    record!(p, T)
end
for c in connections
    update_traces!(c, c.param, dt, T)     # train! only
    forward!(c, c.param, dt, T)
    plasticity!(c, c.param, dt, T)        # train! only
    record!(c, T)
end
```

In a step, the stimuli are applied first and provide inputs to the populations; then the
differential equations of the populations are integrated; finally the population activity is
propagated through the connections, which deliver it to their targets for the next step.
Plasticity (long-term and short-term) runs only under `train!`; `sim!` never changes the
synaptic weights. Both functions take the model as keyword (`sim!(; model, duration = 1s)`) or as
first positional argument (`sim!(model, 1s)`); the default time step is `dt = 0.125ms`.

Using Julia's [pass-by-sharing](https://docs.julialang.org/en/v1/manual/functions/#man-argument-passing),
connections and stimuli keep references to the fields of the populations they target. This allows
`stimulate!` and `forward!` to read and update the population variables directly.

## Installation

JuliaSNN/SpikingNeuralNetworks.jl is available in the General Julia registry.
Install it with `]add SpikingNeuralNetworks`.

You can install the latest version directly from the git repository:

<!-- norun -->
```julia
]add https://github.com/JuliaSNN/SpikingNeuralNetworks.jl
```

To learn how to use the library, follow the [Tutorial](examples.md).
