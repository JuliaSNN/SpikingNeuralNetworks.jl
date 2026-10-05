# Stimuli

```@meta
CurrentModule = SNNModels
```

This page explains how to attach external inputs to a network. The full list of stimulus
models, with their equations and parameters, is in the
[stimulus catalogue](catalogue/stimuli.md).

## Building a stimulus

A stimulus is built from a parameter object, a target population and a target variable:

```
stim = SNN.Stimulus(param, post, sym, [comp]; kwargs...)
```

- `param` selects the stimulus model (`PoissonFixed`, `PoissonLayer`,
  `SpikeTimeStimulusParameter`, `CurrentNoise`, ...).
- `post` is the population that receives the input.
- `sym` is the variable that receives it. For spiking inputs it is resolved by the
  population's `synaptic_target`: on integrate-and-fire neurons `:ge` (or `:he`, `:glu`)
  is the excitatory receptor buffer and `:gi` (or `:hi`, `:gaba`) the inhibitory one. For
  current inputs it is a field of the population, by default `:I`.
- `comp` is the compartment of a dendritic neuron (`:s`, `:d1`, `:d2` for `Tripod`).
  Poisson stimuli take it as the keyword `comp`, layer and spike-time stimuli as the
  fourth positional argument.

The stimulus is added to a model with `compose`, under any name:

```julia
using SpikingNeuralNetworks
SNN.@load_units

E = SNN.IF(N = 100)
noise = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), E, :ge)                  # independent trains
layer = SNN.Stimulus(SNN.PoissonLayer(rate = 10Hz, N = 500), E, :gi;         # shared input layer
                     conn = (p = 0.1, μ = 1.0))
model = SNN.compose(; E, noise, layer)
SNN.monitor!(E, :fire)
SNN.sim!(; model, duration = 500ms)
```

At every step of `sim!`/`train!` the stimuli are applied first, after the clock has
advanced by `dt` and before the populations are integrated.

## Choosing the target neurons

How the targeted neurons are chosen depends on the stimulus:

- `PoissonStimulus` (`PoissonFixed`, `PoissonInterval`, `PoissonVariable`): keyword
  `neurons = :ALL` (default), a vector or range of indices, or `neurons = :p_post` with
  `p_post = 0.2` to target a random 20% of the population.
- `CurrentStimulus`: keyword `neurons = :ALL` or a vector of indices.
- `PoissonStimulusLayer` and `SpikeTimeStimulus`: the targets are given by the
  connectivity `conn`, either a NamedTuple passed to `sparse_matrix` (e.g.
  `(p = 0.1, μ = 1.0, rule = :Fixed)`) or, for spike-time stimuli, a weight matrix of size
  `post.N x N`. `SpikeTimeStimulusIdentity` connects input `j` to neuron `j`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
first20 = SNN.Stimulus(SNN.PoissonFixed(rate = 1kHz), E, :ge; neurons = 1:20)
random = SNN.Stimulus(SNN.PoissonFixed(rate = 1kHz), E, :ge; neurons = :p_post, p_post = 0.2)
SNN.neurons(random)          # the sampled indices
current = SNN.Stimulus(SNN.CurrentNoise(E; I_base = 200pA), E; neurons = [1, 2, 3])
```

## Changing a stimulus during a simulation

Stimuli can be modified between `sim!` calls (or from a `perturbation!` function):

- `set_active!(stim, false)` / `set_active!(stim, true)` switches a `PoissonStimulus` off
  and on; it also works for `PoissonStimulusLayer` (ignored in SNNModels 1.8.4).
- `set_intervals!(stim, [[t1, t2], ...])` replaces the activity intervals of a
  `PoissonInterval` stimulus.
- `set_variable!(stim, key, value)` sets `param.variables[key]` of a `PoissonVariable`
  stimulus, broadcasts `value` into an array field of the parameter (e.g. `:rates` of
  `PoissonLayerHet`, `:I_base` of `CurrentNoise`), or replaces a scalar field of a mutable
  parameter (e.g. `:rate` of `PoissonFixed`, `PoissonInterval`, `PoissonLayer`; this failed in
  SNNModels 1.8.4).
- `update_spikes!(stim, spikes, start_time)` and `shift_spikes!(stim, delay)` replace (sorted
  by time) or shift the spike list of a `SpikeTimeStimulus`; empty lists are allowed.

All of them also accept a `StimulusGroup` and are then applied to every element.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 50)
f(t, v) = v[:rate]
stim = SNN.Stimulus(SNN.PoissonVariable(variables = Dict{Symbol,Any}(:rate => 1kHz), rate = f), E, :ge)
pulse = SNN.Stimulus(SNN.PoissonInterval(rate = 2kHz, intervals = [[50ms, 100ms]]), E, :ge)
model = SNN.compose(; E, stim, pulse)
SNN.monitor!(E, :fire)
SNN.sim!(; model, duration = 200ms)
SNN.set_variable!(stim, :rate, 3kHz)              # stronger background
SNN.set_intervals!(pulse, [[250ms, 300ms]])       # new pulse
SNN.sim!(; model, duration = 200ms)
SNN.set_active!(stim, false)                      # background off
SNN.sim!(; model, duration = 200ms)
```

## Multicompartment targets

`MultiCompartmentStimulusGroup(param, post, sym, comps)` creates one Poisson stimulus per
compartment and bundles them in a `StimulusGroup`:

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Tripod(N = 10)
group = SNN.MultiCompartmentStimulusGroup(SNN.PoissonFixed(rate = 200Hz), E, :glu, [:d1, :d2])
soma = SNN.Stimulus(SNN.PoissonFixed(rate = 100Hz), E, :glu; comp = :s)
model = SNN.compose(; E, group, soma)
SNN.sim!(; model, duration = 100ms)
```

## Recording stimuli

Stimuli that have their own spikes (`PoissonStimulusLayer`, `SpikeTimeStimulus`) can be
monitored like populations, e.g. `monitor!(stim, :fire)` and `spiketimes(stim)`; see
[Recordings](recordings.md).

The API of every stimulus type is documented in the [stimulus catalogue](catalogue/stimuli.md).
