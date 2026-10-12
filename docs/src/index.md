# SpikingNeuralNetworks.jl

Simulate networks of spiking neurons in Julia: point and dendritic neuron models, conductance and
receptor synapses, delays, long- and short-term plasticity, Poisson and spike-time inputs, and
recording, analysis and plotting of the results.

```@example index
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.1, μ = 2nS))
drive = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), E, :ge)
model = SNN.compose(; E, EE, drive, silent = true)
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
| learn by example: neurons, plasticity, save and rerun | Tutorials: [Populations](tutorials/populations.md), [Plasticity](tutorials/plasticity.md), [Simulations](tutorials/simulations.md) |
| understand populations, connections, stimuli | [Populations](populations.md), [Stimuli](stimuli.md) |
| run, train, change precision or threads, save | [Running simulations](simulation.md) |
| record and plot | [Recordings](recordings.md), [Visualization](visualization.md) |
| add plasticity | [Plasticity](plasticity.md) |
| find a model and its equations | [Model catalogue](catalogue/index.md) |
| write a new model | [Model extensions](models_ext.md) |
| upgrade from 1.x | [What changes from v1 to v2](#What-changes-from-v1-to-v2), [V2 changelog](release_notes.md) |

## What changes from v1 to v2

Version 2 is faster (an E/I network runs 8 to 12 times faster than 1.9, synapses use a third of the
memory) and adds `Float64` models, opt-in threads and rule switching. Keyword-built models run
unchanged; what needs attention:

**Saved files**
- Files saved with version 1 cannot be loaded: rebuild the model from its configuration and save it again.

**Plasticity**
- Rule types are part of the synapse type: install another rule with `with_plasticity`,
  `without_plasticity` or `change_plasticity(model, :name; LTP, STP)`; `update_plasticity!` keeps the type.
- `vSTDPParameter` is event-driven: read its weights with `plasticity_value(syn, :W)`; the 1.9 rule is `vSTDPDenseParameter`.
- STDP traces are stored lazily: read them with `trace_values(syn, :tpre)` or after `sync_traces!(syn)`.
- `iSTDPTime` is removed: use `iSTDPRate` or `iSTDPPotential`.

**Synapse fields**
- `syn.J`, `syn.index`: call `SNNModels.ensure_transpose!(syn)` first.
- `syn.ρ`: read with `SNNModels.efficacy(syn)`, call `SNNModels.materialize_efficacy!(syn)` before writing.
- `syn.param.delaytime` has length 1 when all delays are equal; delays round up to whole steps.

**Results**
- Seeded runs with Poisson inputs, delays at `dt = 0.1ms`, or STDP weights differ from 1.9 (same statistics).
- `sim!`/`train!` flush subnormal numbers to zero: pass `flush_subnormals = false` for IEEE behaviour.

**Custom code**
- Positional constructors of the changed structs need the new fields; keyword constructors are unchanged.
- Do not write `pop.fire` of an integrate-and-fire population after its `integrate!` within a step.
- For `Float64` build populations with `VFT = Vector{Float64}` and rules with `FT = Float64`.

Every change, with its migration: [V2 changelog](release_notes.md) and
[`CHANGELOG_2.0.0.md`](https://github.com/JuliaSNN/SNNModels.jl/blob/main/CHANGELOG_2.0.0.md) of SNNModels.
