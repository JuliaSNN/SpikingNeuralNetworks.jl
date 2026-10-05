# Synapse and receptor models

```@meta
CurrentModule = SNNModels
```

A *synapse model* describes how the spikes delivered by the connections are turned into a
synaptic current in the postsynaptic neurons. It is a parameter struct
(`AbstractSynapseParameter`) stored in the population: field `synapse` of the generalized
integrate-and-fire neurons (`IF`, `AdEx`, `ExtendedIF`), fields `soma_syn` and `dend_syn` of the
dendritic neurons (`Tripod`, `BallAndStick`). The connection (`SpikingSynapse`, see
[Connections](connections.md)) only adds weights to an input buffer; all kinetics live in the
population.

## How spikes reach a synapse model

Each population owns three objects built from its synapse model:

| Object | Built by | Content |
|:-------|:---------|:--------|
| `receptors` (`receptors_s`, `receptors_d1`, ... in dendritic models) | `synaptic_receptors` | `NamedTuple` of `Vector{Float32}` input buffers, by default `(glu, gaba)`; one per receptor group for `MultiReceptorSynapse` |
| `synvars` (`synvars_s`, ...) | `synaptic_variables` | the state variables (conductances, rise variables), an `AbstractSynapseVariable` |
| `syn_curr` (columns of `is` in dendritic models) | | the synaptic current, written by `synaptic_current!` |

In one simulation step (`sim!` or `train!`), the order is:

1. `integrate!` of the population calls `update_synapses!`: the buffers are added to the
   synaptic state, the kinetics are integrated over `dt`, and the buffers are zeroed;
   then `synaptic_current!` computes the current from the state and the membrane potential;
   then the membrane equations are integrated.
2. `forward!` of each `SpikingSynapse` adds `W[s] * ρ[s]` (weight times short-term plasticity
   factor) to the buffer entry of the postsynaptic neuron for every presynaptic spike. These
   inputs are consumed at the next step.

The receptor symbol given to the connection selects the buffer. `get_synapse_symbol` maps
`:ge` and `:he` to `:glu`, `:gi` and `:hi` to `:gaba`; any other symbol is used as is (for
example `:glu`, `:gaba`, or a receptor group such as `:AMPA` of a `MultiReceptorSynapse`).
For dendritic neurons, the fourth argument of `SpikingSynapse` selects the compartment
(`:s`, `:d1`, `:d2` for `Tripod`, `:s`, `:d` for `BallAndStick`); see `synaptic_target`.

All synaptic weights, buffers and state variables are `Float32`.

**Sign convention.** The neuron equations subtract the synaptic current,
``C\,dV/dt = \ldots - I_{syn}``. Conductance-based models return
``I_{syn} = \sum_r g_r (V - E_r)``; current-based models return ``I_{syn} = -(g_e - g_i)``.
In both cases weights are positive for excitatory and inhibitory connections; the sign of the
effect comes from the receptor.

## Summary

| Model | Type | Variables | Integration | Works with |
|:------|:-----|:----------|:------------|:-----------|
| `DeltaSynapse` | current, instantaneous | `ge`, `gi` | none (one-step pulse) | `IF`, `AdEx`, `ExtendedIF` |
| `CurrentSynapse` | current, single exponential | `ge`, `gi` | forward Euler | point neurons, dendritic neurons |
| `SingleExpSynapse` | conductance, single exponential | `ge`, `gi` | forward Euler | point neurons, dendritic neurons |
| `DoubleExpSynapse` | conductance, rise and decay | `ge`, `gi`, `he`, `hi` | forward Euler | all (default of `IF`, `AdEx`) |
| `DoubleExpCurrentSynapse` | current, rise and decay | `ge`, `gi`, `he`, `hi` | forward Euler | all |
| `ReceptorSynapse` | conductance, list of receptors, NMDA block | `g`, `h` (`N x n_rec`) | exponential Euler | all (default of `Tripod`, `BallAndStick`) |
| `MultiReceptorSynapse` | as `ReceptorSynapse`, one input per receptor group | `g`, `h` | exponential Euler | point neurons, dendritic neurons |
| `Confavreux2025Synapse` | conductance, AMPA + filtered NMDA + GABA | `gAMPA`, `gNMDA`, `gGABA` | forward Euler | all |

"Point neurons" are the generalized integrate-and-fire populations; `DeltaSynapse` lacks the
five-argument `synaptic_current!` method used by `Tripod` and `BallAndStick` and fails there.

## Delta synapse

`DeltaSynapse()` has no parameters. The weights received during the previous step are applied
as a current for one integration step and then discarded:

```math
I_{syn}(t) = -\left(\sum_{k \in \text{exc}} w_k - \sum_{k \in \text{inh}} w_k\right).
```

With the forward Euler update of `IF`, a weight ``w`` produces a voltage jump of
``R\,w\,dt/\tau_m``, which depends on `dt`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS), synapse = SNN.DeltaSynapse())
```

```@autodocs
Modules = [SNNModels]
Pages   = ["synapse/synapses/DeltaSynapse.jl"]
```

## Current synapse

Single-exponential current-based synapse.

```math
\frac{dg_e}{dt} = -\frac{g_e}{\tau_e} + \sum_k w_k\,\delta(t - t_k), \qquad
\frac{dg_i}{dt} = -\frac{g_i}{\tau_i} + \sum_k w_k\,\delta(t - t_k), \qquad
I_{syn} = -(g_e - g_i)
```

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `τe` | `6ms` | ms | decay time constant of the excitatory current |
| `τi` | `2ms` | ms | decay time constant of the inhibitory current |

Integration: the input is added to `ge`/`gi`, then one forward Euler step of the decay.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS),
           synapse = SNN.CurrentSynapse(τe = 5ms, τi = 10ms))
```

```@autodocs
Modules = [SNNModels]
Pages   = ["synapse/synapses/CurrentSynapse.jl"]
```

## Single-exponential conductance synapse

```math
\frac{dg_e}{dt} = -\frac{g_e}{\tau_e} + \sum_k w_k\,\delta(t - t_k), \qquad
\frac{dg_i}{dt} = -\frac{g_i}{\tau_i} + \sum_k w_k\,\delta(t - t_k)
```
```math
I_{syn} = g_{syn,e}\, g_e\,(V - E_e) + g_{syn,i}\, g_i\,(V - E_i)
```

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `τe` | `6ms` | ms | excitatory decay time constant |
| `τi` | `0.5ms` | ms | inhibitory decay time constant |
| `E_i` | `-75mV` | mV | inhibitory reversal potential |
| `E_e` | `0mV` | mV | excitatory reversal potential |
| `gsyn_e` | `1.0` | - | excitatory conductance scaling (weights in nS) |
| `gsyn_i` | `1.0` | - | inhibitory conductance scaling |

Integration: input added to `ge`/`gi`, then one forward Euler step of the decay.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS),
           synapse = SNN.SingleExpSynapse(τe = 5ms, τi = 10ms))
```

```@autodocs
Modules = [SNNModels]
Pages   = ["synapse/synapses/SingleExpSynapse.jl"]
```

## Double-exponential conductance synapse

The default synapse of `IF` and `AdEx`. Each spike increments a rise variable ``h`` that drives
the conductance ``g``:

```math
\frac{dh_e}{dt} = -\frac{h_e}{\tau_{re}} + \sum_k w_k\,\delta(t - t_k), \qquad
\frac{dg_e}{dt} = -\frac{g_e}{\tau_{de}} + h_e
```

(same for ``h_i, g_i`` with ``\tau_{ri}, \tau_{di}``), and
``I_{syn} = g_{syn,e} g_e (V - E_e) + g_{syn,i} g_i (V - E_i)``. The kernel is not normalised:
a unit increment of ``h`` gives
``g(t) = \frac{\tau_r \tau_d}{\tau_d - \tau_r}\left(e^{-t/\tau_d} - e^{-t/\tau_r}\right)``.

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `τre` | `1ms` | ms | excitatory rise time constant |
| `τde` | `6ms` | ms | excitatory decay time constant |
| `τri` | `0.5ms` | ms | inhibitory rise time constant |
| `τdi` | `2ms` | ms | inhibitory decay time constant |
| `E_i` | `-75mV` | mV | inhibitory reversal potential |
| `E_e` | `0mV` | mV | excitatory reversal potential |
| `gsyn_e` | `1.0` | - | excitatory conductance scaling |
| `gsyn_i` | `1.0` | - | inhibitory conductance scaling |

Integration: forward Euler; the input is added to `he`/`hi`, then `g` is updated with the new
`h`, then `h` decays.

```julia
using SpikingNeuralNetworks
SNN.@load_units
syn = SNN.DoubleExpSynapse(τre = 1ms, τde = 6ms, τri = 0.5ms, τdi = 2ms)
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS), synapse = syn)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["synapse/synapses/DoubleExpSynapse.jl"]
```

## Double-exponential current synapse

Same kinetics as `DoubleExpSynapse`, but `ge` and `gi` are currents (pA) and
``I_{syn} = -(g_e - g_i)``. Fields: `τre = 1ms`, `τde = 6ms`, `τri = 0.5ms`, `τdi = 2ms`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS),
           synapse = SNN.DoubleExpCurrentSynapse())
```

```@autodocs
Modules = [SNNModels]
Pages   = ["synapse/synapses/DoubleExpCurrentSynapse.jl"]
```

## Receptor-based synapses

`ReceptorSynapse` and `MultiReceptorSynapse` combine an arbitrary list of receptors
(`Receptor`), each with its own kinetics, reversal potential and peak conductance, and an
optional voltage-dependent magnesium block for NMDA receptors.

### Receptor kinetics

For receptor ``r`` with rise and decay time constants ``\tau_r``, ``\tau_d``, peak
conductance ``g_0`` and reversal potential ``E_{rev}``:

```math
\frac{dh_r}{dt} = -\frac{h_r}{\tau_r} + \alpha_r \sum_k w_k\,\delta(t - t_k), \qquad
\frac{dg_r}{dt} = -\frac{g_r}{\tau_d} + h_r, \qquad
\alpha_r = \frac{1}{\tau_r} - \frac{1}{\tau_d}
```

so that a unit weight produces ``g_r(t) = e^{-t/\tau_d} - e^{-t/\tau_r}``. The receptor
conductance scale is ``g_{syn} = g_0 \cdot \mathrm{norm\_synapse}(\tau_r, \tau_d)``, where
`norm_synapse` is the inverse of the peak of the difference of exponentials,

```math
t_p = \frac{\tau_r \tau_d}{\tau_d - \tau_r}\ln\frac{\tau_d}{\tau_r}, \qquad
\mathrm{norm\_synapse} = \left(e^{-t_p/\tau_d} - e^{-t_p/\tau_r}\right)^{-1},
```

hence a unit weight gives a peak conductance of ``g_0`` nS. The current is

```math
I_{syn} = \sum_r g_{syn,r}\, g_r\,(V - E_{rev,r})\, B_r(V), \qquad
B_r(V) = \begin{cases} 1 & nmda_r = 0 \\
\left(1 + \frac{[\mathrm{Mg}]}{b}\, e^{k V}\right)^{-1} & \text{otherwise.} \end{cases}
```

The block ``B`` (`nmda_gating`) has the Jahr and Stevens (1990) form; its parameters are the
fields of `NMDAVoltageDependency`.

Integration (exponential Euler, exact decay over one step, receptor by receptor):
`h += α w`; `g = exp(-dt/τd) (g + dt h)`; `h = exp(-dt/τr) h`. The current uses the membrane
potential at the beginning of the step.

### `Receptor` fields

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `name` | `"Receptor"` | | label |
| `E_rev` | `0.0` | mV | reversal potential |
| `τr` | `-1.0` | ms | rise time constant (non-positive means unset) |
| `τd` | `-1.0` | ms | decay time constant (non-positive means unset) |
| `g0` | `0.0` | nS | peak conductance per unit weight |
| `gsyn` | `g0 * norm_synapse(τr, τd)` if `g0 > 0`, else `0` | nS | computed |
| `α` | `(τd - τr) / (τd τr)` | 1/ms | computed |
| `τr⁻`, `τd⁻` | `1/τr`, `1/τd` if positive, else `0` | 1/ms | computed |
| `nmda` | `0.0` | | non-zero enables the magnesium block |
| `target` | `:none` | | receptor group, used by `MultiReceptorSynapse` |

`ReceptorVoltage` is an alias of `Receptor`. `Receptors(...)` is a function that builds a
`ReceptorArray = Vector{Receptor{Float32}}`; `Receptors(AMPA, NMDA, GABAa, GABAb)` and
`Receptors(glu::Glutamatergic, gaba::GABAergic)` return the four receptors in this order.
`Glutamatergic(AMPA, NMDA)` and `GABAergic(GABAa, GABAb)` are convenience pairs.

### `NMDAVoltageDependency` fields

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `b` | `3.36` | mM | |
| `k` | `-0.077` | 1/mV | voltage sensitivity |
| `mg` | `1.0` | mM | extracellular magnesium concentration |

The defaults are attributed in the code to Eyal et al. (2018). `SomaNMDA` and `EyalNMDA` are
predefined instances with these values.

### `ReceptorSynapse`

Spikes on the `glu` input drive the receptors listed in `glu_receptors`, spikes on `gaba`
those in `gaba_receptors`.

| Field | Default | Meaning |
|:------|:--------|:--------|
| `syn` | `SomaReceptors` | `ReceptorArray` |
| `NMDA` | `NMDAVoltageDependency()` | magnesium block |
| `glu_receptors` | `[1, 2]` | indices driven by `glu` |
| `gaba_receptors` | `[3, 4]` | indices driven by `gaba` |

A positional form `ReceptorSynapse(syn, NMDA; kwargs...)` is also available.

### `MultiReceptorSynapse`

Each receptor is driven by the input buffer named by its `target`; the population gets one
buffer per distinct target (computed by `infer_receptors`), and a connection addresses a group
by name. Fields: `syn = SomaReceptors`, `NMDA = NMDAVoltageDependency()`, and the computed
`receptors = infer_receptors(syn)`. Since 1.8.4 there is a positional constructor
`MultiReceptorSynapse(syn; kwargs...)`, equivalent to `MultiReceptorSynapse(; syn, kwargs...)`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
recs = SNN.Receptors(SNN.Receptor(E_rev = 0mV, τr = 0.5ms, τd = 3ms, g0 = 1nS, target = :AMPA),
                     SNN.Receptor(E_rev = -70mV, τr = 0.5ms, τd = 6ms, g0 = 1nS, target = :GABA))
P = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS),
           synapse = SNN.MultiReceptorSynapse(recs))
E = SNN.Poisson(N = 20, param = SNN.PoissonParameter(10Hz))
EP = SNN.SpikingSynapse(E, P, :AMPA; conn = (p = 0.5, μ = 1.0))
SNN.sim!([E, P], [EP]; duration = 100ms)
```

### Predefined receptor sets

`E_rev` in mV, time constants in ms, `g0` in nS.

| Name | Type | Receptors (E_rev, τr, τd, g0) | Indices |
|:-----|:-----|:------------------------------|:--------|
| `SomaReceptors` | `ReceptorArray` | AMPA (0, 1, 6, 0.7); NMDA (0, 1, 100, 0.15, block); GABAa (-70, 0.5, 10, 2.0); GABAb (-90, 30, 400, 0.006) | |
| `SomaSynapse` | `ReceptorSynapse` | `SomaReceptors`, `NMDA = SomaNMDA` (equal to `ReceptorSynapse()`) | glu `[1, 2]`, gaba `[3, 4]` |
| `TripodSomaSynapse` | `ReceptorSynapse` | AMPA (0, 0.26, 2.0, 0.73); GABAa (-70, 0.1, 15.0, 0.38); `NMDA = EyalNMDA` | glu `[1]`, gaba `[2]` |
| `TripodDendSynapse` | `ReceptorSynapse` | AMPA (0, 0.26, 2.0, 0.73); NMDA (0, 8, 35, 1.31, block); GABAa (-70, 4.8, 29, 0.27); GABAb (-90, 30, 400, 0.006); `NMDA = EyalNMDA` | glu `[1, 2]`, gaba `[3, 4]` |
| `SomaNMDA`, `EyalNMDA` | `NMDAVoltageDependency` | `b = 3.36`, `k = -0.077`, `mg = 1` | |

The dendritic AMPA/NMDA values are attributed to Eyal et al. (2018), Front. Cell. Neurosci. 12,
doi:10.3389/fncel.2018.00181, and the GABA values to Miles et al. (1996), Neuron 16(4):815-823
(references as cited in `SNNUtils/src/models/quaresima2022.jl`); the somatic AMPA values are
labelled "Duarte" in the code without a reference.

`SNNModels` also exports `NMDA_CANAHP` and `Synapse_CANAHP`, but their definitions are commented
out, so these names are not defined in 1.8.4.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS),
           synapse = SNN.ReceptorSynapse())          # AMPA + NMDA + GABAa + GABAb
syn = SNN.ReceptorSynapse(SNN.SomaReceptors, SNN.SomaNMDA; glu_receptors = [1], gaba_receptors = [3])
SNN.nmda_gating(-70.0f0, SNN.SomaNMDA)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["synapse/receptors.jl", "synapse/receptor_types.jl", "synapse/synapses/ReceptorSynapse.jl"]
```

## Confavreux 2025 synapse

AMPA, NMDA and GABA conductances; the NMDA conductance is a low-pass filtered copy of the AMPA
conductance and the excitatory current mixes the two with weight ``\alpha``. No magnesium block.

```math
\frac{dg_{AMPA}}{dt} = -\frac{g_{AMPA}}{\tau_{AMPA}} + x_{glu}, \qquad
\frac{dg_{GABA}}{dt} = -\frac{g_{GABA}}{\tau_{GABA}} + x_{gaba}, \qquad
\tau_{NMDA}\frac{dg_{NMDA}}{dt} = g_{AMPA} - g_{NMDA}
```
```math
I_{syn} = \left(\alpha\, g_{AMPA} + (1-\alpha)\, g_{NMDA}\right)(V - E_e) + g_{GABA}\,(V - E_i)
```

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `τAMPA` | `5ms` | ms | AMPA decay time constant |
| `τNMDA` | `100ms` | ms | NMDA filter time constant |
| `τGABA` | `10ms` | ms | GABA decay time constant |
| `E_i` | `-80mV` | mV | inhibitory reversal potential |
| `E_e` | `0mV` | mV | excitatory reversal potential |
| `α` | `0.23` | - | AMPA fraction of the excitatory conductance |

Integration: forward Euler (`gAMPA`, then `gGABA`, then `gNMDA` with the updated `gAMPA`). The
input ``x`` enters the Euler step multiplied by `dt`, so a spike of weight ``w`` increments
`gAMPA` by ``w\,dt``, unlike the other models where the increment is ``w``. The model is named
after Confavreux et al. (2025); the full reference is not given in the code.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS),
           synapse = SNN.Confavreux2025Synapse())
```

```@autodocs
Modules = [SNNModels]
Pages   = ["synapse/synapses/Confraveux2025.jl"]
```

## Interface and generic functions

To add a synapse model, subtype `AbstractSynapseParameter` and implement
`synaptic_variables`, `update_synapses!`, `synaptic_current!` (five-argument form, plus the
three-argument form if needed) and optionally `synaptic_receptors`.

```@autodocs
Modules = [SNNModels]
Pages   = ["synapse/synapses.jl", "synapse/synaptic_targets.jl"]
```
