# Synapse and receptor models

```@meta
CurrentModule = SNNModels
```

A synapse model turns the weights delivered by connections into a synaptic current. It is the
field `synapse` of `IF` and `AdEx` (`soma_syn`, `dend_syn` of `Tripod`, `BallAndStick`); the target
symbol of a connection picks the input (`:ge`, `:he` -> `:glu`; `:gi`, `:hi` -> `:gaba`; other names
as given). The neuron equations subtract ``I_{syn}``: conductance models give
``I_{syn} = \sum_r g_r (V - E_r)``, current models ``I_{syn} = -(g_e - g_i)``; weights are positive.

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

## Delta synapse

No parameters; the weights of the previous step act as a current for one step:

```math
I_{syn}(t) = -\left(\sum_{k \in \text{exc}} w_k - \sum_{k \in \text{inh}} w_k\right).
```

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

Default of `IF` and `AdEx`:

```math
\frac{dh_e}{dt} = -\frac{h_e}{\tau_{re}} + \sum_k w_k\,\delta(t - t_k), \qquad
\frac{dg_e}{dt} = -\frac{g_e}{\tau_{de}} + h_e
```

(same for ``h_i, g_i``), ``I_{syn} = g_{syn,e} g_e (V - E_e) + g_{syn,i} g_i (V - E_i)``.

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

A list of receptors (`Receptor`), each with its kinetics, reversal potential, peak conductance
``g_0`` and an optional NMDA magnesium block:

```math
\frac{dh_r}{dt} = -\frac{h_r}{\tau_r} + \alpha_r \sum_k w_k\,\delta(t - t_k), \qquad
\frac{dg_r}{dt} = -\frac{g_r}{\tau_d} + h_r, \qquad
\alpha_r = \frac{1}{\tau_r} - \frac{1}{\tau_d}
```

with ``g_{syn} = g_0\,\mathrm{norm\_synapse}(\tau_r, \tau_d)`` (a unit weight peaks at ``g_0`` nS) and

```math
I_{syn} = \sum_r g_{syn,r}\, g_r\,(V - E_{rev,r})\, B_r(V), \qquad
B_r(V) = \begin{cases} 1 & nmda_r = 0 \\
\left(1 + \frac{[\mathrm{Mg}]}{b}\, e^{k V}\right)^{-1} & \text{otherwise.} \end{cases}
```

``B`` (`nmda_gating`) is the Jahr and Stevens (1990) block.

### `Receptor` fields

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `name` | `"Receptor"` | | label |
| `E_rev` | `0.0` | mV | reversal potential |
| `τr` | `-1.0` | ms | rise time constant (non-positive means unset) |
| `τd` | `-1.0` | ms | decay time constant (non-positive means unset) |
| `g0` | `0.0` | nS | peak conductance per unit weight |
| `nmda` | `0.0` | | non-zero enables the magnesium block |
| `target` | `:none` | | receptor group, used by `MultiReceptorSynapse` |

`Receptors(...)` builds a `ReceptorArray`.

### `NMDAVoltageDependency` fields

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `b` | `3.36` | mM | |
| `k` | `-0.077` | 1/mV | voltage sensitivity |
| `mg` | `1.0` | mM | extracellular magnesium concentration |

Defaults from Eyal et al. (2018) (`SomaNMDA`, `EyalNMDA`).

### `ReceptorSynapse`

`glu` drives the receptors in `glu_receptors`, `gaba` those in `gaba_receptors`.

| Field | Default | Meaning |
|:------|:--------|:--------|
| `syn` | `SomaReceptors` | `ReceptorArray` |
| `NMDA` | `NMDAVoltageDependency()` | magnesium block |
| `glu_receptors` | `[1, 2]` | indices driven by `glu` |
| `gaba_receptors` | `[3, 4]` | indices driven by `gaba` |

### `MultiReceptorSynapse`

Each receptor is driven by the input named by its `target`; a connection addresses a group by name.

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

Dendritic AMPA/NMDA: Eyal et al. (2018); GABA: Miles et al. (1996).

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

NMDA is a low-pass filtered copy of AMPA; no magnesium block.

```math
\frac{dg_{AMPA}}{dt} = -\frac{g_{AMPA}}{\tau_{AMPA}} + \sum_k w_k \delta(t - t_k), \qquad
\frac{dg_{GABA}}{dt} = -\frac{g_{GABA}}{\tau_{GABA}} + \sum_k w_k \delta(t - t_k), \qquad
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

```@autodocs
Modules = [SNNModels]
Pages   = ["synapse/synapses.jl", "synapse/synaptic_targets.jl"]
```
