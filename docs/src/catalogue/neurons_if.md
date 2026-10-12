# Integrate-and-fire neurons

```@meta
CurrentModule = SNNModels
```

`IF`, `AdEx` and `ExtendedIF`. `IF` and `AdEx` combine a neuron model (`param`), a
[synapse model](synapses.md) (`synapse`, which gives ``I_{syn}``) and spike parameters (`spike`).

```@autodocs
Modules = [SNNModels]
Pages   = ["populations/populations.jl", "generalized_if/gif.jl"]
```

## `IF`

```math
\begin{aligned}
\tau_m \frac{dv}{dt} &= -(v - E_l) + R\,(I - w) - R\, I_{syn} \\
\tau_w \frac{dw}{dt} &= a\,(v - E_l) - w
\end{aligned}
```

Spike when ``v > V_t``: ``v \leftarrow V_r``, ``w \leftarrow w + b``, refractory for `spike.τabs`.
`τw = 0` (default) disables adaptation.

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `C`, `gl` | unset | pF, nS | give `τm = C / gl`, `R = 1nS / gl` |
| `τm` | `15ms` | ms | membrane time constant |
| `R` | `0.06` | GΩ | membrane resistance |
| `Vt` | `-50mV` | mV | spike threshold |
| `Vr` | `-60mV` | mV | reset potential |
| `El` | `-70mV` | mV | leak reversal potential |
| `a` | `0` | nS | subthreshold adaptation |
| `b` | `0` | pA | spike-triggered adaptation increment |
| `τw` | `0` | ms | adaptation time constant |

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100, param = SNN.IFParameter(C = 281pF, gl = 40nS, a = 4nS, b = 80.5pA, τw = 144ms))
E.I .= 1nA
SNN.monitor!(E, [:v, :w, :fire])
SNN.sim!([E]; duration = 200ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["generalized_if/if.jl"]
```

## `AdEx`

```math
\begin{aligned}
\tau_m \frac{dv}{dt} &= -(v - E_l) + \Delta_T \exp\!\left(\frac{v - \theta}{\Delta_T}\right)
                        - R\, I_{syn} - R\, w + R\, I \\
\tau_w \frac{dw}{dt} &= a\,(v - E_l) - w \\
\tau_A \frac{d\theta}{dt} &= V_t - \theta
\end{aligned}
```

Spike when ``v \geq 0`` mV: ``v \leftarrow V_r`` at the next step, ``w \leftarrow w + b``,
``\theta \leftarrow \theta + A_t``, refractory for `spike.τabs`.

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `C` | `281pF` | pF | membrane capacitance |
| `gl` | `40nS` | nS | leak conductance |
| `τm`, `R` | `C / gl`, `1nS / gl` | ms, GΩ | membrane time constant and resistance |
| `Vt` | `-50mV` | mV | threshold of the exponential term |
| `Vr` | `-70.6mV` | mV | reset potential |
| `El` | `-70.6mV` | mV | leak reversal potential |
| `ΔT` | `2mV` | mV | slope factor |
| `τw` | `144ms` | ms | adaptation time constant |
| `a` | `4nS` | nS | subthreshold adaptation |
| `b` | `80.5pA` | pA | spike-triggered adaptation increment |

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.AdEx(N = 10, param = SNN.AdExParameter(), spike = SNN.PostSpike(τabs = 2ms, At = 2mV, τA = 30ms))
E.I .= 1nA
SNN.monitor!(E, [:v, :w, :fire])
SNN.sim!([E]; duration = 500ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["generalized_if/adex.jl"]
```

## Membrane parameters

`C`, `gl`, `R` and `τm` are tied by ``\tau_m = C / g_L`` and ``R = 1\,\mathrm{nS} / g_L``:

| given | result |
|---|---|
| nothing | default pair |
| one value | `ArgumentError` |
| two values | the other two are derived |
| more than two | accepted if consistent within 0.1 % |

```julia
using SpikingNeuralNetworks
SNN.@load_units
p = SNN.AdExParameter(gl = 40nS, τm = 20ms)     # C = 800pF
SNN.@update! p τm = 10ms                         # gl kept: C = 400pF
q = SNN.with_membrane(p; C = 281pF, gl = 40nS)   # new pair: τm = 7.025ms
```

```@autodocs
Modules = [SNNModels]
Pages   = ["generalized_if/membrane.jl"]
```

## `PostSpike`

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `At` | `0mV` | mV | adaptive-threshold increment per spike |
| `τA` | `10ms` | ms | relaxation time of the threshold |
| `AP_membrane` | `10mV` | mV | somatic potential during the action potential (dendritic models) |
| `τabs` | `1ms` | ms | absolute refractory period |
| `up` | `1ms` | ms | action-potential duration (dendritic models) |

```@autodocs
Modules = [SNNModels]
Pages   = ["spike/postspike.jl"]
```

## Heterogeneous parameters

```julia
using SpikingNeuralNetworks
SNN.@load_units
param = SNN.make_heterogeneous(SNN.AdExParameter(), 50; El = SNN.SNNModels.Uniform(-72mV, -68mV))
E = SNN.AdEx(N = 50, param = param)
SNN.sim!([E]; duration = 100ms)
```

## `ExtendedIF`

Excitatory, PV-like and SST-like conductances with an excitation-SST interaction; no `synapse` field.

```math
\begin{aligned}
C_m \frac{dv}{dt} &= g_l (E_l - v) + g_E (E_e - v) + g_{PV} (E_i - v) + g_{SST} (E_i - v)
                    - \alpha\, g_E\, g_{SST} (E_e - v) + I \\
\frac{dg_E}{dt} &= -\frac{g_E}{\tau_e}, \qquad
\frac{dg_{PV}}{dt} = -\frac{g_{PV}}{\tau_i}, \qquad
\frac{dg_{SST}}{dt} = -\frac{g_{SST}}{\tau_i}
\end{aligned}
```

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `Cm` | `250pF` | pF | membrane capacitance |
| `Vt` | `-40mV` | mV | spike threshold |
| `Vr` | `-65mV` | mV | reset potential |
| `El` | `-70mV` | mV | leak reversal potential |
| `gl` | `10nS` | nS | leak conductance |
| `τe` | `6ms` | ms | decay of `g_Exc` |
| `τi` | `20ms` | ms | decay of `g_PV` and `g_SST` |
| `E_i` | `-75mV` | mV | inhibitory reversal potential |
| `E_e` | `0mV` | mV | excitatory reversal potential |
| `τabs` | `5ms` | ms | absolute refractory period |
| `α` | `0` | 1/nS | excitation-SST interaction strength |

Targets: `:g_Exc` (`:ge`, `:glu`), `:g_PV` (`:gi`, `:gaba`), `:g_SST`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.ExtendedIF(N = 10, param = SNN.ExtendedIFParameter(α = 0.01))
E.I .= 400pA
SNN.monitor!(E, [:v, :fire])
SNN.sim!([E]; duration = 100ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["generalized_if/if_extended.jl"]
```
