# Integrate-and-fire neurons

```@meta
CurrentModule = SNNModels
```

This page documents the generalized integrate-and-fire (IF) family: the leaky/adaptive `IF`,
the adaptive exponential `AdEx` and the three-conductance `ExtendedIF`, together with the
machinery they share (the integration pipeline of `AbstractGeneralizedIF`, the spike
parameters `PostSpike`, and `make_heterogeneous`).

`IF` and `AdEx` are built from three parameter objects:

- `param`: the neuron model (`IFParameter`, `AdExParameter`);
- `synapse`: the synapse model, any `AbstractSynapseParameter` (default `DoubleExpSynapse()`),
  see [Synapse and receptor models](synapses.md);
- `spike`: the spike parameters (`PostSpike()`).

The synapse model determines the synaptic current ``I_{syn}`` (field `syn_curr`) that enters
the membrane equation and the target symbols that connections can use (`:ge`, `:glu`, `:he`
for excitation and `:gi`, `:gaba`, `:hi` for inhibition with the two-receptor models).

## Integration pipeline of generalized IF populations

`integrate!(p::AbstractGeneralizedIF, param, dt)` (file `generalized_if/gif.jl`) runs, in this
order:

1. `update_synapses!(p, p.synapse, p.receptors, p.synvars, dt)`: the spikes written by the
   connections into `p.receptors` during the previous step are added to the synaptic variables,
   which are integrated over `dt`; the receptor buffers are then zeroed;
2. `synaptic_current!(p, p.synapse, p.synvars)`: computes `p.syn_curr` from the synaptic
   variables and the membrane potential;
3. `update_neuron!(p, param, dt)`: integrates the membrane (and adaptation) equations, detects
   spikes, applies reset and refractoriness.

All schemes are forward Euler with the step `dt` of `sim!`/`train!` (default `0.125ms`).
`ExtendedIF` defines its own `integrate!`.

```@autodocs
Modules = [SNNModels]
Pages   = ["populations/populations.jl", "generalized_if/gif.jl"]
```

## Leaky and adaptive integrate-and-fire: `IF`

```math
\begin{aligned}
\tau_m \frac{dv}{dt} &= -(v - E_l) + R\,(I - w) - R\, I_{syn} \\
\tau_w \frac{dw}{dt} &= a\,(v - E_l) - w
\end{aligned}
```

When ``v > V_t`` the neuron fires, ``v \leftarrow V_r``, ``w \leftarrow w + b`` and the neuron is
refractory for `spike.τabs`. The adaptation current is integrated only when `τw > 0` (the
default `τw = 0` gives a plain leaky integrate-and-fire neuron).

Integration (`update_neuron!`), per neuron and step:
- if `tabs > 0`: `fire = false`, `tabs -= 1`, ``v`` is not integrated (it stays at ``V_r``);
- else `v += dt / τm * (-(v - El) + R * (-w + I) - R * syn_curr)`; if `v > Vt`: `fire = true`,
  `v = Vr`, `tabs = round(Int, τabs / dt)`;
- then, if `τw > 0`, for every neuron `w += b` if it fired and
  `w += dt * (a * (v - El) - w) / τw`.

`tabs` starts at 1, so every neuron skips the first step.

`IFParameter` fields:

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `C` | `-1pF` (sentinel) | pF | membrane capacitance, only used to compute `τm` |
| `gl` | `-1nS` (sentinel) | nS | leak conductance, only used to compute `τm` and `R` |
| `τm` | `C / gl` if `C > 0 && gl > 0`, else `15ms` | ms | membrane time constant |
| `Vt` | `-50mV` | mV | spike threshold |
| `Vr` | `-60mV` | mV | reset potential |
| `El` | `-70mV` | mV | leak reversal potential |
| `R` | `1nS / gl` if `gl > 0`, else `0.06` | GΩ | membrane resistance |
| `ΔT` | `2mV` | mV | slope factor (not used by `IF`) |
| `a` | `0` | nS | subthreshold adaptation |
| `b` | `0` | pA | spike-triggered adaptation increment |
| `τw` | `0` | ms | adaptation time constant (`0` disables adaptation) |

Of `PostSpike`, `IF` uses only `τabs`.

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

## Adaptive exponential integrate-and-fire: `AdEx`

```math
\begin{aligned}
\tau_m \frac{dv}{dt} &= -(v - E_l) + \Delta_T \exp\!\left(\frac{v - \theta}{\Delta_T}\right)
                        - R\, I_{syn} - R\, w + R\, I \\
\tau_w \frac{dw}{dt} &= a\,(v - E_l) - w \\
\tau_A \frac{d\theta}{dt} &= V_t - \theta
\end{aligned}
```

A spike is detected when ``v \geq 0`` mV (not at `Vt`, which is the threshold of the
exponential term). At a spike ``v`` is set to 20 mV for that step, ``w \leftarrow w + b``,
``\theta \leftarrow \theta + A_t`` and the refractory counter is set to
`round(Int, τabs / dt)`; at the next step ``v`` is reset to ``V_r``. ``A_t``, ``\tau_A`` and
`τabs` are fields of `spike::PostSpike`; with the default `At = 0mV` the threshold stays at
``V_t``. A negative `ΔT` removes the exponential term.

Integration (`update_neuron!`), per neuron and step: reset `v = Vr` if the neuron fired in the
previous step; `fire = false`; `tabs -= 1`; if `tabs > 0` stop (``v``, ``w``, ``θ`` frozen);
`w += dt * (a * (v - El) - w) / τw` (old ``v``); the Euler step of ``v``;
`θ += dt * (Vt - θ) / τA`; spike test and spike updates.

`AdExParameter` fields (Brette and Gerstner, 2005):

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `C` | `281pF` | pF | membrane capacitance, only used to compute `τm` |
| `gl` | `40nS` | nS | leak conductance (the code notes that the paper uses 30 nS) |
| `Vt` | `-50mV` | mV | threshold of the exponential term, resting value of ``θ`` |
| `Vr` | `-70.6mV` | mV | reset potential |
| `El` | `-70.6mV` | mV | leak reversal potential |
| `τm` | `C / gl` (7.025 ms) | ms | membrane time constant |
| `R` | `nS / gl` (0.025 GΩ) | GΩ | membrane resistance |
| `ΔT` | `2mV` | mV | slope factor |
| `τw` | `144ms` | ms | adaptation time constant |
| `a` | `4nS` | nS | subthreshold adaptation |
| `b` | `80.5pA` | pA | spike-triggered adaptation increment |

`AdEx` is the only point-neuron model that accepts heterogeneous parameters
(`AdExParameter{Vector{Float32}}`, built with `make_heterogeneous`).

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

## Membrane parameters: the pair rule

`IFParameter` and `AdExParameter` (and through it `Tripod` and `BallAndStick`) store `C`, `gl`,
`R` and `τm`, tied by ``\tau_m = C / g_L`` and ``R = 1\,\mathrm{nS} / g_L``. `IF` and `AdEx`
integrate with `τm` and `R`, the multicompartment models with `C` and `gl`, so the four values
are kept consistent:

| given | result |
|---|---|
| nothing | default pair (AdEx `C = 281pF, gl = 40nS`; IF `τm = 15ms, R = 0.06`) |
| one value | `ArgumentError`: a given value is never combined with a default |
| (C, gl), (C, R), (C, τm), (gl, τm), (R, τm) | the other two are derived |
| more than two | accepted if consistent within 0.1 %, otherwise `ArgumentError` |

Changing one value later keeps the others consistent:

| changed | kept | recomputed |
|---|---|---|
| `τm` | `gl`, `R` | `C = τm gl` |
| `C` | `gl`, `R` | `τm = C / gl` |
| `gl` or `R` | `C` | `τm`, and `R` or `gl` |

The rule applies to `@update!`, to property assignment on the mutable `AdExParameter`
(`p.τm = 20ms`), to `with_membrane` and to the sampled fields of `make_heterogeneous`. Two values
changed together (`with_membrane(p; τm = 20ms, C = 281pF)`) define a new pair. In an `@update!`
block the assignments are applied one after the other, each with the rule above.

```julia
using SpikingNeuralNetworks
SNN.@load_units
p = SNN.AdExParameter(gl = 40nS, τm = 20ms)     # C = 800pF
SNN.@update! p τm = 10ms                             # gl kept: C = 400pF
q = SNN.with_membrane(p; C = 281pF, gl = 40nS)   # new pair: τm = 7.025ms
```

```@autodocs
Modules = [SNNModels]
Pages   = ["generalized_if/membrane.jl"]
```

## Spike parameters: `PostSpike`

| Field | Default | Units | Used by | Meaning |
|:------|:--------|:------|:--------|:--------|
| `At` | `0mV` | mV | `AdEx`, `Tripod`, `BallAndStick` | adaptive-threshold increment per spike |
| `τA` | `10ms` | ms | `AdEx`, `Tripod`, `BallAndStick` | relaxation time of the threshold to `Vt` |
| `AP_membrane` | `10mV` | mV | `Tripod`, `BallAndStick` | somatic potential during the action potential |
| `τabs` | `1ms` | ms | all | absolute refractory period |
| `up` | `1ms` | ms | `Tripod`, `BallAndStick` | action-potential duration, added to `τabs` |

```@autodocs
Modules = [SNNModels]
Pages   = ["spike/postspike.jl"]
```

## Heterogeneous parameters: `make_heterogeneous`

`make_heterogeneous(param, N; field = distribution, ...)` returns a parameter object of the same
type in which every field is a vector of length `N`; the fields passed as keywords are sampled
with `rand(distribution, N)`, the others are copies of the value in `param`. Only `AdEx`
supports such parameters in SNNModels 1.8.4 (`IF` raises a `MethodError` when simulated with
vector parameters).

```julia
using SpikingNeuralNetworks
SNN.@load_units
param = SNN.make_heterogeneous(SNN.AdExParameter(), 50; El = SNN.SNNModels.Uniform(-72mV, -68mV))
E = SNN.AdEx(N = 50, param = param)
SNN.sim!([E]; duration = 100ms)
```

The docstring of `make_heterogeneous` is listed with the integration pipeline above.

## Three-conductance integrate-and-fire: `ExtendedIF`

A conductance-based IF neuron with an excitatory conductance and two inhibitory conductances
(PV-like and SST-like, same reversal potential and time constant), plus a multiplicative
excitation-SST interaction term. It has no `synapse` field; its conductances are population
fields.

```math
\begin{aligned}
C_m \frac{dv}{dt} &= g_l (E_l - v) + g_E (E_e - v) + g_{PV} (E_i - v) + g_{SST} (E_i - v)
                    - \alpha\, g_E\, g_{SST} (E_e - v) + I \\
\frac{dg_E}{dt} &= -\frac{g_E}{\tau_e}, \qquad
\frac{dg_{PV}}{dt} = -\frac{g_{PV}}{\tau_i}, \qquad
\frac{dg_{SST}}{dt} = -\frac{g_{SST}}{\tau_i}
\end{aligned}
```

Spike when ``v > V_t``, reset to ``V_r``, refractory for `τabs` (a field of the parameter, not
of `PostSpike`). Integration: forward Euler, conductances first, then the membrane.

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `Cm` | `250pF` | pF | membrane capacitance |
| `Vt` | `-40mV` | mV | spike threshold |
| `Vr` | `-65mV` | mV | reset potential |
| `El` | `-70mV` | mV | leak reversal potential |
| `gl` | `10nS` | nS | leak conductance (marked as arbitrary in the code) |
| `τe` | `6ms` | ms | decay of `g_Exc` |
| `τi` | `20ms` | ms | decay of `g_PV` and `g_SST` |
| `E_i` | `-75mV` | mV | inhibitory reversal potential |
| `E_e` | `0mV` | mV | excitatory reversal potential |
| `τabs` | `5ms` | ms | absolute refractory period |
| `α` | `0` | 1/nS | excitation-SST interaction strength |

Connections target one of the three conductances: `SpikingSynapse(pre, E, :g_Exc; conn)` (also
`:g_PV`, `:g_SST`; `:ge`/`:glu` map to `:g_Exc` and `:gi`/`:gaba` to `:g_PV`). (In SNNModels 1.8.4
there was no `synaptic_target` method for `ExtendedIF`.)

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

## Models present in the source tree but not loaded

The following files exist in `SNNModels/src/populations` but their `include` is commented out
in `populations/populations.jl`; the types are not available in SNNModels 1.8.4 and the files
refer to abstract types that no longer exist in the package.

- `generalized_if/if_CANAHP.jl`: `IF_CANAHP` / `IF_CANAHPParameter`, an integrate-and-fire
  neuron with a calcium variable, a non-specific calcium-activated cationic current (CAN) and a
  calcium-activated after-hyperpolarisation potassium current (AHP), with NMDA receptors. The
  docstring links the preprint https://www.biorxiv.org/content/10.1101/2022.07.26.501548v1.
- `adex/adex_multitimescale.jl`: `AdExMultiTimescale` / `AdExMultiTimescaleParameter`, an AdEx
  neuron with an arbitrary number of double-exponential synaptic timescales assigned to
  glutamatergic or GABAergic receptors, and an adaptive threshold.
