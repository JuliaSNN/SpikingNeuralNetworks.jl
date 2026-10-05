# Rate models

```@meta
CurrentModule = SNNModels
```

SNNModels provides two firing-rate populations, `Rate` and `WilsonCowan`, which in version
1.8.4 have identical dynamics, and `HetRec`, a stochastically spiking population with
heterogeneous dendritic time constants whose firing probability is a sigmoid of a filtered
somatic variable.

## Rate units: `Rate`

```math
\frac{dx}{dt} = -x + g + I, \qquad r = \tanh(x)
```

``t`` is in ms, so the time constant is fixed to 1 ms; there are no parameters
(`RateParameter` has no fields). ``g`` is the input written by `RateSynapse`
(``g_i \mathrel{+}= \sum_j W_{ij} r_j`` at every step, see [Connections](connections.md)),
``I`` an external input. Integration: forward Euler, `x += dt * (-x + g + I)`, `r = tanh(x)`.

Neither `Rate` nor `RateSynapse` resets ``g`` between steps: with a `RateSynapse` the input
``g`` is the running sum of all past inputs ``W r``. `Rate` has no `fire` field; record `:x`,
`:r` or `:g`.

| Field | Default | Meaning |
|:------|:--------|:--------|
| `N` | `100` | number of units |
| `x` | `0.5randn(N)` | internal state |
| `r` | `tanh.(x)` | output rate in ``(-1, 1)`` |
| `g` | zeros | synaptic input (target of `RateSynapse`) |
| `I` | zeros | external input |

```julia
using SpikingNeuralNetworks
SNN.@load_units
R = SNN.Rate(N = 100)
syn = SNN.RateSynapse(R, R; μ = 1.0, p = 0.2)
model = SNN.compose(; R, syn)
SNN.monitor!(R, [:r])
SNN.sim!(model, 50ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["populations/rate.jl"]
```

## `WilsonCowan`

Despite its name, `WilsonCowan` implements the same single-variable dynamics as `Rate`
(``dx/dt = -x + g + I``, ``r = \tanh x``, forward Euler, 1 ms time constant); it does not
implement the coupled excitatory-inhibitory equations of Wilson and Cowan (1972). Its
parameter type `WCParameter` has no fields, and no `synaptic_target` method is defined, so
`RateSynapse` cannot target it in SNNModels 1.8.4.

```julia
using SpikingNeuralNetworks
SNN.@load_units
W = SNN.WilsonCowan(N = 10)
W.I .= 0.2
SNN.monitor!(W, [:r])
SNN.sim!([W]; duration = 20ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["populations/wilsoncowan.jl"]
```

## Heterogeneous dendritic timescales: `HetRec`

`HetRec` is created with `Population(HetRecParameter(...); N)`. Each of the `N` neurons owns
`Nd` leaky dendritic compartments with time constants sampled from `τd`. The soma of neuron
``i`` reads its own dendrites and, with probability `overlap`, each dendrite of every other
neuron. Input connections target the dendrites through the receptors `:glu` and `:gaba`
(postsynaptic size `N * Nd`), filtered by a current-based `CurrentSynapse`
(``\tau_e = 6`` ms, ``\tau_i = 2`` ms). Spikes are stochastic.

```math
\begin{aligned}
\tau_d\, \frac{dv_d}{dt} &= -v_d + g_E - g_I \\
\tau_m\, \frac{dv_s^i}{dt} &\approx \sum_{d \in \mathcal{D}_i} \left(v_d - v_s^i\right) \\
P(\text{spike of } i \text{ in } dt) &= r_i\, \sigma\!\left(k\, (v_s^i - a_i)\right) dt,
\qquad \sigma(x) = \frac{1}{1 + e^{-x}}
\end{aligned}
```

``r_i`` (sampled from `rate`, in spikes per ms) is the maximal rate, ``k`` = `steepness`, and
``a_i`` (`trace`) is an adaptive baseline: it decays with `τrate`, relaxes towards ``v_s`` by
`(v_s - trace) / τrate` per non-refractory step (no `dt` factor in the code), and increases by
1 at each spike.

Integration per step: synapse update, Euler step of the dendrites, then for each neuron one
sequential relaxation step `v_s += (v_d - v_s) * dt / τm` per connected dendrite (with ``k``
connected dendrites the effective time constant is about ``τ_m / k``), refractory counter,
baseline update, Bernoulli spike draw; a spike sets the refractory counter to
`round(Int, τabs / dt)`.

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `Nd` | `2` | - | dendrites per neuron |
| `overlap` | `0.5` | - | probability of reading another neuron's dendrite |
| `τd` | `Uniform(10, 100)` | ms | distribution of dendritic time constants |
| `rate` | `Uniform(0, 1)` | 1/ms | distribution of maximal rates |
| `τabs` | `5ms` | ms | absolute refractory period |
| `steepness` | `1` | 1/mV | slope of the sigmoid |
| `τm` | `20ms` | ms | somatic time constant |
| `τrate` | `100ms` | ms | time constant of the adaptive baseline |

Reference not given in the code.

```julia
using SpikingNeuralNetworks
SNN.@load_units
H = SNN.Population(SNN.HetRecParameter(Nd = 3, overlap = 0.2); N = 20)
P = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
s = SNN.SpikingSynapse(P, H, :glu; conn = (μ = 1, p = 0.2))
model = SNN.compose(; P, H, s)
SNN.monitor!(H, [:fire, :v_s])
SNN.sim!(model, 200ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["populations/hetrec.jl"]
```
