# Rate models

```@meta
CurrentModule = SNNModels
```

`Rate` and `WilsonCowan` (rate units, same dynamics) and `HetRec` (stochastic spiking with
heterogeneous dendritic time constants).

| Model | Reference | Integration |
|:------|:----------|:------------|
| `Rate` | none | forward Euler |
| `WilsonCowan` | named after Wilson and Cowan (1972), Biophys. J. 12:1-24, [10.1016/S0006-3495(72)86068-5](https://doi.org/10.1016/S0006-3495(72)86068-5); see the note | forward Euler |
| `HetRec` | none | forward Euler, stochastic spike per step |

## Rate units: `Rate`

```math
\frac{dx}{dt} = -x + g + I, \qquad r = \tanh(x)
```

Time constant 1 ms, no parameters. ``g`` is the input of the step (written by `RateSynapse`,
reset after each step), ``I`` an external input. Record `:x` or `:r`.

**Integration:** forward Euler, `x += dt (-x + g + I)`, then `r = tanh(x)`, then `g = 0`.

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

Same dynamics and integration as `Rate`.

**Reference:** Wilson and Cowan (1972), Biophys. J. 12:1-24, [10.1016/S0006-3495(72)86068-5](https://doi.org/10.1016/S0006-3495(72)86068-5).

!!! note "Implementation note"
    The population only carries the name: it does not implement the coupled excitatory/inhibitory equations of the reference.

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

Each neuron owns `Nd` dendrites with time constants sampled from `τd`; its soma also reads each
dendrite of the other neurons with probability `overlap`. Inputs target the dendrites (`:glu`,
`:gaba`, size `N * Nd`).

```math
\begin{aligned}
\tau_d\, \frac{dv_d}{dt} &= -v_d + g_E - g_I \\
\tau_m\, \frac{dv_s^i}{dt} &= \sum_{d \in \mathcal{D}_i} \left(v_d - v_s^i\right) \\
P(\text{spike of } i \text{ in } dt) &= r_i\, \sigma\!\left(k\, (v_s^i - a_i)\right) dt,
\qquad \sigma(x) = \frac{1}{1 + e^{-x}}
\end{aligned}
```

``r_i`` (from `rate`) is the maximal rate, ``k`` = `steepness`, ``a_i`` an adaptive baseline
(time constant `τrate`, +1 per spike).

**Integration:** forward Euler. Per step: dendrites `v_d += dt (-v_d - is) / τd`, soma `v_s += dt/τm Σ_d (v_d - v_s)` (with ``k`` connected
dendrites the soma relaxes to their mean with time constant ``τ_m/k``), baseline decay, then one Bernoulli trial of probability
``r_i σ(k (v_s - a_i)) dt``; refractory for `round(τabs / dt)` steps. Reference: none.

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
