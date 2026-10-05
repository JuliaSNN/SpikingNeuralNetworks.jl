# Spike sources

```@meta
CurrentModule = SNNModels
```

Spike sources are populations without input dynamics: they produce spikes that connections
propagate to other populations. They differ from stimuli (see [Stimuli](stimuli.md)), which
write directly into a target population without a connection object. `Identity` is a relay
population that turns its input into spikes.

## Poisson population: `Poisson`

In each step neuron ``i`` fires with probability

```math
P(\text{spike}) = \nu_i\, dt ,
```

the Bernoulli approximation of a Poisson process (accurate for ``\nu_i dt \ll 1``). The rate is
set by the parameter type:

| Parameter type | Field | Default | Meaning |
|:---------------|:------|:--------|:--------|
| `PoissonHomoParameter` | `rate` | `1Hz` | same rate for all neurons |
| `PoissonHetParameter` | `rate` | `1Hz` (must be a vector of length `N`) | one rate per neuron |

`PoissonParameter(rate)` returns a `PoissonHomoParameter` for a scalar and a
`PoissonHetParameter` for a vector. `Population(param::PoissonParameter; N)` builds the
population. Integration: `rand!(randcache)`, `fire[i] = randcache[i] < rate * dt`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
P  = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
Ph = SNN.Population(SNN.PoissonParameter(collect(range(1Hz, 50Hz, length = 100))); N = 100)
SNN.monitor!([P, Ph], [:fire])
SNN.sim!([P, Ph]; duration = 1s)
```

## Population with a shared fluctuating rate: `VariablePoisson`

`VariablePoisson(; N, param = VariablePoissonParameter(...))` (no `Population` method, `param`
is required). One scalar noise ``\eta``, low-pass filtered uniform noise
(``\xi_t \in [-0.5, 0.5]``), modulates the total rate ``\nu`` of the population; each neuron
fires with probability ``\nu\, dt / N``:

```math
\begin{aligned}
\eta &\leftarrow \eta\,(1 - dt/\tau) + \xi_t\, dt/\tau \\
\nu &= \left[\tfrac{r_0}{2}\, F(\beta \eta) + \rho\right]_+, \qquad
      F(x) = \begin{cases} x & x > 0 \\ 1 & x \le 0 \end{cases} \\
\rho &\leftarrow \rho + (r_0 - \nu)\, dt / 400\,\text{ms}
\end{aligned}
```

``\rho`` starts at ``r_0``; the slow feedback drives the mean of ``\nu`` to ``r_0``. ``F`` is
discontinuous at 0 as written in the code.

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `β` | `0` | - | gain of the noise |
| `τ` | `50ms` | ms | noise filter time constant |
| `r0` | `1kHz` | 1/ms | target total rate of the population |

```julia
using SpikingNeuralNetworks
SNN.@load_units
P = SNN.VariablePoisson(N = 100, param = SNN.VariablePoissonParameter(r0 = 2kHz))
SNN.monitor!(P, [:fire])
SNN.sim!([P]; duration = 500ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["populations/poisson.jl"]
```

## Inhomogeneous Poisson population: `InhomogeneousPoisson`

`Population(InhomogeneousPoissonParam(...); N)`. Same rate process as `VariablePoisson`, but
independent for every neuron and with rate ``\nu_i`` per neuron (not divided by `N`); the
feedback time constant is a parameter, and the spike probability is
``1 - e^{-\nu_i dt}``:

```math
\begin{aligned}
\eta_i &\leftarrow \eta_i\,(1 - dt/\tau) + \xi_{i,t}\, dt/\tau \\
\nu_i &= \left[\tfrac{r_0}{2}\, F(\beta \eta_i) + \rho_i\right]_+ \\
\rho_i &\leftarrow \rho_i + (r_0 - \nu_i)\, dt / \tau_{rate} \\
P(\text{spike}) &= 1 - e^{-\nu_i dt}
\end{aligned}
```

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `β` | `0` | - | gain of the noise |
| `τ` | `50ms` | ms | noise filter time constant |
| `r0` | `1kHz` | 1/ms | target rate of each neuron |
| `rate_timescale` | `400ms` | ms | time constant of the rate feedback |

```julia
using SpikingNeuralNetworks
SNN.@load_units
P = SNN.Population(SNN.InhomogeneousPoissonParam(r0 = 10Hz, β = 5); N = 100)
SNN.monitor!(P, [:fire])
SNN.sim!([P]; duration = 500ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["populations/inhomogeneous_poisson.jl"]
```

## Relay population: `Identity`

Each step, neuron ``i`` fires if its input `g` is positive; `spikecount` stores that input,
`h` accumulates it, and `g` is reset to zero. Any target symbol used by a connection is mapped
to `g`. Because populations are integrated before connections are forwarded, a spike arriving
in step ``t`` is re-emitted in step ``t + 1``.

```julia
using SpikingNeuralNetworks
SNN.@load_units
P = SNN.Poisson(N = 10, param = SNN.PoissonParameter(50Hz))
Id = SNN.Identity(N = 10)
s = SNN.SpikingSynapse(P, Id, :g; conn = (μ = 1, p = 1.0))
model = SNN.compose(; P, Id, s)
SNN.monitor!(Id, [:fire])
SNN.sim!(model, 100ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["populations/identity.jl"]
```
