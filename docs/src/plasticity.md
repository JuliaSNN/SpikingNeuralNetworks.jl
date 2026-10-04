# Plasticity

```@meta
CurrentModule = SpikingNeuralNetworks
```

Synapses (`SpikingSynapse`) can carry a long-term plasticity rule (`LTPParam`) and a short-term plasticity rule (`STPParam`):

```julia
using SpikingNeuralNetworks
SNN.@load_units

E = SNN.IF(N = 400)
I = SNN.IF(N = 100)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.1, μ = 3.0),
                        LTPParam = SNN.STDPGerstner(A_pre = 1e-2, A_post = -1.05e-2, Wmax = 20pF))
IE = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 5.0), LTPParam = SNN.iSTDPRate(r = 5Hz))
model = SNN.compose(; E, I, EE, IE)

SNN.train!(model = model, duration = 10s)   # weights change
SNN.sim!(model = model, duration = 10s)     # weights frozen
```

!!! warning "Plasticity runs only under `train!`"
    `train!` calls `update_traces!` and `plasticity!` of every connection at each step; `sim!` never does. A synapse with an `LTPParam` or `STPParam` keeps its weights (and its STP variables) constant under `sim!`. Use `train!` to learn and `sim!` to test with frozen weights.

!!! danger "Results obtained with `iSTDPRate` before SNNModels 1.8.2 are wrong"
    See [Inhibitory STDP](@ref "Inhibitory STDP") and [Release notes](release_notes.md).

## Rules at a glance

| Rule | Type | Traces | Notes |
|---|---|---|---|
| `STDPGerstner` | pair, additive | exact decay | signed amplitudes, `A_post < 0` is LTD (default) |
| `STDPTriplet` | triplet, all-to-all | exact decay (4 traces) | Pfister and Gerstner 2006, Auryn `MinimalTriplet` |
| `STDPWeightDependent` | pair, soft bounds | exact decay | Gütig et al. 2003, Auryn `STDPwd`; `η` is a relative rate |
| `STDPConfavreux2025` | pair with rate terms | exact decay | |
| `STDPMexicanHat` | zero-integral kernel | Euler | event-driven weight passes |
| `STDPSymmetric`, `STDPAntiSymmetric` | structured inhibition (Festa et al. 2024) | Euler | |
| `iSTDPRate`, `iSTDPPotential` | inhibitory (Vogels et al. 2011) | Euler | |
| `vSTDPParameter` | voltage-based (Clopath et al. 2010) | Euler | |

## Event-driven pair and triplet STDP

`STDPGerstner`, `STDPConfavreux2025`, `STDPWeightDependent` and `STDPTriplet` share event-driven kernels that follow the ordering of Auryn (Zenke and Gerstner 2014). Per time step, with `fireJ` the presynaptic and `fireI` the postsynaptic spikes of this step:

1. **Pre-spike pass.** For every presynaptic spike, walk its outgoing synapses: LTD, computed from the postsynaptic trace of the target neuron.
2. **Post-spike pass.** For every postsynaptic spike, walk its incoming synapses: LTP, computed from the presynaptic trace of the source neuron.
3. **Trace increment.** Every neuron that fired adds 1 to its trace(s).
4. **Trace decay.** Every trace is multiplied by `exp(-dt/τ)` (exact decay, one precomputed factor per time constant).

Consequences:

- Traces are read *before* the spikes of the current step are added: a trace holds ``\sum_{m<n} e^{-(t_n-t_m)/τ}`` over the *earlier* spikes of the neuron. A pre and a post spike in the same step do not interact with each other (the earlier history of each still does). This is the convention of Auryn and Brian2 (`w` updated before the trace increment).
- Only the weights touched in the step (the synapses of neurons that fired) are updated and clamped to `[Wmin, Wmax]`. All weights are clamped once at the first step.
- Cost per step is O(N) multiplications plus O(spikes x fan-out); there is no scan of all synapses and no `exp` per neuron. The loops are serial (no threading).
- The kernels agree with Brian2 (relative weight difference 2e-6), Auryn (2e-7) and the analytic pair kernels.

For one pre/post pair with ``Δt = t_{post} - t_{pre}``, `STDPGerstner` gives

```math
Δw = \begin{cases} A_{pre}\, e^{-Δt/τ_{pre}} & Δt > 0 \\ A_{post}\, e^{Δt/τ_{post}} & Δt < 0 \end{cases}
```

and 0 if both spikes fall in the same step.

### Sign conventions

- `STDPGerstner` amplitudes are signed and applied once. `A_pre > 0` potentiates pre-before-post pairs; `A_post < 0` depresses post-before-pre pairs. The default is `A_pre = 1e-4`, `A_post = -1e-4`. Any sign combination is accepted (anti-Hebbian: `A_pre < 0 < A_post`).
- `STDPTriplet` amplitudes are all positive; the sign is explicit in the update (LTD subtracts, LTP adds).
- `STDPWeightDependent`: `η` and `α` are positive; LTD has the sign built into the update. `μ_plus = μ_minus = 0` is additive STDP with hard bounds, `1` multiplicative STDP.
- Units: time in ms, weights in the units of `W` (pF for conductance-based synapses in the default parameters). Rescale the amplitudes to the weight scale of your network.

!!! warning "`STDPGerstner` before SNNModels 1.8.2"
    The amplitudes were applied twice (effective ``A^2``, sign lost), so a negative `A_post` potentiated. Parameter sets tuned with older versions must be rescaled (old `A = 5e-2` is a new amplitude of `2.5e-3`). See [Release notes](release_notes.md).

## Rule reference

Docstrings of all rules and of their variable types are in the [API Reference](api_reference.md) (section
"Plasticity rules and variables"): `STDPGerstner`, `STDPTriplet`, `STDPTripletVariables`,
`STDPWeightDependent`, `STDPConfavreux2025`, `STDPMexicanHat`, `STDPVariables`.

The weight-update kernels are validated against analytic kernels, Brian2 and Auryn in the umbrella repository (`papers/JuliaSNN_publication/validation/stdp`). The tutorial `examples/tutorials/STDP_kernel.jl` plots the kernels of all rules.

## Inhibitory STDP

`iSTDPRate` and `iSTDPPotential` implement the inhibitory plasticity of Vogels et al. (2011) with Euler-integrated traces.

!!! danger "Bug fixed in SNNModels 1.8.2: `iSTDPRate` potentiation hit the wrong synapses"
    In `iSTDPRate` (and the former `iSTDPTime`) the potentiation applied at a postsynaptic spike used a `@turbo` loop that reassigned its loop variable. LoopVectorization ignored the reassignment, so potentiation was applied to the synapses stored at positions `rowptr[i]:rowptr[i+1]-1` of the column-ordered arrays (synapses onto unrelated postsynaptic neurons) instead of the synapses onto the spiking neuron `i`. The depression branch was correct and `iSTDPPotential` was not affected.

    Affected: SpikingNeuralNetworks.jl from commit 680a30c (2025-01-06, v1.0.0) and SNNModels 1.5.0 to 1.8.1. Simulations that used `iSTDPRate` or `iSTDPTime` with those versions give different results and should be rerun.

Docstrings: `iSTDPRate`, `iSTDPPotential`, `iSTDPTime`, `iSTDPVariables` in the [API Reference](api_reference.md).

## Hebbian Synaptic Plasticity

See the rules above and the voltage-based `vSTDPParameter`. Short-term plasticity (`MarkramSTPParameter`, ...) acts on the efficacy `ρ` and is also updated only under `train!`.

## Heterosynaptic Plasticity

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.SNNModels.AbstractMetaPlasticity 
```

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.SNNModels.MetaPlasticityParameter
```


```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.SNNModels.AbstractSpikingSynapseParameter
```
