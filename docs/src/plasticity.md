# Plasticity

```@meta
CurrentModule = SpikingNeuralNetworks
```

Synapses (`SpikingSynapse`) can carry a long-term plasticity rule (keyword `LTPParam`, acting on the weights `W`) and a short-term plasticity rule (keyword `STPParam`, acting on the efficacy `ρ`). The equations, parameters and defaults of every rule are in the catalogue page [Plasticity rules](catalogue/plasticity_rules.md); this page explains how plasticity is run.

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
    `train!` calls `update_traces!` (before `forward!`) and `plasticity!` (after `forward!`) of every connection at each step; `sim!` never does. A synapse with an `LTPParam` or `STPParam` keeps its weights (and its STP variables and efficacy `ρ`) constant under `sim!`. Use `train!` to learn and `sim!` to test with frozen weights.

## Switching and replacing rules

- `SNN.set_LTP!(syn, false)` / `SNN.set_STP!(syn, false)` deactivate the long-term / short-term rule of a synapse (they set the `active` flag of `syn.LTPVars` / `syn.STPVars`); `true` reactivates it. They are no-ops for connections that are not sparse synapses.
- `SNN.change_plasticity!(syn; LTP = rule, STP = rule)` replaces a rule and re-creates its state (traces are reset).
- The traces are fields of `syn.LTPVars` / `syn.STPVars` and can be recorded, e.g. `SNN.monitor!(syn, [:tpre, :tpost], :LTPVars)` for the STDP rules (see [Recordings](recordings.md)).

!!! danger "Results obtained with `iSTDPRate` before SNNModels 1.8.2 are wrong"
    See [Inhibitory STDP](#Inhibitory-STDP) below and [Release notes](release_notes.md).

## Rules at a glance

| Rule | Type | Traces | Notes |
|---|---|---|---|
| `STDPGerstner` | pair, additive | exact decay | signed amplitudes, `A_post < 0` is LTD (default) |
| `STDPTriplet` | triplet, all-to-all | exact decay (4 traces) | Pfister and Gerstner 2006, Auryn `MinimalTriplet` |
| `STDPWeightDependent` | pair, soft bounds | exact decay | Gütig et al. 2003, Auryn `STDPwd`; `η` is a relative rate |
| `STDPConfavreux2025` | pair with rate terms | exact decay | |
| `STDPMexicanHat` | Mexican-hat kernel in Δt | Euler | event-driven weight passes; same-step spikes interact |
| `STDPSymmetric`, `STDPAntiSymmetric` | structured inhibition (Festa et al. 2024) | Euler | |
| `iSTDPRate`, `iSTDPPotential` | inhibitory (Vogels et al. 2011) | Euler | same-step pre and post spikes interact |
| `iSTDPTime` | - | - | parameters only, no update defined |
| `vSTDPParameter` | voltage-based (Clopath et al. 2010) | Euler | LTP applied every step |
| `MarkramSTPParameter`, `MarkramSTPParameterHet` | short-term (Tsodyks-Markram) | exact, event-driven | updated in `update_traces!` |
| `MarkramSTPParameterTimestep` | short-term (Tsodyks-Markram) | Euler | |

## Event-driven pair and triplet STDP

`STDPGerstner`, `STDPConfavreux2025`, `STDPWeightDependent` and `STDPTriplet` share event-driven kernels that follow the ordering of Auryn (Zenke and Gerstner 2014). Per time step, with `fireJ` the presynaptic and `fireI` the postsynaptic spikes of this step:

1. **Pre-spike pass.** For every presynaptic spike, walk its outgoing synapses and apply the post-before-pre term, computed from the postsynaptic trace of the target neuron (LTD with the default signs).
2. **Post-spike pass.** For every postsynaptic spike, walk its incoming synapses and apply the pre-before-post term, computed from the presynaptic trace of the source neuron (LTP with the default signs).
3. **Trace increment.** Every neuron that fired adds 1 to its trace(s).
4. **Trace decay.** Every trace is multiplied by `exp(-dt/τ)` (exact decay, one precomputed factor per time constant).

Consequences:

- Traces are read *before* the spikes of the current step are added: a trace holds ``\sum_{m<n} e^{-(t_n-t_m)/τ}`` over the *earlier* spikes of the neuron. A pre and a post spike in the same step do not interact with each other (the earlier history of each still does). This is the convention of Auryn and Brian2 (`w` updated before the trace increment).
- Only the weights touched in the step (the synapses of neurons that fired) are updated and clamped to `[Wmin, Wmax]`. All weights are clamped once at the first step.
- The trace decay is applied to every neuron at every step (not lazily at spikes).
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
- `STDPConfavreux2025`: the signs are carried by `κ`, `γ`, `α`, `β`; with the defaults both pairings potentiate.
- Units: time in ms, weights in the units of `W` (pF for conductance-based synapses in the default parameters). Rescale the amplitudes to the weight scale of your network.

!!! warning "`STDPGerstner` before SNNModels 1.8.2"
    The amplitudes were applied twice (effective ``A^2``, sign lost), so a negative `A_post` potentiated. Parameter sets tuned with older versions must be rescaled (old `A = 5e-2` is a new amplitude of `2.5e-3`). See [Release notes](release_notes.md).

## Rule reference

Equations, parameter tables, defaults and the API of all rules and of their variable types are in [Plasticity rules](catalogue/plasticity_rules.md).

The weight-update kernels are validated against analytic kernels, Brian2 and Auryn in the umbrella repository (`papers/JuliaSNN_publication/validation/stdp`). The tutorial `examples/tutorials/STDP_kernel.jl` plots the kernels of all rules.

## Inhibitory STDP

`iSTDPRate` and `iSTDPPotential` implement the inhibitory plasticity of Vogels et al. (2011) with Euler-integrated traces.

!!! danger "Bug fixed in SNNModels 1.8.2: `iSTDPRate` potentiation hit the wrong synapses"
    In `iSTDPRate` (and the former `iSTDPTime`) the potentiation applied at a postsynaptic spike used a `@turbo` loop that reassigned its loop variable. LoopVectorization ignored the reassignment, so potentiation was applied to the synapses stored at positions `rowptr[i]:rowptr[i+1]-1` of the column-ordered arrays (synapses onto unrelated postsynaptic neurons) instead of the synapses onto the spiking neuron `i`. The depression branch was correct and `iSTDPPotential` was not affected.

    Affected: SpikingNeuralNetworks.jl from commit 680a30c (2025-01-06, v1.0.0) and SNNModels 1.5.0 to 1.8.1. Simulations that used `iSTDPRate` or `iSTDPTime` with those versions give different results and should be rerun.

`iSTDPTime` holds parameters only: no update is defined for it, so it cannot be used as `LTPParam`. Equations and parameters: [Plasticity rules](catalogue/plasticity_rules.md).

## Voltage-based and short-term plasticity

`vSTDPParameter` implements voltage-based STDP (Clopath et al. 2010). Short-term plasticity (`MarkramSTPParameter`, `MarkramSTPParameterHet`, `MarkramSTPParameterTimestep`) acts on the efficacy `ρ` of each synapse: a presynaptic spike adds `W[s] * ρ[s]` to the target. It is also updated only under `train!`; the event-driven variants update `ρ` in `update_traces!`, before the spike is transmitted. See [Plasticity rules](catalogue/plasticity_rules.md).

## Heterosynaptic plasticity and metaplasticity

Rules that act on whole connections rather than on single synapses (synaptic normalisation, aggregate scaling, synaptic turnover) are separate connection objects added to the model; they are described in [Metaplasticity](catalogue/metaplasticity.md).
