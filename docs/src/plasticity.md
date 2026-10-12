# Plasticity

```@meta
CurrentModule = SpikingNeuralNetworks
```

After this page you can add long- and short-term plasticity to a synapse, switch rules on, off or
to another rule, and read weights and traces correctly. Equations and parameters of every rule are
in [Plasticity rules](catalogue/plasticity_rules.md).

```julia
using SpikingNeuralNetworks
SNN.@load_units

E = SNN.IF(N = 400)
I = SNN.IF(N = 100)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.1, μ = 3.0),
                        LTPParam = SNN.STDPGerstner(A_pre = 1e-2, A_post = -1.05e-2, Wmax = 20pF))
IE = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 5.0), LTPParam = SNN.iSTDPRate(r = 5Hz))
drive = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), E, :ge)
model = SNN.compose(; E, I, EE, IE, drive, silent = true)

SNN.train!(model, 1s)   # weights change
SNN.sim!(model, 1s)     # weights frozen
```

`LTPParam` acts on the weights `W`, `STPParam` on the efficacy `ρ` (a presynaptic spike adds
`W * ρ` to the target). Plasticity runs only under `train!`: `sim!` never changes weights or
short-term variables. Rules that act on whole connections (normalisation, scaling, turnover) are
separate components, see [Metaplasticity](catalogue/metaplasticity.md).

## Rules

| Rule | Kind |
|:-----|:-----|
| `STDPGerstner` | pair STDP, additive |
| `STDPTriplet` | triplet STDP (Pfister and Gerstner 2006) |
| `STDPWeightDependent` | pair STDP with soft bounds (Gütig et al. 2003) |
| `STDPConfavreux2025` | pair STDP with rate terms |
| `STDPMexicanHat` | Mexican-hat kernel in ``Δt`` |
| `STDPSymmetric`, `STDPAntiSymmetric` | structured inhibitory plasticity (Festa et al. 2024) |
| `iSTDPRate`, `iSTDPPotential` | inhibitory plasticity (Vogels et al. 2011) |
| `vSTDPParameter` | voltage-based STDP (Clopath et al. 2010), event-driven |
| `vSTDPDenseParameter` | the same rule, clock-driven |
| `MarkramSTPParameter`, `MarkramSTPParameterHet`, `MarkramSTPParameterTimestep` | short-term plasticity (Tsodyks-Markram) |

## Switching rules

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 0.5nS))
drive = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), E, :ge)
model = SNN.compose(; E, EE, drive, silent = true)

model = SNN.change_plasticity(model, :EE; LTP = SNN.STDPGerstner())       # new model, plastic EE
SNN.train!(model, 500ms)
SNN.set_LTP!(model.syn.EE, false)                                         # pause the rule
SNN.update_plasticity!(model.syn.EE; LTP = SNN.STDPGerstner(A_pre = 2e-4)) # same type, new parameters
model = SNN.change_plasticity(model, :EE; LTP = SNN.NoLTP())              # weights frozen for good
```

The rule types are type parameters of `SpikingSynapse`, so a rule of another type needs a new
synapse:

- `with_plasticity(syn; LTP, STP)` returns a new synapse with the same connectivity and weights;
  a rule left to `nothing` is kept, `NoLTP()` / `NoSTP()` switch that part off.
  `without_plasticity(syn)` switches both off.
- `change_plasticity(model, :name; LTP, STP)` returns a new model with that synapse replaced.

The old and the new synapse share their weights: use only the new one. In place,
`update_plasticity!(syn; LTP, STP)` accepts a rule of the current type (new parameters, fresh
state), and `set_LTP!(syn, false)` / `set_STP!(syn, false)` pause a rule.

## Reading weights and traces

Some rules store their state lazily: the STDP traces are kept as a value times a common scale,
and `vSTDPParameter` writes potentiation into `W` only at presynaptic spikes. Read them through
the accessors:

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 0.5nS), LTPParam = SNN.STDPGerstner())
drive = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), E, :ge)
model = SNN.compose(; E, EE, drive, silent = true)
SNN.train!(model, 200ms)

W = SNN.plasticity_value(EE, :W)              # current weights
tpre = SNN.trace_values(EE, :tpre)            # current presynaptic traces
SNN.sync_traces!(EE)                          # or: write the values into the stored fields
EE.LTPVars.tpre == tpre
```

`monitor!`, `record`, `matrix` and `save_model` already read through the accessors.

## Pair and triplet STDP

`STDPGerstner`, `STDPConfavreux2025`, `STDPWeightDependent` and `STDPTriplet` follow the
ordering of Auryn and Brian2. In each step:

1. for each presynaptic spike, the post-before-pre term is applied to its outgoing synapses;
2. for each postsynaptic spike, the pre-before-post term is applied to its incoming synapses;
3. the neurons that fired add 1 to their traces, and all traces decay by `exp(-dt/τ)`.

Traces are read before the spikes of the step are added, so a pre and a post spike in the same
step do not interact. Only the weights of neurons that fired are updated and clamped to
`[Wmin, Wmax]`. For one pair with ``Δt = t_{post} - t_{pre}``, `STDPGerstner` gives

```math
Δw = \begin{cases} A_{pre}\, e^{-Δt/τ_{pre}} & Δt > 0 \\ A_{post}\, e^{Δt/τ_{post}} & Δt < 0 \end{cases}
```

Signs: `STDPGerstner` amplitudes are signed (`A_pre > 0` potentiates pre-before-post, `A_post < 0`
depresses post-before-pre; defaults `1e-4`, `-1e-4`). `STDPTriplet` and `STDPWeightDependent` use
positive amplitudes with the sign in the update; `STDPConfavreux2025` carries the signs in `κ`, `γ`,
`α`, `β`.
Amplitudes are in the units of `W`; rescale them to the weights of your network.

## Inhibitory, voltage-based and short-term plasticity

- `iSTDPRate` and `iSTDPPotential` (Vogels et al. 2011) use Euler-integrated traces. Results of
  `iSTDPRate` with SNNModels 1.5.0 to 1.8.1 are wrong (see [Release notes](release_notes.md)).
- `vSTDPParameter` (Clopath et al. 2010) is event-driven: read its weights with
  `plasticity_value(syn, :W)`. `vSTDPDenseParameter` updates `W` every step and reproduces the
  rule of SNNModels 1.9.
- Short-term plasticity acts on the efficacy `ρ`; the event-driven Markram rules update it before
  the spike is transmitted.
