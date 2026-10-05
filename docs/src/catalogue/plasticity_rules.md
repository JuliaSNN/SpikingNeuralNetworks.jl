# Plasticity rules

```@meta
CurrentModule = SNNModels
```

This page lists every synaptic plasticity rule of SNNModels 1.8.4 that acts on a sparse synapse
(`SpikingSynapse`): long-term rules, passed with the keyword `LTPParam`, and short-term rules,
passed with `STPParam`. For how to use them in a network see [Plasticity](../plasticity.md);
for the rules that act on whole connections (normalisation, aggregate scaling, turnover) see
[Metaplasticity](metaplasticity.md).

!!! warning "Plasticity runs only under `train!`"
    `train!` calls, for every connection and every time step, `update_traces!` before
    `forward!` and `plasticity!` after it. `sim!` calls neither: weights, traces and STP
    variables stay frozen.

## Common structure

A rule is a parameter object (subtype of `LTPParameter` or `STPParameter`) and a state
object created by `plasticityvariables(param, Npre, Npost)`:

| Rule (parameter type) | State type | Family | Traces |
|---|---|---|---|
| `STDPGerstner` | `STDPVariables` | pair STDP, additive | exact decay, event-driven |
| `STDPConfavreux2025` | `STDPVariables` | pair STDP with rate terms | exact decay, event-driven |
| `STDPWeightDependent` | `STDPVariables` | pair STDP, soft bounds | exact decay, event-driven |
| `STDPTriplet` | `STDPTripletVariables` | triplet STDP | exact decay, event-driven |
| `STDPMexicanHat` | `STDPVariables` | Mexican-hat kernel | Euler |
| `STDPSymmetric` | `STDPStructuredVariables` | symmetric difference of exponentials | Euler |
| `STDPAntiSymmetric` | `STDPStructuredVariables` | antisymmetric exponentials | Euler |
| `iSTDPRate` | `iSTDPVariables` | inhibitory STDP, target rate | Euler |
| `iSTDPPotential` | `iSTDPVariables` | inhibitory STDP, membrane potential | Euler |
| `iSTDPTime` | `iSTDPVariables` | parameters only, no update defined | - |
| `vSTDPParameter` | `vSTDPVariables` | voltage-based STDP | Euler |
| `MarkramSTPParameter` (= `MarkramSTPParameterEvent`) | `MarkramSTPVariables` | short-term plasticity | exact, event-driven |
| `MarkramSTPParameterHet` | `MarkramSTPVariables` | short-term plasticity, per-neuron parameters | exact, event-driven |
| `MarkramSTPParameterTimestep` | `MarkramSTPVariables` | short-term plasticity | Euler |
| `NoLTP`, `NoSTP` | `NoVariables` | no plasticity (defaults) | - |

Every state has an `active` flag; the rule runs only if it is `true`. Switch a rule with
`set_LTP!(syn, false)` / `set_STP!(syn, false)`, replace it with
`change_plasticity!(syn; LTP = ..., STP = ...)`.

Notation: synapse ``s`` connects presynaptic neuron ``j`` to postsynaptic neuron ``i``, its
weight is ``w_{ij}`` (`W[s]`), time is in ms and weights are in the units of `W` (pF for the
default conductance-based synapses). The traces of the event-driven rules jump by 1 at a spike.
Amplitudes must be scaled to the weight scale of the network.

## Event-driven implementation (Auryn ordering)

`STDPGerstner`, `STDPConfavreux2025`, `STDPWeightDependent` and `STDPTriplet` share the
kernels of `sparse_plasticity/STDP_kernels.jl`, which follow Auryn's `STDPConnection` /
`TripletConnection` (Zenke and Gerstner 2014). At step ``n`` (time ``t_n``), with `fireJ` the
presynaptic and `fireI` the postsynaptic spikes of the step, `plasticity!` does:

1. **Pre-spike pass.** For every presynaptic neuron ``j`` that fired, every outgoing
   synapse: ``w \leftarrow \mathrm{clamp}(w + f_{pre}(w, i, j), W_{min}, W_{max})``.
2. **Post-spike pass.** For every postsynaptic neuron ``i`` that fired, every incoming
   synapse: ``w \leftarrow \mathrm{clamp}(w + f_{post}(w, i, j), W_{min}, W_{max})``.
3. **Trace increment.** Every neuron that fired adds 1 to its trace(s).
4. **Trace decay.** Every trace is multiplied by ``e^{-dt/τ}`` (one precomputed factor per time
   constant; the decay is done every step, not lazily).

The traces read in steps 1 and 2 therefore hold
``\sum_{m<n} e^{-(t_n - t_m)/τ}`` over the *earlier* spikes only: a pre and a post spike in the
same step do not interact with each other (their earlier history does). This is the
convention of Auryn and of Brian2 code that updates `w` before incrementing the trace. Only
weights of neurons that fired are updated and clamped; all weights are clamped once on the
first plasticity step. The loops are serial. The kernels were validated against Brian2
(relative weight difference 2e-6), Auryn (2e-7) and the analytic pair kernels.

`STDPMexicanHat`, the structured rules, the iSTDP rules and `vSTDPParameter` integrate their
traces with forward Euler and use their own orderings, described in each section.

## STDPGerstner

Additive pair-based STDP with all-to-all spike interaction (Gerstner et al. 1996, Auryn
`STDPConnection`). Traces ``x_{pre}`` (``τ_{pre}``) and ``x_{post}`` (``τ_{post}``).

```math
\begin{aligned}
\text{presynaptic spike of } j:&\quad w_{ij} \mathrel{+}= A_{post}\, x_{post,i} \\
\text{postsynaptic spike of } i:&\quad w_{ij} \mathrel{+}= A_{pre}\, x_{pre,j}
\end{aligned}
```

For a single pair with ``Δt = t_{post} - t_{pre}``:

```math
Δw = \begin{cases} A_{pre}\, e^{-Δt/τ_{pre}} & Δt > 0 \\ A_{post}\, e^{Δt/τ_{post}} & Δt < 0 \end{cases}
```

and ``Δw = 0`` if both spikes fall in the same step. The amplitudes are signed and applied
once: `A_pre > 0` potentiates pre-before-post pairs, `A_post < 0` depresses post-before-pre
pairs (Hebbian, the default). Any sign combination is accepted.

| Field | Default | Units | Meaning |
|---|---|---|---|
| `A_post` | `-1e-4` | weight | change per unit ``x_{post}`` at a presynaptic spike (`< 0`: LTD) |
| `A_pre` | `1e-4` | weight | change per unit ``x_{pre}`` at a postsynaptic spike (`> 0`: LTP) |
| `τpre` | `20ms` | ms | presynaptic trace time constant |
| `τpost` | `20ms` | ms | postsynaptic trace time constant |
| `Wmax` | `30pF` | weight | upper bound |
| `Wmin` | `0pF` | weight | lower bound |

!!! warning "Behaviour change in SNNModels 1.8.2"
    Before 1.8.2 the amplitudes were applied twice (effective ``A^2``, sign lost) and the default
    `A_post` was positive. Now each amplitude is applied once and the default `A_post` is `-1e-4`.
    Rescale old parameter sets: old `A = 5e-2` is a new amplitude of `2.5e-3`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
rule = SNN.STDPGerstner(A_pre = 1e-2, A_post = -1.05e-2, τpre = 16.8ms, τpost = 33.7ms, Wmax = 50)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 10.0), LTPParam = rule)
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 1s)
```

## STDPConfavreux2025

Pair-based STDP with constant rate terms (named "Confavreux et al. 2025" in the code; full
reference not given in the code). Same traces and ordering as `STDPGerstner`.

```math
\begin{aligned}
\text{presynaptic spike of } j:&\quad w_{ij} \mathrel{+}= η\,(κ\, x_{post,i} + α) \\
\text{postsynaptic spike of } i:&\quad w_{ij} \mathrel{+}= η\,(γ\, x_{pre,j} + β)
\end{aligned}
```

For one pair, ``Δw = ηγ e^{-Δt/τ_{pre}}`` (``Δt > 0``) and ``Δw = ηκ e^{Δt/τ_{post}}``
(``Δt < 0``); every presynaptic spike also adds ``ηα`` and every postsynaptic spike ``ηβ``.
The signs are carried by `κ`, `γ`, `α`, `β`; with the defaults both pairings potentiate.

| Field | Default | Units | Meaning |
|---|---|---|---|
| `η` | `0.01` | weight | learning rate |
| `α` | `0` | - | constant term at each presynaptic spike |
| `β` | `0` | - | constant term at each postsynaptic spike |
| `κ` | `1` | - | weight of the post-before-pre term (read at a pre spike) |
| `γ` | `1` | - | weight of the pre-before-post term (read at a post spike) |
| `τpre`, `τpost` | `20ms` | ms | trace time constants |
| `Wmin`, `Wmax` | `0pF`, `30pF` | weight | bounds |

```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
rule = SNN.STDPConfavreux2025(η = 1e-3, κ = -1.0, γ = 1.0, α = -0.1)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 1.0), LTPParam = rule)
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 500ms)
```

## STDPWeightDependent

Weight-dependent (soft-bound) pair STDP (Gütig et al. 2003; Morrison, Diesmann and Gerstner
2008; Auryn `STDPwdConnection`). With ``\tilde W = W_{max} - W_{min}``:

```math
\begin{aligned}
\text{presynaptic spike (LTD)}:&\quad w \mathrel{-}= η\, α\, \tilde W^{1-μ_-}\, (w - W_{min})^{μ_-}\, x_{post} \\
\text{postsynaptic spike (LTP)}:&\quad w \mathrel{+}= η\, \tilde W^{1-μ_+}\, (W_{max} - w)^{μ_+}\, x_{pre}
\end{aligned}
```

(the factors ``(w - W_{min})`` and ``(W_{max} - w)`` are floored at 0). `μ = 0` is additive
STDP with hard bounds and amplitudes ``η \tilde W`` and ``-ηα\tilde W``; `μ = 1` is
multiplicative STDP. With `Wmin = 0` the rule equals Auryn's `STDPwdConnection`
(`learning_rate = η`, `param_alpha = α`).

| Field | Default | Units | Meaning |
|---|---|---|---|
| `η` | `1e-3` | - | relative learning rate |
| `α` | `1` | - | LTD/LTP asymmetry |
| `μ_plus` | `1` | - | LTP weight-dependence exponent |
| `μ_minus` | `1` | - | LTD weight-dependence exponent |
| `τpre`, `τpost` | `20ms` | ms | trace time constants |
| `Wmax`, `Wmin` | `30pF`, `0pF` | weight | bounds |

```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
rule = SNN.STDPWeightDependent(η = 1e-3, α = 1.05, μ_plus = 0.0, μ_minus = 1.0)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 10.0), LTPParam = rule)
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 500ms)
```

## STDPTriplet

All-to-all triplet STDP (Pfister and Gerstner 2006; Auryn `MinimalTripletConnection` /
`TripletConnection`). Presynaptic traces ``r_1`` (``τ_+``), ``r_2`` (``τ_x``), postsynaptic
traces ``o_1`` (``τ_-``), ``o_2`` (``τ_y``), all read before this step's spikes are added
(the ``t - ε`` of the paper):

```math
\begin{aligned}
\text{presynaptic spike of } j:&\quad w_{ij} \mathrel{-}= o_{1,i}\,(A_2^- + A_3^-\, r_{2,j}) \\
\text{postsynaptic spike of } i:&\quad w_{ij} \mathrel{+}= r_{1,j}\,(A_2^+ + A_3^+\, o_{2,i})
\end{aligned}
```

The amplitudes are positive and the signs explicit. The defaults are documented in the code as
the minimal all-to-all model fitted to the hippocampal culture data (Table 4 of Pfister and
Gerstner 2006), equal to Auryn's `MinimalTripletConnection` with `eta = 1`; with
`A3_minus = 0` the trace ``r_2`` (and `τ_x`) has no effect.

| Field | Default | Units | Meaning |
|---|---|---|---|
| `A2_plus` | `5.3e-3` | weight | pair LTP amplitude |
| `A3_plus` | `8e-3` | weight | triplet LTP amplitude |
| `A2_minus` | `3.5e-3` | weight | pair LTD amplitude |
| `A3_minus` | `0` | weight | triplet LTD amplitude |
| `τ_plus` | `16.8ms` | ms | presynaptic trace ``r_1`` |
| `τ_minus` | `33.7ms` | ms | postsynaptic trace ``o_1`` |
| `τ_x` | `946ms` | ms | presynaptic trace ``r_2`` |
| `τ_y` | `40ms` | ms | postsynaptic trace ``o_2`` |
| `Wmax`, `Wmin` | `30pF`, `0pF` | weight | bounds |

```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
rule = SNN.STDPTriplet(A2_plus = 5.3e-2, A3_plus = 8e-2, A2_minus = 3.5e-2)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 10.0), LTPParam = rule)
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 500ms)
```

## STDPMexicanHat

Kernel ``Δw = A\,(1 - z)\,e^{-z/\sqrt 2}`` with ``z = [\ln(x_{pre}/x_{post})]^2``, where the pre-
and postsynaptic traces have the same time constant ``τ``. For an isolated pair
``z ≈ (Δt/τ)^2``: potentiation for ``|Δt| < τ``, depression beyond. Reference not given in the
code.

Update order per step: (1) Euler decay of all traces, then +1 for the neurons that fired (so
same-step spikes interact); (2) pre-spike pass and (3) post-spike pass, each applying ``Δw`` to
every synapse of a firing neuron whose two traces are non-zero (a `NaN` kernel value is
replaced by 0); (4) clamp of the touched weights.

| Field | Default | Units | Meaning |
|---|---|---|---|
| `A` | `0.1` | weight | amplitude |
| `τ` | `20ms` | ms | trace time constant |
| `Wmax`, `Wmin` | `30pF`, `0pF` | weight | bounds |

```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 1.0), LTPParam = SNN.STDPMexicanHat(A = 1e-3))
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 500ms)
```

## STDPSymmetric and STDPAntiSymmetric

Structured rules cited in the code from Festa, Cusseddu and Gjorgjieva (2024), "Structured
stabilization in recurrent neural circuits through inhibitory synaptic plasticity". Both have
pair kernels whose integral is ``A_x - A_y`` (zero with the defaults) and constant terms
`αpre`, `αpost` added at every pre / post spike.

Symmetric kernel (``Δt = t_{post} - t_{pre}``):

```math
Δw(Δt) = \frac{A_x}{2τ_x} e^{-|Δt|/τ_x} - \frac{A_y}{2τ_y} e^{-|Δt|/τ_y}
```

Antisymmetric kernel:

```math
Δw(Δt) = \begin{cases} \dfrac{A_x}{τ_x} e^{-Δt/τ_x} & Δt > 0 \\[1ex] -\dfrac{A_y}{τ_y} e^{Δt/τ_y} & Δt < 0 \end{cases}
```

Implementation: per step (1) pre-spike pass and (2) post-spike pass using the traces before
this step's spikes, (3) Euler decay of all traces then +1 for the neurons that fired, (4)
clamp of the touched weights (all weights once on the first step).

| Field | `STDPSymmetric` | `STDPAntiSymmetric` | Units | Meaning |
|---|---|---|---|---|
| `A_x` | `3e-2` | `3e-2` | weight·ms | narrow / potentiation amplitude |
| `A_y` | `3e-2` | `3e-2` | weight·ms | wide / depression amplitude |
| `τ_x` | `50ms` | `50ms` | ms | time constant of the ``x`` traces |
| `τ_y` | `500ms` | `50ms` | ms | time constant of the ``y`` traces |
| `αpre`, `αpost` | `0pF` | `0pF` | weight | constant change per pre / post spike |
| `Wmax`, `Wmin` | `30pF`, `0pF` | `30pF`, `0pF` | weight | bounds |

```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 1.0),
                         LTPParam = SNN.STDPAntiSymmetric(A_x = 1e-2, A_y = 1e-2))
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 500ms)
```

## Inhibitory STDP: iSTDPRate, iSTDPPotential, iSTDPTime

Rules of Vogels et al. (2011), with Euler-integrated traces (time constant ``τ_y``) that jump
by 1 at a spike.

`iSTDPRate` (target rate ``r``, ``α = 2 r τ_y``):

```math
Δw_{ij} = η\,(y_i - α) \ \text{at a presynaptic spike}, \qquad Δw_{ij} = η\, x_j \ \text{at a postsynaptic spike}
```

`iSTDPPotential` replaces the postsynaptic spike trace by a low-pass filter of the membrane
potential, ``τ_y\, dy_i/dt = V_i - y_i``, and the target by a reference potential ``v_0``:
``Δw_{ij} = η (y_i - v_0)`` at a presynaptic spike, ``Δw_{ij} = η x_j`` at a postsynaptic spike.
The trace starts at 0 mV.

Update order per step: presynaptic pass (Euler decay of ``x_j``, +1 if ``j`` fired, then the
pre-spike update of its outgoing synapses), then postsynaptic pass (Euler update of ``y_i``, +1
if ``i`` fired for `iSTDPRate`, then the post-spike update of its incoming synapses). Every
touched weight is clamped to `[Wmin, Wmax]`. Because the presynaptic trace is incremented
before the postsynaptic pass, a pre and a post spike in the same step interact.

`iSTDPTime` holds only parameters: no `plasticity!` method is defined for it and it cannot
be used as `LTPParam`.

| Field | `iSTDPRate` | `iSTDPPotential` | `iSTDPTime` | Units | Meaning |
|---|---|---|---|---|---|
| `η` | `0.01pA` | `0.001pA` | `0.01pA` | weight | learning rate |
| `r` | `3Hz` | - | - | 1/ms | target postsynaptic rate |
| `v0` | - | `-50mV` | - | mV | reference potential |
| `τy` | `50ms` | `200ms` | `50ms` | ms | trace time constant |
| `Wmax` | `243pF` | `243pF` | `243pF` | weight | upper bound |
| `Wmin` | `0.01pF` | `0.01pF` | `0.01pF` | weight | lower bound |

!!! danger "Bug fixed in SNNModels 1.8.2"
    In SNNModels 1.5.0 - 1.8.1 the potentiation of `iSTDPRate` (and of the former `iSTDPTime`
    update) at a postsynaptic spike was applied to the synapses stored at CSC positions
    `rowptr[i]:rowptr[i+1]-1` instead of the incoming synapses of the spiking neuron. Since 1.8.2
    it walks the incoming synapses of `i` (`s = index[k]` for `k` in `rowptr[i]:rowptr[i+1]-1`).
    See the [Release notes](../release_notes.md).

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 80)
I = SNN.Poisson(N = 20, param = SNN.PoissonParameter(20Hz))
IE = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 5.0), LTPParam = SNN.iSTDPRate(r = 5Hz))
SNN.train!(model = SNN.compose(; E, I, IE), duration = 500ms)
```

## vSTDPParameter

Voltage-based STDP in the form of Clopath et al. (2010). ``V_i`` is the postsynaptic membrane
potential (`v_post` of the synapse), ``u_i`` and ``v_i`` are low-pass filters of it, ``x_j`` is a
presynaptic trace and ``[z]_+ = \max(z, 0)``:

```math
\begin{aligned}
τ_x \frac{dx_j}{dt} &= -x_j + S_j, \qquad τ_u \frac{du_i}{dt} = V_i - u_i, \qquad τ_v \frac{dv_i}{dt} = V_i - v_i \\
Δw_{ij} &= -A_{LTD}\,[u_i - θ_{LTD}]_+ \quad \text{at each presynaptic spike} \\
Δw_{ij} &= A_{LTP}\, x_j\,[v_i - θ_{LTD}]_+\,[V_i - θ_{LTP}]_+ \quad \text{at every step}
\end{aligned}
```

``S_j`` is 1 in the step in which ``j`` fires, so a spike increases ``x_j`` by ``dt/τ_x``. Update
order per step: Euler update of ``x``, then of ``u`` and ``v``; for each presynaptic neuron
(threaded over chunks), LTD if it fired with clamping at `Wmin`, then the LTP term on all its
outgoing synapses with clamping at `Wmax`. The LTP increment is added once per step (it is not
multiplied by `dt`) and the traces start at 0 mV.

| Field | Default | Units | Meaning |
|---|---|---|---|
| `A_LTD` | `8e-4` | weight/mV | LTD amplitude |
| `A_LTP` | `1.4e-3` | weight/mV² | LTP amplitude |
| `θ_LTD` | `-70mV` | mV | LTD threshold (also applied to ``v`` in the LTP term) |
| `θ_LTP` | `-49mV` | mV | LTP threshold on ``V`` |
| `τu` | `20ms` | ms | LTD filter time constant |
| `τv` | `7ms` | ms | LTP filter time constant |
| `τx` | `15ms` | ms | presynaptic trace time constant |
| `Wmax`, `Wmin` | `30pF`, `0.1pF` | weight | bounds |

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
P = SNN.AdEx(N = 10, param = SNN.AdExParameter())
syn = SNN.SpikingSynapse(E, P, :ge; conn = (p = 0.2, μ = 1.0), LTPParam = SNN.vSTDPParameter())
SNN.train!(model = SNN.compose(; E, P, syn), duration = 200ms)
```

## Short-term plasticity: Tsodyks-Markram

Each presynaptic neuron ``j`` has a utilisation ``u_j`` and available resources ``x_j``; the
efficacy of its outgoing synapses is ``ρ_s = u_j x_j`` and a spike adds ``W_s ρ_s`` to the target
conductance (Tsodyks and Markram 1997; Markram, Wang and Tsodyks 1998; the code follows the
formulation of Mongillo, Barak and Tsodyks 2008). Between spikes

```math
\frac{du}{dt} = \frac{U - u}{τ_F}, \qquad \frac{dx}{dt} = \frac{1 - x}{τ_D}.
```

**`MarkramSTPParameter` (alias of `MarkramSTPParameterEvent`) and `MarkramSTPParameterHet`**:
event-driven and exact. At a presynaptic spike, with ``Δ`` the interval from the previous
spike of the same neuron,

```math
\begin{aligned}
u^- &= U - (U - u)\,e^{-Δ/τ_F}, & x^- &= 1 - (1 - x)\,e^{-Δ/τ_D}, \\
ρ &= u^- x^-, & & \\
u &\leftarrow u^- + U\,(1 - u^-), & x &\leftarrow x^- - u\,x^- .
\end{aligned}
```

This is done in `update_traces!`, i.e. before `forward!`, so the spike of the current step is
transmitted with the new efficacy; the first spike of a neuron is transmitted with ``ρ = U``.
`MarkramSTPParameterHet` uses per-presynaptic-neuron vectors `U[j]`, `τF[j]`, `τD[j]` (no
defaults, no `Wmax`/`Wmin`).

**`MarkramSTPParameterTimestep`**: clock-driven. In `plasticity!` (after `forward!`): jumps
``u \leftarrow u + U(1-u)``, ``x \leftarrow x - u x`` for the neurons that fired, Euler relaxation of
``u`` and ``x`` for all neurons, then ``ρ = u x``. The spike of step ``n`` is transmitted with the
efficacy of step ``n - 1``.

| Field | Default | Units | Meaning |
|---|---|---|---|
| `τD` | `200ms` | ms | recovery (depression) time constant |
| `τF` | `1500ms` | ms | facilitation time constant |
| `U` | `0.2` | - | baseline utilisation |
| `Wmax`, `Wmin` | `1pF`, `0pF` | weight | not used by the update |

Under `sim!` the efficacy `ρ` is not updated and keeps its current value (1 for a new synapse).

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
P = SNN.IF(N = 10)
syn = SNN.SpikingSynapse(E, P, :ge; conn = (p = 0.2, μ = 1.0),
                         STPParam = SNN.MarkramSTPParameter(τD = 200ms, τF = 1500ms, U = 0.2))
SNN.train!(model = SNN.compose(; E, P, syn), duration = 500ms)
extrema(syn.ρ)
```

## Files present but not loaded

The source tree contains two files under `connections/sparse_plasticity/` that are not
included by SNNModels 1.8.4: `CaRule.jl` (an unfinished calcium-based rule, `CaPlasticityParameter`)
and `dump.jl` (draft types for triplet and nonlinear STDP rules). They define nothing usable.

## API

### Common interface

```@autodocs
Modules = [SNNModels]
Pages   = ["connections/sparse_plasticity.jl"]
```

### Event-driven kernels (internal)

```@autodocs
Modules = [SNNModels]
Pages   = ["sparse_plasticity/STDP_kernels.jl"]
```

### Pair STDP rules

```@autodocs
Modules = [SNNModels]
Pages   = ["sparse_plasticity/STDP_traces.jl", "sparse_plasticity/STDP_weight_dependent.jl"]
```

### Triplet STDP

```@autodocs
Modules = [SNNModels]
Pages   = ["sparse_plasticity/STDP_triplet.jl"]
```

### Structured STDP

```@autodocs
Modules = [SNNModels]
Pages   = ["sparse_plasticity/STDP_structured.jl"]
```

### Inhibitory STDP

```@autodocs
Modules = [SNNModels]
Pages   = ["sparse_plasticity/iSTDP.jl"]
```

### Voltage-based STDP

```@autodocs
Modules = [SNNModels]
Pages   = ["sparse_plasticity/vSTDP.jl"]
```

### Short-term plasticity

```@autodocs
Modules = [SNNModels]
Pages   = ["sparse_plasticity/STP.jl"]
```
