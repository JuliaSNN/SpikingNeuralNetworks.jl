# Plasticity rules

```@meta
CurrentModule = SNNModels
```

Long-term rules (`LTPParam`, act on `W`) and short-term rules (`STPParam`, act on `ρ`) of
`SpikingSynapse`; they run only under `train!`. Usage: [Plasticity](../plasticity.md). Implementation details to check when reproducing a
published result (trace and amplitude conventions, update order, clamping, lazy weights): [Plasticity: conventions and implementation notes](plasticity_conventions.md).

## Common structure


| Rule (parameter type) | State type | Family | Reference | Integration |
|---|---|---|---|---|
| `STDPGerstner` | `STDPVariables` | pair STDP, additive | Gerstner et al. (1996) | event-driven, traces exact decay |
| `STDPConfavreux2025` | `STDPVariables` | pair STDP with rate terms | citation not recorded | event-driven, traces exact decay |
| `STDPWeightDependent` | `STDPVariables` | pair STDP, soft bounds | Gütig et al. (2003) | event-driven, traces exact decay |
| `STDPTriplet` | `STDPTripletVariables` | triplet STDP | Pfister and Gerstner (2006) | event-driven, traces exact decay |
| `STDPMexicanHat` | `STDPVariables` | Mexican-hat kernel | none | event-driven, traces forward Euler |
| `STDPSymmetric` | `STDPStructuredVariables` | symmetric difference of exponentials | Festa et al. (2024) | event-driven, traces forward Euler |
| `STDPAntiSymmetric` | `STDPStructuredVariables` | antisymmetric exponentials | Festa et al. (2024) | event-driven, traces forward Euler |
| `iSTDPRate` | `iSTDPVariables` | inhibitory STDP, target rate | Vogels et al. (2011) | event-driven, traces forward Euler |
| `iSTDPPotential` | `iSTDPVariables` | inhibitory STDP, membrane potential | Vogels et al. (2011) | clock-driven, traces forward Euler |
| `vSTDPParameter` | `vSTDPVariables` | voltage-based STDP, lazy weights | Clopath et al. (2010) | forward Euler, event-driven evaluation |
| `vSTDPDenseParameter` | `vSTDPDenseVariables` | voltage-based STDP, clock-driven | Clopath et al. (2010) | forward Euler, every step |
| `MarkramSTPParameter` (= `MarkramSTPParameterEvent`), `MarkramSTPParameterHet` | `MarkramSTPVariables` | short-term plasticity (`Het`: per-neuron parameters) | Tsodyks and Markram (1997); Mongillo et al. (2008) | exact solution at presynaptic spikes |
| `MarkramSTPParameterTimestep` | `MarkramSTPVariables` | short-term plasticity | same | forward Euler, every step |
| `NoLTP`, `NoSTP` | `NoVariables` | no plasticity (defaults) | - | - |

References (full citations):

- Gerstner, Kempter, van Hemmen and Wagner (1996), Nature 383:76-78, [10.1038/383076a0](https://doi.org/10.1038/383076a0).
- Gütig, Aharonov, Rotter and Sompolinsky (2003), J. Neurosci. 23:3697-3714, [10.1523/JNEUROSCI.23-09-03697.2003](https://doi.org/10.1523/JNEUROSCI.23-09-03697.2003).
- Pfister and Gerstner (2006), J. Neurosci. 26:9673-9682, [10.1523/JNEUROSCI.1425-06.2006](https://doi.org/10.1523/JNEUROSCI.1425-06.2006).
- Clopath, Büsing, Vasilaki and Gerstner (2010), Nat. Neurosci. 13:344-352, [10.1038/nn.2479](https://doi.org/10.1038/nn.2479).
- Vogels, Sprekeler, Zenke, Clopath and Gerstner (2011), Science 334:1569-1573, [10.1126/science.1211095](https://doi.org/10.1126/science.1211095).
- Festa, Cusseddu and Gjorgjieva (2024), Structured stabilization in recurrent neural circuits through inhibitory synaptic plasticity, bioRxiv, [10.1101/2024.10.12.618014](https://doi.org/10.1101/2024.10.12.618014).
- Tsodyks and Markram (1997), PNAS 94:719-723, [10.1073/pnas.94.2.719](https://doi.org/10.1073/pnas.94.2.719); Markram, Wang and Tsodyks (1998), PNAS 95:5323-5328, [10.1073/pnas.95.9.5323](https://doi.org/10.1073/pnas.95.9.5323);
  Mongillo, Barak and Tsodyks (2008), Science 319:1543-1546, [10.1126/science.1150769](https://doi.org/10.1126/science.1150769).

Synapse ``s`` connects ``j`` to ``i`` with weight ``w_{ij}`` (`W[s]`); traces jump by 1 at a spike (`vSTDP`: by ``1/τ_x``).
Traces are stored lazily as a vector times a scale: read traces and weights with `trace_values` and `plasticity_value`.

Same-step spikes: in the pair rules with exact decay and in the triplet rule a presynaptic and a postsynaptic spike of the same step do not interact
(traces are read before the spikes of the step are added); they interact in `iSTDPRate`, `STDPMexicanHat` and the voltage rules.

## STDPGerstner

Conventions: [Plasticity: conventions and implementation notes](plasticity_conventions.md#Pair-STDP).

Additive pair STDP (Gerstner et al. 1996), traces ``x_{pre}``, ``x_{post}``; update order:
[Pair and triplet STDP](../plasticity.md#Pair-and-triplet-STDP).

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

Amplitudes are signed: `A_pre > 0` potentiates, `A_post < 0` depresses (default).

| Field | Default | Units | Meaning |
|---|---|---|---|
| `A_post` | `-1e-4` | weight | change per unit ``x_{post}`` at a presynaptic spike (`< 0`: LTD) |
| `A_pre` | `1e-4` | weight | change per unit ``x_{pre}`` at a postsynaptic spike (`> 0`: LTP) |
| `τpre` | `20ms` | ms | presynaptic trace time constant |
| `τpost` | `20ms` | ms | postsynaptic trace time constant |
| `Wmax` | `30pF` | weight | upper bound |
| `Wmin` | `0pF` | weight | lower bound |

**Integration:** event-driven; traces jump by 1 and decay exactly by `exp(-dt/τ)` per step.
**Reference:** Gerstner et al. (1996) (table above).

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

Conventions: [Plasticity: conventions and implementation notes](plasticity_conventions.md#Pair-STDP).

Pair STDP with constant rate terms.

```math
\begin{aligned}
\text{presynaptic spike of } j:&\quad w_{ij} \mathrel{+}= η\,(κ\, x_{post,i} + α) \\
\text{postsynaptic spike of } i:&\quad w_{ij} \mathrel{+}= η\,(γ\, x_{pre,j} + β)
\end{aligned}
```

The signs are carried by `κ`, `γ`, `α`, `β`.

| Field | Default | Units | Meaning |
|---|---|---|---|
| `η` | `0.01` | weight | learning rate |
| `α` | `0` | - | constant term at each presynaptic spike |
| `β` | `0` | - | constant term at each postsynaptic spike |
| `κ` | `1` | - | weight of the post-before-pre term (read at a pre spike) |
| `γ` | `1` | - | weight of the pre-before-post term (read at a post spike) |
| `τpre`, `τpost` | `20ms` | ms | trace time constants |
| `Wmin`, `Wmax` | `0pF`, `30pF` | weight | bounds |

**Integration:** as `STDPGerstner`. **Reference:** citation not recorded; the rule is named after Confavreux et al. (2025).

## STDPWeightDependent

Conventions: [Plasticity: conventions and implementation notes](plasticity_conventions.md#Weight-dependent-STDP).

Soft-bound pair STDP (Gütig et al. 2003). With ``\tilde W = W_{max} - W_{min}``:

```math
\begin{aligned}
\text{presynaptic spike (LTD)}:&\quad w \mathrel{-}= η\, α\, \tilde W^{1-μ_-}\, (w - W_{min})^{μ_-}\, x_{post} \\
\text{postsynaptic spike (LTP)}:&\quad w \mathrel{+}= η\, \tilde W^{1-μ_+}\, (W_{max} - w)^{μ_+}\, x_{pre}
\end{aligned}
```

`μ = 0` is additive STDP with hard bounds, `μ = 1` multiplicative STDP.

| Field | Default | Units | Meaning |
|---|---|---|---|
| `η` | `1e-3` | - | relative learning rate |
| `α` | `1` | - | LTD/LTP asymmetry |
| `μ_plus` | `1` | - | LTP weight-dependence exponent |
| `μ_minus` | `1` | - | LTD weight-dependence exponent |
| `τpre`, `τpost` | `20ms` | ms | trace time constants |
| `Wmax`, `Wmin` | `30pF`, `0pF` | weight | bounds |

**Integration:** as `STDPGerstner`. **Reference:** Gütig et al. (2003) (table above).

## STDPTriplet

Conventions: [Plasticity: conventions and implementation notes](plasticity_conventions.md#Triplet-STDP).

All-to-all triplet STDP (Pfister and Gerstner 2006), traces ``r_1``, ``r_2`` (pre), ``o_1``, ``o_2`` (post):

```math
\begin{aligned}
\text{presynaptic spike of } j:&\quad w_{ij} \mathrel{-}= o_{1,i}\,(A_2^- + A_3^-\, r_{2,j}) \\
\text{postsynaptic spike of } i:&\quad w_{ij} \mathrel{+}= r_{1,j}\,(A_2^+ + A_3^+\, o_{2,i})
\end{aligned}
```

Defaults: minimal all-to-all model (Table 4 of Pfister and Gerstner 2006). `nearest = true` selects the nearest-spike interaction: a spike sets its traces to 1 instead of adding 1.

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
| `nearest` | `false` | | `true`: nearest-spike interaction |
| `Wmax`, `Wmin` | `30pF`, `0pF` | weight | bounds |

**Integration:** as `STDPGerstner` (four traces, exact decay). **Reference:** Pfister and Gerstner (2006) (table above).

!!! note "Implementation note"
    The defaults were not re-checked against Table 4 of the paper; `τ_x = 946 ms` has no effect while `A3_minus = 0`.


## STDPMexicanHat

Conventions: [Plasticity: conventions and implementation notes](plasticity_conventions.md#Pair-STDP).

``Δw = A\,(1 - z)\,e^{-z/\sqrt 2}`` with ``z = [\ln(x_{pre}/x_{post})]^2 ≈ (Δt/τ)^2``: potentiation for
``|Δt| < τ``, depression beyond.

| Field | Default | Units | Meaning |
|---|---|---|---|
| `A` | `0.1` | weight | amplitude |
| `τ` | `20ms` | ms | trace time constant |
| `Wmax`, `Wmin` | `30pF`, `0pF` | weight | bounds |

**Integration:** event-driven; traces jump by 1 and decay by forward Euler (factor `1 - dt/τ`). **Reference:** none.

!!! note "Implementation note"
    The spikes of the step are added to the traces before the weight update, so same-step pre and post spikes interact; for one pair the Euler traces give
    an effective time constant of about `τ - dt/2`. A `NaN` kernel value (log of a zero trace) is replaced by 0.

## STDPSymmetric and STDPAntiSymmetric

Structured rules of Festa, Cusseddu and Gjorgjieva (2024, table above); kernel integral ``A_x - A_y``, plus
constant terms `αpre`, `αpost` per spike.

**Integration:** event-driven; per step the pre pass, the post pass, then forward-Euler decay (factor `1 - dt/τ`) and `+1` of the traces.

Symmetric kernel (``Δt = t_{post} - t_{pre}``):

```math
Δw(Δt) = \frac{A_x}{2τ_x} e^{-|Δt|/τ_x} - \frac{A_y}{2τ_y} e^{-|Δt|/τ_y}
```

Antisymmetric kernel:

```math
Δw(Δt) = \begin{cases} \dfrac{A_x}{τ_x} e^{-Δt/τ_x} & Δt > 0 \\[1ex] -\dfrac{A_y}{τ_y} e^{Δt/τ_y} & Δt < 0 \end{cases}
```

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

## Inhibitory STDP: iSTDPRate, iSTDPPotential

Conventions: [Plasticity: conventions and implementation notes](plasticity_conventions.md#Inhibitory-STDP).

Vogels et al. (2011, table above).

**Integration:** traces jump by 1 and decay by forward Euler (factor `1 - dt/τ_y`); `iSTDPRate` is event-driven, `iSTDPPotential` low-passes the
postsynaptic potential of every neuron at every step.

`iSTDPRate` (target rate ``r``, ``α = 2 r τ_y``):

```math
Δw_{ij} = η\,(y_i - α) \ \text{at a presynaptic spike}, \qquad Δw_{ij} = η\, x_j \ \text{at a postsynaptic spike}
```

`iSTDPPotential` uses ``τ_y\, dy_i/dt = V_i - y_i`` and the target potential ``v_0``:
``Δw_{ij} = η (y_i - v_0)`` at a presynaptic spike, ``η x_j`` at a postsynaptic spike.

| Field | `iSTDPRate` | `iSTDPPotential` | Units | Meaning |
|---|---|---|---|---|
| `η` | `0.01pA` | `0.001pA` | weight | learning rate |
| `r` | `3Hz` | - | 1/ms | target postsynaptic rate |
| `v0` | - | `-50mV` | mV | reference potential |
| `τy` | `50ms` | `200ms` | ms | trace time constant |
| `Wmax` | `243pF` | `243pF` | weight | upper bound |
| `Wmin` | `0.01pF` | `0.01pF` | weight | lower bound |

!!! note "Implementation note"
    `iSTDPRate` lets a presynaptic and a postsynaptic spike of the same step interact: traces decay, the spikes of the step are added, and
    postsynaptic spikes read the presynaptic trace including them. The depression is the constant ``-2 η r τ_y`` per presynaptic spike, so `r = 0` only potentiates.
    The Euler traces shorten the effective time constant by about `dt/2`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 80)
I = SNN.Poisson(N = 20, param = SNN.PoissonParameter(20Hz))
IE = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 5.0), LTPParam = SNN.iSTDPRate(r = 5Hz))
SNN.train!(model = SNN.compose(; E, I, IE), duration = 500ms)
```

## vSTDPParameter and vSTDPDenseParameter

Conventions: [Plasticity: conventions and implementation notes](plasticity_conventions.md#Voltage-based-STDP).

Clopath et al. (2010, table above). `vSTDPParameter` is event-driven (read `W` with `plasticity_value(syn, :W)`);
`vSTDPDenseParameter` is clock-driven. ``V_i`` is the
postsynaptic potential of the target compartment, ``u_i``, ``v_i`` its low-pass filters, ``[z]_+ = \max(z, 0)``:

```math
\begin{aligned}
τ_x \frac{dx_j}{dt} &= -x_j + \sum_k δ(t - t_j^k), \qquad τ_u \frac{du_i}{dt} = V_i - u_i, \qquad τ_v \frac{dv_i}{dt} = V_i - v_i \\
Δw_{ij} &= -A_{LTD}\,[u_i - θ_{LTD}]_+ \quad \text{at each presynaptic spike} \\
\frac{dw_{ij}}{dt} &= A_{LTP}\, x_j\,[v_i - θ_{LTD}]_+\,[V_i - θ_{LTP}]_+
\end{aligned}
```

| Field | Default | Units | Meaning |
|---|---|---|---|
| `A_LTD` | `8e-4` | weight/mV | LTD amplitude |
| `A_LTP` | `1.4e-3` | weight/mV² | LTP amplitude |
| `θ_LTD` | `-70mV` | mV | LTD threshold (also applied to ``v`` in the LTP term) |
| `θ_LTP` | `-49mV` | mV | LTP threshold on ``V`` |
| `τu` | `20ms` | ms | LTD filter time constant |
| `τv` | `7ms` | ms | LTP filter time constant |
| `τx` | `15ms` | ms | presynaptic trace time constant |
| `delay_uv` | `0ms` | ms | delay of the potential that drives ``u`` and ``v`` |
| `Wmax`, `Wmin` | `30pF`, `0.1pF` | weight | bounds |

**Integration:** forward Euler for ``x``, ``u``, ``v`` (``x`` includes the spikes of the step); LTD at presynaptic spikes, then LTP as the rate integrated over `dt`
(`dt A_LTP x ...`). `vSTDPParameter` evaluates the same Euler recursion event-driven: filters and drive every step, weights materialised at presynaptic spikes, once per
`epoch` times `τx`, and on demand (`Float64` accumulators). `vSTDPDenseParameter` updates every weight every step.

!!! note "Implementation note"
    The trace ``x`` jumps by ``1/τ_x`` (units 1/ms), LTP is integrated with `dt` and LTD acts per spike without `dt`; the LTP sum depends on `dt` (8 % larger at `dt = 0.1 ms` than at 0.025 ms
    in the protocol of the conventions page). `syn.W` of `vSTDPParameter` holds the weights of the last materialisation: read `plasticity_value(syn, :W)`; the event rule needs `A_LTP >= 0`.
    `delay_uv` (default 0) follows the ModelDB implementation of Clopath et al. and NEST `delay_u_bars`; it is not in the paper.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
P = SNN.AdEx(N = 10, param = SNN.AdExParameter())
syn = SNN.SpikingSynapse(E, P, :ge; conn = (p = 0.2, μ = 1.0), LTPParam = SNN.vSTDPParameter())
SNN.train!(model = SNN.compose(; E, P, syn), duration = 200ms)
SNN.plasticity_value(syn, :W)   # current weights (syn.W is lazy for the event-driven rule)
```

## Short-term plasticity: Tsodyks-Markram

Conventions: [Plasticity: conventions and implementation notes](plasticity_conventions.md#Short-term-plasticity).

Presynaptic utilisation ``u_j`` and resources ``x_j`` set the efficacy ``ρ_s = u_j x_j`` (Tsodyks and
Markram 1997; Mongillo, Barak and Tsodyks 2008). Between spikes

```math
\frac{du}{dt} = \frac{U - u}{τ_F}, \qquad \frac{dx}{dt} = \frac{1 - x}{τ_D}.
```

At a presynaptic spike, ``Δ`` after the previous one (`MarkramSTPParameter`, `MarkramSTPParameterHet`, exact):

```math
\begin{aligned}
u^- &= U - (U - u)\,e^{-Δ/τ_F}, & x^- &= 1 - (1 - x)\,e^{-Δ/τ_D}, \\
ρ &= u^+ x^-, & & \\
u &\leftarrow u^- + U\,(1 - u^-), & x &\leftarrow x^- - u\,x^- .
\end{aligned}
```

`MarkramSTPParameterHet` takes per-neuron vectors `U`, `τF`, `τD`; `MarkramSTPParameterTimestep` is
clock-driven (forward Euler).

**Integration:** `MarkramSTPParameter` and `MarkramSTPParameterHet`: exact solution between spikes, evaluated at presynaptic spikes only; `MarkramSTPParameterTimestep`: forward Euler of
``u`` and ``x`` at every step, jumps at spikes. **Reference:** Tsodyks and Markram (1997); Mongillo et al. (2008) (table above).

| Field | Default | Units | Meaning |
|---|---|---|---|
| `τD` | `200ms` | ms | recovery (depression) time constant |
| `τF` | `1500ms` | ms | facilitation time constant |
| `U` | `0.2` | - | baseline utilisation |
| `Wmax`, `Wmin` | `1pF`, `0pF` | weight | not used by the update |

!!! note "Implementation note"
    The efficacy is ``u^+ x^-`` (Mongillo et al. 2008): the spike is transmitted with the utilisation after the jump, so the first spike from rest has
    ``ρ = U(2 - U)``, not ``U`` (the value with ``u^-``).

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
