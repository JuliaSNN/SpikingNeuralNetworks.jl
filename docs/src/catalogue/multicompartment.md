# Multicompartment neurons

```@meta
CurrentModule = SNNModels
```

`Tripod` (AdEx soma, two passive dendrites `d1`, `d2`) and `BallAndStick` (one dendrite `d`).

| Group | Population field | Type | Default |
|:------|:-----------------|:-----|:--------|
| morphology | `param` | `DendNeuronParameter` | `TripodParameter()` / `BallAndStickParameter()` |
| soma | `adex` | `AdExParameter` | `AdExParameter()` |
| spike shape, refractoriness | `spike` | `PostSpike` | `PostSpike()` |
| synapses | `soma_syn`, `dend_syn` | `AbstractSynapseParameter` | `TripodSomaSynapse`, `TripodDendSynapse` |

A connection selects the compartment (`:s`, `:d1`, `:d2`; `:s`, `:d`) with its fourth argument:

```julia
using SpikingNeuralNetworks
SNN.@load_units
T = SNN.Tripod(N = 10)
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
I = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
E_d1 = SNN.SpikingSynapse(E, T, :glu, :d1; conn = (p = 0.2, μ = 2.0))   # first dendrite
E_d2 = SNN.SpikingSynapse(E, T, :glu, :d2; conn = (p = 0.2, μ = 2.0))   # second dendrite
I_s  = SNN.SpikingSynapse(I, T, :gaba, :s; conn = (p = 0.2, μ = 2.0))   # soma
model = SNN.compose(; E, I, T, E_d1, E_d2, I_s)
SNN.monitor!(T, [:v_s, :v_d1, :v_d2, :fire])
SNN.sim!(; model, duration = 500ms)
```

## Equations

For dendrite ``k`` (``k = 1, 2`` for `Tripod`, ``k = 1`` for `BallAndStick`), with somatic
parameters from `adex` and spike parameters from `spike`:

```math
\begin{aligned}
C \frac{dV_s}{dt} &= g_L (E_L - V_s) + g_L \Delta_T\, e^{(V_s - \theta)/\Delta_T} - w_s
    - I_{syn,s} - \sum_k g_{ax,k}\,(V_s - V_{d,k}) + I \\
C_{d,k} \frac{dV_{d,k}}{dt} &= g_{m,k} (E_{L,k} - V_{d,k}) - I_{syn,d,k}
    + g_{ax,k}\,(V_s - V_{d,k}) + I_d \\
\tau_w \frac{dw_s}{dt} &= a (V_s - E_L) - w_s \\
\tau_A \frac{d\theta}{dt} &= V_t - \theta
\end{aligned}
```

``I_{syn}`` are the currents of the synapse models (clamped to ``\pm 1500`` pA in `Tripod`,
``\pm 1000`` pA in `BallAndStick`); ``C_{d,k}``, ``g_{m,k}``, ``g_{ax,k}`` come from `create_dendrite`.

A spike is emitted when ``V_s`` reaches `Vspike` (default ``0`` mV): ``V_s`` is held at
`spike.AP_membrane` for `spike.up`, then at `adex.Vr` for `spike.τabs`; ``w_s \mathrel{+}= b``,
``\theta \mathrel{+}= A_t``. During the back-propagation (`up`) and refractory (`τabs`) periods the
dendrites are integrated in full (leak, synapses, axial current, external input) with the soma held at its
clamp value, and ``w_s`` is frozen. `Vspike` is compared with the predicted potential. Integration: Heun
method (`scheme = HeunScheme()`); the split scheme of the original code is `PaperScheme`, see
[TripodPaper](#TripodPaper).

## Parameters

### Morphology: `DendNeuronParameter`

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `ds` | `[(200um, 400um), (200um, 400um)]` | cm | one entry per dendrite: a length or a `(min, max)` range sampled per neuron in 1 μm steps |
| `physiology` | `human_dend` | | cable properties (`Physiology`) |
| `d` | `4um` | cm | dendrite diameter (any `Real`, stored as `Float32`): ``g_{ax} \propto d^2``, ``g_m \propto d``, ``C \propto d`` |
| `El_dend` | `nothing` | mV | dendritic leak reversal; `nothing` uses `adex.El`. Read at construction only |
| `geometry` | `[(:s=>:d1), (:s=>:d2)]` | | compartment graph, stored for bookkeeping |
| `type` | `TripodNeuron()` if `length(ds) == 2`, `BallAndStickNeuron()` if `length(ds) == 1` | | other lengths raise an error |

Shortcuts: `TripodParameter`, `BallAndStickParameter`. Predefined fixed lengths:
`distal_distal` (400, 400 um), `proximal_proximal` (150, 150 um), `proximal_distal` (150, 400 um),
`distal_proximal` (400, 150 um), `soma_only` (0, 0 um); the first word is the length of `d1`. The random
ranges are `random_proximal_distal` and `random_proximal_proximal`, `proximal`, `all_lengths`. Dendrites longer than 1000 um raise an
error (a length written without units is read as cm).

### Cable properties: `Physiology`

| Field | Meaning | `human_dend` | `mouse_dend` |
|:------|:--------|:-------------|:-------------|
| `Ri` | axial resistivity | 200 Ω cm | 200 Ω cm |
| `Rd` | specific membrane resistance | 38907 Ω cm² | 1700 Ω cm² |
| `Cd` | specific membrane capacitance | 0.5 μF/cm² | 1 μF/cm² |

Sources: Eyal et al. (2016), eLife 5:e16553 (human); Dasika et al. (2007), J Neurophysiol 97:3449 (mouse).

For a cylinder of length ``l`` and diameter ``d``:

```math
G_{ax} = \frac{\pi d^2}{4 R_i l} \;(\texttt{G\_axial}), \qquad
G_m = \frac{\pi d l}{R_d} \;(\texttt{G\_mem}), \qquad
C = C_d\, \pi d l \;(\texttt{C\_mem}).
```

```julia
using SpikingNeuralNetworks
SNN.@load_units
SNN.G_axial(Ri = 200Ω*cm, d = 4um, l = 200um)       # ≈ 31.4 nS
d = SNN.create_dendrite(10, (150um, 300um); physiology = SNN.mouse_dend)
d.C[1], d.gm[1], d.gax[1]
```

### Spike: `PostSpike` fields used

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `AP_membrane` | `20mV` | mV | somatic potential during the spike |
| `up` | `1ms` | ms | duration of the clamped spike (back-propagation period) |
| `τabs` | `2ms` | ms | absolute refractory period after `up` |
| `At` | `0mV` | mV | threshold increment ``A_t`` per spike |
| `τA` | `30ms` | ms | relaxation time constant of ``\theta`` |

### Synapses

Defaults: `soma_syn = TripodSomaSynapse` (AMPA + GABAa), `dend_syn = TripodDendSynapse`
(AMPA + NMDA + GABAa + GABAb); see [Synapse and receptor models](synapses.md). Mouse and paper sets: `TripodDendSynapseMouse` (with `DuarteGluDend`, `EyalNMDA32`), `MouseNMDA` (the Mg-block
values printed in the preprint, opt-in), and the sets of the original code `TripodSomaSynapsePaper`,
`TripodDendSynapsePaper`, `TripodDendSynapsePaperMouse`. Select with `dend_syn = TripodDendSynapseMouse`.

## Tripod

Reference: Quaresima et al. (2023), J. Physiol.; preprint bioRxiv 10.1101/2022.09.05.506197 (its Eq. 1
omits ``g_L`` on the exponential term).

```julia
using SpikingNeuralNetworks
SNN.@load_units
T = SNN.Tripod(N = 10, param = SNN.TripodParameter(ds = SNN.random_proximal_distal),
               adex = SNN.AdExParameter(b = 0pA, a = 0nS))
T.I .= 300pA
SNN.monitor!(T, [:v_s, :fire])
SNN.sim!([T]; duration = 200ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["multicompartment/tripod.jl", "multicompartment/tripod_paper.jl"]
```

### TripodPaper

`TripodPaper(; N = 100, physiology = :human, config = :distal_proximal, d = 4um, kwargs...)` is a `Tripod` with the parameters and the integration scheme (`PaperScheme`) of the original code
(`aquaresima/tripod_neuron`, commit 8698e1a). `physiology` is `:human` or `:mouse`; `config` is
`:distal_distal`, `:proximal_proximal`, `:proximal_distal`, `:distal_proximal` or `:soma_only`; `kwargs`
replace any `Tripod` field. Use it to reproduce the published results; the default `Tripod` is the library's
own scheme.

| | `TripodPaper` | default `Tripod` |
|:--|:--|:--|
| Integration | split two-stage scheme of the original, frozen axial current (`PaperScheme`) | full Heun |
| NMDA Mg block | `exp32` approximation (`EyalNMDA32`, b 3.36, gamma 0.077) | `exp256` |
| Spike onset | fires when ``V \ge V_t`` (-50.4 mV) at the start of the step | predicts ``V + dt\,k_2`` against `Vspike` (0 mV) |
| Receptor update | the original discrete update with its dt bias (`exact = false`) | exact update, peak = `g0` at any dt |
| Dendrite lengths | fixed configurations (400 / 150 um) | random 150 to 400 um |
| dt | 0.1 ms | 0.125 ms |
| Other | original GABAa (-75 mV, soma ``\tau_r`` 0.3 ms), soma AMPA ``\tau_r`` 0.25 ms, BAP 20 mV, ``\tau_{abs}`` 2.1 ms | `PostSpike` 20 mV / 1 ms / 2 ms |

`PaperScheme` handles `ReceptorSynapse` only. Run `TripodPaper` at `dt = 0.1ms`: the scheme and the
receptor update reproduce the original at that step (subthreshold step and single EPSP within 7.6e-6 mV,
identical spike times for a 1200 pA step and a frozen Poisson input, Fig. 8 plateau durations identical).

## BallAndStick

```julia
using SpikingNeuralNetworks
SNN.@load_units
B = SNN.BallAndStick(N = 10, param = SNN.BallAndStickParameter(ds = [300um]))
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
syn = SNN.SpikingSynapse(E, B, :glu, :d; conn = (p = 0.2, μ = 2.0))
model = SNN.compose(; E, B, syn)
SNN.sim!(; model, duration = 200ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["multicompartment/ballandstick.jl"]
```

## Dendrites, morphology and parameters

```@autodocs
Modules = [SNNModels]
Pages   = ["multicompartment/dendrite.jl", "multicompartment/dendneuron_parameter.jl"]
```
