# Multicompartment neurons

```@meta
CurrentModule = SNNModels
```

SNNModels provides two dendritic neuron models, both subtypes of `AbstractDendriteIF`
(itself a subtype of `AbstractGeneralizedIF`):

- `Tripod`: an adaptive exponential soma coupled to two passive dendrites (`d1`, `d2`);
- `BallAndStick`: the same soma coupled to one passive dendrite (`d`).

Both are configured by three groups of parameters:

| Group | Population field | Type | Default |
|:------|:-----------------|:-----|:--------|
| morphology | `param` | `DendNeuronParameter` | `TripodParameter()` / `BallAndStickParameter()` |
| soma | `adex` | `AdExParameter` | `AdExParameter()` |
| spike shape, refractoriness | `spike` | `PostSpike` | `PostSpike()` |
| synapses | `soma_syn`, `dend_syn` | `AbstractSynapseParameter` | `TripodSomaSynapse`, `TripodDendSynapse` |

Each compartment has its own synapse state and input buffers, so a connection selects the
compartment with the fourth argument of `SpikingSynapse`:

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

Valid targets are `:s`, `:d1`, `:d2` for `Tripod` and `:s`, `:d` for `BallAndStick`; the
receptor symbol is mapped as for point neurons (`:ge`/`:glu` excitatory, `:gi`/`:gaba`
inhibitory; see [Synapse and receptor models](synapses.md)).

## Equations

For dendrite ``k`` (``k = 1, 2`` for `Tripod`, ``k = 1`` for `BallAndStick`), with somatic
parameters from `adex` and spike parameters from `spike`:

```math
\begin{aligned}
C \frac{dV_s}{dt} &= g_L (E_L - V_s) + \Delta_T\, e^{(V_s - \theta)/\Delta_T} - w_s
    - I_{syn,s} - \sum_k g_{ax,k}\,(V_s - V_{d,k}) + I \\
C_{d,k} \frac{dV_{d,k}}{dt} &= g_{m,k} (E_L - V_{d,k}) - I_{syn,d,k}
    + g_{ax,k}\,(V_s - V_{d,k}) + I_d \\
\tau_w \frac{dw_s}{dt} &= a (V_s - E_L) - w_s \\
\tau_A \frac{d\theta}{dt} &= V_t - \theta
\end{aligned}
```

- ``I_{syn,s}``, ``I_{syn,d,k}`` are the currents of the somatic and dendritic synapse models,
  computed once per step with the potentials at the beginning of the step and clamped to
  ``\pm 1500`` pA (`Tripod`) or ``\pm 1000`` pA (`BallAndStick`). With the default
  `ReceptorSynapse` they include AMPA, NMDA (with magnesium block
  ``B(V) = 1/(1 + [\mathrm{Mg}]/b\; e^{kV})``), GABAa and GABAb conductances.
- ``C_{d,k}``, ``g_{m,k}``, ``g_{ax,k}`` are the capacitance, leak and axial conductance of the
  dendrite (from `create_dendrite`); the dendritic leak reversal is the somatic ``E_L``.
- ``I`` (`Tripod.I`) and ``I_d`` (`Tripod.I_d`, the same current injected in both dendrites) are
  external currents. In `BallAndStick` the fields `Is` and `Id` exist but are not used by the
  equations.
- As implemented, the exponential term is not multiplied by ``g_L`` (the standard AdEx term is
  ``g_L \Delta_T e^{(V-\theta)/\Delta_T}``), and ``\theta`` is a dynamic threshold of the
  exponential term only.

**Spike and refractoriness.** A spike is emitted when the predicted somatic potential
``V_s + dt\,\dot V_s`` reaches ``-10`` mV (hard-coded). Then ``V_s`` is set to `AP_membrane`,
``w_s \mathrel{+}= b``, ``\theta \mathrel{+}= A_t`` (`spike.At`), and a refractory counter is set to
`round((up + τabs)/dt)` steps. During the first `up` ms the soma is held at `AP_membrane`
(back-propagating action potential), during the next `τabs` ms at `adex.Vr`. In both periods
the dendrites only relax towards the soma through the axial term (forward Euler) and ``w_s`` is
frozen. ``\theta`` relaxes to `adex.Vt` with time constant `spike.τA` at every step.

## Integration scheme

1. `update_synapses!` advances the synaptic state of every compartment (scheme of the synapse
   model; exponential Euler for `ReceptorSynapse`).
2. Heun (explicit trapezoidal) method for ``(V_s, V_{d,k}, w_s)``: the derivatives are evaluated
   at the current state and at the Euler-predicted state, and the state is advanced by
   ``\frac{dt}{2}(k_1 + k_2)``. In the predicted adaptation derivative the code uses
   `v_s + Δv` and `w_s + Δv` without the factor `dt`, whereas the voltage equations use
   `v + Δv dt`.
3. Spike detection, reset and refractoriness as described above; ``\theta`` by forward Euler.

## Parameters

### Morphology: `DendNeuronParameter`

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `ds` | `[(200um, 400um), (200um, 400um)]` | cm | one entry per dendrite: a length or a `(min, max)` range sampled per neuron in 1 μm steps |
| `physiology` | `human_dend` | | cable properties (`Physiology`) |
| `geometry` | `[(:s=>:d1), (:s=>:d2)]` | | compartment graph, stored for bookkeeping |
| `type` | `TripodNeuron()` if `length(ds) == 2`, `BallAndStickNeuron()` if `length(ds) == 1` | | other lengths raise an error |

`TripodParameter(; ds = [(200um, 400um), (200um, 400um)], physiology = human_dend,
geometry = [(:s=>:d1), (:s=>:d2)])` and `BallAndStickParameter(; ds = [(150um, 400um)],
physiology = human_dend, geometry = [(:s=>:d)])` are shortcuts.
`Population(param::DendNeuronParameter; N, ...)` builds a `Tripod` or a `BallAndStick` from
`param.type`. The dendrite diameter is always the `create_dendrite` default, `4um`.

Predefined length ranges: `proximal_distal = [(150um, 400um), (150um, 400um)]`,
`proximal_proximal = [(150um, 300um), (150um, 300um)]` (two dendrites);
`proximal = [(150um, 300um)]`, `all_lengths = [(150um, 400um)]` (one dendrite).

### Cable properties: `Physiology`

| Field | Meaning | `human_dend` | `mouse_dend` |
|:------|:--------|:-------------|:-------------|
| `Ri` | axial resistivity | 200 Ω cm | 200 Ω cm |
| `Rd` | specific membrane resistance | 38907 Ω cm² | 1700 Ω cm² |
| `Cd` | specific membrane capacitance | 0.5 μF/cm² | 1 μF/cm² |

References for these values are not given in the code. Geometry helpers, for a cylinder of
length ``l`` and diameter ``d``:

```math
G_{ax} = \frac{\pi d^2}{4 R_i l} \;(\texttt{G\_axial}), \qquad
G_m = \frac{\pi d l}{R_d} \;(\texttt{G\_mem}), \qquad
C = C_d\, \pi d l \;(\texttt{C\_mem}).
```

`create_dendrite(l; d = 4um, physiology = human_dend)` returns `(gm, gax, C, l, d)` for one
dendrite (lengths above 500 μm raise an error; `l <= 0` gives a disconnected compartment with
`gax = 0`), and `create_dendrite(N, l; ...)` returns a `Dendrite` with vectors of `N` values and
`El = -70.6mV`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
SNN.G_axial(Ri = 200Ω*cm, d = 4um, l = 200um)       # ≈ 31.4 nS
d = SNN.create_dendrite(10, (150um, 300um); physiology = SNN.mouse_dend)
d.C[1], d.gm[1], d.gax[1]
```

### Soma: `AdExParameter` fields used

`C` (pF), `gl` (nS), `El` (mV), `ΔT` (mV), `Vt` (mV, resting value of ``\theta``), `Vr` (mV,
reset during the refractory period), `a` (nS), `b` (pA), `τw` (ms). The defaults of
`AdExParameter` are `C = 281pF`, `gl = 40nS`, `El = -70.6mV`, `ΔT = 2mV`, `Vt = -50mV`,
`Vr = -70.6mV`, `a = 4nS`, `b = 80.5pA`, `τw = 144ms`. `Vt` is not the spike threshold.

### Spike: `PostSpike` fields used

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `AP_membrane` | `10mV` | mV | somatic potential during the spike |
| `up` | `1ms` | ms | duration of the clamped spike (back-propagation period) |
| `τabs` | `1ms` | ms | absolute refractory period after `up` |
| `At` | `0mV` | mV | threshold increment ``A_t`` per spike |
| `τA` | `10ms` | ms | relaxation time constant of ``\theta`` |

### Synapses

Defaults: `soma_syn = TripodSomaSynapse` (AMPA + GABAa), `dend_syn = TripodDendSynapse`
(AMPA + NMDA + GABAa + GABAb); see [Synapse and receptor models](synapses.md) for the
receptor values. Any synapse model with a five-argument `synaptic_current!` method can be
used; `DeltaSynapse` cannot.

## Tripod

State variables: `v_s`, `v_d1`, `v_d2` (mV, initialised uniformly between `Vr` and `Vt`),
`w_s` (pA), `θ` (mV), `fire`, `tabs`, external currents `I`, `I_d` (pA); synaptic variables
`synvars_s`, `synvars_d1`, `synvars_d2` and input buffers `receptors_s`, `receptors_d1`,
`receptors_d2`.

Reference: Quaresima A. et al. (2023), "The Tripod neuron: a minimal structural reduction of
the dendritic tree", J. Physiol. (the reference is not given in the code).

```julia
using SpikingNeuralNetworks
SNN.@load_units
T = SNN.Tripod(N = 10, param = SNN.TripodParameter(ds = SNN.proximal_distal),
               adex = SNN.AdExParameter(b = 0pA, a = 0nS))
T.I .= 300pA
SNN.monitor!(T, [:v_s, :fire])
SNN.sim!([T]; duration = 200ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["multicompartment/tripod.jl"]
```

## BallAndStick

State variables: `v_s`, `v_d` (mV), `w_s` (pA), `θ` (mV), `fire`, `tabs`; `synvars_s`,
`synvars_d`, `receptors_s`, `receptors_d`.

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

## Multipod (not loaded)

`src/populations/multicompartment/multipod.jl` is present in the source tree but not loaded by
SNNModels 1.8.4 (its `include` is commented out). It implements an AdEx soma coupled to an
arbitrary number `Nd` of passive dendrites with receptor-based synapses, integrated with the
Heun method, but it depends on names that no longer exist in the loaded code (`AdExSoma`,
`synapsearray`), so it cannot be used. The exported name `MultipodNeurons` of that file is
therefore not available either.
