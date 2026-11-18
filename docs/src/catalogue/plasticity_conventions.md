# Plasticity: conventions and implementation notes

```@meta
CurrentModule = SNNModels
```

What the plasticity rules of the library implement, which choices were made, what they are sensitive to, and
what looks like a bug but is intended. Written as a reproducibility report: if you implement a published
plasticity model with these types, compare the points below with your reference. Equations and parameters:
[Plasticity rules](plasticity_rules.md); usage: [Plasticity](../plasticity.md).

## How to read this page

Each statement carries an evidence tag, dated 2026-11-18 (day of the check). Sources and tests are those of SNNModels 2.3.0
(the plasticity files are unchanged from 2.2.0 except `delay_uv`, `nearest` and docstrings):

| Tag | Meaning |
|---|---|
| `[source: file]` | a docstring or code comment in `src/connections/sparse_plasticity/` or `metaplasticity/` (`kernels` = `STDP_kernels.jl`, `traces` = `STDP_traces.jl`, `triplet`, `wdep` = `STDP_weight_dependent.jl`, `vSTDP`, `iSTDP`, `norm` = `normalization.jl`, `aggr` = `aggregate_scaling.jl`, `STP`) |
| `[test: file]` | a test in SNNModels `test/syn/` (`vstdp.jl`, `vstdp_delay.jl`, `stdp_triplet.jl`, `lazy_traces_euler.jl`, `istdp_kernel.jl`, `stdp_reference.jl`, `lazy_traces.jl`, `lazy_weights.jl`, `metaplasticity.jl`) |
| `[measured here]` | a short Julia experiment run for this page with SNNModels 2.2.0 (sections up to the voltage rule's dt table, inhibitory, pair) or 2.3.0 (`delay_uv`, `nearest`), protocol given in the text |
| `[derived]` | follows from the equations |

"Rationale not recorded" means the code and tests state the behaviour but not why it was chosen.
Weights are `Float32`, times in ms.

## Conventions shared by all rules

| Statement | Evidence |
|---|---|
| Plasticity runs only under `train!`. `sim!` never changes weights, traces or short-term state. | `[source: sparse_plasticity.jl, traces]` |
| Pair rules (`STDPGerstner`, `STDPConfavreux2025`, `STDPWeightDependent`) and `STDPTriplet`: traces jump by 1, decay exactly by `exp(-dt/τ)`, are read before the spikes of the step are added. Chosen to follow the ordering of Auryn `System::run` and Brian2 `on_pre`/`on_post`. | `[source: kernels]` |
| `iSTDPRate`, `iSTDPPotential`, `STDPMexicanHat`, `STDPSymmetric`, `STDPAntiSymmetric`, `vSTDP*`: Euler traces, factor `1 - dt/τ`. Why Euler rather than exact for these rules: rationale not recorded; behaviour tested against a dense recursion. | `[source: iSTDP, traces, vSTDP]`, `[test: lazy_traces_euler.jl]` |
| Traces are stored lazily as vector times a scalar `scale`; `syn.LTPVars.tpre` is not the trace. Read `trace_values`, `plasticity_value`, or `sync_traces!` first. A `sync_traces!` rounds once, so a run that syncs is not bit-identical to one that does not. | `[source: kernels, traces]`, `[test: lazy_traces.jl]` |
| `scale` is kept in `[1e-6, 1]` so stored values stay far from the `Float32` limit; the stored vector absorbs it below that. | `[source: kernels: _SCALE_MIN]` |
| Bounds `Wmin`, `Wmax` are applied to touched weights only (synapses of neurons that fired) plus once to all weights at the first step (pair rules). A weight outside the bounds that is never touched stays outside. | `[source: kernels: _initial_clamp!, _clamp_touched!]` |
| A voltage rule reads `v_post`, the potential of the target compartment of the synapse (the compartment passed when the synapse is built, e.g. `:d1` reads the potential of dendrite `d1` of a multicompartment target); one filter pair per synapse object. | `[source: spiking_synapse.jl: fields g, v_post; vSTDP docstring]` |
| Amplitude units: pair rules weight per unit trace; voltage rules weight/mV (LTD) and weight/mV^2 (LTP). Defaults of `Wmin` differ between rules (0 pF; `vSTDP` 0.1 pF; `iSTDP` 0.01 pF): rationale not recorded. | `[source: traces, vSTDP, iSTDP]` |

## Pair STDP

`STDPGerstner`, `STDPConfavreux2025`, `STDPMexicanHat`. Definition `[source: traces]`: `STDPGerstner`: pre spike `w += A_post x_post[i]`, post spike `w += A_pre x_pre[j]`.
`STDPConfavreux2025`: `w += η(κ x_post + α)` and `w += η(γ x_pre + β)`. `STDPMexicanHat`: `w += A(1-z)e^(-z/√2)`,
`z = ln(x_pre/x_post)^2`, at every pre and post spike with both traces nonzero.

| Choice or behaviour | Evidence |
|---|---|
| Per step: pre pass, post pass, trace `+= 1`, exact decay. A pre and a post spike in the same step give no weight change; the history before does. | `[source: kernels]`, `[measured here]`: `STDPGerstner` A = ±1e-2, tau = 20 ms, dt = 0.1: same step 0.0, post one step after pre +0.00995, pre one step after post -0.00995 |
| Amplitudes are signed and applied once; they multiply a trace, so the update does not carry `dt`. | `[source: traces]` |
| `STDPMexicanHat` is the exception to the ordering: spikes of the step are added (Euler) before the update, so same-step spikes interact. Its `NaN` result (log of a zero trace) is replaced by 0. | `[source: traces]` |
| `STDPConfavreux2025`: `α`, `β` are per-spike terms independent of the traces. | `[source: traces]` |

Sensitivity: for a single pair, the Euler traces of `STDPMexicanHat` give an effective time constant of about
`τ - dt/2` (below). Same-step coincidences contribute nothing in the exact-decay rules, so with a large `dt`
(many spikes per step) the number of effective pairs falls. `[derived]`

## Triplet STDP

`STDPTriplet`. Definition `[source: triplet]`: pre spike `w -= o1 (A2_minus + A3_minus r2)`, post spike
`w += r1 (A2_plus + A3_plus o2)`; `r1` (`τ_plus`), `r2` (`τ_x`) presynaptic, `o1` (`τ_minus`), `o2` (`τ_y`)
postsynaptic; all-to-all interaction (default) or nearest-spike (`nearest = true`).

| Choice or behaviour | Evidence |
|---|---|
| Traces read before this step's spikes are added, `r2` and `o2` before their own increment (the `t - ε` of the definition); same-step pre and post spikes do not interact (`[measured here]`: same step 0.0 with the defaults). | `[source: triplet]` |
| Defaults: the all-to-all minimal parameter set of Table 4 of Pfister and Gerstner (2006); the values were not re-checked against the paper. Equal to Auryn `MinimalTripletConnection` with `eta = 1`. | `[source: triplet]`; paper side not verified |
| `τ_x = 946 ms` (full-model value) is a placeholder while `A3_minus = 0`. | `[source: triplet]` |
| `nearest = false` (default): a spike adds 1 to the traces of its neuron, every earlier spike contributes. `nearest = true` (the nearest-spike model of Pfister and Gerstner 2006): a spike sets the traces of its neuron to 1, only the most recent spike contributes; the weight updates and the order are the same. | `[source: triplet, kernels: _spike_set!]`, `[test: stdp_triplet.jl]` (closed-form updates for both, event-driven equals dense) |

`[measured here]` (2.3.0, defaults, `dt = 0.1`, `w0 = 1`): `dW` for one pre and one post spike 5 ms later is 0.0039357 for both; for 3 pre spikes 10 ms apart then a post spike 5 ms after the last, all-to-all 0.0073028 against nearest 0.0039357 (the earlier pre spikes no longer count); for one pre spike then 3 post spikes 10 ms apart, 0.012356 against 0.011261. The choice changes the result as soon as spikes are not isolated pairs.

Fragility: set `A3_minus = 0` to switch the triplet LTD off; a non-positive `τ_x` or `τ_y` is not a valid time
constant for these exact-decay traces. `[derived]`

## Weight-dependent STDP

`STDPWeightDependent`. Definition `[source: wdep]`, `W̃ = Wmax - Wmin`: pre spike `w -= η α W̃^(1-μ_minus) (w - Wmin)^μ_minus x_post`; post spike
`w += η W̃^(1-μ_plus) (Wmax - w)^μ_plus x_pre`. Same order, traces and same-step behaviour as the pair rules.

- `η` is relative to the weight range (the factor `W̃^(1-μ)` makes the update scale with it); for `Wmin = 0` it is Auryn `STDPwdConnection` with `learning_rate = η`. `[source: wdep]`
- `μ = 0` is additive with hard bounds, amplitudes `η W̃` (LTP) and `-η α W̃` (LTD); `μ = 1` multiplicative. `[source: wdep]`
- Fragility: with `μ_plus > 0` the update vanishes at `Wmax`; a weight initialised above `Wmax` is clamped on the first step. `[derived]`, `[source: kernels]`

## Voltage-based STDP

`vSTDPParameter` (event-driven) and `vSTDPDenseParameter` (clock-driven) `[source: vSTDP]`:

```math
τ_x \dot x_j = -x_j + \sum_k δ(t-t_j^k),\ \ τ_u \dot u_i = V_i - u_i,\ \ τ_v \dot v_i = V_i - v_i,
\qquad Δw = -A_{LTD}[u_i-θ_{LTD}]_+ \ \text{(pre spike)},\ \ \dot w = A_{LTP}\, x_j [v_i-θ_{LTD}]_+[V_i-θ_{LTP}]_+ .
```

| Choice or behaviour | Evidence |
|---|---|
| `x` jumps by `1/τx` (units 1/ms), not 1; LTP is a rate integrated as `dt A_LTP x ...`; LTD is per spike without `dt`. Rationale recorded only as "potentiation is a rate"; whether the update is invariant to `dt` is quantified below. | `[source: vSTDP]` |
| `θ_LTD` thresholds both `u` (LTD) and `v` (in the LTP term); `θ_LTP` the instantaneous `V`. | `[source: vSTDP]` |
| Order per step: Euler `x` (this step's spikes included), Euler `u` and `v` driven by the current `v_post`, LTD at pre spikes (clamp at `Wmin`), LTP on all outgoing synapses (clamp at `Wmax`). A pre spike therefore gets LTD with this step's `u` and LTP with an `x` that already contains it. | `[source: vSTDP]`, `[test: vstdp.jl]` (independent Float64 reference, same order) |
| `u`, `v` are set to the current `v_post` at the first `plasticity!` call; `x` starts at 0. | `[source: vSTDP]` |
| `delay_uv = 0 ms` (default): `u`, `v` are driven by the current `v_post`. For `delay_uv > 0` they are driven by the potential `D = max(1, round(delay_uv/dt))` steps earlier (at least one step even if `delay_uv < dt/2`); the LTP term keeps the current potential; the delay buffer is filled with the current potential at the first step and whenever `D` changes (new `dt` or `delay_uv`). This is the library's way to handle delayed-voltage filtering: it follows the ModelDB implementation of Clopath et al. 2010 and NEST `delay_u_bars`, and is not in the paper. Event-driven and dense rules have the same option. | `[source: vSTDP]`, `[test: vstdp_delay.jl]` (closed form after a potential step; event equals dense; `dt` change) |
| Event-driven rule: `syn.W` holds the weights of the last materialisation; LTP is added at the pre spike of each neuron, every `epoch` times `τx` and on demand (`sync_traces!`, a change of `dt`, `τx`, `A_LTP`, `Wmax`, `epoch`). Read `plasticity_value(syn, :W)`. The trace is stored as an amplitude: `plasticity_value(syn, :LTPVars, :x, t)`. | `[source: vSTDP]`, `[test: lazy_weights.jl]` |
| Event and dense rules agree: same equations and order. One known difference: a weight initially above `Wmax` is clamped at the first step by the dense rule and at the first materialisation by the event rule. | `[source: vSTDP: Bounds]` |

Constraints `[source: vSTDP]`: the event rule needs `A_LTP >= 0` (the single clamp at `Wmax` commutes with the
accumulation only for non-negative increments; use `vSTDPDenseParameter` otherwise), `dt < τx`, `0 < epoch <= 25`
(units of `τx`, because the `Float64` accumulator loses its digits after about 35 `τx`).

Sensitivities `[measured here]` (protocol: 3 presynaptic neurons, 40 presynaptic and 25 postsynaptic spike times on a
0.5 ms grid over 3 s, `V(t)` = -70 mV + 4 mV sinusoid + a 1 ms rectangular spike of +90 mV followed by -8 mV decaying
with 15 ms at every postsynaptic spike, evaluated at each step; `A_LTD = 8e-4`, `A_LTP = 5e-4`, `w0 = 0.5`, dense rule):

| `dt` (ms) | 0.025 | 0.05 | 0.1 | 0.2 | 0.5 |
|---|---|---|---|---|---|
| `dW` LTD only (ratio to `dt = 0.025`) | 1 | 0.999 | 0.996 | 0.992 | 0.975 |
| `dW` LTP only (ratio) | 1 | 1.026 | 1.077 | 1.183 | 1.519 |
| `dW` both (ratio) | 1 | 0.921 | 0.760 | 0.437 | -0.610 |

- The LTD sum is nearly independent of `dt`; the LTP sum is not: for this protocol it is 8 % larger at `dt = 0.1` than at 0.025, and net depression turns into net potentiation at `dt = 0.5`. The cause was not isolated (candidates: sampling of the 1 ms spike waveform, `x` including the spike of the step, Euler filters); only the protocol above was tested. Consequence: tune amplitudes at the `dt` you will use, or check the sum at two time steps.
- Initial filter values: starting `u`, `v` at 0 mV instead of the postsynaptic potential changes `dW` of synapse 1 from -0.0089 to -0.0596 (6.7 times more depression) in this protocol, because the LTD term `[u - θ_LTD]_+` is large until `u` relaxes to the potential.
- Epoch: event minus dense weights differ by at most 2.4e-7 for `epoch` = 1, 10 and 25 at `dt = 0.1` (the `Float32` rounding of the weights, `dW` about 9e-3); default `epoch = 10` (rationale not recorded; `[test: vstdp.jl]` compares against a Float64 reference).

Delay `[measured here]` (2.3.0, `dt = 0.1`, `τu = 20 ms`, LTD only, `A_LTD = 1e-2`, `V` steps from -70 to -40 mV at t = 10 ms, one presynaptic spike):

| spike at t (ms) | `delay_uv` 0 | 1 ms | 5 ms |
|---|---|---|---|
| 10.5, `u` (mV), `dW` | -69.11, -0.0089 | -70.00, 0 | -70.00, 0 |
| 12.0 | -67.00, -0.0300 | -68.39, -0.0161 | -70.00, 0 |
| 20.0 | -58.08, -0.1192 | -59.01, -0.1099 | -63.23, -0.0677 |

Event-driven and dense weights agree to the digits shown. A delay of 1 ms removes the whole response of a spike 0.5 ms after the step and reduces the LTD of a spike 10 ms after it by 8 %; set `delay_uv` to the value of your reference implementation and check it at your `dt`.

Looks like a bug but is intended: `syn.W` does not change between presynaptic spikes (event rule); LTD has no `dt`;
`A_LTP < 0` raises an error for the event rule.

## Inhibitory STDP

`iSTDPRate` `[source: iSTDP]`: pre spike `w += η (y[i] - 2 r τy)`, post spike `w += η x[j]`; `x`, `y` jump by 1 and decay
by Euler `1 - dt/τy`; `r` in 1/ms (give it with `Hz`). `iSTDPPotential`: `y` low-passes `v_post` (set to `v_post` at the first call),
pre spike `w += η (y[i] - v0)`, post spike `w += η x[j]`, plain traces.

| Choice or behaviour | Evidence |
|---|---|
| Order: pre spikes read `y` of the previous step; traces decay, then this step's spikes are added; post spikes read `x` including this step's spikes. A pre and a post spike in the same step interact. Rationale not recorded; behaviour is. | `[source: iSTDP]`; `[measured here]`: `iSTDPRate`, `η = 1e-2`, `r = 0`, same step +0.0100, post one step after pre +0.00995, pre one step after post +0.0100 |
| Euler against exact decay: for a pair at lag 20 ms with `τy = 20 ms`, `iSTDPRate` gives 0.9994, 0.9975, 0.9874, 0.9745, 0.9478 of the exact `e^-1` for `dt` = 0.025, 0.1, 0.5, 1, 2 ms, equal to `(1 - dt/τ)^n e^1` (about `1 - dt/(2τ)`). | `[measured here]` (against `STDPGerstner`), `[derived]` |
| The depression is the constant `-2 η r τy` per pre spike: with `r = 0` (above) the rule only potentiates. | `[source: iSTDP]` |
| A bug in SNNModels 1.5.0 to 1.8.1: at post spikes a `@turbo` loop reassigned its loop variable and updated synapses at CSC positions `rowptr[i]:rowptr[i+1]-1`, onto unrelated postsynaptic neurons; fixed in 1.8.2. Results of `iSTDPRate` from those versions are wrong; `iSTDPPotential` was not affected. | `[source: iSTDP]`, `[test: istdp_kernel.jl]` (plain CSC reference loop) |
| `iSTDPRate` traces are stored scaled (`trace_values`); `iSTDPPotential` traces are not. | `[source: iSTDP]` |

## Normalisation and scaling

| Choice or behaviour | Evidence |
|---|---|
| `SynapseNormalization` acts every `max(1, round(τ/dt))` steps under `train!`, after the `plasticity!` of the connections listed before it, on the materialised (current) weights. It restores the sum of the incoming weights to the value at construction: multiplicative `w *= W0/W1`, additive `w += (W0 - W1)/n`. | `[source: norm]`, `[test: metaplasticity.jl]` |
| It does not clamp to the `Wmin`, `Wmax` of any plasticity rule. `[measured here]`: four weights `[0.9, 1.2, 1.2, 1.2]` with a rule `Wmin = 0.9`, normaliser built at sum 4: after one `AdditiveNorm` step `[0.775, 1.075, 1.075, 1.075]`, the first is below `Wmin` until the rule touches it. | `[source: norm]`, `[measured here]` |
| `AggregateScaling`: `τ_a ẏ = -y + Σδ` (a spike adds `1/τ_a`; `y` in 1/ms against the rate `Y`), `τ_e Ẇ^T = (1 - W^T/Wmax)(1 - y/Y)`; every `round(τ/dt)` steps `w ← w μ + Wmin`, `μ = max(W^T - n Wmin, 0)/W^t`. `Wmin` is an offset added to every weight, not a clamp; `Wmax` is a soft bound of the target sum. `y` is updated in `forward!` (also under `sim!`), `W^T` in `plasticity!` (only `train!`). | `[source: aggr]` |
| Both normalisers are separate components: neither changes the state of the plasticity rule, and the rule does not know about them. | `[source: norm, aggr]` |

## Short-term plasticity

`[source: STP]`: efficacy `ρ = u^+ x^-` per presynaptic neuron, with `u^+ = u^- + U(1-u^-)`, `x ← x^- - u^+ x^-`, where `u^-`, `x^-` are the
values relaxed since the previous spike (exact for `MarkramSTPParameter`, `MarkramSTPParameterHet`; Euler for `MarkramSTPParameterTimestep`).

- The spike is transmitted with the new efficacy (`update_traces!` runs before `forward!`); from rest the first spike has `ρ = U(2 - U)`, not `U`. The formulation is that of Mongillo, Barak and Tsodyks (2008). `[source: STP]`
- `Wmin`, `Wmax` are not used; under `sim!` `ρ` keeps its last value. `[source: STP]`

## Symbols of the published definitions

Only mappings that the library docstrings state or that follow from the equations; the paper side was not re-read.

| Definition | Library symbols | Evidence |
|---|---|---|
| Clopath et al. 2010 | `A_LTD`, `A_LTP`, `θ_LTD` (θ-), `θ_LTP` (θ+), `τu`, `τv`, `τx`; a paper trace that jumps by 1 corresponds to `A_LTP τx` here, because the library `x` is the paper trace divided by `τx`; the visual-cortex fit is given in the docstring as `A_LTD = 14e-5, A_LTP = 8e-5, θ_LTD = -70.6mV, θ_LTP = -45.3mV, τu = 10ms, τv = 7ms` (from the NEST `clopath_synapse` example) | `[source: vSTDP]` (the defaults are "not referenced in the code"), `[derived]` |
| Pfister and Gerstner 2006 | `A2_plus`, `A3_plus`, `A2_minus`, `A3_minus`, `τ_plus`, `τ_minus`, `τ_x`, `τ_y`; traces `r1, r2, o1, o2` | `[source: triplet]` |
| Gütig et al. 2003 | `η = λ`, `α`, `μ_plus = μ_minus = μ`, `τpre = τpost = τ`, `Wmin = 0`, `Wmax = 1` reduces to LTP `λ(1-w)^μ x_pre`, LTD `-λ α w^μ x_post` | `[source: wdep docstring]` |
| Vogels et al. 2011 | `η`, `r` (target rate), `τy`, `α = 2 r τy` | `[source: iSTDP]` |

## Checklist when reproducing a published result

| Compare | In the library |
|---|---|
| Trace and amplitude convention | pair and triplet traces jump by 1; `vSTDP` trace by `1/τx`; LTD per spike, LTP integrated with `dt`; convert amplitudes in your code |
| Time step | LTP sum of `vSTDP` depends on `dt` (table above); Euler traces shorten `τ` by `dt/2`; `dt < τx` for the event rule |
| Bounds and clamping | touched weights only (pair rules, event `vSTDP`) or all outgoing weights each step (dense `vSTDP`); normalisation never clamps; one-off clamp at the first step |
| Update order | LTD before LTP; same-step pre/post: none in the exact-decay pair rules and triplet, interaction in `iSTDPRate`, `STDPMexicanHat` and the voltage rules |
| Compartment voltage | `v_post` of the target compartment; one filter pair per synapse object |
| Initial filter values | `u`, `v`, `iSTDPPotential` `y` start at the potential; initial 0 changes `vSTDP` results (6.7 times in the protocol above) |
| Delays | `delay_uv` on the voltage filters (default 0), `round(delay_uv/dt)` steps, at least one; synaptic delays are those of the synapse |
| Normalisation | listed synapses, period, additive or multiplicative, clamps after it |
| Lazy weights and traces | `plasticity_value`, `trace_values`, `sync_traces!`; only under `train!` |

## Design guideline

The library implements the most correct formulation of each rule and carries no options, presets or forked code paths whose
only purpose is to imitate another implementation (policy of the library maintainer, 2026-11-18). The options that exist, `delay_uv`
and `nearest`, are defined by a published or reference model, not by a particular code base. Reproducing a published result
that uses other conventions is done in the reproducing code, by converting parameters: for example `A_LTD = dt a` for a rule
written with a `dt` factor in the LTD term, or `A_LTP = a τx` for a trace that jumps by 1 (`[derived]`). Each conversion is a
statement of the form above and should be checked at the `dt` of the study.
