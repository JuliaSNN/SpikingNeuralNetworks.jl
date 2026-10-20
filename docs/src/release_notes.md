# V2 changelog

Changes of version 2, its known limitations, and the bugs fixed during version 1.

## 2.0.1: SpikingNeuralNetworks 2.0.1, SNNModels 2.0.2

- Perturbation records are stored compactly (SNNModels 2.0.2): one `PerturbationStore` /
  `PerturbationSpikeStore` per (object, variable, condition) instead of one `Dict` per window, with
  `perturbation_record` output unchanged. Records written by earlier versions must be converted once with
  `compact_perturbation_records!` (then saved again); readers and new recordings refuse the old layout.
  New accessor `perturbation_windows`. Window numbers are never reused after a deletion.
- Re-exported: `perturbation_windows`, `compact_perturbation_records!`, `PerturbationStore`,
  `PerturbationSpikeStore`. Compat `SNNModels = "2.0.2"`.
- Documentation: performance comparison page (v1, v2, Auryn, Brian2); `monitor_rate` written with units;
  COBA tutorial with the Brette et al. 2007 weights.

## 2.0.0: SNNModels 2.0.0, SpikingNeuralNetworks 2.0.0, SNNUtils 1.0.0, SNNPlots 1.0.0, CalciumSpike 1.0.0

All packages move to the next major version together. SNNModels 2 changes the types of synapses, stimuli and
populations (concretely typed plasticity rules, spike lists, lazy synapse arrays); saved models of 1.9.1 and
earlier are refused at load time. Full release summaries: `CHANGELOG.md` of SNNModels, SNNUtils and SNNPlots; every
change with its migration: `CHANGELOG_2.0.0.md` of SNNModels.

- Faster: E/I balanced network 8 to 12 times faster than 1.9.1, synapse memory 29 to 9 bytes per synapse, event-driven vSTDP, ring-buffer delays, one spike list per population, vectorised Tripod/BallAndStick, flush-to-zero of subnormals, no allocation per step.
- New: Float64 by population type, `with_plasticity`/`without_plasticity`/`change_plasticity`, opt-in threading (`neurons_per_thread`).
- Breaking: synapse/stimulus/population struct fields, `iSTDPTime` removed, delays rounded up to whole steps, random streams changed, `vSTDPParameter` is event-driven, Tripod/BallAndStick use `param.physiology`.
- Removed code that was not loaded: Multipod, IF_CANAHP, AdExMultiTimescale, `VariableInputParameter`, stale SNNUtils parameter sets, SNNPlots Plots.jl backend and their examples.

Upgrading: see [What changes from v1 to v2](index.md#What-changes-from-v1-to-v2).

## Known limitations

- `Float32` only: `HetRec` (refuses a `Float64` step), `CurrentStimulus`, `BalancedStimulus` and
  `Turnover` (refuse a `Float64` target), and the rate and FORCE connections.
- The clock accumulates rounding at a step that is not exact in `Float32` (e.g. `dt = 0.1ms`):
  `get_time` drifts, and a `SpikeTimeStimulus` spike can be delivered one step off.
- Delays are counted in steps: after a change of `dt` mid-run, spikes in flight are dropped (if the
  delay in steps changes) or keep their old step count.
- A record closed by a plasticity switch is not reopened by a later switch (`clear_records!` first).
- Some components still do per-step dense work: `iSTDPPotential`, `vSTDPDenseParameter`,
  `MarkramSTPParameterTimestep`, `ActivityDependentTurnover`, `AggregateScaling`.
- `Tripod`/`BallAndStick` reproduce 1.9 bit for bit only on x86-64 with FMA.
- Files saved with version 1 cannot be loaded (`load_model` throws an `ArgumentError`).

## Bugs fixed in version 1

### SNNModels 1.9.0, SNNPlots 0.2.11, SNNUtils 0.2.10, SpikingNeuralNetworks 1.3.0

Bug fixes in SNNModels, SNNPlots, SNNUtils and SpikingNeuralNetworks found while documenting
every public symbol. Several of them change simulation results.

#### Breaking changes

These are released as a minor version (1.9.0), but code that ran with 1.8.x can fail or
give different results. Check this list before upgrading.

| Change | Code affected | Migration |
|---|---|---|
| `AdExParameter`/`IFParameter`: one membrane value alone (`C`, `gl`, `R` or `τm`) is an `ArgumentError` | `IFParameter(τm = 20ms)`, `AdExParameter(C = 200pF)` | give a pair; to keep the 1.8 neuron: `IFParameter(τm = x, R = 0.06)`, `AdExParameter(C = x, gl = 40nS)` |
| More than two membrane values must agree within 0.1 % (`τm = C/gl`, `R = 1nS/gl`) | e.g. `IFParameter(C = 281, gl = 40, τm = 20, R = 0.1)` | keep the pair the model integrates with: `τm, R` for `IF`/`AdEx`, `C, gl` for `Tripod`/`BallAndStick` |
| Changing one membrane value updates the others (`@update!`, `p.τm = x`, `make_heterogeneous`) | `@update! cfg.adex.τm = x` on a Tripod or BallAndStick (no effect before, now `C = τm gl`); `p.C = x` on an `AdExParameter` (now also changes `τm`) | intended in most cases; to change two values at once use `with_membrane(p; τm = …, C = …)` |
| A Tripod or BallAndStick built with `AdExParameter(gl = …, τm = …)` now uses `C = τm gl` | e.g. `gl = 40nS, τm = 20ms`: C 281 → 800 pF | give `C, gl` explicitly to keep the 1.8 neuron |
| Tsodyks-Markram STP update order (Mongillo 2008) | every model with `MarkramSTPParameter*` | none in code; parameters tuned with ≤ 1.8.4 are not equivalent (see below) |
| Tripod/BallAndStick somatic equation, Heun step and refractoriness | every Tripod/BallAndStick model | none in code; results change (see below) |
| Exported names that were never defined are no longer exported | SNNModels: `HUMAN`, `MOUSE`, `MultiRecetorSynapse`, `NMDA_CANAHP`, `PSParam`, `BSParam`, `SpikeTime`, `SpikingSynapseDelay`, `Synapse_CANAHP`, `autocorrelogram`, `filter_populations`, `gcamp6_kernel`, `get_path`, `get_synapse_symbols`, `isi_cv`, `no_PlasticityVariables`, `no_STDPParameter`, `record_plast!`, `synapsearray`; SNNPlots: `default_colors`, `nature_figure`, `plot_model`, `plot_stimulus`, `plot_connections` | these names could not be used before either; remove them from `using … :` lists |
| SNNPlots `stdp_test`/`stdp_kernel` measure a single pre/post pair | kernels computed with SNNPlots ≤ 0.2.10 | recompute; the old kernel included an extra causal pairing |
| SNNUtils `quaresima2022_nar` returns Tripod keyword arguments (it used the removed `AdExSoma`) | code that called it (it threw before) | pass the result to `Tripod(; …)` |

!!! danger "Behaviour change: Tripod and BallAndStick"
    - The somatic exponential term is now ``g_L \Delta_T e^{(V-\theta)/\Delta_T}``, as in the AdEx
      model and in the published Tripod model (Quaresima et al. 2023, Eq. 1, and its code); it
      lacked ``g_L`` (40 times smaller at `gl = 40nS`). Rheobase drops by about 29 % (Tripod with
      default parameters: 1134 -> 814 pA; paper parameters: 1097 -> 782 pA), the rate at 1500 pA
      rises from 24 to 44 Hz (19 to 32 Hz), the first ISI shortens from 18.7 to 10.9 ms, the
      adaptation current reaches about 1.5 times higher values; subthreshold dendritic EPSPs and
      NMDA plateaus are unchanged (< 0.1 mV). BallAndStick changes in the same way.
    - The Heun stage is now consistent (the adaptation stage lacked `dt`, the synaptic currents
      were not evaluated at the predicted state): spike times move by up to 7 ms over 500 ms and
      the scheme converges with `dt`.
    - The refractory counters keep at least one step per period: with `up = τabs = 0.1ms` at the
      default `dt = 0.125ms` the neuron used to fire at about 1 kHz.
    - In a 400-cell Tripod E/I network the E rate doubles (12.6 -> 24.8 Hz) and the
      interneuron rates double as well.
    - New: `Vspike` (spike detection threshold, default -10 mV), dendritic `El` used, `Is`/`Id`
      of BallAndStick applied.

!!! danger "Behaviour change: Tsodyks-Markram short-term plasticity"
    `MarkramSTPParameter` (and the `Het` and `Timestep` variants) now follow Mongillo, Barak &
    Tsodyks (2008): at a presynaptic spike ``u`` jumps first and the same ``u^+`` sets both the
    transmitted efficacy ``u^+ x`` and the depletion of ``x``. Before, the efficacy used ``u``
    before the jump while the depletion used ``u`` after it, which depressed more than either
    published model. Efficacies are higher (first spike from rest ``U(2-U)`` instead of ``U``),
    and for some parameters the character of the synapse changes: with ``U = 0.3``,
    ``τ_F = 500`` ms, ``τ_D = 70`` ms a 20 Hz train facilitated before (0.30 -> 0.44) and
    depresses now (0.51 -> 0.47).

!!! danger "Behaviour change: membrane parameters of AdExParameter and IFParameter (pair rule)"
    `C`, `gl`, `R` and `τm` are now always consistent (``τ_m = C/g_L``, ``R = 1\,\mathrm{nS}/g_L``).
    `AdEx` and `IF` integrate with `τm` and `R`; `Tripod` and `BallAndStick` (through their
    `adex::AdExParameter`) with `C` and `gl`. Before, these were four independent fields after
    construction, so the same parameter object could describe two different membranes.
    - Construction takes a pair of independent values, any of (C, gl), (C, R), (C, τm),
      (gl, τm), (R, τm), and derives the rest, or none and uses the default pair (AdEx
      `C = 281pF, gl = 40nS`; IF `τm = 15ms, R = 0.06`). A single value is an `ArgumentError`
      (before, it was combined with the defaults: `IFParameter(τm = 20ms)` used `R = 0.06`,
      `AdExParameter(τm = 20ms)` kept `C = 281pF` for Tripod). More than two values must agree
      within 0.1 %.
    - Changing one value (`@update!`, `p.τm = x` on the mutable `AdExParameter`,
      `with_membrane`, `make_heterogeneous`) follows a fixed rule: `τm` keeps `gl` and changes
      `C`; `C` keeps `gl` and changes `τm`; `gl` or `R` keeps `C` and changes `τm`. Two values
      given together define a new pair (`with_membrane(p; τm = 20ms, C = 281pF)`).
    - Consequences: `@update! cfg.adex.τm = x` on a Tripod or BallAndStick used to have no
      effect at run time (only `C` and `gl` are read); it now sets ``C = τ_m g_L``. A Tripod
      built with `AdExParameter(gl = 40nS, τm = 20ms)` used `C = 281pF` and now uses
      `C = 800pF`. A parameter set such as `C = 281, gl = 40, τm = 20, R = 0.1` (two different
      membranes) is now rejected and must be fixed by choosing the intended pair.
    - New: `with_membrane`, `resolve_membrane`, `membrane_update`; `@snn_kw` structs can
      specialise `snn_kw_finalize` to derive dependent fields.

!!! warning "Other behaviour changes"
    - HH and MorrisLecar flag one spike per action potential (they flagged about 60 per action
      potential at `dt = 0.01ms`).
    - `Rate`/`WilsonCowan` reset their input `g` every step (it accumulated all past inputs).
    - `Confavreux2025Synapse`: a spike of weight `w` increments the conductance by `w` (it was
      `w dt`).
    - `AggregateScaling`: time constants in ms, rate units, correct summed weight; `HetRec`
      baseline with `dt`; `AdditiveNorm` restores the summed weight.
    - vSTDP and iSTDPPotential voltage traces start at the membrane potential (no initial
      spurious depression); vSTDP weights are unchanged otherwise.
    - Recording: exact time axis of `record`/`interpolated_record`, sampling period rounded
      to the nearest step.
    - SNNPlots `stdp_test`/`stdp_kernel` measure a single pre/post pair (the kernel was shifted
      by an extra causal pairing).
    - BalancedStimulus, FORCE/PINning connections, `train!` with IZ/HH/MorrisLecar or without
      connections, connections onto MorrisLecar/ExtendedIF/WilsonCowan, several analysis,
      IO and SNNUtils functions now work; 35 exported-but-undefined names were removed.

### Earlier releases

Two bugs of earlier releases change results:

- `iSTDPRate` (SNNModels 1.5.0 to 1.8.1, SpikingNeuralNetworks from v1.0.0): potentiation at a
  postsynaptic spike was applied to the wrong synapses (a `@turbo` loop ignored a reassigned loop
  variable). Rerun simulations that used `iSTDPRate` with these versions.
- `STDPGerstner` (before SNNModels 1.8.2): amplitudes were applied twice (effective ``A^2``, sign
  lost). Old amplitude `A` corresponds to `±A^2` now.

Notes of SpikingNeuralNetworks 1.2.1, SNNModels 1.8.2 to 1.8.4 and SNNPlots 0.2.10:
[release notes of SpikingNeuralNetworks 1.3.0](https://github.com/JuliaSNN/SpikingNeuralNetworks.jl/blob/5a6dffb03664c6ac593e0bcb3777870597eead1b/docs/src/release_notes.md).
