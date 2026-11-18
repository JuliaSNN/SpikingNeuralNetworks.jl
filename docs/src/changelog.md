# Changelog

All history of the library, newest release first. The rest of the documentation describes the current
state only. Under each release, *Fixed* lists the changes that alter simulation or analysis results, *Improved* all other
changes. Complete per-package changelogs: `CHANGELOG.md` of SNNModels, SNNUtils, SNNPlots and CalciumSpike, and
[`CHANGELOG_2.0.0.md`](https://github.com/JuliaSNN/SNNModels.jl/blob/main/CHANGELOG_2.0.0.md) of SNNModels (every 2.0.0 change with its migration).

## 2.1.0 (SpikingNeuralNetworks 2.1.0; SNNModels 2.1.0 to 2.3.0; SNNUtils 1.1.0; CalciumSpike 1.1.0)

### Fixed

- **Receptor kinetics (SNNModels 2.2.0).** `ReceptorSynapse` and `MultiReceptorSynapse` (soma and dendrites of `Tripod` and
  `BallAndStick`, point neurons with receptors) peak at `g0` at any `dt`. Before, the peak was `c(dt) g0` with
  `c = α dt / (1 - exp(-α dt))`, `α = 1/τr - 1/τd` (AMPA and GABAa 17 to 23 % too high at `dt` 0.1 to 0.125 ms, NMDA 5 to 6 %, GABAb
  below 0.2 %). To reproduce an old run: `g0_new = c(dt) g0_old`, per receptor, at the `dt` of the old run. `exact = false`
  keeps the old discrete update (used by `TripodPaper`).
- **Tripod and BallAndStick soma clamp (SNNModels 2.1.0).** During the back-propagating spike (`up`) and the refractory period
  (`τabs`) the dendrites only relaxed axially towards the clamped soma, without leak or synaptic current (NMDA included), so one
  somatic spike could end an NMDA plateau. They now integrate the full dendritic equation with the Heun scheme (plateau of paper
  Fig. 8 protocol 28.7 ms to 54.2 ms; network rate +1.5 to +2 % for `Tripod`, below 0.3 % for `BallAndStick`).
- **GABA and AMPA defaults of `Tripod` and `BallAndStick` (SNNModels 2.1.0).** Somatic and dendritic GABAa `E_rev` -70 to -75 mV,
  somatic GABAa `τr` 0.1 to 0.3 ms, somatic AMPA `τr` 0.26 to 0.25 ms (values of the published Tripod code).
- **`PostSpike` defaults (SNNModels 2.1.0).** `τabs` 1 to 2 ms, `τA` 10 to 30 ms, `AP_membrane` 10 to 20 mV, and `Vspike` of `Tripod`
  and `BallAndStick` -10 to 0 mV. The refractory period of `IF` and `AdEx` doubles (default `IF` benchmark: 315 to 238 Hz).
- **Length presets (SNNModels 2.1.0).** `proximal_distal` and `proximal_proximal` are the fixed lengths (150/400 um, 150/150 um) of the
  paper; the random ranges are `random_proximal_distal` and `random_proximal_proximal`.
- **`IZ` (SNNModels 2.2.0).** Default `ge`, `gi` start at zero; they were random with means 40 and 200 and every default `IZ` diverged.
- **`HH` (SNNModels 2.2.0).** The 0/0 point of the rate functions (`v` exactly -50, -23 or -48 mV) gives its limit instead of `NaN`.
- **`score_spikes` (SNNUtils 1.0.1).** The counting window follows `window` and `delay`; outputs differ from earlier ones only for
  `delay > 10 ms`, `delay = nothing`, or windows ending past `offset_times[end] + 100`.

### Improved

- **Multipod (SNNModels 2.3.0).** New `Multipod` (soma with 1 to 64 passive dendrites), `MultipodParameter`, `MultipodPaper`,
  `declare_multipod`, `set_dendrite_length!`, `ndendrites`; equal to `Tripod` for two dendrites, bit for bit. Re-exported as `SNN.<name>`.
- **vSTDP delay and triplet option (SNNModels 2.3.0).** `delay_uv` of `vSTDPParameter` and `vSTDPDenseParameter`; `nearest` of
  `STDPTriplet`. Voltage rules on a `Tripod`, `Multipod` or `BallAndStick` compartment read that compartment.
- **TripodPaper (SNNModels 2.1.0).** `TripodPaper`, `PaperScheme` (the integration scheme of the published code, field `scheme` of
  `Tripod`), `HeunScheme`, mouse synapses (`DuarteGluDend`, `TripodDendSynapseMouse`, `MouseNMDA`), `NMDAVoltageDependency(approx)`,
  `ReceptorSynapse(decay)`, dendrite diameter `d`, `El_dend`, 1000 um length guard. Re-exported as `SNN.<name>`.
- **Synapses and API (SNNModels 2.2.0).** Field `exact` of `ReceptorSynapse`; `SynapseNormalization` about 7 times faster; `firing_rate` of
  a silent population returns zeros; positional `AggregateScalingParameter` accepts `Wmax`; `record` on an empty record and `monitor!` on a
  key already monitored give explicit messages; dendrite lengths of any `Real` type; thread-safe receptor registry.
- **SNNUtils 1.1.0.** Submodule `SNNUtils.SpikeTimit` (Spike TIMIT reader, migrated from SpikeTimit.jl; remove that package from the
  environment). `create_dataset` stores `Transcription` objects, `select_words` samples without replacement by default,
  `get_ordered_spikes` takes seconds.
- **CalciumSpike 1.1.0.** `seed` keyword of `calcium_trace`, `generate_synthetic_data`, `biophysical_calcium`; compat `SNNModels = "1.7.2, 2"`.
- Compat `SNNModels = "2.3"`.

## 2.0.1 (SpikingNeuralNetworks 2.0.1; SNNModels 2.0.1 to 2.0.3; SNNUtils 1.0.1)

### Fixed

- **`STTC` (SNNModels 2.0.1).** Both trains are restricted to the interval and the total time is the union of the windows (Cutts and Eglen 2014); 2.0.0
  returned values above 1 in absolute value and `NaN`.
- **Perturbation windows (SNNModels 2.0.2).** Window numbers are never reused; after deleting window 2 of `{1, 2, 3}` the next flush overwrote window 3.

### Improved

- **Perturbation records (SNNModels 2.0.2).** One `PerturbationStore` / `PerturbationSpikeStore` per (object, variable, condition); `perturbation_windows`,
  `compact_perturbation_records!` (one-time conversion of older files). Output of `perturbation_record` unchanged.
- **`:fire` records (SNNModels 2.0.3).** Neuron ids stored as `Int32`; `monitor_time` sizes the `:fire` buffer; `get_git_commit_hash(; dir)` and provenance of
  `write_config` from outside the repository.
- **`monitor_rate` (SNNModels 2.0.1)** carries units (`50Hz`); a plain number is an `ArgumentError`.
- Performance comparison page.

## 2.0.0 (SNNModels 2.0.0, SpikingNeuralNetworks 2.0.0, SNNUtils 1.0.0, SNNPlots 1.0.0, CalciumSpike 1.0.0)

### Fixed

- **Random streams.** Poisson stimuli and populations draw one random number per spike: same laws, different streams, so seeded 1.9 runs are not reproduced.
- **Delays.** Delays are rounded up to whole steps (`ceil(delay / dt)`); at `dt` such as 0.1 ms, 1.9.1 delivered 22 to 80 % of the events one step late.
- **Plasticity.** STDP weights differ from 1.9.1 by `Float32` rounding (lazy traces); `vSTDPParameter` is event-driven (the 1.9.1 rule is `vSTDPDenseParameter`).
- **Tripod and BallAndStick** use `param.physiology`.

### Improved

- **Speed and memory.** E/I balanced network 8 to 12 times faster than 1.9.1; synapse memory 29 to 9 bytes per synapse; event-driven vSTDP, ring-buffer
  delays, one spike list per population, vectorised Tripod and BallAndStick, flush-to-zero of subnormals, no allocation per step.
- **New.** `Float64` by population type; `with_plasticity`, `without_plasticity`, `change_plasticity`; opt-in threading (`neurons_per_thread`).
- **Removed.** `iSTDPTime`; code that was not loaded (Multipod of 1.x, `IF_CANAHP`, `AdExMultiTimescale`, `VariableInputParameter`, stale SNNUtils parameter sets, SNNPlots Plots.jl backend).
- **Upgrading from 1.x.** Files saved with version 1 are refused at load: rebuild the model from its configuration. Read plasticity state with `plasticity_value`,
  `trace_values`, `sync_traces!`. Call `SNNModels.ensure_transpose!(syn)` before `syn.J` / `syn.index`; read `syn.ρ` with `SNNModels.efficacy`.
  `syn.param.delaytime` has length 1 when all delays are equal. For `Float64` build populations with `VFT = Vector{Float64}` and rules with `FT = Float64`.
  `flush_subnormals = false` restores IEEE behaviour. Positional constructors of the changed structs need the new fields. Do not write `pop.fire` of an
  integrate-and-fire population between its `integrate!` and the end of the step.

## 1.9.0 (SNNModels 1.9.0, SNNPlots 0.2.11, SNNUtils 0.2.10, SpikingNeuralNetworks 1.3.0)

### Fixed

- **Tripod and BallAndStick.** The somatic exponential term is `gL ΔT exp((V - θ)/ΔT)` (it lacked `gL`): rheobase -29 % (1134 to 814 pA), rate at 1500 pA 24 to 44 Hz,
  E rate of a 400-cell E/I network doubled. The Heun stages are consistent (spike times moved by up to 7 ms over 500 ms). Refractory counters keep at least one
  step per period. New: `Vspike` (default -10 mV then), dendritic `El` used, `Is`/`Id` of `BallAndStick` applied.
- **Tsodyks-Markram STP.** `MarkramSTPParameter*` follow Mongillo et al. (2008): the same `u^+` sets the efficacy `u^+ x` and the depletion of `x`; before, the efficacy used `u`
  before the jump (first spike from rest `U(2 - U)` instead of `U`; with `U = 0.3`, `τF = 500` ms, `τD = 70` ms a 20 Hz train facilitated before and depresses now).
- **Membrane parameters.** `C`, `gl`, `R`, `τm` of `AdExParameter` and `IFParameter` are consistent (`τm = C/gl`, `R = 1nS/gl`): construction takes a pair, one value alone is an
  `ArgumentError`, more than two must agree within 0.1 %; changing one value updates the others (`τm` keeps `gl`, `C` keeps `gl`, `gl` or `R` keeps `C`). `@update! cfg.adex.τm = x` on a
  `Tripod` had no effect and sets `C = τm gl` now; `AdExParameter(gl = 40nS, τm = 20ms)` gives `C = 800pF` (was 281 pF). New: `with_membrane`, `resolve_membrane`, `membrane_update`.
- `HH` and `MorrisLecar` flag one spike per action potential (about 60 at `dt = 0.01 ms`); `Rate` and `WilsonCowan` reset `g` every step; `Confavreux2025Synapse` increments by `w` (was `w dt`).
- `AggregateScaling` time constants in ms, rate units, correct summed weight; `HetRec` baseline with `dt`; `AdditiveNorm` restores the summed weight.
- vSTDP and `iSTDPPotential` voltage traces start at the membrane potential; time axis of `record` / `interpolated_record` exact; SNNPlots `stdp_test` / `stdp_kernel` measure one pair (the kernel included an extra causal pairing).
- SNNUtils `quaresima2022_nar` returns `Tripod` keyword arguments.

### Improved

- `BalancedStimulus`, FORCE and PINning connections, `train!` with `IZ`/`HH`/`MorrisLecar` or without connections, connections onto `MorrisLecar`/`ExtendedIF`/`WilsonCowan`, several analysis, IO and SNNUtils functions work;
  35 exported but undefined names were removed.

## Earlier 1.x releases

### Fixed

- **`iSTDPRate` (SNNModels 1.5.0 to 1.8.1).** Potentiation at a postsynaptic spike was applied to the wrong synapses; rerun simulations that used it.
- **`STDPGerstner` (before SNNModels 1.8.2).** Amplitudes were applied twice (effective `A^2`, sign lost); old amplitude `A` corresponds to `±A^2` now.

### Improved

- Notes of SpikingNeuralNetworks 1.2.1, SNNModels 1.8.2 to 1.8.4 and SNNPlots 0.2.10:
  [release notes of SpikingNeuralNetworks 1.3.0](https://github.com/JuliaSNN/SpikingNeuralNetworks.jl/blob/5a6dffb03664c6ac593e0bcb3777870597eead1b/docs/src/release_notes.md).
