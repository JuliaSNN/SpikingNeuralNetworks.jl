# Release notes

## SpikingNeuralNetworks 1.2.1

Requires SNNModels 1.8.4, SNNPlots 0.2.10 and SNNUtils 0.2.9 or later (within the same major
version). Documentation updated for the changes below; `matrix_record` is exported.

## Changes in SNNModels 1.8.4

- `get_git_commit_hash` no longer throws outside a git repository (cluster jobs run from a copied
  directory); `write_config` records `"unknown"` and the run continues.

## Changes in SNNModels 1.8.3

- `CurrentStimulus` with `CurrentNoise` on a neuron subset (e.g. `neurons = 51:100` on a 100-neuron
  population) indexed its random cache by neuron id and read past its end; fixed.
- The positional constructor `MultiReceptorSynapse(syn)` now works and matches the keyword form.
- `SomaNMDA`: a first definition (`b = 3.57, k = -0.062`) that was always overridden by
  `NMDAVoltageDependency()` (`b = 3.36, k = -0.077`) was removed. The effective value is unchanged.

## Changes in SNNPlots 0.2.10

- `using SNNPlots` now applies the SNNPlots Makie theme (no grid, frameless legends, Okabe-Ito
  palette). Before, the theme was set only during precompilation and never reached user sessions;
  with older versions call `SNNPlots.@makie_default` after loading.

## Changes in SNNModels 1.8.2

SNNModels 1.8.2 (released 2026-10-05) rewrites the trace-based STDP rules, fixes a bug in the inhibitory STDP and replaces the connectivity generator. Two changes alter simulation results.

!!! danger "Bug fix: `iSTDPRate` potentiated the wrong synapses (SNNModels 1.5.0 - 1.8.1)"
    In `iSTDPRate` (and the former `iSTDPTime`) the loop applying potentiation at a postsynaptic spike used `@turbo` and reassigned its loop variable (`st = index[st]`). LoopVectorization ignored the reassignment, so potentiation was applied to the synapses stored at positions `rowptr[i]:rowptr[i+1]-1` of the column-ordered arrays, that is to synapses onto unrelated postsynaptic neurons, instead of to the synapses onto the spiking neuron `i`. The depression branch was correct. `iSTDPPotential` was not affected.

    **Affected versions:** SpikingNeuralNetworks.jl from commit 680a30c (2025-01-06, v1.0.0) and SNNModels v1.5.0 to v1.8.1.

    The error is structured, not random: because CSC storage is ordered by presynaptic neuron, the potentiation of all outgoing synapses of an inhibitory neuron `j` was driven by the spikes of a handful of postsynaptic neurons with indices close to `j * Npost / Npre`. Models whose neurons are laid out in index order (tonotopic maps, assemblies as contiguous blocks) were most exposed, because the bug imposed a fixed index-to-index pattern of inhibitory potentiation. Population-averaged rates may change little; per-neuron weights and rate spread do. The regression test is `test/syn/istdp_kernel.jl` in SNNModels. In one index-independent auditory-cortex model (8 networks, neuron positions assigned independently of index), rerun with the buggy and the fixed kernel on identical networks, the mechanistic signature was present in the PV-to-Exc weights but the effect on activity was small (rates changed by less than 1%, model ranking unchanged); other models, especially index-ordered ones, should be checked individually.

    **What to do:** rerun simulations that used `iSTDPRate` (or `iSTDPTime`) as `LTPParam` with these versions. Weight trajectories, balance of excitation and inhibition and anything downstream of them change. Packages that use the rule (SNNUtils models, tutorials, the recordings page) change accordingly.

!!! warning "Behaviour change: `STDPGerstner` amplitude and default"
    `A_pre` and `A_post` were applied twice (the trace was incremented by `A` and multiplied by `A` again), so the effective amplitude was ``A^2`` and the sign was lost: a negative `A_post` potentiated. Now each amplitude is applied once and keeps its sign. The default `A_post` is `-1e-4` (LTD). Both signs are accepted (`A_post < 0` is Hebbian LTD). To reproduce an old parameter set with amplitude `A`, use `|A|^2` with the intended sign (old `5e-2` becomes `2.5e-3`).

### New features

- `STDPTriplet`: minimal all-to-all triplet rule (Pfister and Gerstner 2006, Table 4 defaults; equal to Auryn `MinimalTriplet`), with `STDPTripletVariables`.
- `STDPWeightDependent`: weight-dependent soft-bound pair rule (Gütig et al. 2003; Auryn `STDPwd` form, `η` is a relative rate).
- Event-driven kernels (`sparse_plasticity/STDP_kernels.jl`) shared by `STDPGerstner`, `STDPConfavreux2025`, `STDPWeightDependent`, `STDPTriplet` (see [Plasticity](plasticity.md)). Validated against Brian2 (2e-6), Auryn (2e-7) and analytic kernels.
- `sparse_matrix` builds the sparse matrix directly: no dense `Npost x Npre` array. 1e5 x 1e5 at `p = 1e-3` takes 0.22 s and 0.11 GiB.

### Behaviour changes

- Plasticity still runs only under `train!`, not `sim!`; the documentation now says so.
- Same-step pre and post spikes do not interact in the pair and triplet rules (traces are read before the current step's spikes; earlier history is kept), as in Auryn and Brian2.
- `STDPGerstner`, `STDPConfavreux2025`, `STDPMexicanHat`: no threading, only touched weights are clamped (plus one clamp of all weights at the first step).
- `sparse_matrix` returns `SparseMatrixCSC{Float32,Int}` from a different random stream: seeded networks do not reproduce earlier realisations (identical degree and weight statistics). The old generator remains as the non-exported `SNNModels.sparse_matrix_dense_legacy`.
- Autapses of recurrent `SpikingSynapse(pre, pre, ...)` are removed structurally; no zero-weight self synapses remain that plasticity could grow.
- Synaptic data (weights, delays, STP `ρ`) are `Float32` at all constructor boundaries, whatever the element type of the input.

### Performance

Measured on the science workstation (Julia 1.12.6), 4000 `Identity` neurons, recurrent `p = 0.02`, `plasticity!` alone, microseconds per step at 10 Hz:

| rule | old, 1 thread | old, 24 threads | new, 1 thread |
|---|---|---|---|
| `STDPGerstner` | 543 | 191 | 7.5 |
| `STDPMexicanHat` | 711 | - | 25 |

A full `train!` second of a 4000-neuron Poisson population projecting to an IF population with `STDPGerstner` (320,000 synapses) takes 0.21 s against 4.59 s before (1 thread), and 0.17 s without plasticity. `sparse_matrix` for 2e4 x 2e4 at `p = 1e-3` (Bernoulli): 0.01 s and 5 MiB against 13.5 s and 19.9 GiB.

### Migration checklist

1. Rerun every simulation that used `iSTDPRate` or `iSTDPTime`.
2. Rescale `STDPGerstner` amplitudes (`A_new = ±A_old^2`) and check the sign of `A_post`.
3. Expect different (statistically equivalent) connectivity for seeded networks.
4. Make sure plastic networks are run with `train!`.
