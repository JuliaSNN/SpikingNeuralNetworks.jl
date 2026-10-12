# SNNUtils models

```@meta
CurrentModule = SNNUtils
```

The folder `src/models/` of [SNNUtils.jl](https://github.com/JuliaSNN/SNNUtils.jl) contains
`stp_het.jl`, the only file of the folder: short-term-plasticity parameters sampled from
experimental fits, expressed with the SNNModels types.

## `stp_het.jl`: heterogeneous short-term plasticity

Loaded with SNNUtils. At load time the table `data/recordings/dsxc_mouse_fit_results.csv`
(Tsodyks-Markram fits of mouse cortical connections; its columns, such as `pre_cell.cre_type` and
`stp_induction_50hz`, follow the Allen Institute synaptic-physiology dataset) is read into
`SNNUtils.stp_data` and filtered: facilitation and recovery time constants below 10 s (then
converted to ms), fit error below its 75th percentile, and ``0.2 < U < 0.8``. The subsets
`exc_exc_stp`, `exc_inh_stp`, `inh_exc_stp`, `inh_inh_stp` select the presynaptic
(`synapse_type`) and postsynaptic (`post_cell.cell_class`) classes.

[`sample_stp_params`](@ref) draws ``(\tau_F, \tau_D, U)`` (and the fitted amplitude `w`) from a
table; [`sample_stp_campagnola`](@ref) does the same for one connection class. Each parameter is
drawn from an independent random row. The function name refers to Campagnola et al. (2022,
Science), the Allen Institute survey of synaptic connectivity and dynamics in mouse and human
neocortex; the source file does not state the reference.

| Output | Column | Units | Meaning |
| :--- | :--- | :--- | :--- |
| `τF` | `fit_tau_fac` | ms | facilitation time constant |
| `τD` | `fit_tau_rec` | ms | recovery (depression) time constant |
| `U` | `fit_U` | - | utilisation of synaptic efficacy |
| `w` | `fit_w` | as in the table | fitted amplitude (only with `weights = true`) |

The vectors `τD`, `τF`, `U` (converted to `Float32`) match the fields of `MarkramSTPParameterHet`
(see [Plasticity rules](plasticity_rules.md)).

```julia
using SpikingNeuralNetworks, SNNUtils
p = sample_stp_campagnola(100, :exc_exc)          # τF, τD, U, w for 100 samples
q = sample_stp_params(100)                        # whole table, without weights
stp = SNN.MarkramSTPParameterHet(τD = Float32.(q.τD), τF = Float32.(q.τF), U = Float32.(q.U))
```

```@autodocs
Modules = [SNNUtils]
Pages   = ["models/stp_het.jl"]
```
