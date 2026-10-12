# SNNUtils models

```@meta
CurrentModule = SNNUtils
```

Heterogeneous short-term plasticity parameters sampled from Tsodyks-Markram fits of mouse cortical
connections (Allen Institute synaptic physiology; Campagnola et al. 2022). [`sample_stp_params`](@ref)
samples the whole table, [`sample_stp_campagnola`](@ref) one connection class (`:exc_exc`,
`:exc_inh`, `:inh_exc`, `:inh_inh`).

| Output | Column | Units | Meaning |
| :--- | :--- | :--- | :--- |
| `τF` | `fit_tau_fac` | ms | facilitation time constant |
| `τD` | `fit_tau_rec` | ms | recovery (depression) time constant |
| `U` | `fit_U` | - | utilisation of synaptic efficacy |
| `w` | `fit_w` | as in the table | fitted amplitude (only with `weights = true`) |

`τD`, `τF`, `U` are the fields of `MarkramSTPParameterHet` ([Plasticity rules](plasticity_rules.md)).

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
