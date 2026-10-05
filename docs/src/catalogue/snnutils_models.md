# SNNUtils models

```@meta
CurrentModule = SNNUtils
```

The folder `src/models/` of [SNNUtils.jl](https://github.com/JuliaSNN/SNNUtils.jl) collects
parameter sets of published or in-house networks: receptor kinetics, neuron parameters,
connectivity rules and plasticity parameters, expressed with the SNNModels types. They are
parameter collections, not new model types: the dynamics are those of the SNNModels populations,
synapses and plasticity rules they instantiate (see the other pages of the
[Model catalogue](index.md)).

Only `stp_het.jl` is loaded by SNNUtils 0.2.9. All other files are present in the source tree but
are **not included** by `SNNUtils.jl` (nor by `models/models.jl`, which is itself not included).
Several of them use types that no longer exist in SNNModels 1.8.4 and cannot be loaded. The
table summarises the status, tested with SNNModels 1.8.4 and SNNUtils 0.2.9.

| File | Content | Reference | Loaded | Usable with SNNModels 1.8.4 |
| :--- | :--- | :--- | :--- | :--- |
| `stp_het.jl` | short-term-plasticity parameters sampled from experimental fits | see below | yes | yes |
| `quaresima2022.jl` | receptor sets of the Tripod neuron (`quaresima2022`, `EyalGluDend`, `MilesGabaDend`, ...) | receptor references cited in the file | no | yes, with `include` |
| `quaresima_2024_updown.jl` | dendritic glutamatergic receptors with variable NMDA/AMPA ratio | reference not given in the code | no | partly (needs `quaresima2022.jl`; `quaresima2022_nar` fails) |
| `quaresima2023.jl` | iSTDP, vSTDP and connectivity parameters | reference not given in the code | no | plasticity: yes; connectivity: see note (1) |
| `duarte2019.jl` | PV, SST and AdEx cell parameters | Duarte & Morrison (2019), inferred from the name | no | no (`IFParameterGsyn`, `AdExParameterGsyn` undefined) |
| `lkd2014.jl` | AdEx E and IF PV parameters | Litwin-Kumar & Doiron (2014), cited in the file | no | no (keywords and types removed) |
| `connections.jl` | connectivity rules of several networks | names refer to Duarte & Morrison and Litwin-Kumar & Doiron; not cited | no | loads; see note (1) |
| `dendrite_STM.jl` | dendritic short-term-memory network parameters | reference not given in the code | no | no (`AdExSoma`, `IFParameterGsyn` undefined) |
| `mongillo_WM2008.jl` | `Mongillo2008()` network constructor | Mongillo, Barak & Tsodyks (2008), inferred from the name | no | no (`IFCurrent`, `IFCurrentDeltaParameter` undefined) |
| `learning/structs.jl` | legacy plasticity parameter structs | Clopath 2010, Vogels 2011, Gütig 2003 named in comments | no | definitions load, not used by SNNModels |
| `models.jl` | includes the first six files above | - | no | no (fails at `duarte2019.jl`) |

(1) The connection rules store the distribution as a type (`dist = Normal`, `dist = LogNormal`),
while `SpikingSynapse(...; conn = rule)` in SNNModels 1.8.4 expects its name as a `Symbol`
(`dist = :Normal`); passing these rules unchanged raises a `TypeError`. Rules without `dist` (for
example `lkd2014_dend.If_to_E`) work; for the others, replace the field, e.g.
`merge(rule, (dist = nameof(rule.dist),))`.

## How to use the files that are not loaded

The usable files can be evaluated in the user session with `include`, after bringing the SNNModels
types (receptors, `EyalNMDA`, plasticity rules) and `Distributions` into scope. The path of the
installed package is `pkgdir(SNNUtils)`:

```julia
using SpikingNeuralNetworks, SNNUtils
using SpikingNeuralNetworks.SNNModels                     # EyalNMDA, receptor and plasticity types
using SNNUtils.Distributions, SNNUtils.Parameters         # Normal, LogNormal, @with_kw
SNN.@load_units
models_dir = joinpath(pkgdir(SNNUtils), "src", "models")
for file in ("quaresima2022.jl", "quaresima_2024_updown.jl", "quaresima2023.jl",
             "connections.jl", joinpath("learning", "structs.jl"))
    Base.include(@__MODULE__, joinpath(models_dir, file))   # `include(...)` at the REPL
end
quaresima2022.dend_syn              # receptors of the Tripod dendrites
EyalEquivalentNAR(1.8)              # dendritic receptors with NMDA/AMPA ratio 1.8
quaresima2023.plasticity.vstdp      # vSTDPParameter
rule = lkd2014_dend.E_to_Ed           # (p = 0.2, μ = 15.78, dist = Normal)
rule = merge(rule, (dist = nameof(rule.dist),))   # dist = :Normal, as SNNModels expects
E = SNN.IF(N = 100)
EE = SNN.SpikingSynapse(E, E, :ge, conn = rule)
```

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

## `quaresima2022.jl`: Tripod receptors

Receptor parameter sets used with the Tripod dendritic neuron (the file name refers to the Tripod
neuron model of Quaresima et al.; the file does not cite that paper). Each set is a
`Glutamatergic` (AMPA, NMDA) or `GABAergic` (GABA``_A``, GABA``_B``) pair of `Receptor`s with
reversal potential `E_rev` (mV), rise and decay time constants `τr`, `τd` (ms) and peak
conductance `g0` (nS); with the default ``g_0 = 0`` the receptor conductance `gsyn` is zero.

| Name | Receptor 1 (`E_rev`, `τr`, `τd`, `g0`) | Receptor 2 | Reference cited in the file |
| :--- | :--- | :--- | :--- |
| `KochGlu` | AMPA: 0, 0.2, 25.0, 0.73 | `ReceptorVoltage(gsyn = -1)` (other fields default) | Koch, *Biophysics of Computation* (1999) |
| `EyalGluDend` | AMPA: 0, 0.26, 2.0, 0.73 | NMDA: 0, 8, 35.0, 1.31 (`nmda = 1`) | Eyal et al. (2018), Front. Cell. Neurosci. |
| `EyalGluDend_nonmda` | AMPA: 0, 0.25, 2.0, 0 | NMDA: 0, 8, 35.0, 0 (`nmda = 0`) | Eyal et al. (2018) |
| `MilesGabaDend` | GABA``_A``: -70, 4.8, 29.0, 0.27 | GABA``_B``: -90, 30, 400.0, 0.006 | Miles et al. (1996), Neuron |
| `MilesGabaSoma` | GABA``_A``: -75, 0.5, 6.0, 0.265 | `Receptor(τr = -1)` (``g_0 = 0``: no conductance) | Miles et al. (1996) |
| `DuarteGluSoma` | AMPA: 0, 0.25, 2.0, 0.73 | `ReceptorVoltage(τr = -1)` (``g_0 = 0``: no conductance) | reference not given in the code |

`quaresima2022` bundles `dends = [(150um, 400um), (150um, 400um)]` (length ranges of the two
dendrites), `soma_syn = Receptors(DuarteGluSoma, MilesGabaSoma)`,
`dend_syn = Receptors(EyalGluDend, MilesGabaDend)` and `NMDA = EyalNMDA` (the SNNModels NMDA
magnesium-block parameters). `EyalNMDA` is re-exported by the file but defined in SNNModels.

## `quaresima_2024_updown.jl`: variable NMDA/AMPA ratio

Requires the definitions of `quaresima2022.jl`.
- `EyalGluNAR(NAR = 1.8, τd = 35ms)`: dendritic `Glutamatergic` receptors with AMPA
  ``g_0 = 0.73\,(1 + \mathrm{NAR}_0 - \mathrm{NAR})`` (``\mathrm{NAR}_0 = 1.31/0.73``), rise 0.25 ms,
  decay 2 ms, and NMDA ``g_0 = 0.73\,\mathrm{NAR}``, rise 8 ms, decay `τd`.
- `EyalEquivalentNAR(NAR, τd = 35)`: `Receptors(EyalGluNAR(NAR, τd), MilesGabaDend)`.
- `quaresima2022_nar(nar, τ = 35ms)`: Tripod configuration with these receptors; it uses the
  removed type `AdExSoma` and fails with SNNModels 1.8.4.
- `quaresima2022_nonmda` is exported but not defined.

Reference not given in the code.

## `quaresima2023.jl`: plasticity and connectivity

`quaresima2023.plasticity` contains
- `iSTDP_rate = iSTDPRate(η = 0.2, τy = 5ms, r = 10Hz, Wmax = 243.4pF, Wmin = 0.1pF)`;
- `iSTDP_potential = iSTDPPotential(η = 0.2, v0 = -70mV, τy = 5ms, Wmax = 243.4pF, Wmin = 0.1pF)`;
- `vstdp = vSTDPParameter(A_LTD = 14.0f-4, A_LTP = 8.0f-4, θ_LTD = -60.0, θ_LTP = -25.0, τu = 15.0,
  τv = 45.0, τx = 20.0, Wmin = 2.78, Wmax = 41.4)`.

`quaresima2023.connectivity` contains connection rules `(p, μ, dist, σ)` between an excitatory
population (`E`, dendrites `Ed`) and two inhibitory populations (`If`, `Is`), all with
``p = 0.2``: `E_to_Ed` (Normal, ``\mu = 10.78``, ``\sigma = 1``) and log-normal rules with
``\mu = \log(\cdot)`` of 16.0 (`E_to_If`, `E_to_Is`, `Is_to_Ed`), 16.8 (`If_to_E`), 5.83
(`If_to_Is`), 16.2 (`If_to_If`, `Is_to_Is`), 5.47 (`Is_to_If`), with ``\sigma = 0``.
`ballstick_network` is exported but not defined. Reference not given in the code. The connectivity
rules need the conversion of note (1).

## `duarte2019.jl`: PV, SST and AdEx cells

Parameters "without NMDA and GABA``_B`` synapses" (comment in the file) of a PV interneuron
(``\tau_m = 104.52\,\mathrm{pF}/9.75\,\mathrm{nS}``, ``E_L = -64.33`` mV, ``V_t = -38.97`` mV,
``V_r = -57.47`` mV, refractory 0.5 ms), an SST interneuron (``\tau_m = 102.86/4.61`` ms,
``E_L = -61`` mV, ``V_t = -34.4`` mV, ``V_r = -47.11`` mV, refractory 1.3 ms, adaptation
``b = 80.5`` pA, ``\tau_w = 144`` ms) and an AdEx excitatory cell (``E_L = -76.43`` mV,
``\tau_m = 116.5/4.64`` ms, ``V_t = -44.45`` mV, refractory 2.05 ms), with excitatory and
inhibitory synaptic rise/decay times and peak conductances. The name suggests Duarte & Morrison
(2019, PLoS Comput. Biol.); the file does not cite it. It uses `IFParameterGsyn` and
`AdExParameterGsyn`, which do not exist in SNNModels 1.8.4.

## `lkd2014.jl`: Litwin-Kumar and Doiron (2014)

Cited in the file: Litwin-Kumar, A., & Doiron, B. (2014). Formation and maintenance of neuronal
assemblies through synaptic plasticity. *Nature Communications*, 5. `LKD2014` holds an AdEx
excitatory neuron (``E_L = -70`` mV, ``V_t = -52`` mV, ``\tau_m = 300\,\mathrm{pF}/15\,\mathrm{nS}``,
``V_r = -60`` mV, refractory 1 ms, adaptive threshold ``A_t = 10`` mV, ``E_E = 0`` mV,
``E_I = -75`` mV) and an IF PV neuron (``E_L = -62`` mV, ``V_t = -52`` mV, ``V_r = -57.47`` mV,
``\tau_m = 20`` ms), with excitatory/inhibitory rise and decay times 1/6 ms and 0.5/2 ms.
`LKD2014SingleExp` is the single-exponential-synapse variant. The parameter constructors receive
keywords (`τabs`, `τri`, `τde`, `E_i`, `At`, ...) that `AdExParameter`/`IFParameter` no longer
have, and the types `AdExSinExpParameter`, `IFSinExpParameter` do not exist: the file cannot be
loaded with SNNModels 1.8.4.

## `connections.jl`: connectivity rules

NamedTuples of connection rules: each entry `(p, μ, dist, σ)` gives the connection probability,
the mean weight (for `LogNormal`, the log-mean) and the weight distribution, as accepted by the
`conn` keyword of `SpikingSynapse` in SNNModels. Population names: `E`/`Es`/`Ed` (excitatory,
soma, dendrite), `If` (fast, PV-like), `Is` (slow, SST-like).

| Name | Content |
| :--- | :--- |
| `duarte_types` | population fractions `[0.8, 0.2 * 0.65, 0.2 * 0.35]` (E, PV, SST) |
| `pv_only`, `sst_only` | fractions `[0.8, 0.0, 0.2]` and `[0.8, 0.2, 0.0]` |
| `duartemorrison2017_dend`, `duartemorrison2017_soma` | identical rule sets, ``p`` from 0.168 to 0.60 |
| `lkd2014_dend`, `lkd2014_dend_upinh`, `lkd2014_soma` | ``p = 0.2`` (0.4 for `upinh`, except `E_to_Ed`) |
| `lkd2014_soma_j(j0)` | function: `lkd2014_soma` with E-to-E mean `j0` |
| `quaresima2023_dend` | log-normal rules with small ``\sigma`` |
| `no_connections` | all ``p = 0`` |

The names refer to Duarte & Morrison and to Litwin-Kumar & Doiron (2014); the file does not cite
them. See note (1) at the top of the page for the `dist` field.

## `dendrite_STM.jl`: dendritic short-term memory network

A single `let` block defining `dendritic_stp_network`, with an excitatory dendritic neuron
(two dendrites of 150-400 µm, Eyal NMDA kinetics, AdEx soma with ``C = 281`` pF, ``g_L = 40`` nS,
``a = 4`` nS, ``b = 10.5`` pA, ``\tau_w = 144`` ms), PV and SST parameters as in `duarte2019.jl`,
iSTDP and vSTDP parameters, an `STPParameter()` entry, connectivity rules and the rates of the
background noise inputs (4.0, 2.5 and 3.5 kHz). Reference not given in the code. It uses
`AdExSoma` and `IFParameterGsyn`, which do not exist in SNNModels 1.8.4.

## `mongillo_WM2008.jl`: synaptic theory of working memory

`Mongillo2008(; n_assemblies = 1, n_neurons = 800)` builds an 8000 E / 2000 I network of
current-based integrate-and-fire neurons (E: ``\tau_m = 15`` ms, ``V_t = 20`` mV, ``V_r = 16`` mV;
I: ``\tau_m = 10`` ms, ``V_r = 13`` mV; refractory 2 ms), random connections with ``p = 0.2``,
uniform delays 1-5 ms, short-term plasticity on E-to-E synapses, Gaussian external currents
(mean 19.8) and `n_assemblies` potentiated assemblies of `n_neurons` neurons. The name refers to
Mongillo, Barak & Tsodyks (2008), *Synaptic theory of working memory*, Science; the file does not
cite it. It uses `IFCurrent` and `IFCurrentDeltaParameter`, which do not exist in SNNModels 1.8.4,
so calling `Mongillo2008()` fails.

## `learning/structs.jl`: legacy plasticity structs

Plain parameter structs from an earlier version of the library, not used by SNNModels:
`STDP` (voltage-based rule, comment "Clopath 2010"), `ISTDP` (inhibitory STDP, comment
"Vogel 2011", i.e. Vogels et al. 2011), `TripletRule` (triplet STDP amplitudes and time
constants; reference not given in the code) and `NLTAH` (comment "Gutig 2003"). The corresponding
rules of SNNModels are `vSTDPParameter`, `iSTDPRate`/`iSTDPPotential` and `STDPTriplet`
(see [Plasticity rules](plasticity_rules.md)).
