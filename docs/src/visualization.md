# Visualization (SNNPlots)

```@meta
CurrentModule = SNNPlots
```

Plotting is provided by [SNNPlots.jl](https://github.com/JuliaSNN/SNNPlots.jl), which is loaded and
re-exported (in part) by `SpikingNeuralNetworks`. Since version 0.2 all SNNPlots functions use
[Makie](https://docs.makie.org). SNNPlots depends on `Makie` only, so a Makie backend must be loaded
to display or save figures: `using CairoMakie` (static figures, also on headless machines) or
`using GLMakie` (interactive windows). Figures can be built without a backend; `save` and the
display need one.

`SpikingNeuralNetworks` re-exports `raster`, `raster!`, `vecplot`, `vecplot!`, `@makie_default` and
`okabe_ito_10`. The other SNNPlots functions are reachable as `SNN.stdp_kernel`, and so on.
(`raster!` was not exported by SNNPlots 0.2.10, so `SNN.raster!` was undefined.)

## Theme

When SNNPlots is loaded it applies its Makie theme with [`makie_default!`](@ref): figure size
600x400, axes without grid lines, legends without frame (label size 10), and the
colour-blind-safe Okabe-Ito palette [`okabe_ito_10`](@ref) as colour cycle for lines and bands.
Since SNNPlots 0.2.10 the theme is applied in the package `__init__`, i.e. every time the package
is loaded; before 0.2.10 it was applied only at precompilation time and was lost in the user
session. Call `SNNPlots.makie_default!()` (or the equivalent macro [`@makie_default`](@ref)) to
restore it after changing the theme with `set_theme!`.

```julia
using SpikingNeuralNetworks, CairoMakie
SNNPlots.makie_default!()          # restore the SNNPlots theme
fig = Figure()
ax = Axis(fig[1, 1])
for k in 1:3
    lines!(ax, 1:10, k .* (1:10))  # colours cycle through okabe_ito_10
end
```

## Raster plots

[`raster`](@ref) creates a new figure, [`raster!`](@ref) draws into an existing axis. Both accept a
population (or stimulus), a `NamedTuple` of populations such as `model.pop` (stacked vertically,
separated by dashed lines and labelled by population name), or a `Spiketimes` vector. The
populations must record spikes, which is enabled with `monitor!(pop, [:fire])`. The time window
is given in ms (`0:1s`, `[0, 500ms]`); with populations the time axis is shown in seconds, with a
`Spiketimes` vector in ms (labelled "Time (ms)"; the label wrongly read "Time (s)" in SNNPlots
0.2.10). `names` relabels the populations, extra keyword arguments go to `scatter!`, and the
y-limits are set on the axis given to `raster!`. At most 200 000 spikes are drawn; larger sets
are randomly subsampled with a warning.

```julia
using SpikingNeuralNetworks, CairoMakie
SNN.@load_units
E = SNN.IF(N = 80, name = "E")
I = SNN.IF(N = 20, name = "I")
input = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz), name = "input")
inE = SNN.SpikingSynapse(input, E, :ge, conn = (p = 0.2, μ = 3.0))
inI = SNN.SpikingSynapse(input, I, :ge, conn = (p = 0.2, μ = 3.0))
EI = SNN.SpikingSynapse(E, I, :ge, conn = (p = 0.2, μ = 2.0))
IE = SNN.SpikingSynapse(I, E, :gi, conn = (p = 0.2, μ = 5.0))
model = SNN.compose(; E, I, input, inE, inI, EI, IE)
SNN.monitor!(model.pop, [:fire])
SNN.sim!(; model, duration = 1s)

fig, ax, plt = SNN.raster(model.pop, 0:1s)        # all populations, new figure

fig2 = Figure()
ax2 = Axis(fig2[1, 1], xlabel = "Time (s)", ylabel = "Neuron")
SNN.raster!(ax2, model.pop.E, 0:500ms)           # one population into an existing axis
```

## Recorded variables

[`vecplot`](@ref) and [`vecplot!`](@ref) plot the time course of any recorded variable (`:v`,
`:w`, synaptic conductances, ...). The variable must be recorded with `monitor!(pop, [sym])`. The
interpolated record is sampled on `interval` (ms, default the whole record); `neurons` selects
the neurons, `pop_average = true` plots the mean (with `ribbon = true`, also the 20-80 % band),
`add_spikes = true` marks the spikes as 20 mV peaks on the trace. Three-dimensional records
(one value per compartment) need `sym_id`. The x tick labels are in seconds.

```julia
using SpikingNeuralNetworks, CairoMakie
SNN.@load_units
E = SNN.IF(N = 5, param = SNN.IFParameter(gl = 10nS, C = 200pF))
stim = SNN.CurrentStimulus(E; param = SNN.CurrentNoise(E; I_base = 300pA))
model = SNN.compose(; E, stim)
SNN.monitor!(E, [:v, :fire])
SNN.sim!(; model, duration = 300ms)

fig, ax, plt = SNN.vecplot(E, :v; interval = 0:1:300ms, neurons = 1:2,
                           add_spikes = true, ylabel = "V (mV)")
fig2 = Figure()
ax2 = Axis(fig2[1, 1], ylabel = "V (mV)")
SNN.vecplot!(ax2, E, :v; pop_average = true, ribbon = true)
```

`vecplot(P::Array, sym)` (one panel per population) and `vecplot(p, Any[:v, :w])` (one panel per
variable) return a `Figure` with stacked panels; `vecplot(p, [:v, :w])` draws all variables in
one axis with their names as legend labels. `factor` multiplies the trace by a number, by
another recorded variable (`factor = :g`) or by a `neurons x time` matrix. (In SNNPlots 0.2.10
the multi-panel methods called the Plots.jl API, the legend labels were `"nothing"`, and the
`Symbol`/`Matrix` forms of `factor` threw.)

## STDP kernels

[`stdp_kernel`](@ref) (new figure) and [`stdp_kernel!`](@ref) (existing axis) plot the weight change
of a long-term plasticity rule as a function of the spike-time difference
``\Delta t = t_{post} - t_{pre}``. Each point is a separate two-neuron `train!` run of 400 ms
([`stdp_test`](@ref)); the default grid has 40 points, so reduce `ΔTs` for a quick look. Each run
measures exactly one pre/post pair. (In SNNPlots 0.2.10 the measured synapse also drove the
postsynaptic `Identity` neuron, which added a causal pairing at ``\Delta t \approx 0.1`` ms to
every point: for `STDPGerstner()` the kernel was positive at ``\Delta t = -10`` ms.)

```julia
using SpikingNeuralNetworks, CairoMakie
SNN.@load_units
fig = SNN.stdp_kernel(SNN.STDPGerstner(); ΔTs = [-40.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 40.0])
dw = SNNPlots.stdp_test(SNN.STDPGerstner(); ΔT = 10ms)   # single pairing
```

## Spatial networks

[`plot_spatial_connectivity`](@ref) and [`plot_connection_distances`](@ref) plot the layout and the
distance dependence of spatially embedded networks with three populations named `:Exc`, `:PV`,
`:SST`. They take the connectivity and configuration objects of a specific spatial-network
workflow (fields `points`, `links`, `network`, `spatial`, or distance histograms `ds`, `rs`); see
their docstrings for the expected layout. They are not generic plots of an SNNModels model.

## Exported names

The export lists of SNNPlots 0.2.10 contained names that were not defined in the package
(`plot_model`, `plot_stimulus`, `plot_connections`, `stp_plot`, `plot_weights`, `plot_activity`,
`dendrite_gplot`, `soma_gplot`, `stdp_weight_decorrelated`, `default_colors`, `nature_figure`);
they were removed. `plot` and `plot!` are Makie's; `save_model` and
`load_model` are the SNNModels functions; `inch` and `pt` are `Measures` lengths, while `cm` is the
SNNModels length unit (`cm == 1.0f0`). The files `backend/plots.jl`, `other_plots.jl`,
`extra_plots.jl` and `old_plot.jl` of the source tree belong to the former Plots.jl interface and
are not loaded.

## API

```@autodocs
Modules = [SNNPlots]
Pages   = ["SNNPlots.jl", "backend/makie.jl", "raster.jl", "vecplot.jl", "stdp_plots.jl", "spatial.jl"]
```
