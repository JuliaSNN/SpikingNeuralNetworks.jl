# Visualization (SNNPlots)

```@meta
CurrentModule = SNNPlots
```

After this page you can draw raster plots, traces of recorded variables and STDP kernels.

[SNNPlots.jl](https://github.com/JuliaSNN/SNNPlots.jl) plots with [Makie](https://docs.makie.org);
load a backend to display or save figures: `using CairoMakie` (files, headless machines) or
`using GLMakie` (interactive windows). `raster`, `raster!`, `vecplot`, `vecplot!`, `@makie_default`
and `okabe_ito_10` are re-exported; the other functions are reachable as `SNN.stdp_kernel`, etc.

## Theme

Loading SNNPlots applies its Makie theme ([`makie_default!`](@ref)): 600x400 figures, no grid
lines, frameless legends, and the colour-blind-safe Okabe-Ito palette [`okabe_ito_10`](@ref).
Call `SNNPlots.makie_default!()` (or [`@makie_default`](@ref)) to restore it after `set_theme!`.

```@example visualization
using SpikingNeuralNetworks, CairoMakie
SNNPlots.makie_default!()          # restore the SNNPlots theme
fig = Figure()
ax = Axis(fig[1, 1])
for k in 1:3
    lines!(ax, 1:10, k .* (1:10))  # colours cycle through okabe_ito_10
end
fig
```

## Raster plots

[`raster`](@ref) creates a new figure, [`raster!`](@ref) draws into an existing axis. Both accept a
population (or stimulus) that records `:fire`, a `NamedTuple` of populations such as `model.pop`
(stacked, labelled by name), or a `Spiketimes` vector. The time window is in ms (`0:1s`); the axis
is in seconds for populations, in ms for a `Spiketimes` vector. `names` relabels the populations,
other keywords go to `scatter!`. Above 200 000 spikes the plot is subsampled, with a warning.

```@example visualization
using SpikingNeuralNetworks, CairoMakie
SNN.@load_units
E = SNN.IF(N = 80, name = "E")
I = SNN.IF(N = 20, name = "I")
input = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz), name = "input")
inE = SNN.SpikingSynapse(input, E, :ge, conn = (p = 0.2, μ = 3.0))
inI = SNN.SpikingSynapse(input, I, :ge, conn = (p = 0.2, μ = 3.0))
EI = SNN.SpikingSynapse(E, I, :ge, conn = (p = 0.2, μ = 2.0))
IE = SNN.SpikingSynapse(I, E, :gi, conn = (p = 0.2, μ = 5.0))
model = SNN.compose(; E, I, input, inE, inI, EI, IE, silent = true)
SNN.monitor!(model.pop, [:fire])
SNN.sim!(; model, duration = 1s)

fig, ax, plt = SNN.raster(model.pop, 0:1s)        # all populations, new figure

fig2 = Figure()
ax2 = Axis(fig2[1, 1], xlabel = "Time (s)", ylabel = "Neuron")
SNN.raster!(ax2, model.pop.E, 0:500ms)           # one population into an existing axis
fig
```

## Recorded variables

[`vecplot`](@ref) and [`vecplot!`](@ref) plot the time course of any recorded variable (`:v`,
`:w`, synaptic conductances, ...). The variable must be recorded with `monitor!(pop, [sym])`. The
interpolated record is sampled on `interval` (ms, default the whole record); `neurons` selects
the neurons, `pop_average = true` plots the mean (with `ribbon = true`, also the 20-80 % band),
`add_spikes = true` marks the spikes as 20 mV peaks on the trace. Three-dimensional records
(one value per compartment) need `sym_id`. The x tick labels are in seconds.

```@example visualization
using SpikingNeuralNetworks, CairoMakie
SNN.@load_units
E = SNN.IF(N = 5, param = SNN.IFParameter(gl = 10nS, C = 200pF))
stim = SNN.CurrentStimulus(E; param = SNN.CurrentNoise(E; I_base = 300pA))
model = SNN.compose(; E, stim, silent = true)
SNN.monitor!(E, [:v, :fire])
SNN.sim!(; model, duration = 300ms)

fig, ax, plt = SNN.vecplot(E, :v; interval = 0:1:300ms, neurons = 1:2,
                           add_spikes = true, ylabel = "V (mV)")
fig2 = Figure()
ax2 = Axis(fig2[1, 1], ylabel = "V (mV)")
SNN.vecplot!(ax2, E, :v; pop_average = true, ribbon = true)
fig
```

`vecplot(P::Array, sym)` (one panel per population) and `vecplot(p, Any[:v, :w])` (one panel per
variable) return a `Figure` with stacked panels; `vecplot(p, [:v, :w])` draws all variables in
one axis with their names as legend labels. `factor` multiplies the trace by a number, by
another recorded variable (`factor = :g`) or by a `neurons x time` matrix.

## STDP kernels

[`stdp_kernel`](@ref) (new figure) and [`stdp_kernel!`](@ref) (existing axis) plot the weight change
of a long-term plasticity rule as a function of the spike-time difference
``\Delta t = t_{post} - t_{pre}``. Each point is a separate two-neuron `train!` run of 400 ms
([`stdp_test`](@ref)) measuring one pre/post pair; the default grid has 40 points, so reduce `ΔTs`
for a quick look.

```@example visualization
using SpikingNeuralNetworks, CairoMakie
SNN.@load_units
fig = SNN.stdp_kernel(SNN.STDPGerstner(); ΔTs = [-40.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 40.0])
dw = SNNPlots.stdp_test(SNN.STDPGerstner(); ΔT = 10ms)   # single pairing
fig
```

## Spatial networks

[`plot_spatial_connectivity`](@ref) and [`plot_connection_distances`](@ref) plot the layout and the
distance dependence of spatially embedded networks with three populations named `:Exc`, `:PV`,
`:SST`. They take the connectivity and configuration objects of a specific spatial-network
workflow (fields `points`, `links`, `network`, `spatial`, or distance histograms `ds`, `rs`); see
their docstrings for the expected layout. They are not generic plots of an SNNModels model.

## API

```@autodocs
Modules = [SNNPlots]
Pages   = ["SNNPlots.jl", "backend/makie.jl", "raster.jl", "vecplot.jl", "stdp_plots.jl", "spatial.jl"]
```
