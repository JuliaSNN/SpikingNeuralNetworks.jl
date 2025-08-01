using Revise
using SpikingNeuralNetworks
using DrWatson
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using Random
using StatsBase
using SparseArrays
using Distributions
@unpack rates, interval =
    DrWatson.load(datadir("zeus", "ExpData", "ACrates.jld2")) |> dict2ntuple
sound = rates[1]

##
ss = 50
spikes = SNN.sample_spikes(500, sound[ss, :], interval)
r = 0:1:500
fr, r = firing_rate(spikes, interval = r, τ = 1ms, interpolate = false)
fr
plot(r, fr)
plot!(1:501, sound[ss, :])
##

sound = mean(rates)
interval = 1:500
a = SNN.sample_inputs(70, sound, interval)
shift_spikes!(a, 0.1s)
plot(
    raster(a),
    heatmap(sound, c = :viridis, size = (800, 400), cbar = false),
    size = (800, 400),
)
