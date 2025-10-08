using Revise
using SpikingNeuralNetworks
using DrWatson
SNN.@load_units;
using SNNUtils
using Statistics
using Random
using StatsBase
using SparseArrays
using Distributions
using StatsPlots
using Logging

# %%
# Instantiate a  Symmetric STDP model with these parameters:
include("parameters.jl")

WW = zeros(2, 4, length(NSSTs))
path = datadir("Lagzi2022_AssemblyFormation", "mixed_inh")
Threads.@threads for t in eachindex(NSSTs)
    @unpack stim_τ, stim_rate = config
    info = (τ = stim_τ, rate = stim_rate, signal = :off, NSST = NSSTs[t])
    !isfile(get_path(path = path, name = "Model_sst", info = info)) && continue
    @show info
    model = load_model(path, "Model_sst", info).model
    #recurrent
    WW[1, 1, t] = mean(model.syn.E1_to_E1.W)
    WW[2, 1, t] = mean(model.syn.E2_to_E2.W)
    #later
    WW[1, 2, t] = mean(model.syn.E1_to_E2.W)
    WW[2, 2, t] = mean(model.syn.E2_to_E1.W)
    #SST rerrent
    WW[1, 3, t] = mean(model.syn.SST1_to_E1.W)
    WW[2, 3, t] = mean(model.syn.SST2_to_E2.W)
    #SST laral
    WW[1, 4, t] = mean(model.syn.SST1_to_E2.W)
    WW[2, 4, t] = mean(model.syn.SST2_to_E1.W)
    # fr, r = firing_rate(model.pop, interval = 0s:5ms:200s,  τ=40ms, interpolate=false)
end
##
W = mean(WW, dims = 1)[1, :, :]
p1, p2 = plot(), plot()
p1=plot!(p1, NSSTs, W[1, :], label = "Recurrent", lw = 3)
plot!(
    NSSTs,
    W[2, :],
    label = "Lateral",
    lw = 3,
    title = "Average EE weight",
    xlabel = "% NSST",
    ylabel = "Weight",
)
p2 = plot!(p2, NSSTs, W[3, :], label = "Recurrent", lw = 3)
plot!(
    NSSTs,
    W[4, :],
    label = "Lateral",
    lw = 3,
    title = "Average SST weight",
    xlabel = "% NSSTs",
    ylabel = "Weight",
)
plot(p1, p2, layout = (2, 1), size = (500, 700), margin = 5Plots.mm)



##
WEE = W[1, :, :]
WSST = W[4, :, :]
p1 = heatmap(
    stim_rates,
    τs,
    WEE',
    c = :roma,
    xlabel = "Stim rate",
    ylabel = "τ",
    title = "E1 to E1",
    size = (800, 800),
    margin = 5Plots.mm,
    yscale = :log10,
)
p2 = heatmap(
    stim_rates,
    τs,
    WSST',
    c = :roma,
    xlabel = "Stim rate",
    ylabel = "τ",
    title = "SST1 to E2",
    size = (800, 800),
    margin = 5Plots.mm,
    yscale = :log10,
)

WEE = (W[1, :, :] - W[2, :, :]) ./ W[1, :, :]
WSST = (W[4, :, :] - W[3, :, :]) ./ W[4, :, :]
p3 = heatmap(
    stim_rates,
    τs,
    WEE',
    c = :bluesreds,
    xlabel = "Stim rate",
    ylabel = "τ",
    title = "lateral",
    size = (800, 800),
    margin = 5Plots.mm,
    yscale = :log10,
    clims = (-1, 1),
)
p4 = heatmap(
    stim_rates,
    τs,
    WSST',
    c = :redsblues,
    xlabel = "Stim rate",
    ylabel = "τ",
    title = " ",
    size = (800, 800),
    margin = 5Plots.mm,
    yscale = :log10,
    clims = (-1, 1),
)
# heatmap(stim_rates, τs, W1[4,:,:], c=:roma, xlabel="Stim rate", ylabel="τ", title="E1 to E1", size=(800, 800), margin=5Plots.mm, yscale=:log10)
# #%%
plot(p1, p2, p3, p4, layout = (2, 2), size = (900, 800), margin = 5Plots.mm)

##


info = (τ = 100ms, rate = 0.5)
train_model = load_data(path, "Model_sst", info).model
p1 = raster(
    train_model.pop,
    200s:205s,
    every = 1,
    size = (800, 500),
    margin = 5Plots.mm,
    link = :x,
    legend = :none,
    yrotation = 0,
)

info = (τ = 100ms, rate = 0.5, signal = :off)
test_model = load_data(path, "Model_sst", info).model
p2 = raster(
    test_model.pop,
    200s:205s,
    every = 1,
    size = (800, 500),
    margin = 5Plots.mm,
    link = :x,
    legend = :none,
    yrotation = 0,
)


plot(p1, p2, layout = (2, 1), size = (1200, 800), margin = 5Plots.mm)
