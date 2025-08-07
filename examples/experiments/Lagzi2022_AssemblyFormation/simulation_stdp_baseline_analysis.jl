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
##
include("parameters.jl")
include("plots.jl")
# %%
# Instantiate a  Symmetric STDP model with these parameters:
# Inspect models over a grid of parameters:
# NSSTs = 0:10:100
# synapses = [:ampa, :nmda]
# %%
WW = zeros(2, 4, length(NSSTs), length(synapses))
frs = Matrix{Any}(undef, length(NSSTs), length(synapses))
path = datadir("zeus", "Lagzi2022_AssemblyFormation", "mixed_inh", "baseline")
models = Matrix{Any}(undef, length(NSSTs), length(synapses))
Threads.@threads for n in eachindex(NSSTs)
    for s in eachindex(synapses)
        syn = synapses[s]
        nsst = NSSTs[n]
        @unpack stim_τ, stim_rate = config
        info = (τ = stim_τ, rate = stim_rate, NSST = nsst, syn = syn)
        !isfile(get_path(path = path, name = "Model_sst", info = info)) && continue
        model = load_model(path, "Model_sst", info).model
        #recurrent
        WW[1, 1, n, s] = mean(model.syn.E1_to_E1.W)
        WW[2, 1, n, s] = mean(model.syn.E2_to_E2.W)
        #lateras
        WW[1, 2, n, s] = mean(model.syn.E1_to_E2.W)
        WW[2, 2, n, s] = mean(model.syn.E2_to_E1.W)
        #SST renusrent
        WW[1, 3, n, s] = mean(model.syn.SST1_to_E1.W)
        WW[2, 3, n, s] = mean(model.syn.SST2_to_E2.W)
        #SST lanesal
        WW[1, 4, n, s] = mean(model.syn.SST1_to_E2.W)
        WW[2, 4, n, s] = mean(model.syn.SST2_to_E1.W)
        # fr, r = firing_rate(model.pop, interval = 0s:5ms:200s,  τ=40ms, interpolate=false)
        models[n, s] = model
    end
end
W = mean(WW, dims = 1)[1, :, :, :]
p1, p2 = plot(), plot()
p1=plot!(
    p1,
    NSSTs,
    W[1, :, 1],
    label = "Recurrent - $(synapses[1])",
    lw = 3,
    title = "Exc to Exc",
)
p1=plot!(
    NSSTs,
    W[1, :, 2],
    label = "Recurrent - $(synapses[2])",
    lw = 3,
    title = "Exc to Exc",
)
p1=plot!(
    NSSTs,
    W[2, :, 1],
    label = "Lateral - $(synapses[1])",
    lw = 3,
    title = "Exc to Exc",
)
p1=plot!(
    NSSTs,
    W[2, :, 2],
    label = "Lateral - $(synapses[2])",
    lw = 3,
    title = "Exc to Exc",
)

p2=plot!(
    p2,
    NSSTs,
    W[3, :, 1],
    label = "Recurrent - $(synapses[1])",
    lw = 3,
    title = "SST to Exc",
)
p2=plot!(
    NSSTs,
    W[3, :, 2],
    label = "Recurrent - $(synapses[2])",
    lw = 3,
    title = "SST to Exc",
)
p2=plot!(
    NSSTs,
    W[4, :, 1],
    label = "Lateral - $(synapses[1])",
    lw = 3,
    title = "SST to Exc",
)
p2=plot!(
    NSSTs,
    W[4, :, 2],
    label = "Lateral - $(synapses[2])",
    lw = 3,
    title = "SST to Exc",
)
plot(p1, p2, layout = (2, 1), size = (800, 700), margin = 5Plots.mm, legend = :outerright)

##
# nsst = 30
# syn = :nmda
NSSTs = 10:10:90
path = datadir("zeus", "Lagzi2022_AssemblyFormation", "mixed_inh", "baseline")
for s in eachindex(synapses)
    for n in eachindex(NSSTs)
        global path, config
        syn = synapses[s]
        nsst = NSSTs[n]
        info = (τ = stim_τ, rate = stim_rate, NSST = nsst, syn = syn)
        data = load_data(path, "Model_sst", info)
        isnothing(data) && continue
        model = data.model
        config = data.config
        raster(
            model.pop,
            400s:405s,
            every = 10,
            size = (800, 500),
            margin = 5Plots.mm,
            link = :x,
            legend = :none,
            yrotation = 0,
        )

        plot_path =
            plotsdir("AssemblyLagzi2022", "baseline_model", "syn-$(syn)_nsst-$(nsst)") |>
            mkpath
        learning_plot(plot_path, model, config)
    end
end
