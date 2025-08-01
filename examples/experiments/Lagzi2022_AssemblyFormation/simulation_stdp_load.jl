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
include("plots.jl")

WW = zeros(2, 4, length(stim_rates), length(τs))
frs  = Matrix{Any}(undef, length(stim_rates), length(τs))
path = datadir("zeus", "Lagzi2022_AssemblyFormation", "mixed_inh", "baseline")
Threads.@threads for t in eachindex(τs)
    for r in eachindex(stim_rates)
        for s in []
        stim_rate = stim_rates[r]
        stim_τ = τs[t]
        info = (τ= stim_τ, rate=stim_rate, signal=:off)
        !isfile(get_path(path=path, name="Model_sst", info=info)) && continue
        model = load_model(path, "Model_sst", info).model
        #recurrent
        WW[1,1,r,t] = mean(model.syn.E1_to_E1.W)
        WW[2,1,r,t] = mean(model.syn.E2_to_E2.W)
        #lateral
        WW[1,2,r,t] = mean(model.syn.E1_to_E2.W)
        WW[2,2,r,t] = mean(model.syn.E2_to_E1.W)
        #SST recurrent
        WW[1,3,r,t] = mean(model.syn.SST1_to_E1.W)
        WW[2,3,r,t] = mean(model.syn.SST2_to_E2.W)
        #SST lateral
        WW[1,4,r,t] = mean(model.syn.SST1_to_E2.W)
        WW[2,4,r,t] = mean(model.syn.SST2_to_E1.W)
        # fr, r = firing_rate(model.pop, interval = 0s:5ms:200s,  τ=40ms, interpolate=false)
    end
end
W = mean(WW, dims=1)[1,:,:,:]
p1, p2 = plot(), plot()
r = 3
p1=plot!(p1,τs, W[1,r,:], label="Recurrent", lw=3)
plot!(τs, W[2,r,:], label= "Lateral", lw=3, title="Average EE weight", xlabel="τ (ms)", ylabel="Weight")
p2 =plot!(p2, τs, W[3,r,:], label="Recurrent", lw=3)
plot!(τs, W[4,r,:], label= "Lateral", lw=3, title="Average SST weight", xlabel="τ (ms)", ylabel="Weight")
plot(p1,p2, layout=(2,1), size=(500, 700), margin=5Plots.mm, xscale=:log10)



##
WEE = W[1,:,:]
WSST = W[4,:,:] 
p1 = heatmap(stim_rates, τs, WEE', c=:roma, xlabel="Stim rate", ylabel="τ", title="E1 to E1", size=(800, 800), margin=5Plots.mm, yscale=:log10, )
p2 = heatmap(stim_rates, τs, WSST', c=:roma, xlabel="Stim rate", ylabel="τ", title="SST1 to E2", size=(800, 800), margin=5Plots.mm, yscale=:log10)

WEE = (W[1,:,:] - W[2,:,:])./W[1,:,:]
WSST = (W[4,:,:] - W[3,:,:])./ W[4,:,:]
p3 = heatmap(stim_rates, τs, WEE', c=:bluesreds, xlabel="Stim rate", ylabel="τ", title="lateral", size=(800, 800), margin=5Plots.mm, yscale=:log10, clims=(-1,1))
p4 = heatmap(stim_rates, τs, WSST', c=:redsblues, xlabel="Stim rate", ylabel="τ", title=" ", size=(800, 800), margin=5Plots.mm, yscale=:log10, clims=(-1,1))
# heatmap(stim_rates, τs, W1[4,:,:], c=:roma, xlabel="Stim rate", ylabel="τ", title="E1 to E1", size=(800, 800), margin=5Plots.mm, yscale=:log10)
# #%%
plot(p1,p2, p3, p4, layout=(2,2), size=(900, 800), margin=5Plots.mm)

##

for t in eachindex(τs)
    for r in eachindex(stim_rates)

#

info = (τ= 100ms, rate= 0.5)
train_model = load_data(path, "Model_sst", info).model
p1 = raster(train_model.pop, 200s:205s, every=1, size=(800, 500), margin=5Plots.mm, link=:x, legend=:none, yrotation=0)

info = (τ= 100ms, rate= 0.5, signal=:off)
test_model = load_data(path, "Model_sst", info).model
p2 = raster(test_model.pop, 200s:205s, every=1, size=(800, 500), margin=5Plots.mm, link=:x, legend=:none, yrotation=0)


plot(p1, p2, layout=(2,1), size=(1200, 800), margin=5Plots.mm)