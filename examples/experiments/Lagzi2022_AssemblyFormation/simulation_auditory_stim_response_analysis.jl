using Revise
using DrWatson
using SpikingNeuralNetworks
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
include("protocol.jl")
include("plots.jl")
include("experiments_config.jl")

root = datadir("helix", "Lagzi2022_AssemblyFormation", "mixed_inh")
@assert isdir(root)

# ResponsePath = datadir("helix", "Lagzi2022_AssemblyFormation", "mixed_inh", "SoundResponseExp5_LowInput")
# @assert isdir(ResponsePath)

SYNAPSE = :nmda
TRAIN = false

function test_experiment(ResponsePath, SYNAPSE = :nmda, TRAIN = false)
    recordings = []
    models = []
    for t in eachindex(NSSTs)
        try
            @unpack stim_τ, stim_rate = config
            # # Set model info
            info = (
                τ = stim_τ,
                rate = stim_rate,
                train = TRAIN,
                signal = :off,
                NSST = NSSTs[t],
                syn = SYNAPSE,
            )
            exp_config = (
                delay = 1s,
                repetitions = 40,
                warmup = 20s,
                target_pop = :E1,
                input_strength = 200,
            )
            rec_name =
                joinpath(ResponsePath, savename("SoundResponseRecordings", info, "jld2"))
            model = load_model(ResponsePath, "SoundResponse", info)
            rec_name = joinpath(
                ResponsePath,
                "recordings",
                savename("SoundResponseRecordings", info, "jld2"),
            )
            push!(recordings, (load(rec_name) |> dict2ntuple))
            push!(models, model)
        catch
        end
    end
    (
        plot = response_plot(recordings, [model.model for model in models]),
        recordings = recordings,
        models = models,
    )
end

plots = map(response_experiments_names) do exp
    ExpPath = joinpath(root, getfield(response_experiments, exp).name)
    @info "Running experiment: $exp"
    p = test_experiment(ExpPath, SYNAPSE, TRAIN).plot
    plot!(p, title = exp, legend = false)
end
plots = filter(x->x!=nothing, plots)
p = plot(plots..., layout = (1, length(plots)), size = (length(plots)*400, 1400))
savefig(p, plotsdir("SoundResponsePlots_$SYNAPSE.pdf"))



# ##
# data = datadir("helix", "Lagzi2022_AssemblyFormation", "mixed_inh", experiments[3]) |> x->test_experiment(x).plot

# data = datadir("helix", "Lagzi2022_AssemblyFormation", "mixed_inh", experiments[5]) |> x->test_experiment(x).models

# info = (τ= 100ms, rate=0.5, signal=:off, NSST=50)
# ExpPath = datadir("helix", "Lagzi2022_AssemblyFormation", "mixed_inh")
# load_data(ExpPath,  )
# raster(data[1].model.pop, 100s:105s)
# data[1].exp_config
# plot()

# data = load_model("/pasteur/appa/homes/aquaresi/spiking/network_models/data/helix/Lagzi2022_AssemblyFormation/mixed_inh/HighNoise/Model_sst-NSST=10-rate=0.5-τ=100.0.model.jld2")

# data.config |> dump|> print
# # write config to file:
# for path in ["HighNoise", "Noise_0.8", "DoubleSizeNetwork"]
#         path = datadir("helix","Lagzi2022_AssemblyFormation", "mixed_inh", path)
#         data = load_model(path)
#         open(path, "w") do io
#                 println(io, "config = ", data.config)
#         end
# end

# Write the configuration to a file
# Helper function to write a single value to the file

# Helper function to write the entire configuration to the file

# response
# plot(EEs..., layout=(size(EEs)..., 1), size=(800, 1400))
# ##
# plot()
# for recordings in recordings_noplast
#     plot(recordings.recordings[:,:,1])
# end
# plot!()


# size(recordings_noplast[1].recordings)

# plots=map(recordings_noplast) do recs
#         @unpack recordings, info, rec_interval = recs
#         plot(rec_interval,mean(recordings[:,:,1], dims=2)[:,1], label="E1")
#         plot!(rec_interval,mean(recordings[:,:,2], dims=2)[:,1], label="E2")
#         plot!(rec_interval,mean(recordings[:,:,3], dims=2)[:,1], label="PV")
#         plot!(rec_interval,mean(recordings[:,:,4], dims=2)[:,1], label="SST1")
# end
# plot(plots..., layout=(size(plots)..., 1), size=(800, 1400))
##

## Compare  the SST1_to_E1 synaptic weights of the models 

# @unpack model, TTL, sim_interval = models[2]
# rec_interval = 0:3s
# frs, r, names  = firing_rate(model.pop, interval=0:5:3s, τ=20ms)
# recordings = record_sound_response(model; TTL, sim_interval, rec_interval)


# ##
# sound_stim = deepcopy(SNN.sample_inputs(200, sound, interval))
# model = models[end-5]
# model.pop.SST1.N
# model.stim.inh_noise.param.rate .=0.5
# exp_config = (delay=1s, repetitions=1, warmup=10s, inter_trial_interval=5s, target_pop =:E1 )
# stimulus_TTL, sim_interval = test_sound_response(model, sound_stim, plasticity=false; exp_config...)
# raster(model.pop, 0.1s:4s)
# recordings = record_sound_response(model, sim_interval, exp_config.rec_interval)
# frs, r, names = firing_rate(model.pop; interval=0:10:15s, τ=10ms)
# # rec_name = joinpath(path, savename("SoundResponseRecordings", info, ".jld2"))
# # save(rec_name, @strdict recordings=recordings info=info rec_interval=exp_config.rec_interval)
# # recs = load(rec_name)["recordings"]

# ##

# plot(rec_interval,mean(recordings[:,:,1], dims=2)[:,1], label="E1")
# plot!(rec_interval,mean(recordings[:,:,2], dims=2)[:,1], label="E2")
# plot!(rec_interval,mean(recordings[:,:,3], dims=2)[:,1], label="PV")
# plot!(rec_interval,mean(recordings[:,:,4], dims=2)[:,1], label="SST1")
# ##

# # stimulus_TTL
# ##
# # T = get_time(mytime)
# #     shift_spikes!(stim, 1s) 
# #     sim!(model=model, duration=3s, time=mytime)


# # W, r = record(model.syn.PV_to_E1, :W) 
# # W =mean(W, dims=1)[1,:]
# # plot(r, W)
# # #   |> plot

##

t = 6
@unpack rates, interval = load(datadir("ExpData", "ACrates.jld2")) |> dict2ntuple
sound = mean(rates)
@unpack stim_τ, stim_rate = config
info = (τ = stim_τ, rate = stim_rate, signal = :off, NSST = NSSTs[t], plasticity = false)
data = load_model(ResponsePath, "SoundResponse", info)
@unpack model = data
model.stim.inh_noise.param.rate .= 0.8
model.stim.exc_noise1.param.rate .= 0.8
model.stim.exc_noise2.param.rate .= 0.8
model.syn.SST1_to_E1.W .*= 0#1/mean(model.syn.SST1_to_E1.W)
model.syn.E1_to_E1.W .*= 1/mean(model.syn.E1_to_E1.W)
model.syn.E2_to_E2.W .*= 1/mean(model.syn.E1_to_E1.W)
model.syn.E1_to_E2.W .*= 0.5/mean(model.syn.E1_to_E1.W)
model.syn.E2_to_E1.W .*= 0.5/mean(model.syn.E1_to_E1.W)
exp_config =
    (delay = 1s, repetitions = 10, warmup = 10s, target_pop = :E1, input_strength = 200)

SNN.monitor(model.pop.E1, [:v, :ge, :gi])
sound_stim = deepcopy(SNN.sample_inputs(exp_config.input_strength, sound, interval))
TTL, sim_interval =
    test_sound_response(model, sound_stim, plasticity = false; exp_config...)
##
p1 = vecplot(model.pop.E1, :v, neurons = 1:400, r = 0:5:5s, pop_average = true)
SNN.vecplot!(p1, model.pop.E1, :ge, neurons = 1:400, r = 0:1:5s, pop_average = true)
SNN.vecplot!(
    p1,
    model.pop.E1,
    :gi,
    neurons = 1:400,
    r = 0:5:2s,
    pop_average = true,
    factor = -1,
)
plot!(ylims = :auto)

frPV, r = firing_rate(model.pop.PV; interval = 0:5:3s, τ = 20ms)
frSST, r = firing_rate(model.pop.SST1; interval = 0:5:3s, τ = 20ms)
frE1, r = firing_rate(model.pop.E1; interval = 0:5:3s, τ = 20ms)
p2 = plot(r, mean(frPV, dims = 1)[1, :], label = "PV")
p2 = plot!(r, mean(frSST, dims = 1)[1, :], label = "SST")
p2 = plot!(r, mean(frE1, dims = 1)[1, :], label = "E1")
plot(p1, p2, layout = (2, 1), size = (800, 800))

##
raster(model.pop, 6s:12s)

##
histogram(model.syn.E1_to_E1.W)
histogram!(model.syn.E1_to_E2.W)
histogram(model.syn.SST1_to_E1.W, bins = 0:0.4:20)
histogram!(model.syn.SST1_to_E2.W, bins = 0:0.4:20)
##
# sound_stim = deepcopy(SNN.sample_inputs(200, sound, interval))
# model = models[end-5]
# model.pop.SST1.N
# model.stim.inh_noise.param.rate .=0.5
# exp_config = (delay=1s, repetitions=1, warmup=10s, inter_trial_interval=5s, target_pop =:E1 )
# stimulus_TTL, sim_interval = test_sound_response(model, sound_stim, plasticity=false; exp_config...)
# raster(model.pop, 0.1s:4s)
# recordings = record_sound_response(model, sim_interval, exp_config.rec_interval)
# frs, r, names = firing_rate(model.pop; interval=0:10:15s, τ=10ms)
# # rec_name = joinpath(path, savename("SoundResponseRecordings", info, ".jld2"))
# # save(rec_name, @strdict recordings=recordings info=info rec_interval=exp_config.rec_interval)
# # recs = load(rec_name)["recordings"]


# # stimulus_TTL
# ##
# # T = get_time(mytime)
# #     shift_spikes!(stim, 1s) 
# #     sim!(model=model, duration=3s, time=mytime)


# # W, r = record(model.syn.PV_to_E1, :W) 
# # W =mean(W, dims=1)[1,:]
# # plot(r, W)
# # #   |> plot

rec_interval = 0:4s
recordings = record_sound_response(model; TTL, sim_interval, rec_interval)
plot(rec_interval, mean(recordings[:, :, 1], dims = 2)[:, 1], label = "E1")
plot!(rec_interval, mean(recordings[:, :, 3], dims = 2)[:, 1], label = "PV")
plot!(rec_interval, mean(recordings[:, :, 4], dims = 2)[:, 1], label = "SST")
vline!([1s, 1.5s], label = "", ls = :dash, lc = :black)
