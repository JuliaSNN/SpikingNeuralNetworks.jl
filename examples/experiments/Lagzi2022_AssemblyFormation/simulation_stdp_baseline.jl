using SpikingNeuralNetworks
using DrWatson
SNN.@load_units;
using SNNUtils
using Statistics
using Distributions

include("models.jl")
include("parameters.jl")
##
# Instantiate a  Symmetric STDP model with these parameters:

exp_name = length(ARGS) > 0 ? ARG[1] : "baseline"
path =
    datadir("helix", "Lagzi2022_AssemblyFormation/mixed_inh", experiments[exp_name].name) |>
    mkpath

for syn in [:nmda, :ampa]
    Threads.@threads for t in eachindex(NSSTs)
        local_config =
            (; config..., adex_param = getfield(adex_model, syn), NSST = NSSTs[t])
        info = (
            NSST = NSSTs[t],
            τ = local_config.stim_τ,
            rate = local_config.stim_rate,
            syn = syn,
        )
        model = nothing
        if isfile(get_path(path = path, name = "Model_sst", info = info))
            @info "Loading model $(exp_name) with NSST = $(NSSTs[t]), synapse = $(syn)"
            model = load_model(path, "Model_sst", info).model
        else
            @info "Training model $(exp_name) with NSST = $(NSSTs[t]), synapse = $(syn)"
            model = network(local_config = local_config, type = :sst)
            train!(model = model, duration = 500s, pbar = true)
            save_model(
                path = path,
                name = "Model_sst",
                model = model,
                info = info,
                config = local_config,
            )
            clear_records(model)
        end
        @info "Running signal off model $(exp_name) with NSST = $(NSSTs[t])"
        info = (; info..., signal = :off)
        model.stim.exc_noise1.param.rate .= 0.8*config.stim_rate
        model.stim.exc_noise2.param.rate .= 0.8*config.stim_rate
        model.stim.signal_signal_E1.param.active[1] = false
        model.stim.signal_signal_E2.param.active[1] = false
        isfile(get_path(path = path, name = "Model_sst", info = info)) && continue
        train!(model = model, duration = 500s, pbar = true)
        save_model(
            path = path,
            name = "Model_sst",
            model = model,
            info = info,
            config = local_config,
        )
    end
end
