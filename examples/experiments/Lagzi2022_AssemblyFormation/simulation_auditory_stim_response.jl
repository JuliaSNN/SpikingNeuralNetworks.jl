using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Logging
using Statistics

# %%
# Instantiate a  Symmetric STDP model with these parameters:


root = datadir("zeus", "Lagzi2022_AssemblyFormation", "mixed_inh")
@assert isdir(root)
include("experiments_config.jl")
include("parameters.jl")
include("protocol.jl")

# experiment_name = length(ARGS) > 0 ? ARG[1] : experiment_name = "Test"

models = Dict{String,Any}()
for n in eachindex(running_exps)
        exp_config = getfield(response_experiments, running_exps[n])
        @info "Running experiment: $(exp_config.name)"
        Threads.@threads for t in eachindex(exp_config.NSSTs)
                for syn in [:ampa, :nmda]
                        NSST = exp_config.NSSTs[t]
                        @unpack stim_τ, stim_rate = config
                        info = (τ = stim_τ, 
                                rate = stim_rate, 
                                NSST = NSSTs[t], 
                                syn = syn,
                                signal = :off)

                        # Update the model parameters as in the config
                        load_path = joinpath(root, exp_config.load_path) 
                        @assert isdir(load_path)
                        path_response = joinpath(root, exp_config.name) |> mkpath

                        name = get_path(;path=load_path, name="Model_sst", info=info)
                        data = load_model(load_path, "Model_sst", info)
                        push!(models, name=>data)
                end
        end
end

Threads.@threads for n in eachindex(running_exps)
        exp_config = getfield(response_experiments, running_exps[n])
        @info "Running experiment: $(exp_config.name)"
        Threads.@threads for t in eachindex(exp_config.NSSTs)
                for syn in [:ampa, :nmda]
                        NSST = exp_config.NSSTs[t]
                        @unpack stim_τ, stim_rate = config
                        info = (
                                τ = stim_τ, 
                                rate = stim_rate, 
                                NSST = NSSTs[t], 
                                syn = syn,
                                signal = :off
                                )

                        # Update the model parameters as in the config
                        load_path = joinpath(root, exp_config.load_path) 
                        @assert isdir(load_path)
                        path_response = joinpath(root, exp_config.name) |> mkpath

                        name = get_path(;path=load_path, name="Model_sst", info=info)
                        data = deepcopy(models[name])

                        model = data.model
                        network_config = data.config
                        base_model = update_model_parameters!(model; exp_config, config=network_config)

                        sound_stim = deepcopy(SNN.sample_inputs(exp_config.input_strength, sound, interval))
                        @unpack rec_interval = exp_config

                        # Set model info
                        info = (;info...,
                                train = exp_config.train,
                        )
                        @info "Running experiment with $(info)"
                        model, TTL, sim_interval = nothing, nothing, nothing
                        if exp_config.force || !isfile(get_path(path = path_response, name = "SoundResponse", info = info))
                                # Test sound response without train and save model
                                #------------------------------------------------------------------------------
                                @info "Model not found, running experiment with $(info)"
                                model = deepcopy(base_model)
                                clear_records(model)
                                TTL, sim_interval = test_sound_response(
                                        model,
                                        sound_stim;
                                        exp_config...,
                                )
                                save_model(;
                                        path = path_response,
                                        model = model,
                                        name = "SoundResponse",
                                        info = info,
                                        sound = sound_stim,
                                        TTL = TTL,
                                        sim_interval = sim_interval,
                                        rec_interval = rec_interval,
                                        config = exp_config,
                                        network_config = network_config,
                                )
                        else
                                @info "Model found, loading $(info)"
                                @unpack model, TTL, sim_interval =
                                        load_data(; path = path_response, name = "SoundResponse", info)
                        end
                        # Record sound response and save recordings
                        recordings = record_sound_response(model; TTL, sim_interval, rec_interval)
                        rec_name = joinpath(path_response, "recordings", savename("SoundResponseRecordings", info, "jld2"))
                        dirname(rec_name) |> mkpath
                        save(
                        rec_name,
                        @strdict recordings = recordings info = info rec_interval = rec_interval
                        )
                end
        end
end

