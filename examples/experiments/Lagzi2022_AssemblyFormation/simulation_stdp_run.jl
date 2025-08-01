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

include("models.jl")
include("parameters.jl")
# Instantiate a  Symmetric STDP model with these parameters:


path = datadir("Lagzi2022_AssemblyFormation", "baseline_model")
Threads.@threads for t in eachindex(τs)
    for r in eachindex(stim_rates)
        stim_rate = stim_rates[r]
        stim_τ = τs[t]
        local_config = (; config...,
            stim_τ = stim_τ,
            stim_rate = stim_rate,
            I_noise = (1-stim_rate)*ext_rate,
            signal_param = Dict(:X => 2.0f0,
                :σ => 0.4kHz,
                :dt => 0.125f0,
                :θ => 1/stim_τ,
                :μ => stim_rate*ext_rate
                ),
        )
        info = (τ= local_config.stim_τ, rate=local_config.stim_rate)
        @show get_path(path=path, name="Model_sst", info=info)
        isfile(get_path(path=path, name="Model_sst", info=info)) && continue
        model = network(local_config=local_config, type= :sst)
        train!(model=model, duration=1000s, pbar=true)
        save_model(path=path,name="Model_sst", model=model, info=info, config=config)
    end
end



path = datadir("Lagzi2022_AssemblyFormation")
Threads.@threads for t in eachindex(τs)
    for r in eachindex(stim_rates)
        stim_rate = stim_rates[r]
        stim_τ = τs[t]
        info = (τ= stim_τ, rate=stim_rate)
        model = load_model(path, "Model_sst", info).model

        info = (τ= stim_τ, rate=stim_rate, signal=:off)
        model.stim.exc_noise1.param.rate.=2kHz
        model.stim.exc_noise2.param.rate.=2kHz
        model.stim.signal_signal_E1.param.active[1] = false
        model.stim.signal_signal_E2.param.active[1] = false
        SNN.monitor(model.pop, [:fire])

        SNN.monitor(model.syn.E1_to_E1, [:W], sr=10Hz)
        SNN.monitor(model.syn.E1_to_E2, [:W], sr=10Hz)
        SNN.monitor(model.syn.E2_to_E1, [:W], sr=10Hz)
        SNN.monitor(model.syn.E2_to_E2, [:W], sr=10Hz)

        SNN.monitor(model.syn.SST1_to_E1, [:W], sr=10Hz)
        SNN.monitor(model.syn.SST1_to_E2, [:W], sr=10Hz)
        SNN.monitor(model.syn.SST2_to_E1, [:W], sr=10Hz)
        SNN.monitor(model.syn.SST2_to_E2, [:W], sr=10Hz)

        SNN.monitor(model.syn.PV_to_E1, [:W], sr=10Hz)
        SNN.monitor(model.syn.PV_to_E2, [:W], sr=10Hz)

        no_ext_info = (τ= stim_τ, rate=stim_rate, signal=:off)
        isfile(get_path(path=path, name="Model_sst", info=no_ext_info)) && continue
        train!(model=model, duration=500s, pbar=true)
        save_model(path=path,name="Model_sst", model=model, info=no_ext_info, config=config)
    end
end 

path = datadir("Lagzi2022_AssemblyFormation")
Threads.@threads for t in eachindex(τs)
    for r in eachindex(stim_rates)
        stim_rate = stim_rates[r]
        stim_τ = τs[t]
        info = (τ= stim_τ, rate=stim_rate)
        model = load_model(path, "Model_sst", info).model

        info = (τ= stim_τ, rate=stim_rate, signal=:off)
        model.stim.exc_noise1.param.rate.=2kHz
        model.stim.exc_noise2.param.rate.=2kHz
        model.stim.signal_signal_E1.param.active[1] = false
        model.stim.signal_signal_E2.param.active[1] = false
        SNN.monitor(model.pop, [:fire])

        SNN.monitor(model.syn.E1_to_E1, [:W], sr=10Hz)
        SNN.monitor(model.syn.E1_to_E2, [:W], sr=10Hz)
        SNN.monitor(model.syn.E2_to_E1, [:W], sr=10Hz)
        SNN.monitor(model.syn.E2_to_E2, [:W], sr=10Hz)

        SNN.monitor(model.syn.SST1_to_E1, [:W], sr=10Hz)
        SNN.monitor(model.syn.SST1_to_E2, [:W], sr=10Hz)
        SNN.monitor(model.syn.SST2_to_E1, [:W], sr=10Hz)
        SNN.monitor(model.syn.SST2_to_E2, [:W], sr=10Hz)

        SNN.monitor(model.syn.PV_to_E1, [:W], sr=10Hz)
        SNN.monitor(model.syn.PV_to_E2, [:W], sr=10Hz)

        no_ext_info = (τ= stim_τ, rate=stim_rate, signal=:off)
        isfile(get_path(path=path, name="Model_sst", info=no_ext_info)) && continue
        train!(model=model, duration=500s, pbar=true)
        save_model(path=path,name="Model_sst", model=model, info=no_ext_info, config=config)
    end
end 