
function test_sound_response(
    model,
    sound_stim;
    train = false,
    inter_trial_interval = 5s,
    delay = 1s,
    repetitions = 3,
    warmup = 10s,
    target_pop = :E1,
    kwargs...,
)
    @info "Running sound response test with:
    plasticity: $(train), delay: $delay, repetitions: $repetitions, warmup: $warmup"

    sim!(model = model, duration = warmup)
    # Monitor firing rates and synaptic weights
    monitor(model.pop, [:fire])
    for syn in model.syn
        !isa(syn.param, no_STDPParameter) && monitor(syn, [:W], sr = 5Hz)
    end

    sound_stim = SpikeTimeStimulus(
        getfield(model.pop, target_pop),
        :he,
        param = SpikeTimeParameter(sound_stim),
    )
    shift_spikes!(sound_stim, delay)

    TTL = []
    mytime=Time()
    for i = 1:repetitions
        T = get_time(mytime)
        push!(TTL, sound_stim.param.spiketimes[1])
        train && train!(
            sound_stim,
            model = model,
            duration = inter_trial_interval,
            time = mytime,
        )
        !train &&
            sim!(sound_stim, model = model, duration = inter_trial_interval, time = mytime)
        shift_spikes!(sound_stim, inter_trial_interval)
    end
    return TTL, [0, get_time(mytime)]
end


function record_sound_response(model; TTL, sim_interval, rec_interval, kwargs...)
    interval = sim_interval[1]:10ms:sim_interval[2]
    frs, r, names = firing_rate(model.pop; interval, τ = 20ms)
    recordings = zeros(length(rec_interval), length(TTL), length(names))
    for t in eachindex(TTL)
        _rec_interval = TTL[t] .+ rec_interval .- 1s
        for i = 1:length(names)
            size(frs[i], 1) == 0 && continue
            recordings[:, t, i] = frs[i][:, _rec_interval]|>x -> mean(x, dims = 1)[1, :]
        end
    end
    return recordings
end
