
function stdp_integral(stdp_param; ΔTs = -101ms:2:101ms, fill = true)
    ΔWs = zeros(Float32, length(ΔTs))
    Threads.@threads for i in eachindex(ΔTs)
        ΔT = ΔTs[i]
        spiketime = [2000ms, 2000ms + ΔT]
        neurons = [1, 2]
        inputs = SpikeTimeParameter(spiketime, neurons)
        w = zeros(Float32, 2, 2)
        w[2, 1] = 10.0f0
        st = Identity(N = max_neurons(inputs))
        stim = SpikeTimeStimulusIdentity(st, :g, param = inputs)
        syn = SpikingSynapse(st, st, :h, w = w, param = stdp_param)
        model = merge_models(pop = st, stim = stim, syn = syn, silent = true)
        SNN.monitor!(model.pop..., [:fire])
        train!(model = model, duration = 3000ms, dt = 0.1ms)
        ΔWs[i] = model.syn[1].W[1] - 10.0f0
    end
    return sum([w * Float32(ΔTs.step) for w in ΔWs])
end

"""
    stdp_kernel(stdp_param; ΔT= -97.5:5:100ms)

    Plot the STDP kernel for the given STDP parameters. 
    
    # Arguments
    - `stdp_param::STDPParameter`: STDP parameters
    - `ΔT::Array{Float32}`: Arrays of time differences between pre and post-synaptic spikes

    # Return
    - `Plots.Plot`: Plot of the STDP kernel

"""
function stdp_kernel(stdp_param; ΔTs = -200.5:5:200ms, fill = true, kwargs...)
    ΔWs = zeros(Float32, length(ΔTs))
    # Threads.@threads 
    for i in eachindex(ΔTs)
        ΔT = ΔTs[i]
        spiketime = [2000ms, 2000ms + ΔT]
        neurons = [1, 2]
        inputs = SpikeTimeParameter(spiketime, neurons)
        st = Identity(N = max_neuron(inputs))
        stim = SpikeTimeStimulusIdentity(st, :g, param = inputs)
        w = zeros(Float32, 2, 2)
        w[2, 1] = 1.0f0
        syn = SpikingSynapse(st, st, :h, w = w, param = stdp_param)
        model = merge_models(pop = st, stim = stim, syn = syn, silent = true)
        SNN.monitor!(model.pop..., [:fire])
        train!(model = model, duration = 3000ms, dt = 0.1ms)
        ΔWs[i] = model.syn[1].W[1] - 1.0f0
    end

    n_plus = findall(ΔTs .>= 0)
    n_minus = findall(ΔTs .< 0)
    # R(X; f=maximum) = [f([x,0]) for x in X]
    plot(
        ΔTs[n_minus],
        ΔWs[n_minus],
        legend = false,
        fill = fill,
        xlabel = "ΔT",
        ylabel = "ΔW",
        title = "STDP",
        size = (500, 300),
        alphafill = 0.5,
        lw = 4,
    )
    plot!(
        ΔTs[n_plus],
        ΔWs[n_plus],
        legend = false,
        fill = fill,
        xlabel = "T_post - T_pre ",
        ylabel = "ΔW",
        title = "STDP",
        size = (500, 300),
        alphafill = 0.5,
        lw = 4,
    )
    plot!(ylims = extrema(ΔWs) .* 1.4, xlims = extrema(ΔTs), framestyle = :zerolines)
    plot!(; kwargs...)
    # plot!(ylims=(-10,10), xlims=extrema(ΔTs))
end

function stdp_weight_correlated(stdp_param, rate1, rate2, τ_cov = 10ms)
    T = 120_000ms
    N_spike1 = T * rate1 |> Int
    N_spike2 = T * rate2 |> Int

    spikes1 = rand(N_spike1) * T
    spikes2 = rand(N_spike2) * T
    for n in eachindex(spikes1)
        N = rand(1:length(spikes2))
        spikes2[N] = spikes1[n] + τ_cov * randn()
    end
    N1 = fill([1], length(spikes1))
    N2 = fill([2], length(spikes2))
    neurons = vcat(N1, N2)
    spiketimes = vcat(spikes1, spikes2)
    inputs = SpikeTimeParameter(spiketimes, neurons)
    st = Identity(N = max_neurons(inputs))
    stim = SpikeTimeStimulusIdentity(st, :g, param = inputs)
    w = zeros(Float32, 2, 2)
    w[2, 1] = 5.0f0
    syn = SpikingSynapse(st, st, :h, w = w, param = stdp_param)
    model = merge_models(st = st, stim = stim, syn = syn, silent = true)
    SNN.monitor!(model.pop..., [:fire])
    train!(model = model, duration = T, dt = 0.1ms)
    return model
    ΔW = model.syn[1].W[1] - 5.0f0
    return ΔW
end

export stdp_kernel, stdp_integral, stdp_weight_correlated

function stdp_pairing()
    plot(ylims = (-4, 4), xlims = (-10, 10), legend = false, frame = :none)
    hline!([-2, 2], lc = :black, lw = 4)
    scatter!([-6], [2.4], markershape = :vline, mc = :black, ms = 18, lw = 4)
    scatter!([1], [-1.6], markershape = :vline, mc = :black, ms = 18, lw = 4)
    annotate!(6, 2.4, text("Pre synaptic", 15, :bottom))
    annotate!(6, -1.6, text("Post synaptic", 15, :bottom))
    annotate!(-2.5, 0, text("ΔT", 15, :bottom))
    plot!([-6, 1], [0, 0], lc = :black, arrow = (:both, :closed, 3))
end

## Measure the weight change for decorrelated spike trains
function stdp_weight_decorrelated(stdp_param, rate1 = 10Hz, rate2 = 10Hz)
    st1 = Poisson(N = 50, param = PoissonParameter(rate = rate1))
    st2 = Poisson(N = 50, param = PoissonParameter(rate = rate2))
    syn = SpikingSynapse(st1, st2, nothing, p = 1.0f0, μ = 1, param = stdp_param)
    model = merge_models(st1 = st1, st2 = st2, syn = syn, silent = true)
    T = 20_000ms
    train!(model = model, duration = T, dt = 0.1ms)
    return (model.syn[1].W .- 1) / T * 60^2
end

#
function plot_iSTDP_activity(network, config; interval = 1s:20ms:15s)
    i_to_e1 = SNN.filter_items(network.syn, condition = p -> occursin("I1_to_E", p.name))
    i_to_e2 = SNN.filter_items(network.syn, condition = p -> occursin("I2_to_E", p.name))
    w_i1 = map(eachindex(i_to_e1)) do i
        w, r_t = record(i_to_e1[i], :W, interpolate = true)
        mean(w, dims = 1)[1, :]
    end |> collect
    w_i2 = map(eachindex(i_to_e2)) do i
        w, r_t = record(i_to_e2[i], :W, interpolate = true)
        mean(w, dims = 1)[1, :]
    end |> collect

    i_to_e = SNN.filter_items(network.syn, condition = p -> occursin("I1_to_E", p.name))
    _, r_t = record(i_to_e[1], :W, interpolate = true)
    p11 = plot(
        r_t ./ 1000,
        w_i1,
        xlabel = "Time (s)",
        ylabel = "Synaptic weight",
        legend = :topleft,
        title = "I to E synapse",
        labels = ["pop 1" "pop 2" "pop 3" "pop 4"],
        lw = 4,
    )
    p12 = plot(
        r_t ./ 1000,
        w_i2,
        xlabel = "Time (s)",
        ylabel = "Synaptic weight",
        legend = :topleft,
        title = "I to E synapse",
        labels = ["pop 1" "pop 2" "pop 3" "pop 4"],
        lw = 4,
    )
    p1 = plot(p11, p12, layout = (2, 1))

    p31 = SNN.stdp_kernel(i_to_e1[1].param, fill = false)
    p32 = SNN.stdp_kernel(i_to_e2[1].param, fill = false)
    @unpack istdp_ratio = config
    annotate!(p31, (0, 1), text("$(round(Int,100*(istdp_ratio)))%", 18, :black))
    annotate!(p32, (0, 1), text("$(round(Int,100*(1-istdp_ratio)))%", 18, :black))
    p3 = plot(p31, p32, layout = (2, 1))

    Epop = SNN.filter_items(network.pop, condition = p -> occursin("E", p.name))
    rates, interval = SNN.firing_rate(Epop, interval = interval, interpolate = false)
    rates = mean.(rates)
    p2 = plot(
        interval ./ 1000,
        rates,
        xlabel = "Time (s)",
        ylabel = "Firing rate (Hz)",
        legend = :topleft,
        title = "Firing rate of the exc. pop",
        lw = 4,
        labels = ["pop 1" "pop 2" "pop 3" "pop 4"],
    )#, yscale=:log10, ylims=(0.1,50))
    p4 = SNN.raster(network.pop, interval, every = 3)
    plot(p3, p1, p4, p2, layout = (2, 2), size = (1800, 1800), margin = 5Plots.mm)
    # plot(p4, p2, layout=(2,1), size=(800,800), margin=5Plots.mm)
end


export stp_plot,
    plot_weights,
    plot_activity,
    dendrite_gplot,
    soma_gplot,
    stdp_kernel,
    stdp_integral,
    stdp_weight_decorrelated,
    plot_iSTDP_activity
