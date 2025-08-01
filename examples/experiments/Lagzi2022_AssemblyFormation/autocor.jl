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

##
plot()
frs = []
my_c = palette(:tab10, 10)
for (n, t) in enumerate([10, 20, 50, 100])
    signal_param = Dict(:X => 2.0f0, :σ => 10Hz, :dt => 0.125f0, :θ => 0.001f0*t, :μ => 5Hz)
    stim = PoissonStimulusVariable(
        variables = copy(signal_param),
        rate = SNN.OrnsteinUhlenbeckProcess,
    )
    E = Identity(N = 100)
    signal = SNN.PoissonStimulus(
        E,
        :g,
        neurons = :ALL,
        μ = 1.0,
        param = stim,
        name = "ExtSignal_E1",
    )
    monitor(signal, [:fire])
    monitor(E, [:fire])
    monitor(E, [:fire, :spikecount])
    model = merge_models(signal = signal, E = E)
    sim!(model = model, duration = 60s)
    fr, r = firing_rate(model.pop.E, interval = 1s:1ms:20s, τ = 10ms)
    xs=1:200
    rs = r[xs] .- 1s
    θ = stim.variables[:θ]
    τ = 1/θ
    mean(fr, dims = 1)[1, :] |>
    x->autocor(x, xs) |> x->plot!(rs, x, label = "t=$τ", c = my_c[n])
    # plot!(r./s,fr[1,r], label="t=$(τ)")
    # push!(frs, fr[1,r])
end
plot!()
##

plot()
dt = 0.125
for (n, t) in enumerate([10, 20, 50, 100])
    signal_param = Dict(:X => 2.0f0, :σ => 200Hz, :dt => 0.125f0, :θ => 1/t*ms, :μ => 1kHz)
    stim = PoissonStimulusVariable(
        variables = copy(signal_param),
        rate = SNN.OrnsteinUhlenbeckProcess,
    )
    Xs = zeros(Float32, 100_000)
    for n in eachindex(Xs)
        Xs[n] = SNN.OrnsteinUhlenbeckProcess(0.0f0, stim)
    end

    xs=1:2000
    Xs |>
    x->autocor(x, xs) |> x->plot!(xs .* 0.125, x, label = "t=$(t*0.01f0)", c = my_c[n])
    θ = stim.variables[:θ]
    plot!(xs .* dt, exp.(-xs .* θ .* dt), c = my_c[n], label = "", ls = :dash)
end

plot!()
