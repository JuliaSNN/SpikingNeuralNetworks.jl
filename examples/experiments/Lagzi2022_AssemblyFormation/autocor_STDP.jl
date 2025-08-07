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
λ = 0.53
η_exc = 25e-2ms
stdp_exc = STDPParameter(A_pre = η_exc, A_post = -λ * η_exc, τpost = 30ms, τpre = 15ms)

stdp_integral(stdp_exc, ΔTs = -201ms:4:201ms)
stdp_kernel(stdp_exc)

path = datadir("Lagzi2022_AssemblyFormation", "autocor_STDP")
τs = [1, 5, 20, 50, 100, 200, 1000]
##
Threads.@threads for n in eachindex(τs)
    τ = τs[n]
    signal_param = Dict(:X => 5Hz, :σ => 5Hz, :dt => 0.125f0, :θ => 1/τ * ms, :μ => 30Hz)
    stim = PoissonStimulusVariable(
        variables = copy(signal_param),
        rate = SNN.OrnsteinUhlenbeckProcess,
    )
    E1 = Identity(N = 200)
    E2 = Identity(N = 200)
    signal1 = SNN.PoissonStimulus(
        E1,
        :g,
        neurons = :ALL,
        N_pre = 100,
        param = deepcopy(stim),
        name = "ExtSignal_E1",
    )
    signal2 = SNN.PoissonStimulus(
        E2,
        :g,
        neurons = :ALL,
        N_pre = 100,
        param = deepcopy(stim),
        name = "ExtSignal_E1",
    )
    W11 = SpikingSynapse(E1, E1, nothing, param = stdp_exc, p = 0.2, μ = 10)
    W12 = SpikingSynapse(E1, E2, nothing, param = stdp_exc, p = 0.2, μ = 10)
    W21 = SpikingSynapse(E2, E1, nothing, param = stdp_exc, p = 0.2, μ = 10)
    W22 = SpikingSynapse(E2, E2, nothing, param = stdp_exc, p = 0.2, μ = 10)
    monitor(W11, [:W], sr = 20Hz)
    monitor(E1, [:fire])
    monitor(E2, [:fire])
    model = merge_models(; signal1, signal2, E1, E2, W11, W22, W12, W21)
    train!(model = model, duration = 200s, pbar = true)
    save_model(path = path, name = "ModelIdentity", model = model, info = Dict(:τ=>τ))
end
##



frs = Matrix{Vector{Float32}}(undef, length(τs), 2)
Ws = Matrix{Float32}(undef, length(τs), 4)
ave_fr = []
# nothing#Vector{Float32}(undef, length(τs))
# fr2 = begin 
Threads.@threads for n in eachindex(τs)
    τ = τs[n]
    info = Dict(:τ=>τ)
    model = load_data(path, "ModelIdentity", info).model
    st = length(spiketimes(model.pop.E1)[1]) ./ 200
    push!(ave_fr, st)
    # push!(ave_fr,average_firing_rate(model.pop.E1, interval=0s:10ms:200s))
    # fr1, r = firing_rate(model.pop.E1, interval=1s:1ms:200s, τ=10ms, interpolate=false) 
    # fr1 = mean(fr1, dims=1)[1]
    # fr2, r = firing_rate(model.pop.E2, interval=1s:1ms:200s, τ=10ms, interpolate=false) 
    # fr2 = mean(fr2, dims=1)[1]
    # @unpack W11, W12, W21, W22 = model.syn
    # W = [mean(W11.W), mean(W12.W), mean(W21.W), mean(W22.W)]
    # frs[n,1] = fr1
    # frs[n,2] = fr2
    # Ws[n,:] = W
end

plot(
    τs,
    ave_fr,
    lw = 4,
    xlabel = "τ (ms)",
    ylabel = "Firing rate",
    title = "Firing rate",
    grid = false,
    xscale = :log,
    guidefontsize = 17,
    tickfontsize = 10,
    size = (400, 400),
)

##
info = Dict(:τ=>100)
model = load_data(path, "ModelIdentity", info).model
raster(model.pop, 195s:1ms:200s, every = 4)
##
W = [Ws[:, 1] .- Ws[:, 2], .+ Ws[:, 4] .- Ws[:, 3]]
p1=plot()
plot!(τs[[1, end]], [0, 0], ls = :dash, color = :black, lw = 2, label = "", grid = false)
plot!(
    τs,
    mean(W),
    ribbon = std(W),
    lw = 4,
    label = "",
    xscale = :log,
    xlabel = "τ (ms)",
    ylabel = "ΔW",
    title = "Weight change",
    grid = false,
)

pal = palette(:roma, 10)
plot()
for n = 1:length(τs)
    fr1 = frs[n, 1]
    fr2 = frs[n, 2]
    crosscor(fr1, fr1, -200:200) |>
    x->plot!(
        -200:200,
        x,
        label = "$(τs[n])",
        legendtitle = "τ (ms)",
        color = pal[n],
        lw = 3,
    )
    crosscor(fr1, fr2, -200:200) |>
    x->plot!(
        -200:200,
        x,
        label = "",
        legendtitle = "τ (ms)",
        color = pal[n],
        lw = 3,
        ls = :dash,
    )
end
p2 = plot!(
    p2,
    grid = false,
    xlabel = "Time (ms)",
    ylabel = "Cross-correlation",
    title = "Cross-correlation of firing rates",
    legend = :bottomright,
)
p = plot(
    p1,
    p2,
    size = (800, 400),
    margin = 5Plots.mm,
    guidefontsize = 17,
    tickfontsize = 10,
    legendfontsize = 10,
)
# crosscor(fr1,fr1, -100:100) |> x->plot(-100:100, x)
png(p, "crosscorrelation.png")
# Ws


# crosscor(mean(fr1), mean(fr1), -100:100) |> x->plot!(-100:100, x)
# plot!(xlabel="Time (ms)")

# mean(fr1) |> autocor |> plot
