using SNNPlots
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter
using Statistics, Random




## 

# Parameters
T = 10.0s               # Total time
# Example spike trains as floating-point times
t_pre = sort(rand(10000) .* T)  # Post-synaptic spikes (50 random times in [0, T])
# t_pre = sort(rand(30) .* T)  # Post-synaptic spikes (50 random times in [0, T])
t_post = t_pre .+ rand(length(t_pre)) * 100 .- 50 #.-200   # Pre-synaptic spikes (75 random times in [0, T])

t_post = sort(rand(99000) .* T)  # Post-synaptic spikes (50 random times in [0, T])



r, cv = compute_covariance_density(Float32.(t_pre), Float32.(t_post), max_lag = 400)
bar(
    r,
    cv,
    legend = false,
    fill = true,
    xlabel = "ΔT(ms)",
    ylabel = "C(τ)",
    size = (500, 300),
    alpha = 0.5,
    color = :black,
    margin = 5SNNPlots.mm,
)
plot!(frame = :origin, yticks = :none)
##

#
z = []
tpre
for x in reverse(2:length(tpre))
    if tpre[x] - tpre[x-1] < 50ms
        # if rand() < 0.5
        push!(z, x)
        # end
    end
end
z
for x in z
    popat!(tpre, x)
end


taus = autocorrelogram(tpre, τ = 400ms)
histogram(
    taus,
    bins = 100,
    legend = false,
    fill = true,
    xlabel = "ΔT(mAs)",
    ylabel = "Autocorrelogram",
    size = (500, 300),
    alpha = 0.5,
    color = :black,
    margin = 5SNNPlots.mm,
    frame = :origin,
    yticks = :none,
)
