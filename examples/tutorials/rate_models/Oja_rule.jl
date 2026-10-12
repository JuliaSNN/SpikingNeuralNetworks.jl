using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

# Recurrent rate network whose weights follow the learning rule of `RateSynapse` (applied under train!)
G = SNN.Rate(; N = 100)
GG = SNN.RateSynapse(G, G; μ = 1.2, p = 1.0)
model = SNN.compose(; G, GG, silent = true)
SNN.monitor!(G, [:r])

W0 = copy(GG.W)
SNN.train!(; model, duration = 100ms)

fig = Figure(size = (600, 500))
ax1 = Axis(fig[1, 1], ylabel = "Rate r")
SNN.vecplot!(ax1, G, :r, neurons = 1:5)
ax2 = Axis(fig[2, 1], xlabel = "Weight", ylabel = "Count")
hist!(ax2, W0, label = "before", bins = 50)
hist!(ax2, GG.W, label = "after", bins = 50)
axislegend(ax2)
save(joinpath(mktempdir(), "oja_rule.png"), fig)
