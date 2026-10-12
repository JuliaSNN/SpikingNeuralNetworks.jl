# A population of independent Poisson neurons: raster, firing rates and ISI CV2.
using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

poisson_pop = SNN.Population(SNN.PoissonParameter(5Hz); N = 1000)
model = SNN.compose(; poisson_pop, silent = true)

SNN.monitor!(model.pop, [:fire])
SNN.sim!(model; duration = 20s, dt = 1.125ms)

fr, _ = SNN.firing_rate(model.pop.poisson_pop; interval = 1ms:10ms:SNN.get_time(model), time_average = true)
cvisi = SNN.ISI_CV2(model.pop.poisson_pop; interval = 0ms:10ms:SNN.get_time(model))

fig = Figure(size = (900, 600))
SNN.raster!(Axis(fig[1, 1:2], title = "Poisson neurons"), model.pop, 10s:11s)
hist!(Axis(fig[2, 1], xlabel = "Firing rate (Hz)"), fr, color = :black)
hist!(Axis(fig[2, 2], xlabel = "ISI CV2"), filter(isfinite, cvisi), color = :black)
save(joinpath(mktempdir(), "poisson_population.png"), fig)
