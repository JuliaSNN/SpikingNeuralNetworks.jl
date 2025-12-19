using BenchmarkTools
poisson_pop = SNN.Population(N=1000,SNN.PoissonParameter(rate=5Hz))
model = SNN.compose(; poisson_pop)

SNN.monitor!(model.pop, [:fire])
@btime SNN.sim!(model, duration = 100s, dt = 1.125ms)

SNN.raster(model.pop, 10s:11s; figsize = (900, 300), title = "Raster plot of Poisson neurons",)

fr = SNN.firing_rate(model.pop, 1ms:10ms:SNN.get_time(model), time_average=true)
histogram(fr[1][1])

cvisi = SNN.ISI_CV2(model.pop.poisson_pop; interval = 0ms:10ms:SNN.get_time(model))

histogram(cvisi, bins=0.9:0.005:1.1, xlabel="ISI CV2", ylabel="Number of neurons", title="ISI CV2 distribution of Poisson neurons")