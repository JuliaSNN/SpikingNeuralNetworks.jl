# Layer of HetRec units with heterogeneous dendritic time constants, driven by Poisson input.
using SpikingNeuralNetworks
using CairoMakie, Distributions
SNN.@load_units

het = SNN.Population(SNN.HetRecParameter(Nd = 5, τd = LogNormal(4.0f0, 1.5f0), overlap = 0.00,
                                         rate = Normal(4Hz, 1Hz), steepness = 10.0f0, τm = 20.0ms,
                                         τabs = 5.0ms, τrate = 100.0ms))
poisson = SNN.Population(SNN.PoissonParameter(10Hz), N = 100)
W = ones(Float32, het.N * het.param.Nd, poisson.N)
ss = SNN.SpikingSynapse(poisson, het, :glu, conn = W)
model = SNN.compose(; het, poisson, ss, silent = true)
SNN.sim!(model, 30s)               # transient
SNN.reset_time!(model)
SNN.monitor!(model.pop.het, [:v_s, :v_d, :fire, :trace, :τd, :r])
SNN.monitor!(model.pop, :fire)
SNN.sim!(model, 10s)

interval = 0s:50ms:SNN.get_time(model)
fig = Figure(size = (900, 700))
ax = Axis(fig[1, 1], title = "Voltage and trace of unit 1")
for name in (:v_s, :trace)
    t, r = SNN.record(het, name, range = true, interval = interval)
    lines!(ax, r, t(1, r), label = string(name))
end
axislegend(ax)
fr, r = SNN.firing_rate(het; interval, time_average = true)
hist!(Axis(fig[1, 2], title = "Firing rate (Hz)"), fr)
hist!(Axis(fig[2, 1], title = "Timescales distribution"), het.τd, bins = 200, color = :blue)
SNN.raster!(Axis(fig[2, 2], title = "Raster"), het)
save(joinpath(mktempdir(), "hetrec_layer.png"), fig)
