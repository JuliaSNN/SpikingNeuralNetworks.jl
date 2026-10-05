using Makie, CairoMakie
using SNNModels
using Distributions

het = SNNModels.HetRecParameter(Nd=5, τd =LogNormal(4.0f0, 1.5f0), overlap=0.00, rate=Normal(4Hz, 1Hz), steepness=10.0f0, τm=20.0ms, τabs=5.0ms, τrate=100.0ms) |> SNNModels.Population

poisson = SNNModels.Population(SNNModels.PoissonParameter(10Hz), N = 100)
W = ones(Float32, het.N * het.param.Nd, poisson.N)

ss = SNNModels.SpikingSynapse(poisson, het, :glu, conn=W)
model = compose(;het, poisson, ss)
sim!(model, 30s)
reset_time!(model)
monitor!(model.pop.het, [:v_s, :v_d, :fire, :trace, :τd, :r])
monitor!(model.pop, :fire)
sim!(model, 10s, pbar=true)


fig = Figure()
ax = Axis(fig[1,1], title="Voltage and trace")
for (i, name) in enumerate([:v_s, :trace])
    t, r =  SNNModels.record(het, name, range=true, interval = 0s:50ms:get_time(model) )
    lines!(ax, r, t(1,r))
end
fr, r = SNNModels.firing_rate(het, interval=0s:50ms:get_time(model), time_average=true)
ax = Axis(fig[1,2], title="firing rate")
hist!(ax, fr)
ax = Axis(fig[2,1], title="Timescales distribution")
hist!(ax, het.τd, bins=200, color=:blue )
ax = Axis(fig[2,2], title="raster")
SNNPlots.raster!(ax, het)
fig
