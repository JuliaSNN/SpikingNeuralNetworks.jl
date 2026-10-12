# Save a network, load it back, continue it, and rerun it without one of its stimuli.
using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie

function network(; N = 1000)
    E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -60mV), name = "E")
    I = SNN.IF(N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, R = 0.06, El = -50mV), name = "I")
    E_to_I = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 3.0))
    E_to_E = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 0.5))
    I_to_I = SNN.SpikingSynapse(I, I, :gi; conn = (p = 0.2, μ = 4.0))
    I_to_E = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 1.0), LTPParam = SNN.iSTDPRate(r = 4Hz))
    noise = SNN.Stimulus(SNN.PoissonFixed(rate = 2.8kHz), E, :ge)
    noise2 = SNN.Stimulus(SNN.PoissonFixed(rate = 3kHz), E, :gi)
    return SNN.compose(; E, I, E_to_I, E_to_E, I_to_I, I_to_E, noise, noise2, silent = true)
end

path = mktempdir()
config = (N = 1000,)

## Build, train for 1 s and save
model = network(; config...)
SNN.train!(model, 1s)
SNN.save_model(; model, path, name = "network", info = config, config)

## Load and continue: the clock resumes at 1 s
loaded = SNN.load_model(path, "network", config)
model = loaded.model
SNN.monitor!(model.pop, [:fire])
SNN.train!(model, 1s)
@assert SNN.get_time(model.time) ≈ 2000

## Load again and rerun without the inhibitory noise
nogi = SNN.remove_element(SNN.load_model(path, "network", config).model, :noise2)
SNN.monitor!(nogi.pop, [:fire])
SNN.train!(nogi, 1s)

fig = Figure(size = (800, 600))
SNN.raster!(Axis(fig[1, 1], title = "Saved network, continued"), model.pop, 1s:2s)
SNN.raster!(Axis(fig[2, 1], title = "Saved network without inhibitory noise", xlabel = "Time (s)"), nogi.pop, 1s:2s)
save(joinpath(mktempdir(), "load_model.png"), fig)
