using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

# Pair STDP measured on N independent pairs: presynaptic neuron n (E1) fires at 50 ms, postsynaptic
# neuron n (E2) at 50 ms + Δt[n]. Spikes are forced by setting the membrane potential above threshold.
N = 81
Δt = range(-40ms, 40ms, length = N)
E1 = SNN.IF(; N = N, name = "E1")
E2 = SNN.IF(; N = N, name = "E2")
EE = SNN.SpikingSynapse(E1, E2, :ge; conn = Matrix(0.1nS * SNN.SNNModels.LinearAlgebra.I(N)),
                        LTPParam = SNN.STDPGerstner(Wmax = 1nS))
model = SNN.compose(; E1, E2, EE, silent = true)
SNN.monitor!(model.pop, [:fire])
W0 = copy(SNN.plasticity_value(EE, :W))

dt = 0.125ms
for t in 0:dt:150ms
    abs(t - 50ms) < dt / 2 && (E1.v .= 100mV)
    E2.v[abs.(t .- (50ms .+ Δt)) .< dt / 2] .= 100mV
    SNN.train!(; model, duration = dt, dt)
end
ΔW = SNN.plasticity_value(EE, :W) .- W0

fig = Figure(size = (600, 600))
SNN.raster!(Axis(fig[1, 1], ylabel = "Neuron"), model.pop)
ax = Axis(fig[2, 1], xlabel = "Δt = t_post - t_pre (ms)", ylabel = "ΔW (nS)")
scatter!(ax, collect(Δt), ΔW)
save(joinpath(mktempdir(), "stdp_demo.png"), fig)

