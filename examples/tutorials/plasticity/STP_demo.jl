using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

# One presynaptic neuron (Identity, driven by a spike train) contacts one IF neuron through a
# synapse with Tsodyks-Markram short-term plasticity. The efficacy ρ = u x changes spike by spike;
# the weight W does not.
neuron = SNN.IF(; N = 1, synapse = SNN.SingleExpSynapse(τe = 5ms))
input = SNN.Identity(; N = 1)
stp_param = SNN.MarkramSTPParameterHet(τD = [20ms], τF = [200ms], U = [0.4])
# alternatives: SNN.MarkramSTPParameter(τD = 200ms, τF = 200ms, U = 0.4), SNN.MarkramSTPParameterTimestep(...)
syn = SNN.SpikingSynapse(input, neuron, :ge; conn = (μ = 1, p = 1), STPParam = stp_param)

# 9 spikes at 20 Hz from 100 ms, then one spike after a pause
times = [collect(100ms:50ms:500ms); 620ms]
spikes = SNN.SpikeTimeStimulusIdentity(input, :g; param = SNN.SpikeTimeParameter(times, fill(1, length(times))))

SNN.monitor!(syn, [:x, :u], variables = :STPVars, sr = 2kHz)
SNN.monitor!(syn, [:ρ], sr = 2kHz)
SNN.monitor!(neuron, [:ge], variables = :synvars, sr = 2kHz)
model = SNN.compose(; neuron, input, syn, spikes, silent = true)
SNN.train!(model, duration = 800ms)      # STP runs only under train!

r = 50ms:0.5ms:800ms
x = SNN.record(syn, :STPVars_x)
u = SNN.record(syn, :STPVars_u)
ρ = SNN.record(syn, :ρ)
ge = SNN.record(neuron, :synvars_ge)

fig = Figure(size = (600, 400))
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Value", title = "STP dynamics")
lines!(ax, r ./ 1000, ρ(1, r), label = "ρ", color = :black)
lines!(ax, r ./ 1000, x(1, r), label = "x", color = :blue, alpha = 0.7)
lines!(ax, r ./ 1000, u(1, r), label = "u", color = :red, alpha = 0.7)
lines!(ax, r ./ 1000, ge(1, r), label = "ge", color = :green, linewidth = 3)
axislegend(ax, position = :rt)
save(joinpath(mktempdir(), "stp_demo.png"), fig)
