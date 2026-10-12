# CUBA benchmark network (Brette et al. 2007, J. Comput. Neurosci. 23:349): 3200 excitatory and
# 800 inhibitory leaky integrate-and-fire neurons with current-based exponential synapses.
using SpikingNeuralNetworks
using CairoMakie, Statistics
SNN.@load_units

R = 100MΩ                                      # 0.1 GΩ: 1 mV per 10 pA
neuron_param = SNN.IFParameter(; τm = 20ms, R, Vt = -50mV, Vr = -60mV, El = -49mV)
spike = SNN.PostSpike(τabs = 5ms, up = 0ms, At = 0mV, AP_membrane = 0mV)   # no spike shape: reset at once
synapse = SNN.CurrentSynapse(τe = 5ms, τi = 10ms)

E = SNN.Population(; param = neuron_param, synapse, spike, N = 3200, name = "Excitatory")
I = SNN.Population(; param = neuron_param, synapse, spike, N = 800, name = "Inhibitory")
we, wi = 1.62mV / R, 9mV / R                   # weights in pA (inhibitory weights are positive)
EE = SNN.SpikingSynapse(E, E, :glu; conn = (μ = we, p = 0.02, rule = :Bernoulli))
EI = SNN.SpikingSynapse(E, I, :glu; conn = (μ = we, p = 0.02, rule = :Bernoulli))
II = SNN.SpikingSynapse(I, I, :gaba; conn = (μ = wi, p = 0.02, rule = :Bernoulli))
IE = SNN.SpikingSynapse(I, E, :gaba; conn = (μ = wi, p = 0.02, rule = :Bernoulli))
model = SNN.compose(; E, I, EE, EI, IE, II)

E.v .= neuron_param.Vr .+ rand(Float32, size(E.v)) .* (neuron_param.Vt - neuron_param.Vr)
I.v .= neuron_param.Vr .+ rand(Float32, size(I.v)) .* (neuron_param.Vt - neuron_param.Vr)
SNN.monitor!(model.pop, [:fire])
SNN.monitor!(E, [(:v, 1:1)])
SNN.sim!(model = model; duration = 2second)

## Plot
rates, r, names = SNN.firing_rate(model.pop, interval = 0:10ms:2s, pop_average = true)
fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1], ylabel = "Firing rate (Hz)")
for (fr, name) in zip(rates, names)
    lines!(ax1, r ./ 1000, fr, label = name)
end
axislegend(ax1)
ax2 = Axis(fig[2, 1], ylabel = "Neuron")
SNN.raster!(ax2, model.pop, 1.6s:2s)
ax3 = Axis(fig[3, 1], xlabel = "Time (s)", ylabel = "V (mV)")
SNN.vecplot!(ax3, E, :v, neurons = 1, interval = 1.5s:0.5ms:2s, add_spikes = true)
save(joinpath(mktempdir(), "CUBA.png"), fig)
println("E rate: ", mean(rates[1]), " Hz, I rate: ", mean(rates[2]), " Hz")
fig
