# Network of the COBA benchmark (Brette et al. 2007, J. Comput. Neurosci. 23:349): 3200 excitatory and
# 800 inhibitory leaky integrate-and-fire neurons with conductance-based exponential synapses,
# weights 0.6 gL (E) and 6.7 gL (I), gL = 10 nS. From the initial conditions of the benchmark the
# activity is self-sustained, with no external input (about 18 Hz in E, as in Brian2).
using SpikingNeuralNetworks
using CairoMakie, Distributions, Statistics
SNN.@load_units

neuron_param = SNN.IFParameter(; τm = 20ms, R = 100MΩ, Vt = -50mV, Vr = -60mV, El = -60mV)
spike = SNN.PostSpike(τabs = 5ms, up = 0ms, At = 0mV, AP_membrane = 0mV)   # no spike shape: reset at once
synapse = SNN.SingleExpSynapse(τe = 5ms, τi = 10ms, E_e = 0mV, E_i = -80mV)

E = SNN.Population(; param = neuron_param, synapse, spike, N = 3200, name = "Excitatory")
I = SNN.Population(; param = neuron_param, synapse, spike, N = 800, name = "Inhibitory")
we, wi = 6nS, 67nS
delay = Dirac(0.8ms)
EE = SNN.SpikingSynapse(E, E, :glu; conn = (μ = we, p = 0.02, rule = :Bernoulli), delay_dist = delay)
EI = SNN.SpikingSynapse(E, I, :glu; conn = (μ = we, p = 0.02, rule = :Bernoulli), delay_dist = delay)
II = SNN.SpikingSynapse(I, I, :gaba; conn = (μ = wi, p = 0.02, rule = :Bernoulli), delay_dist = delay)
IE = SNN.SpikingSynapse(I, E, :gaba; conn = (μ = wi, p = 0.02, rule = :Bernoulli), delay_dist = delay)
model = SNN.compose(; E, I, EE, EI, IE, II)

for P in (E, I)   # v ~ -65 ± 5 mV, ge ~ N(4, 1.5) gL, gi ~ N(20, 12) gL
    P.v .= -65mV .+ 5mV .* randn(Float32, P.N)
    P.synvars.ge .= (4 .+ 1.5f0 .* randn(Float32, P.N)) .* 10nS
    P.synvars.gi .= (20 .+ 12 .* randn(Float32, P.N)) .* 10nS
end
SNN.monitor!(model.pop, [:fire], monitor_rate = 50Hz)
SNN.monitor!(E, [(:v, 1:1)])
SNN.monitor!(I, [(:ge, 1:1), (:gi, 1:1)], variables = :synvars)

SNN.sim!(model = model; duration = 2.2second)

## Plot
rates, r, names = SNN.firing_rate(model.pop, interval = 0:10ms:2.2s, pop_average = true)
cv2 = SNN.ISI_CV2(model.pop.E, interval = 0.5s:2.2s)
fig = Figure(size = (800, 800))
ax1 = Axis(fig[1, 1:2], ylabel = "Firing rate (Hz)")
for (fr, name) in zip(rates, names)
    lines!(ax1, r ./ 1000, fr, label = name)
end
axislegend(ax1)
ax2 = Axis(fig[2, 1:2], ylabel = "Neuron")
SNN.raster!(ax2, model.pop, 1.8s:2.2s)
ax3 = Axis(fig[3, 1], xlabel = "Time (s)", ylabel = "V (mV)")
SNN.vecplot!(ax3, E, :v, neurons = 1, interval = 1.7s:0.5ms:2.2s, add_spikes = true)
ax4 = Axis(fig[3, 2], xlabel = "Time (s)", ylabel = "Conductance (nS)")
SNN.vecplot!(ax4, I, :ge, variables = :synvars, neurons = 1, interval = 2s:0.5ms:2.2s, label = "ge")
SNN.vecplot!(ax4, I, :gi, variables = :synvars, neurons = 1, interval = 2s:0.5ms:2.2s, label = "gi")
axislegend(ax4)
ax5 = Axis(fig[4, 1:2], xlabel = "ISI CV2", ylabel = "Neurons")
hist!(ax5, filter(isfinite, cv2), color = :black)
save(joinpath(mktempdir(), "COBA.png"), fig)
println("E rate: ", mean(rates[1][r .> 500]), " Hz, I rate: ", mean(rates[2][r .> 500]), " Hz")
fig
