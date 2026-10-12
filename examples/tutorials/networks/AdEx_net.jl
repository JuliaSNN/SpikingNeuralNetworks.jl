# Recurrent network of heterogeneous AdEx (excitatory) and IF (inhibitory) neurons with
# delayed excitatory recurrence. Plots membrane potentials, conductances, a raster and the
# distribution of firing rates.
using SpikingNeuralNetworks
using CairoMakie, Distributions, Statistics
SNN.@load_units

## AdEx neurons with heterogeneous membrane time constant and adaptation
E_uni = SNN.AdExParameter(; El = -50mV)
E_het = SNN.make_heterogeneous(E_uni, 800; τm = Normal(10.0f0, 2.0f0), b = Normal(60.0f0, 4.0f0))
E = SNN.Population(E_het, synapse = SNN.DoubleExpSynapse(); N = 800, name = "Excitatory")
I = SNN.Population(SNN.IFParameter(), synapse = SNN.SingleExpSynapse(); N = 200,
                   name = "Inhibitory", spike = SNN.PostSpike())
EE = SNN.SpikingSynapse(E, E, :he; conn = (μ = 7, p = 0.02), delay_dist = Truncated(Normal(1.5ms, 0.5ms), 0.1ms, Inf))
EI = SNN.SpikingSynapse(E, I, :ge; conn = (μ = 30, p = 0.02))
IE = SNN.SpikingSynapse(I, E, :hi; conn = (μ = 50, p = 0.02))
II = SNN.SpikingSynapse(I, I, :gi; conn = (μ = 10, p = 0.02))
model = SNN.compose(; E, I, EE, EI, IE, II)

SNN.monitor!(E, [(:ge, 1:1), (:gi, 1:1)], variables = :synvars)
SNN.monitor!(E, [(:v, 1:3)])
SNN.monitor!(model.pop, [:fire])
SNN.sim!(model = model; duration = 4second)

## Synchrony: mean spike-time tiling coefficient of 100 excitatory neurons
st = SNN.spiketimes(model.pop.E)
sttc = SNN.STTC([Vector{Float32}(st[n]) for n in 1:100], 50ms, 0:4s)
println("mean STTC: ", mean(sttc))

## Plot
fig = Figure(size = (900, 700))
ax1 = Axis(fig[1, 1], ylabel = "Neuron", title = "Raster")
SNN.raster!(ax1, model.pop, 3.4s:4s)
ax2 = Axis(fig[1, 2], xlabel = "Time (s)", ylabel = "Conductance (nS)")
SNN.vecplot!(ax2, E, :ge, variables = :synvars, neurons = 1, interval = 3.8s:0.5ms:4s, label = "ge")
SNN.vecplot!(ax2, E, :gi, variables = :synvars, neurons = 1, interval = 3.8s:0.5ms:4s, label = "gi")
axislegend(ax2)
ax3 = Axis(fig[2, 1], xlabel = "Time (s)", ylabel = "Membrane potential (mV)")
for n in 1:3
    SNN.vecplot!(ax3, E, :v, neurons = n, interval = 3s:2ms:4s, add_spikes = true)
end
ax4 = Axis(fig[2, 2], xlabel = "Firing rate (Hz)", ylabel = "Neurons")
rates, _ = SNN.firing_rate(model.pop.E, interval = 0:10ms:4s, time_average = true)
hist!(ax4, rates, color = :black)
save(joinpath(mktempdir(), "AdEx_net.png"), fig)
fig
