# AdEx/IF network with depressing short-term plasticity (Tsodyks-Markram) on the excitatory
# recurrent synapses. STP acts only under `train!`.
using SpikingNeuralNetworks
using CairoMakie, Distributions, Statistics, Random
SNN.@load_units
Random.seed!(1234)

E_uni = SNN.AdExParameter(; El = -50mV)
E_het = SNN.make_heterogeneous(E_uni, 1600; τm = Normal(10.0f0, 2.0f0), b = Normal(60.0f0, 4.0f0))
E = SNN.Population(E_het, synapse = SNN.DoubleExpSynapse(); N = 1600, name = "Excitatory")
I = SNN.Population(SNN.IFParameter(), synapse = SNN.SingleExpSynapse(); N = 400,
                   name = "Inhibitory", spike = SNN.PostSpike())
EE = SNN.SpikingSynapse(E, E, :he; conn = (μ = 17, p = 0.02), STPParam = SNN.MarkramSTPParameter(U = 0.5))
EI = SNN.SpikingSynapse(E, I, :ge; conn = (μ = 30, p = 0.02))
IE = SNN.SpikingSynapse(I, E, :hi; conn = (μ = 50, p = 0.02))
II = SNN.SpikingSynapse(I, I, :gi; conn = (μ = 10, p = 0.02))
model = SNN.compose(; E, I, EE, EI, IE, II)

SNN.monitor!(model.pop, [:fire])
SNN.monitor!(E, [(:v, 1:3)])
SNN.monitor!(EE, [:ρ], sr = 50Hz)                  # STP efficacy of every EE synapse
SNN.train!(model = model; duration = 3second)

## Plot
fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1], ylabel = "Neuron")
SNN.raster!(ax1, model.pop, 2s:3s)
ax2 = Axis(fig[2, 1], xlabel = "Time (s)", ylabel = "Mean efficacy ρ")
ρ, r = SNN.record(EE, :ρ, range = true)
lines!(ax2, r ./ 1000, vec(mean(ρ(axes(ρ, 1), r), dims = 1)), color = :black)
ax3 = Axis(fig[3, 1], xlabel = "Time (s)", ylabel = "V (mV)")
SNN.vecplot!(ax3, E, :v, neurons = 1, interval = 2s:1ms:3s, add_spikes = true)
save(joinpath(mktempdir(), "AdEx_STP.png"), fig)
fig
