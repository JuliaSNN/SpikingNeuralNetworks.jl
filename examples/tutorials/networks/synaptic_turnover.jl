# Random structural turnover: every 100 ms each excitatory recurrent synapse is moved to a new,
# uniformly chosen postsynaptic neuron with probability 0.05 (`RandomTurnover`). Turnover is a
# metaplasticity component and runs only under `train!`. For activity-dependent turnover see
# ../plasticity/AdEx_turnover.jl.
using SpikingNeuralNetworks
using CairoMakie, Distributions, Statistics
SNN.@load_units

E_het = SNN.make_heterogeneous(SNN.AdExParameter(; El = -50mV), 800;
                               τm = Normal(10.0f0, 2.0f0), b = Normal(60.0f0, 4.0f0))
E = SNN.Population(E_het, synapse = SNN.DoubleExpSynapse(); N = 800, name = "Excitatory")
I = SNN.Population(SNN.IFParameter(), synapse = SNN.SingleExpSynapse(); N = 200,
                   name = "Inhibitory", spike = SNN.PostSpike())
EE = SNN.SpikingSynapse(E, E, :he; conn = (μ = 2, p = 0.02))
EI = SNN.SpikingSynapse(E, I, :ge; conn = (μ = 30, p = 0.02))
IE = SNN.SpikingSynapse(I, E, :hi; conn = (μ = 50, p = 0.02))
II = SNN.SpikingSynapse(I, I, :gi; conn = (μ = 10, p = 0.02))
turnover = SNN.MetaPlasticity(SNN.RandomTurnover(τ = 100ms, threshold = 0.05f0, μ = 2.0f0), EE)
model = SNN.compose(; E, I, EE, EI, IE, II, turnover)
SNN.monitor!(model.pop, [:fire])

targets_before = [Set(SNN.postsynaptic(EE, j)) for j in 1:E.N]
indegree_before = [length(SNN.presynaptic(EE, i)) for i in 1:E.N]
SNN.train!(model = model; duration = 2second)
targets_after = [Set(SNN.postsynaptic(EE, j)) for j in 1:E.N]
indegree_after = [length(SNN.presynaptic(EE, i)) for i in 1:E.N]
kept = sum(length(intersect(a, b)) for (a, b) in zip(targets_before, targets_after))
println("synapses kept on their original target: ", round(100 * kept / sum(length.(targets_before)), digits = 1), " %")

fig = Figure(size = (700, 600))
ax1 = Axis(fig[1, 1], ylabel = "Neuron")
SNN.raster!(ax1, model.pop, 1s:2s)
ax2 = Axis(fig[2, 1], xlabel = "In-degree of E neurons", ylabel = "Neurons")
hist!(ax2, indegree_before, bins = 0:2:40, label = "before", color = (:black, 0.5))
hist!(ax2, indegree_after, bins = 0:2:40, label = "after 2 s of turnover", color = (:darkred, 0.5))
axislegend(ax2)
save(joinpath(mktempdir(), "synaptic_turnover.png"), fig)
fig
