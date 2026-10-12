"""
Training of a network of AdEx (excitatory) and IF (inhibitory) neurons. The excitatory recurrent
synapses learn with voltage-based STDP (`vSTDPParameter`) and are kept at a constant total input per
neuron by additive normalisation. The script plots the firing rates, a raster, the rate distributions
and the excitatory weights before and after training.
"""

using SpikingNeuralNetworks
using Statistics, Random
using CairoMakie
SNN.@load_units

synapse = SNN.DoubleExpSynapse(τre = 1ms, τde = 4ms, τri = 1ms, τdi = 3ms)
E = SNN.Population(SNN.AdExParameter(; El = -70mV); N = 1000, name = "Excitatory", synapse,
                   spike = SNN.PostSpike(τabs = 5ms))
I = SNN.Population(SNN.IFParameter(; El = -60mV); N = 250, name = "Inhibitory", synapse,
                   spike = SNN.PostSpike(τabs = 5ms))
EE = SNN.SpikingSynapse(E, E, :glu; conn = (μ = 2, σ = 0.0, p = 0.05, rule = :Fixed),
                        LTPParam = SNN.vSTDPParameter(Wmin = 0.1pF))
W0 = copy(SNN.plasticity_value(EE, :W))
stim = SNN.Stimulus(SNN.PoissonLayer(N = 1000, 15Hz), E, :glu; conn = (μ = 2.0, p = 0.1))
Norm = SNN.MetaPlasticity(SNN.AdditiveNorm(τ = 10ms), [EE])
EI = SNN.SpikingSynapse(E, I, :glu; conn = (μ = 2.0, p = 0.05))
IE = SNN.SpikingSynapse(I, E, :gaba; conn = (μ = 2, p = 0.2))
II = SNN.SpikingSynapse(I, I, :gaba; conn = (μ = 2, p = 0.2))
model = SNN.compose(; E, I, EE, EI, IE, II, stim, Norm, silent = true)
SNN.monitor!(model.pop, [:fire])
SNN.monitor!(EE, [:W], sr = 10Hz)
SNN.train!(model = model; duration = 5second)

T = SNN.get_time(model)
fr, r, labels = SNN.firing_rate(model.pop, 0:10ms:T, pop_average = true)
ave_fr, _, _ = SNN.firing_rate(model.pop, 0:10ms:T, time_average = true)
W1 = SNN.plasticity_value(EE, :W)          # current weights (syn.W is updated lazily by vSTDP)

fig = Figure(size = (600, 800))
ax1 = Axis(fig[1, 1:2]; xlabel = "Time (s)", ylabel = "Firing rate (Hz)")
for (k, f) in enumerate(fr)
    lines!(ax1, r ./ 1000, f, label = labels[k])
end
axislegend(ax1; position = :rt)
SNN.raster!(Axis(fig[2, 1:2], title = "Raster"), model.pop, 4s:5s)
hist!(Axis(fig[3, 1], xlabel = "Firing rate (Hz)", title = "Excitatory"), ave_fr[1]; color = :black)
hist!(Axis(fig[3, 2], xlabel = "Firing rate (Hz)", title = "Inhibitory"), ave_fr[2]; color = :black)
ax_W = Axis(fig[4, 1]; xlabel = "Synaptic weight", ylabel = "Fraction of synapses")
bins = range(extrema([W0; W1])..., length = 50)
hist!(ax_W, W0; bins, label = "before", color = :black, normalization = :probability)
hist!(ax_W, W1; bins, label = "after", color = (:darkred, 0.7), normalization = :probability)
axislegend(ax_W; position = :rt)
ax_Wt = Axis(fig[4, 2]; xlabel = "Time (s)", ylabel = "Mean weight")
SNN.vecplot!(ax_Wt, EE, :W, interval = 0s:0.1s:T, pop_average = true)
save(joinpath(mktempdir(), "AdEx_plasticity.png"), fig)
