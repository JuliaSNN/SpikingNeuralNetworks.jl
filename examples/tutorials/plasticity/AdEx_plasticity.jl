"""
This script demonstrates the training of a spiking neural network (SNN) with adaptive exponential integrate-and-fire (AdEx) neurons and integrate-and-fire (IF) neurons. The network includes excitatory and inhibitory connections, with synaptic plasticity modeled using spike-timing-dependent plasticity (STDP). The script monitors synaptic weights and neuronal firing activity, trains the network for 20 seconds, and visualizes the results, including a comparison of synaptic weights before and after training.
"""

using SpikingNeuralNetworks
using Statistics, Random, Plots
using CairoMakie
SNN.@makie_default
SNN.@load_units


## AdEx neuron with fixed external current connections with multiple receptors
synapse = SNN.DoubleExpSynapse(τre = 1ms, τde = 4ms, τri = 1ms, τdi = 3ms)
E = SNN.Population(SNN.AdExParameter(; El = -70mV); N=4000, name = "Excitatory", synapse, spike = SNN.PostSpike(τabs= 5ms))
I = SNN.Population(SNN.IFParameter(; El = -60mV); N=1000, name = "Inhibitory", synapse, spike = SNN.PostSpike(τabs= 5ms))
EE = SNN.SpikingSynapse(
    E,
    E,
    :glu;
    conn = (μ = 2, σ = 0.0, p = 0.02, rule=:Fixed),
    LTPParam = SNN.vSTDPParameter(Wmin = 0.1pF),
)

W0 = copy(EE.W)
stim = SNN.Stimulus(SNN.PoissonLayer(N=1000, 15Hz), E, :glu; conn=(μ = 2.0, p = 0.1))
Norm = SNN.MetaPlasticity(SNN.AdditiveNorm(τ = 10ms), [EE])
EI = SNN.SpikingSynapse(E, I, :glu; conn = (μ = 2., p = 0.02))
IE = SNN.SpikingSynapse(I, E, :gaba; conn = (μ = 2, p = 0.2))
II = SNN.SpikingSynapse(I, I, :gaba; conn = (μ = 2, p = 0.2))
model = SNN.compose(; E, I, EE, EI, IE, II,  stim, Norm)
SNN.monitor!(model.pop, [:fire])
# SNN.monitor!(EE, [(:W, 1:100)], sr=100Hz)
SNN.monitor!(EE, [:W], sr=10Hz)
SNN.train!(model = model; duration = 20second, pbar = true)
fr, r, labels = SNN.firing_rate(model.pop, 1ms:10ms:SNN.get_time(model), pop_average=true) 
ave_fr, _, labels = SNN.firing_rate(model.pop, 1ms:10ms:SNN.get_time(model), time_average=true) 

##
figure = Figure(size=(600, 800))
ax1 = Axis(figure[1, 1:2]; xlabel = "Time (s)", ylabel = "Firing rate (Hz)", title = "Firing rate of Excitatory and Inhibitory populations")
length(r)
fr[1]
series!(ax1, 
        r[:],
    hcat(fr...)';
    labels
)
axislegend(ax1; position = :rt)
ax3 = Axis(figure[2, 1:2])
SNNPlots.raster!(ax3, model.pop, 8s:10s, figsize = (900, 300), title = "Raster plot of Excitatory neurons")
ax21 = Axis(figure[3, 1], xlabel = "Firing rate (Hz)", title = "Average firing rate")
ax22 = Axis(figure[3, 2], xlabel = "Firing rate (Hz)", title = "Average firing rate")
hist!(
    ax21,
    ave_fr[1];
    color = :black,
    normalization=:probability
)
hist!(
    ax22,
    ave_fr[2];
    color = :black,
    normalization=:probability
)

bins = 1.5:0.01:2.5
ax_W0 = Axis(figure[4, 1]; xlabel = "Synaptic strength", ylabel = "Number of synapses")
hist!(ax_W0, W0, label = "Before training", color = :black, bins = bins, normalization=:probability)
hist!(ax_W0, model.syn.EE.W; bins, label="After training", color = :darkred, alpha = 0.9, normalization=:probability)
axislegend(ax_W0; position = :rt)
figure
ax_W1 = Axis(figure[4, 2]; ylabel = "Synaptic weight (nS)",
    xlabel = "Time (s)")
##

SNN.vecplot!(ax_W1,
    EE,
    :W,
    interval= 0s:0.4s:get_time(model),
    pop_average = true,
    legend = :topleft,
)

figure
##