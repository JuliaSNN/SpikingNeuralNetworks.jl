"""
This script demonstrates the training of a spiking neural network (SNN) with adaptive exponential integrate-and-fire (AdEx) neurons and integrate-and-fire (IF) neurons. The network includes excitatory and inhibitory connections, with synaptic plasticity modeled using spike-timing-dependent plasticity (STDP). The script monitors synaptic weights and neuronal firing activity, trains the network for 20 seconds, and visualizes the results, including a comparison of synaptic weights before and after training.
"""

using SpikingNeuralNetworks
SNN.@load_units
using Statistics, Random, Plots


## AdEx neuron with fixed external current connections with multiple receptors
E = SNN.Population(SNN.AdExParameter(; El = -70mV); N=4000, name = "Excitatory", synapse= SNN.DoubleExpSynapse(), spike = SNN.PostSpike(τabs= 5ms))
I = SNN.Population(SNN.AdExParameter(; El = -65mV); N=1000, name = "Excitatory", synapse= SNN.DoubleExpSynapse(), spike = SNN.PostSpike(τabs= 5ms))
EE = SNN.SpikingSynapse(
    E,
    E,
    :glu;
    conn = (μ = 0.1, σ = 0.2, p = 0.02),
    LTPParam = SNN.vSTDPParameter(),
)
W0 = copy(EE.W)
stim = SNN.Stimulus(SNN.PoissonLayer(N=1000, 10Hz), E, :glu; conn=(μ = 10.0, p = 0.5))
Norm = SNN.MetaPlasticity(SNN.AdditiveNorm(τ = 50ms), [EE])
EI = SNN.SpikingSynapse(E, I, :glu; conn = (μ = 2., p = 0.05))
IE = SNN.SpikingSynapse(I, E, :gaba; conn = (μ = 20, p = 0.2))
II = SNN.SpikingSynapse(I, I, :gaba; conn = (μ = 20, p = 0.2))
model = SNN.compose(; E, I, EE, EI, IE, II, Norm, stim)
SNN.monitor!(model.pop, [:fire])
@profview SNN.train!(model = model; duration = 4second, pbar = true)
fr, r = SNN.firing_rate(model.pop, 1ms:10ms:SNN.get_time(model), pop_average=true) 
plot(r,
    fr;
    xlabel = "Time (s)",
    ylabel = "Firing rate (Hz)",
    title = "Firing rate of Excitatory neurons",
    figsize = (900, 300),
)
##
raster(model.pop, 3s:4s, figsize = (900, 300), title = "Raster plot of Excitatory neurons")
## Plots

raster(model.pop; figsize = (900, 300), title = "Raster plot of Excitatory neurons")
##

bins=0:0.1:10.0
p1 = plot()
p1 = W0 |> x -> histogram!(p1, x, label = "Before training", c = :black, bins = bins)
p1 =
    EE.W |>
    x -> histogram!(
        p1,
        x,
        label = "After training",
        c = :red,
        lc = :red,
        xlabel = "Synaptic strength",
        ylabel = "Number of synapses",
        alpha = 0.7,
        line_color = :transparent,
    )
p2 = SNN.vecplot(
    EE,
    [:W],
    pop_average = true,
    legend = :topleft,
    label = "Average weight",
    ylabel = "Synaptic weight (nS)",
    xlabel = "Time (s)",
    lw = 2,
    c = :darkblue,
)
p = plot(
    p1,
    p2,
    layout = (1, 2),
    fg_legend = :transparent,
    size = (900, 500),
    margin = 10Plots.mm,
    legendfontsize = 10,
)
savefig(p, joinpath(SNN.DOCS_ASSETS_PATH, "STDP_net.png"))
