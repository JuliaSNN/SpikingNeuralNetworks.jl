using SpikingNeuralNetworks
SNN.@load_units
using Statistics, Random, Plots



## AdEx neuron with fixed external current connections with multiple receptors
E = SNN.AdExNeuron(; N = 800, param = SNN.AdExParameter(; El = -50mV))

I = SNN.IF(; N = 200, param = SNN.IFParameter())
EE = SNN.SpikingSynapse(E, E, :he; μ = 2, p = 0.02)
EI = SNN.SpikingSynapse(E, I, :ge; μ = 30, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :hi; μ = 50, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; μ = 10, p = 0.02)
model = SNN.merge_models(; E = E, I = I, EE = EE, EI = EI, IE = IE, II = II)

SNN.monitor!(E, [:ge, :gi, :v])
SNN.monitor!(model.pop, [:fire])
SNN.sim!(model = model; duration = 4second)

default(palette = :okabe_ito)
plot(
    SNN.raster(model.pop, [3.4s, 4s]),
    SNN.vecplot(
        E,
        [:ge, :gi],
        neurons = 1,
        r = 3.3s:4s;
        legend = true,
        ylabel = "Conductance (nS)",
    ),
    SNN.vecplot(E, :v, neurons = 1:3, r = 3s:4s, add_spikes = true, lw = 2),
    SNN.firing_rate(model.pop.E, 0:4s, time_average = true)[1] |>
    x->Plots.histogram(
        x,
        ylabel = "Neurons",
        xlabel = "Firing rate (Hz)",
        label = "",
        c = :black,
    ),
    size = (900, 600),
)
