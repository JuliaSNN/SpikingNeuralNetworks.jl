using SpikingNeuralNetworks
SNN.@load_units

## AdEx neuron with fixed external current connections with multiple receptors
E = SNN.AdEx(; N = 800, param = SNN.AdExParameter(; El = -50mV))

I = SNN.IF(; N = 200, param = SNN.IFParameter())
EE = SNN.SpikingSynapse(E, E, :he; μ = 2, p = 0.02)
EI = SNN.SpikingSynapse(E, I, :ge; μ = 30, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :hi; μ = 50, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; μ = 10, p = 0.02)
model = SNN.compose(;  E, I, EE, EI, IE, II)

SNN.monitor!(E, [:ge, :gi, :v])
SNN.monitor!(model.pop, [:fire])
SNN.sim!(model = model; duration = 4second)

default(palette = :okabe_ito)

## Plot
import SpikingNeuralNetworks.SNNPlots: default, plot, histogram, Plots, plot!, savefig
p = plot([SNN.vecplot(E, :v, neurons = n, r = 3s:4s, add_spikes = true, lw = 2, xlabel = "", c=:black) for n in 1:4]..., layout=(4,1), leftmargin=10Plots.mm, rightmargin=10Plots.mm, frame=:none) 
plot!(p, subplot=4,  xticks=(0:1s:4s, 0:1s:4s), xlabel = "Time (s)", ylabel = "Membrane potential (mV)")
p = plot(
    SNN.raster(model.pop, 
        [3.4s, 4s], yrotation=90),
    SNN.vecplot(
        E,
        [:ge, :gi],
        neurons = 1,
        r = 3.8s:4s;
        legend = true,
        xlabel = "Time (s)",
        ylabel = "Conductance (nS)",
    ),
    p,
    
    SNN.firing_rate(model.pop.E, 0:4s, time_average = true)[1] |>
    x->histogram(
        x,
        ylabel = "Neurons",
        xlabel = "Firing rate (Hz)",
        label = "",
        c = :black,
    ),
    size = (900, 600), fg_legend=:transparent, legendfontsize=12
)

p = plot!(p, size=(900,600))
savefig(p, joinpath(SNN.DOCS_ASSETS_PATH, "AdEx_net.png"))