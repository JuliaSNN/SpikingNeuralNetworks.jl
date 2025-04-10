using Plots
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter
using Statistics, Random

function initialize()
    E = SNN.AdEx(; N = 2000, param = AdExParameterSingleExponential(El = -65mV))
    I = SNN.IF(; N = 500, param = SNN.IFParameterSingleExponential())
    EE = SNN.SpikingSynapse(E, E, :ge; μ = 1.0, p = 0.2)
    EI = SNN.SpikingSynapse(E, I, :ge; μ = 10, p = 0.2)
    IE = SNN.SpikingSynapse(I, E, :gi; μ = 5.0, p = 0.2)
    II = SNN.SpikingSynapse(I, I, :gi; μ = 10, p = 0.2)
    inputs = SNN.Poisson(; N = 350, param = SNN.PoissonParameter(rate = 10.5Hz))
    ProjE = SNN.SpikingSynapse(inputs, E, :ge; μ = 5, p = 0.2)
    P = [E, I, inputs]
    C = [EE, EI, IE, II, ProjE]
    return P, C
end
##

P, C = initialize()
E, I = P
SNN.monitor!([E, I], [:fire, :v, :ge])
SNN.sim!(P, C; duration = 3second)
SNN.raster([E, I], [1 * 1000, 3.0 * 1000])
SNN.vecplot(E, [:ge], r = 0.1s:1.2s, neurons = 1:4)
SNN.vecplot(E, [:v], r = 0.1s:1.2s, neurons = 1:5)
i = round(sum(mean(I.records[:fire])), digits = 2)
e = round(sum(mean(E.records[:fire])), digits = 2)

##
rI = []
rE = []
Irange = 0:0.25nA:5nA
for x in Irange
    Random.seed!(10)
    P, C = initialize()
    E, I = P
    SNN.sim!(P, C; duration = 1second)
    SNN.monitor!([E, I], [:fire, :v])
    I.I .= x
    SNN.sim!(P, C; duration = 1second)
    i = round(sum(mean(I.records[:fire])), digits = 2)
    e = round(sum(mean(E.records[:fire])), digits = 2)
    @info "E: $e Hz I: $i Hz"
    push!(rE, e)
    push!(rI, i)
end
plot(
    Irange,
    [rI],
    label = "Inhibitory neurons",
    xlabel = "Input to I neurons (nA)",
    ylabel = "Firing rate (Hz)",
    lc = [:blue],
    lw = 4,
    legend = :topright,
)
plot!(
    twinx(),
    Irange,
    [rE],
    lc = [:red],
    lw = 4,
    label = "Excitatory neurons",
    legend = :topright,
)
##
