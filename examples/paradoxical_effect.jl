using Plots
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter
using Statistics, Random

function initialize()
    E = SNN.AdEx(; N = 2000, param = AdExParameterSingleExponential())
    I = SNN.IF(; N = 500, param = SNN.IFParameterSingleExponential())
    EE = SNN.SpikingSynapse(E, E, :ge; μ = 1.0, p = 0.2)
    EI = SNN.SpikingSynapse(E, I, :ge; μ = 4.0, p = 0.2)
    IE = SNN.SpikingSynapse(I, E, :gi; μ = 10.0, p = 0.2)
    II = SNN.SpikingSynapse(I, I, :gi; μ = 2, p = 0.2)
    inputs = SNN.Poisson(; N = 200, param = SNN.PoissonParameter(rate = 10.5Hz))
    ProjE = SNN.SpikingSynapse(inputs, E, :ge; μ = 5, p = 0.2)
    P = (;E, I, inputs)
    C = (;EE, EI, IE, II, ProjE)
    return merge_models(; P..., C..., silent=true)
end
#


model = initialize()
SNN.monitor!(model.pop, [:fire, :v, :ge])
SNN.sim!(;model, duration = 3second)
SNN.vecplot(model.pop.E, [:ge], r = 0.1s:1.2s, neurons = 1:4)
SNN.vecplot(model.pop.E, [:v], r = 0.1s:1.2s, neurons = 1:5)
fr, r, labels = firing_rate(model.pop, interval=0.1s:1.2s)
e, i, inputs = mean.(fr)
@info "E: $e, I: $i, Inputs: $inputs"
SNN.raster(model.pop, [1 * 1000, 3.0 * 1000])


##
Irange = 0:0.05nA:1nA
rE = zeros(length(Irange))
rI = zeros(length(Irange))
Threads.@threads for x in eachindex(Irange)
    Random.seed!(10)
    model = initialize()
    SNN.sim!(;model, duration = 1second)
    SNN.monitor!(model.pop, [:fire])
    model.pop.I.I .= Irange[x]
    SNN.sim!(;model, duration = 5second)
    fr, r, labels = firing_rate(model.pop, interval=0.0s:10ms:5s)
    e, i, inputs = mean.(fr)
    @info "Input: $x, E: $e, I: $i"
    rE[x] = e
    rI[x] = i
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
inputs = SNN.Poisson(; N = 350, param = SNN.PoissonParameter(rate = 10.5Hz))
@show(inputs)