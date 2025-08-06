using SpikingNeuralNetworks
SNN.@load_units
using Statistics, Random, Plots

using Logging

# Set the logging level to debug
global_logger(ConsoleLogger(stderr, Logging.Info))

## AdEx neuron with fixed external current connections with multiple receptors
E = SNN.AdExNeuron(; N = 500, param = SNN.AdExParameter(; El = -50mV))
I = SNN.IF(; N = 100, param = SNN.IFParameter())
EE = SNN.SpikingSynapse(E, E, :he; μ = 5, σ=0.2, p = 0.02, 
                        LTPParam=SNN.vSTDPParameter())
Norm = SNN.SynapseNormalization(E, [EE], param=SNN.AdditiveNorm(τ=50ms))
EI = SNN.SpikingSynapse(E, I, :ge; μ = 30, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :hi; μ = 50, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; μ = 10, p = 0.02)
model = SNN.compose(; E, I,EE, EI, IE, II, Norm)

W0 = copy(EE.W)
##
SNN.monitor!(EE, [:W], sr=10Hz)
SNN.monitor!(model.pop, [:fire])
SNN.train!(model = model; duration = 20second, pbar=true)

##

bins=0:0.1:10.0
p1 = W0 |> x-> histogram(x, label="Before training", c=:black, bins=bins)
p1 = EE.W |> x-> histogram!(p1, x,label="After training", c=:red, lc=:red, xlabel="Synaptic strength", ylabel="Number of synapses")
p2 = SNN.vecplot(EE, [:W], pop_average=true, legend=:topleft, label="Average weight",
                ylabel="Synaptic weight (nS)", xlabel="Time (s)", lw=2, c=:darkblue)
plot(p1, p2, layout=(2,1), fg_legend=:transparent, size=(800, 700), margin=10Plots.mm, legendfontsize=10)

