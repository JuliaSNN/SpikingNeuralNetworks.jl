"""
This script demonstrates the training of a spiking neural network (SNN) with adaptive exponential integrate-and-fire (AdEx) neurons and integrate-and-fire (IF) neurons. The network includes excitatory and inhibitory connections, with synaptic plasticity modeled using spike-timing-dependent plasticity (STDP). The script monitors synaptic weights and neuronal firing activity, trains the network for 20 seconds, and visualizes the results, including a comparison of synaptic weights before and after training.
"""

using SpikingNeuralNetworks
SNN.@load_units
using Statistics, Random, Plots


## AdEx neuron with fixed external current connections with multiple receptors
E = SNN.AdEx(; N = 500, param = SNN.AdExParameter(; El = -50mV))
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
SNN.monitor!(EE, [:W], sr=100Hz)
SNN.monitor!(model.pop, [:fire])
SNN.train!(model = model; duration = 20second, pbar=true)

## Plots

bins=0:0.1:10.0
p1 = plot()
p1 = W0 |> x-> histogram!(p1, x, label="Before training", c=:black, bins=bins)
p1 = EE.W |> x-> histogram!(p1, x,label="After training", c=:red, lc=:red, xlabel="Synaptic strength", ylabel="Number of synapses", alpha=0.7, line_color=:transparent)
p2 = SNN.vecplot(EE, [:W], pop_average=true, legend=:topleft, label="Average weight",
                ylabel="Synaptic weight (nS)", xlabel="Time (s)", lw=2, c=:darkblue)
p = plot(p1, p2, layout=(1,2), fg_legend=:transparent, size=(900, 500), margin=10Plots.mm, legendfontsize=10)
savefig(p, joinpath(SNN.DOCS_ASSETS_PATH, "STDP_net.png"))