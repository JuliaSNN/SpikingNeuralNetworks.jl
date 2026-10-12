# Paradoxical effect of inhibition-stabilised networks (Tsodyks et al. 1997, J. Neurosci. 17:4382):
# an extra current into the inhibitory neurons lowers their steady-state firing rate.
using SpikingNeuralNetworks
using CairoMakie, Statistics, Random
SNN.@load_units

function initialize()
    synapse = SNN.SingleExpSynapse(τe = 5ms, τi = 5ms, E_e = 0mV, E_i = -80mV)
    spike = SNN.PostSpike(τabs = 2ms)
    E = SNN.IF(; N = 800, param = SNN.IFParameter(C = 200pF, gl = 10nS, Vt = -50mV, Vr = -70mV), synapse, spike, name = "E")
    I = SNN.IF(; N = 200, param = SNN.IFParameter(C = 200pF, gl = 10nS, Vt = -53mV, Vr = -70mV), synapse, spike, name = "I")
    EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.05, μ = 2nS))
    EI = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.05, μ = 2nS))
    IE = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.05, μ = 20nS))
    II = SNN.SpikingSynapse(I, I, :gi; conn = (p = 0.05, μ = 20nS))
    noise = SNN.PoissonLayer(rate = 20Hz, N = 100)
    inE = SNN.Stimulus(noise, E, :ge; conn = (p = 0.1, μ = 4nS))
    inI = SNN.Stimulus(noise, I, :ge; conn = (p = 0.1, μ = 4nS))
    return SNN.compose(; E, I, EE, EI, IE, II, inE, inI, silent = true)
end

Irange = 0:25pA:100pA
rE = zeros(length(Irange))
rI = zeros(length(Irange))
for x in eachindex(Irange)
    Random.seed!(10)
    model = initialize()
    SNN.sim!(model, 500ms)                       # transient
    SNN.monitor!(model.pop, [:fire])
    model.pop.I.I .= Irange[x]                   # extra current into the inhibitory neurons
    SNN.sim!(model, 2second)
    fr, r, labels = SNN.firing_rate(model.pop, interval = 0.5s:10ms:2.5s, pop_average = true)
    rE[x], rI[x] = mean(fr[1]), mean(fr[2])
    @info "Input to I: $(Irange[x]) pA, E: $(round(rE[x], digits = 2)) Hz, I: $(round(rI[x], digits = 2)) Hz"
end

fig = Figure(size = (450, 350))
ax = Axis(fig[1, 1], xlabel = "Current into I neurons (pA)", ylabel = "Firing rate (Hz)")
scatterlines!(ax, Irange, rE, label = "E")
scatterlines!(ax, Irange, rI, label = "I")
axislegend(ax)
save(joinpath(mktempdir(), "ParadoxicalEffect.png"), fig)
fig
