# Balanced E/I network (Zerlaut et al. 2019 parameters) driven by Poisson afferents of increasing
# rate: firing rate and synchrony (spike-time tiling coefficient, STTC) of the excitatory neurons.
# For the transfer function alone see afferent_response.jl.
using SpikingNeuralNetworks
using CairoMakie, Statistics
SNN.@load_units

Zerlaut2019_network = (
    Npop = (E = 800, I = 200),
    exc = SNN.IFParameter(C = 200pF, gl = 10nS, El = -70mV, Vt = -50.0mV, Vr = -70.0mV),
    inh = SNN.IFParameter(C = 200pF, gl = 10nS, El = -70mV, Vt = -53.0mV, Vr = -70.0mV),
    spike_exc = SNN.PostSpike(τabs = 2ms),
    spike_inh = SNN.PostSpike(τabs = 1ms),
    synapse = SNN.SingleExpSynapse(τi = 5ms, τe = 5ms, E_i = -80mV, E_e = 0mV),
    connections = (
        E_to_E = (p = 0.05, μ = 2nS, rule = :Fixed),
        E_to_I = (p = 0.05, μ = 2nS, rule = :Fixed),
        I_to_E = (p = 0.05, μ = 10nS, rule = :Fixed),
        I_to_I = (p = 0.05, μ = 10nS, rule = :Fixed),
    ),
    afferents = (layer = SNN.PoissonLayer(rate = 10Hz, N = 100), conn = (p = 0.1f0, μ = 4.0nS)),
)

function network(config)
    (; afferents, connections, Npop, spike_exc, spike_inh, exc, inh, synapse) = config
    E = SNN.Population(exc; synapse, spike = spike_exc, N = Npop.E, name = "E")
    I = SNN.Population(inh; synapse, spike = spike_inh, N = Npop.I, name = "I")
    afferentE = SNN.Stimulus(afferents.layer, E, :glu, conn = afferents.conn, name = "noiseE")
    afferentI = SNN.Stimulus(afferents.layer, I, :glu, conn = afferents.conn, name = "noiseI")
    synapses = (
        E_to_E = SNN.SpikingSynapse(E, E, :glu, conn = connections.E_to_E, name = "E_to_E"),
        E_to_I = SNN.SpikingSynapse(E, I, :ge, conn = connections.E_to_I, name = "E_to_I"),
        I_to_E = SNN.SpikingSynapse(I, E, :gi, conn = connections.I_to_E, name = "I_to_E"),
        I_to_I = SNN.SpikingSynapse(I, I, :gi, conn = connections.I_to_I, name = "I_to_I"),
    )
    model = SNN.compose(; E, I, afferentE, afferentI, synapses..., silent = true, name = "Balanced network")
    SNN.monitor!(model.pop, [:fire])
    return model
end

νs = exp.(range(log(4), log(40), 6))
results = map(νs) do input_rate
    config = SNN.@update Zerlaut2019_network begin
        afferents.layer.rate = input_rate * Hz
    end
    model = network(config)
    SNN.sim!(; model, duration = 2s)
    frE, _ = SNN.firing_rate(model.pop.E, interval = 1s:10ms:2s, pop_average = true)
    sttc = SNN.STTC(model.pop.E, ΔT = 5ms, interval = 1s:2s)
    @info "afferent $(round(input_rate, digits = 1)) Hz: E $(round(mean(frE), digits = 2)) Hz, STTC $(round(mean(sttc), digits = 3))"
    (rate = mean(frE), sttc = mean(sttc), matrix = sttc)
end

fig = Figure(size = (800, 350))
ax1 = Axis(fig[1, 1], xlabel = "Afferent rate (Hz)", ylabel = "E firing rate (Hz)", xscale = log10)
scatterlines!(ax1, νs, [r.rate for r in results], color = :black)
ax2 = Axis(fig[1, 2], xlabel = "Afferent rate (Hz)", ylabel = "Mean STTC", xscale = log10)
scatterlines!(ax2, νs, [r.sttc for r in results], color = :darkred)
ax3 = Axis(fig[1, 3], title = "STTC, $(round(νs[3], digits = 1)) Hz", xlabel = "Neuron", ylabel = "Neuron")
heatmap!(ax3, results[3].matrix[1:200, 1:200])
save(joinpath(mktempdir(), "weak_strong_afferents.png"), fig)
fig
