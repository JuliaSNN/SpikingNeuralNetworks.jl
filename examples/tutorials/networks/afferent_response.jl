# Transfer function of a balanced E/I network (Zerlaut et al. 2019 parameters): firing rate of
# the excitatory population as a function of the rate of its Poisson afferents.
using SpikingNeuralNetworks
using CairoMakie, Statistics
SNN.@load_units

Zerlaut2019_network = (
    Npop = (E = 800, I = 200),
    exc = SNN.IFParameter(C = 200pF, gl = 10nS, El = -70mV, Vt = -50.0mV, Vr = -70.0mV),
    inh = SNN.IFParameter(C = 200pF, gl = 10nS, El = -70mV, Vt = -53.0mV, Vr = -70.0mV),
    spike = SNN.PostSpike(τabs = 2ms),
    synapse = SNN.SingleExpSynapse(τi = 5ms, τe = 5ms, E_i = -80mV, E_e = 0mV),
    connections = (
        E_to_E = (p = 0.05, μ = 2nS, rule = :Fixed),
        E_to_I = (p = 0.05, μ = 2nS, rule = :Fixed),
        I_to_E = (p = 0.05, μ = 10nS, rule = :Fixed),
        I_to_I = (p = 0.05, μ = 10nS, rule = :Fixed),
    ),
    afferents = (N = 100, rate = 20Hz, conn = (p = 0.1f0, μ = 4.0)),
)

function soma_network(config)
    (; afferents, connections, Npop, synapse, spike) = config
    E = SNN.IF(; N = Npop.E, param = config.exc, synapse, spike, name = "E")
    I = SNN.IF(; N = Npop.I, param = config.inh, synapse, spike, name = "I")
    AfferentParam = SNN.PoissonLayer(rate = afferents.rate, N = afferents.N)
    afferentE = SNN.Stimulus(AfferentParam, E, :ge, conn = afferents.conn, name = "noiseE")
    afferentI = SNN.Stimulus(AfferentParam, I, :ge, conn = afferents.conn, name = "noiseI")
    synapses = (
        E_to_E = SNN.SpikingSynapse(E, E, :ge, conn = connections.E_to_E, name = "E_to_E"),
        E_to_I = SNN.SpikingSynapse(E, I, :ge, conn = connections.E_to_I, name = "E_to_I"),
        I_to_E = SNN.SpikingSynapse(I, E, :gi, conn = connections.I_to_E, name = "I_to_E"),
        I_to_I = SNN.SpikingSynapse(I, I, :gi, conn = connections.I_to_I, name = "I_to_I"),
    )
    model = SNN.compose(; E, I, afferentE, afferentI, synapses..., silent = true, name = "Balanced network")
    SNN.monitor!(model.pop, [:fire])
    return model
end

νa = exp.(range(log(2), log(40), 8))
f_rate = map(νa) do x
    config = SNN.@update Zerlaut2019_network begin
        afferents.rate = x * Hz
    end
    model = soma_network(config)
    SNN.sim!(; model, duration = 2s)
    fr, _ = SNN.firing_rate(model.pop.E, interval = 0.5s:10ms:2s, pop_average = true, time_average = true)
    @info "afferent rate $(round(x, digits = 1)) Hz: E rate $(round(mean(fr), digits = 2)) Hz"
    mean(fr)
end

fig = Figure(size = (400, 400))
ax = Axis(fig[1, 1], xlabel = "Afferent rate (Hz)", ylabel = "E firing rate (Hz)", xscale = log10)
scatterlines!(ax, νa, f_rate, color = :black)
save(joinpath(mktempdir(), "afferent_response.png"), fig)
fig
