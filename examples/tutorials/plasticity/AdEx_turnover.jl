using SpikingNeuralNetworks
using CairoMakie, Statistics, Distributions
SNN.@load_units

# Balanced E/I network on a 1 mm x 1 mm sheet (Zerlaut et al. 2019 parameters) whose E-to-E synapses
# undergo activity-dependent turnover: the least co-active synapses are moved to new targets.
config = (
    Npop = (E = 800, I = 200),
    spatial = (type = :gaussian, σs = (; E = (200um, 200um), I = (100um, 100um)), ϵ = 0.4f0,
               grid_size = Float32.([1mm, 1mm])),
    turnover = (rate = 20Hz, fraction = 0.1f0, τpre = 1s, τpost = 1s, μ = 2.0f0),
    exc = SNN.IFParameter(τm = 200pF / 10nS, El = -70mV, Vt = -50.0mV, Vr = -70.0mV, R = 1 / 10nS),
    inh = SNN.IFParameter(τm = 200pF / 10nS, El = -70mV, Vt = -53.0mV, Vr = -70.0mV, R = 1 / 10nS),
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

    # E-to-E connectivity drawn from the distance between neurons (P: connection probabilities)
    points = SNN.place_populations(Npop, config.spatial.grid_size)
    L, W, P = SNN.compute_connections(:E, :E, points; spatial = config.spatial,
                                      conn = connections.E_to_E, dist = Normal(2, 0.2))
    synapses = (
        E_to_E = SNN.SpikingSynapse(E, E, :glu, conn = W, name = "E_to_E"),
        E_to_I = SNN.SpikingSynapse(E, I, :glu, conn = connections.E_to_I, name = "E_to_I"),
        I_to_E = SNN.SpikingSynapse(I, E, :gaba, conn = connections.I_to_E, name = "I_to_E"),
        I_to_I = SNN.SpikingSynapse(I, I, :gaba, conn = connections.I_to_I, name = "I_to_I"),
    )
    turnover = SNN.ActivityDependentTurnover(; config.turnover...)
    Exc_turnover = SNN.MetaPlasticity(turnover, synapses.E_to_E; p = P, name = "EE_Turnover")
    model = SNN.compose(; E, I, afferentE, afferentI, synapses..., Exc_turnover,
                        name = "Balanced network", silent = true)
    return model, points
end

model, points = network(config)
SNN.monitor!(model.pop, [:fire])
SNN.monitor!(model.syn.Exc_turnover, [:p_rewire], sr = 200Hz)
SNN.train!(model, duration = 5s)          # turnover runs only under train!

fr, r, labels = SNN.firing_rate(model.pop, 0:100ms:5s, pop_average = true)
rates = SNN.firing_rate(model.pop.E, 0:20ms:5s, time_average = true)[1]
th, rt = SNN.record(model.syn.Exc_turnover, :p_rewire, range = true)
exc_x, exc_y = hcat(points.E...)[1, :], hcat(points.E...)[2, :]

fig = Figure(size = (800, 700))
ax1 = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Firing rate (Hz)")
for (k, f) in enumerate(fr)
    lines!(ax1, r ./ 1000, f, label = labels[k])
end
axislegend(ax1)
SNN.raster!(Axis(fig[1, 2]), model.pop, 4s:5s)
ax3 = Axis(fig[2, 1], xlabel = "Time (s)", ylabel = "p_rewire")
lines!(ax3, rt ./ 1000, th(1, rt))
ax4 = Axis(fig[2, 2], xlabel = "x (mm)", ylabel = "y (mm)", title = "Average firing rate (Hz)")
sc = scatter!(ax4, exc_x, exc_y, color = rates, colorrange = (0, 3), colormap = :viridis)
Colorbar(fig[2, 3], sc)
save(joinpath(mktempdir(), "AdEx_turnover.png"), fig)
