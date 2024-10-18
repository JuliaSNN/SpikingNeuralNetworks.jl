using SpikingNeuralNetworks
using Test
SNN.@load_units;

N = 1
E = SNN.Tripod(
    300um,
    150um,
    N = N,
    soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
    dend_syn = Synapse(EyalGluDend, MilesGabaDend),
    NMDA = SNN.EyalNMDA,
    param = SNN.AdExSoma(Er = -65mV),
)


pre = (fire = falses(1), N = 1)
w = 20 * ones(1, 1)
projection_exc_dend = SNN.CompartmentSynapse(pre, E, :d1, :exc, w = w)
projection_inh_dend = SNN.CompartmentSynapse(pre, E, :d1, :inh, w = w)
projection_exc_soma = SNN.CompartmentSynapse(pre, E, :s, :exc, w = w)
projection_inh_soma = SNN.CompartmentSynapse(pre, E, :s, :inh, w = w)


projections = [projection_exc_dend, projection_inh_dend]


SNN.sim!([E], projections, duration = 1000)
#
SNN.monitor(E, [:v_s, :v_d1, :v_d2, :g_d1, :fire, :g_s, :w_s])
SNN.sim!([E], projections, duration = 50)
for p in projections
    p.fireJ[1] = true
end
SNN.sim!([E], projections, duration = 0.1f0)
for p in projections
    p.fireJ[1] = false
end
SNN.sim!([E], projections, duration = 200)
using Plots

plot(
    plot(vcat(E.records[:g_d1]...)[:, 1:2], ylabel = "g_s exc", labels = ["AMPA" "NMDA"]),
    plot(vcat(E.records[:g_d1]...)[:, 3:4], ylabel = "g_s inh", labels = ["GABAa" "GABAb"]),
    SNN.vecplot(E, :v_d1; ylims = (-65, -55)),
    SNN.vecplot(E, :v_d2; ylims = (-65, -55)),
    SNN.vecplot(E, :v_s; ylims = (-65, -55)),
    SNN.vecplot(E, :w_s),
    layout = (6, 1),
    size = (500, 800),
    xticks = (0:500:2500, 0:50:250),
    leftmargin = 5Plots.mm,
)
