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
w = 50 * ones(1, 1)
projection_exc_dend = SNN.CompartmentSynapse(pre, E, :d1, :he, w = w)
projection_inh_dend = SNN.CompartmentSynapse(pre, E, :d1, :hi, w = w)


projections = [projection_exc_dend, projection_inh_dend]


SNN.sim!([E], projections, duration = 1000)
#
SNN.monitor(E, [:v_s, :v_d1, :v_d2, :g_d1, :fire])
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



p1 = plot()
SNN.vecplot!(p1, E, :g_d1, r=1:0.01:250, sym_id=1,label= "AMPA")
SNN.vecplot!(p1,E, :g_d1, r=1:0.01:250, sym_id=2, label= "NMDA")
plot!(p1, ylims=(0,30))
p2 = plot()
SNN.vecplot!(p2,E, :g_d1, r=1:0.01:250, sym_id=3, label= "GABAa")
SNN.vecplot!(p2,E, :g_d1, r=1:0.01:250, sym_id=4, label= "GABAb")
plot!(p2, ylims=(0,30))
plot(
    p1,
    p2,
    SNN.vecplot(E, :v_d1; ylims = (-65, -55), r= 1:0.01:250),
    SNN.vecplot(E, :v_d2; ylims = (-65, -55), r= 1:0.01:250),
    SNN.vecplot(E, :v_s; ylims = (-65, -55), r= 1:0.01:250),
    layout=(5,1),
    size = (500, 800),
    xticks = (0:500:2500, 0:50:250),
    leftmargin = 5Plots.mm,
    legend=:topright,
)
