using SpikingNeuralNetworks
using Test
SNN.@load_units;


N = 1
E = SNN.TripodHet(
    N = N,
    soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
    dend_syn = Synapse(EyalGluDend, MilesGabaDend),
    NMDA = SNN.EyalNMDA,
    param = SNN.AdExSoma(Er = -55mV),
)


pre = (fire = falses(1), N = 1)
w = 20 * ones(1, 1)
projection_exc_soma = SNN.CompartmentSynapse(pre, E, :s, :he, w = w)
projection_inh_soma = SNN.CompartmentSynapse(pre, E, :s, :hi, w = w)


projections = [projection_exc_soma, projection_inh_soma]


SNN.sim!([E], projections, duration = 1000)
#
SNN.monitor!(E, [:v_s, :v_d1, :g_d1, :fire, :ge_s, :gi_s, :w_s])
SNN.sim!([E], projections, duration = 50)
for p in projections
    p.fireJ[1] = true
end
SNN.sim!([E], projections, duration = 0.1f0)
for p in projections
    p.fireJ[1] = false
end
SNN.sim!([E], projections, duration = 200)
using SNNPlots

E.records

vcat(SNN.getrecord(E, :ge_s)...)
vcat(SNN.getrecord(E, :gi_s)...)

plot(
    SNN.vecplot(E, :ge_s, r = 1:0.01:250),
    SNN.vecplot(E, :gi_s, r = 1:0.01:250),
    SNN.vecplot(E, :v_s, r = 1:0.01:250),
    SNN.vecplot(E, :v_d1, r = 1:0.01:250),
    SNN.vecplot(E, :w_s, r = 1:0.01:250),
    layout = (5, 1),
    size = (500, 900),
    linky = true,
    margin = 5Plots.mm,
)
