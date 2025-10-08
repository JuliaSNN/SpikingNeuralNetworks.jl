using SpikingNeuralNetworks
using Test
@load_units;


N = 1
E = TripodHet(
    N = N,
    soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
    dend_syn = Synapse(EyalGluDend, MilesGabaDend),
    NMDA = EyalNMDA,
    param = AdExSoma(El = -55mV),
)


pre = (fire = falses(1), N = 1)
w = 20 * ones(1, 1)
projection_exc_soma = CompartmentSynapse(pre, E, :s, :he, w = w)
projection_inh_soma = CompartmentSynapse(pre, E, :s, :hi, w = w)


projections = [projection_exc_soma, projection_inh_soma]


sim!([E], projections, duration = 1000)
#
monitor!(E, [:v_s, :v_d1, :g_d1, :fire, :ge_s, :gi_s, :w_s])
sim!([E], projections, duration = 50)
for p in projections
    p.fireJ[1] = true
end
sim!([E], projections, duration = 0.1f0)
for p in projections
    p.fireJ[1] = false
end
sim!([E], projections, duration = 200)
using SNNPlots

E.records

vcat(getrecord(E, :ge_s)...)
vcat(getrecord(E, :gi_s)...)

plot(
    vecplot(E, :ge_s, r = 1:0.01:250),
    vecplot(E, :gi_s, r = 1:0.01:250),
    vecplot(E, :v_s, r = 1:0.01:250),
    vecplot(E, :v_d1, r = 1:0.01:250),
    vecplot(E, :w_s, r = 1:0.01:250),
    layout = (5, 1),
    size = (500, 900),
    linky = true,
    margin = 5Plots.mm,
)
