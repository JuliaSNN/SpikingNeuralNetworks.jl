# Define neurons and synapses in the network
N = 1
E = SNN.BallAndStick(
    (150um, 200um);
    N = 1,
    soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
    dend_syn = Synapse(EyalGluDend, MilesGabaDend),
    NMDA = SNN.EyalNMDA,
    param = SNN.AdExSoma(b = 0.0f0, Vr = -50),
)


model = merge_models(Dict(:E => E))
SNN.monitor!(model.pop.E, [:v_s, :v_d, :he_s, :h_d, :ge_s, :g_d])

SNN.sim!(model = model, duration = 1000ms, dt = 0.125)
model.pop.E.v_s[1] = -50mV
model.pop.E.ge_s[1] = 100nS
SNN.integrate!(model.pop.E, model.pop.E.param, 0.125f0)
model.pop.E.ge_s
SNN.sim!(model = model, duration = 1000ms, dt = 0.125)


SNN.vecplot(model.pop.E, :ge_s, dt = 0.125)
