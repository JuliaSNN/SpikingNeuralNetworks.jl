
function test_inputs(μ, rate)
    E = SNN.Identity(N=1)
    rate = 100Hz
    S = SNN.PoissonStimulus(E, :g, x->rate, N=1000, N_pre=10, μ=μ, cells=[1])
    model = SNN.merge_models(pop=Dict("E"=>E), stim=Dict("stim"=>S))

    SNN.monitor(E, [:fire, :h])
    SNN.monitor(S, [:fire])
    S.records[:fire]
    SNN.sim!(model=model, duration=1s, pbar=true, dt=0.125ms)
    return vcat(SNN.getrecord(E, :h)...)[end]/length(SNN.spiketimes(E)[1]), length(SNN.spiketimes(E)[1])
end

## 
# test_inputs(15.f0, 400Hz)


E = SNN.BallAndStick((150um, 300um),
        N = 1,
        soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma), # defines glutamaterbic and gabaergic receptors in the soma
        dend_syn = Synapse(EyalGluDend, MilesGabaDend), # defines glutamaterbic and gabaergic receptors in the dendrites
        NMDA = SNN.EyalNMDA,
        param = SNN.AdExSoma(Vr = -55mV, Vt = -50mV)
        )

my_rate = 4000Hz
S = SNN.PoissonStimulus(E, :h_d, x->my_rate, N=1000, N_pre=10, μ=5.f0, cells=[1])
model = SNN.merge_models(pop=Dict("E"=>E), stim=Dict("stim"=>S))
SNN.monitor(E, [:fire, :h_d, :v_d, :v_s, :g_d])
SNN.monitor(S, [:fire])
S.records[:fire]
SNN.sim!(model=model, duration=10s, pbar=true, dt=0.125ms)
SNN.vecplot(model.pop.E, :v_d, r=0.5s:1s, sym_id=1, dt=0.125, pop_average=true)


##

E = SNN.Tripod(proximal_distal...,
        N = 1,
        soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma), # defines glutamaterbic and gabaergic receptors in the soma
        dend_syn = Synapse(EyalGluDend, MilesGabaDend), # defines glutamaterbic and gabaergic receptors in the dendrites
        NMDA = SNN.EyalNMDA,
        param = SNN.AdExSoma(Vr = -55mV, Vt = -50mV)
        )

my_rate = 4000Hz
S = SNN.PoissonStimulus(E, :h_d1, x->my_rate, N=1000, N_pre=10, μ=5.f0, cells=[1])
model = SNN.merge_models(pop=Dict("E"=>E), stim=Dict("stim"=>S))
SNN.monitor(E, [:fire, :h_d1, :v_d1, :v_s, :g_d1])
SNN.monitor(S, [:fire])
S.records[:fire]
SNN.sim!(model=model, duration=10s, pbar=true, dt=0.125ms)
SNN.vecplot(model.pop.E, :v_d1, r=0.5s:1s, sym_id=1, dt=0.125, pop_average=true)