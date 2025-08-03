using SpikingNeuralNetworks
SNN.@load_units

E = SNN.IF(; N = 1)
rate = 50Hz * 300
stim1 = SNN.PoissonStimulus(E, :ge, param = rate, μ = 1.0f0, neurons = :ALL)
stim2 = SNN.PoissonStimulus(E, :gi, param = rate, μ = 5.1f0, neurons = :ALL)
# E.I = [11]
SNN.monitor!(E, [:v, :fire, :ge, :gi], sr = 1000Hz)

model = merge_models(E, sE = stim1, sI = stim2, silent = true)
SNN.sim!(model = model; duration = 10s)
plot(
    SNN.vecplot(E, :v, r = 0.9s:1ms:10s),
    SNN.vecplot(E, [:ge, :gi], r = 0.9s:1ms:10s),
    layout = (2, 1),
    size = (800, 600),
)
