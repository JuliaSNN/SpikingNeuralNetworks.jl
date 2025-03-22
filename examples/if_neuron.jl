using Plots
using SpikingNeuralNetworks
SNN.@load_units

E = SNN.IF(; N = 1)
r = 50Hz * 300
stim1 = SNN.PoissonStimulus(E, :ge, param = r, μ = 1.0f0, neurons=:ALL)
stim2 = SNN.PoissonStimulus(E, :gi, param = r, μ = 0.1f0, neurons=:ALL)
# E.I = [11]
SNN.monitor(E, [:v, :fire, :ge, :gi], sr = 1 / dt)

model = merge_models(E, sE = stim1, sI = stim2, silent = true)
SNN.sim!(model = model; duration = 1s)
SNN.vecplot(E, :v, r = 0.9s:1ms:1s)
SNN.vecplot(E, :ge, r = 0.9s:1ms:1s)
SNN.vecplot(E, :gi, r = 0.9s:1ms:1s)
