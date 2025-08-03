using SpikingNeuralNetworks
using Distributions
SNN.@load_units

IF_param = SNN.IF_CANAHPParameter()
E = SNN.IF_CANAHP(N = 1, param = IF_param)
SNN.monitor!(E, [:v, :fire, :hi, :he, :g, :h, :I, :syn_curr], sr = 8000Hz)

I_param = SNN.CurrentNoiseParameter(1; I_base = 100pA)
I_stim = SNN.CurrentStimulus(E, param = I_param)

model = SNN.merge_models(; E, I_stim)
SNN.sim!(; model, duration = 5s)
E.he[1] += 0.1
SNN.sim!(; model, duration = 2s)
E.hi[1] += 1
SNN.sim!(; model, duration = 3s)

SNN.vecplot(E, :v, r = 0.8s:1ms:10s)
SNN.vecplot(E, :syn_curr, r = 0.8s:1ms:10s)

p1 = SNN.vecplot(E, :g, sym_id = 1, r = 0.8s:1ms:10s, label="ampa")
p1 = SNN.vecplot!(p1, E, :g, sym_id = 2, r = 0.8s:1ms:10s, label="nmda")
p2 = SNN.vecplot(E, :g, sym_id = 3, r = 0.8s:1ms:10s, label="gaba_a")
p2 = SNN.vecplot!(p2, E, :g, sym_id = 4, r = 0.8s:1ms:10s, label="gaba_b")
p3 = SNN.vecplot(E, :v, r = 0.8s:1ms:10s, lc=:black, label="")
plot(
    p1,
    p2,
    p3,
    layout = (3, 1),
    size = (800, 600),
    xlabel = "Time [s]",
    legend = :topleft,
)
plot!(ylims = :auto)
