E1 = SNN.IF(; N = 1, param = SNN.IFParameter(; El = -65mV, Vr = -55mV))
E2 = SNN.IF(; N = 1, param = SNN.IFParameter(; El = -65mV, Vr = -55mV))
EE = SNN.SpikingSynapse(
    E1,
    E2,
    :ge;
    μ = 60 * 0.27 / 10,
    p = 1,
    delay_dist = Uniform(1ms, 5ms),
)
# PositiveUniform(10ms/0.125, 0.1ms./0.125))

# SNN.monitor!([E, I], [:fire])
# SNN.sim!(P, C; duration = 1second)
SNN.monitor!(EE, [:g, :W])
#
SNN.monitor!(E1, [:v, :fire])
SNN.monitor!(E2, [:v, :fire])
SNN.sim!([E1, E2], [EE]; duration = 1second, dt = 0.125)
E1.v[1] = -20mV
# E1.fire[1] = 1
SNN.sim!([E1, E2], [EE]; duration = 1second, dt = 0.125)

SNN.vecplot(E1, :v, r = 990:0.01:1.1second, dt = 0.125, neurons = 1:1)
SNN.vecplot(EE, :g, r = 990:0.01:1.1second, dt = 0.125, neurons = 1:1)
SNN.vecplot(E2, :v, r = 990:0.01:1.1second, dt = 0.125, neurons = 1:1)
