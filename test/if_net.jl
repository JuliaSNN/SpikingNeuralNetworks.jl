E = SNN.IF(; N = 3200, param = SNN.IFParameter(; El = -49mV))
I = SNN.IF(; N = 800, param = SNN.IFParameter(; El = -49mV))
EE = SNN.SpikingSynapse(E, E, :ge; μ = 60 * 0.27 / 10, p = 0.02)
EI = SNN.SpikingSynapse(E, I, :ge; μ = 60 * 0.27 / 10, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :gi; μ = -20 * 4.5 / 10, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; μ = -20 * 4.5 / 10, p = 0.02)
P = [E, I]
C = [EE, EI, IE, II]

SNN.monitor!([E, I], [:fire])
SNN.sim!(P, C; duration = 1second)
SNN.train!(P, C; duration = 1second)
