E = SNN.HH(;N = 3200)
I = SNN.HH(;N = 800)
EE = SNN.SpikingSynapse(E, E, :ge; σ = 6nS, p = 0.02)
EI = SNN.SpikingSynapse(E, I, :ge; σ = 6nS, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :gi; σ = 67nS, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; σ = 67nS, p = 0.02)
P = [E, I]
C = [EE, EI, IE, II]

SNN.monitor(E, [(:v, [1, 10, 100])])
SNN.sim!(P, C; dt = 0.01ms, duration = 100ms)
