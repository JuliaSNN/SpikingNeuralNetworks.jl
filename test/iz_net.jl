Ne = 800;      Ni = 200
E = SNN.IZ(;N = Ne, param = SNN.IZParameter(;a = 0.02, b = 0.2, c = -65, d = 8))
I = SNN.IZ(;N = Ni, param = SNN.IZParameter(;a = 0.1, b = 0.2, c = -65, d = 2))

EE = SNN.SpikingSynapse(E, E, :v; σ = 0.5,  p = 0.8)
EI = SNN.SpikingSynapse(E, I, :v; σ = 0.5,  p = 0.8)
IE = SNN.SpikingSynapse(I, E, :v; σ = -1.0, p = 0.8)
II = SNN.SpikingSynapse(I, I, :v; σ = -1.0, p = 0.8)
P = [E, I]
C = [EE, EI, IE, II]

SNN.monitor([E, I], [:fire])
for t = 1:1000
    E.I .= 5randn(Ne)
    I.I .= 2randn(Ni)
    SNN.sim!(P, C, 1ms)
end
