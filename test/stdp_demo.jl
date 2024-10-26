N = 100
E1 = SNN.IF(; N = N)
E2 = SNN.IF(; N = N)
EE = SNN.SpikingSynapse(E1, E2, :ge, param = SNN.vSTDPParameter())
for n = 1:E1.N
    SNN.connect!(EE, n, n)
end
SNN.monitor([E1, E2], [:fire])
SNN.monitor(EE, [:W])
SNN.monitor(EE, [:x])


for t = 1:N
    E1.v[t] = 100
    E2.v[N-t+1] = 100
    SNN.train!([E1, E2], [EE], duration = (t - 1) * 0.5ms, dt = 0.1ms)
end
ΔW = EE.records[:W][end]
x = EE.records[:plasticity][:x][end]
