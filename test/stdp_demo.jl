N = 100
E1 = SNN.IF(; N = N)
E2 = SNN.IF(; N = N)
EE = SNN.SpikingSynapse(E1, E2, :ge, LTPParam = SNN.vSTDPParameter())
for n = 1:E1.N
    SNN.connect!(EE, n, n)
end
SNN.monitor!([E1, E2], [:fire])
SNN.monitor!(EE, [:W])
SNN.monitor!(EE, [(:x, [20, 10])], variables=:LTPVars)

for t = 1:N
    E1.v[t] = -40
    E2.v[N-t+1] = -40
    SNN.train!([E1, E2], [EE], duration = 0.5ms, dt = 0.125ms)
end

ΔW = SNN.getrecord(EE, :W)[end]
x = SNN.getrecord(EE, :LTPVars_x)[end]
