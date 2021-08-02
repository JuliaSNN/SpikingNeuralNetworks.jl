G = SNN.Rate(;N = 100)
GG = SNN.RateSynapse(G, G; σ = 1.2, p = 1.0)
SNN.monitor(G, [:r])

SNN.train!([G], [GG]; duration = 100ms)
