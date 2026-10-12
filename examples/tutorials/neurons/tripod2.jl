# Tripod neuron with double-exponential synapses on soma and dendrites, driven by Poisson input
# onto both dendrites.
using SpikingNeuralNetworks
using CairoMakie, Random
SNN.@load_units
Random.seed!(1234)

E = SNN.Population(SNN.TripodParameter(); N = 1,
                   spike = SNN.PostSpike(At = 0, τA = 30ms, τabs = 1, up = 1),
                   adex = SNN.AdExParameter(Vr = -40mV, b = 80pA, a = 0),
                   dend_syn = SNN.DoubleExpSynapse(), soma_syn = SNN.DoubleExpSynapse())

poisson_exc = SNN.PoissonLayer(rate = 10.2Hz, N = 100)
poisson_inh = SNN.PoissonLayer(rate = 3Hz, N = 1000)
exc_conn = (p = 1.0f0, μ = 8.5)     # nS
inh_conn = (p = 1.0f0, μ = 6.0)     # nS
stim_exc1 = SNN.Stimulus(poisson_exc, E, :glu, :d1, conn = exc_conn, name = "noiseE1")
stim_inh1 = SNN.Stimulus(poisson_inh, E, :gaba, :d1, conn = inh_conn, name = "noiseI1")
stim_exc2 = SNN.Stimulus(poisson_exc, E, :glu, :d2, conn = exc_conn, name = "noiseE2")
stim_inh2 = SNN.Stimulus(poisson_inh, E, :gaba, :d2, conn = inh_conn, name = "noiseI2")

model = SNN.compose(; E, stim_exc1, stim_inh1, stim_exc2, stim_inh2, silent = true)
SNN.monitor!(E, [:fire, :v_d1, :v_s, :v_d2, :w_s])
SNN.sim!(model, 5s)

fig = Figure(size = (800, 400))
ax = Axis(fig[1, 1], ylabel = "Voltage (mV)", xlabel = "Time (s)")
SNN.vecplot!(ax, E, :v_d1, neurons = 1, interval = 0s:1ms:5s, label = "Dendrite 1")
SNN.vecplot!(ax, E, :v_d2, neurons = 1, interval = 0s:1ms:5s, label = "Dendrite 2")
SNN.vecplot!(ax, E, :v_s, neurons = 1, interval = 0s:1ms:5s, add_spikes = true, color = :black, label = "Soma")
axislegend(ax)
save(joinpath(mktempdir(), "tripod2.png"), fig)
