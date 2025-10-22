
Random.seed!(1234)
tripod = DendNeuronParameter(
        spike = SNN.PostSpike(At=1, τA=30ms, τabs=10,  up=10), 
        adex= SNN.AdExParameter(Vr=-40mV, b=80pA,a=0),
        dend_syn = SNN.TripodDendSynapse,
        soma_syn = SNN.TripodSomaSynapse,
        )

E = SNNModels.Tripod(;param=tripod, N=1)

poisson_exc = PoissonLayer(
    10.2Hz,    # Mean firing rate (Hz) 
    N = 200, # Neurons in the Poisson Layer
)
poisson_inh = PoissonLayer(
    3Hz,       # Mean firing rate (Hz)
    N = 1000,  # Neurons in the Poisson Layer
)

exc_conn = 
(
    p = 1.0f0,  # Probability of connecting to a neuron
    μ = 8.5,  # Synaptic strength (nS)
)
inh_conn = ( 
    p = 1.0f0,   # Probability of connecting to a neuron
    μ = 5.0,   # Synaptic strength (nS)
)
# Create the Poisson layers for excitatory and inhibitory inputs
stim_exc1 = Stimulus(poisson_exc, E, :glu, :d1, conn=exc_conn, name = "noiseE")
stim_inh1 = Stimulus(poisson_inh, E, :gaba, :d1, conn=inh_conn, name = "noiseI")
stim_exc2 = Stimulus(poisson_exc, E, :glu, :d2, conn=exc_conn, name = "noiseE")
stim_inh2 = Stimulus(poisson_inh, E, :gaba, :d2, conn=inh_conn, name = "noiseI")

model = compose(; E, stim_exc1, stim_inh1, stim_exc2, stim_inh2, silent=true)
SNN.monitor!(E, [:fire, :v_d1, :v_s, :v_d2, :w_s])

#
sim!(model, 0.01s)
#
sim!(model, 5s)
SNN.raster(model.pop,[4s, 5s])
p = SNN.vecplot(E, :v_d1, neurons=1, r=1s:5s)
SNN.vecplot!(p, E, :v_d2, neurons=1, r=1s:5s)
SNN.vecplot!(p, E, :v_s, neurons=1,  r=1s:5s, add_spikes=true)
plot!(ylims=:auto)

scatter!(SNN.spiketimes(E)[1]./1000,mw=10, ones(length(SNN.spiketimes(E)[1])), label="spikes", color=:black)
# SNN.vecplot!(p, E, :w_s, neurons=1,  r=1s:5s, add_spikes=true)
##
