using SNNModels
using BenchmarkTools
@load_units

E = Population(AdExParameter(), N=100, synapse=DoubleExpSynapse(), spike=PostSpike())

poisson_exc = PoissonLayer(
    10.2Hz,    # Mean firing rate (Hz) 
    N = 1000, # Neurons in the Poisson Layer
)
poisson_inh = PoissonLayer(
    3Hz,       # Mean firing rate (Hz)
    N = 1000,  # Neurons in the Poisson Layer
)

exc_conn = 
(
    p = 1.0f0,  # Probability of connecting to a neuron
    μ = 1.0f0,  # Synaptic strength (nS)
)
inh_conn = ( 
        p = 1.0f0,   # Probability of connecting to a neuron
    μ = 4.0f0,   # Synaptic strength (nS)
)
# Create the Poisson layers for excitatory and inhibitory inputs
stim_exc = Stimulus(poisson_exc, E, :glu, conn=exc_conn, name = "noiseE")
stim_inh = Stimulus(poisson_inh, E, :gaba, conn=inh_conn, name = "noiseI")


model = compose(; E, stim_exc, stim_inh, silent=true)
# @profview sim!(model, 50s)
@btime sim!(model, 10s)
#   
#   240.436 ms (799501 allocations: 15.86 MiB) on DellTower
##
Random.seed!(1234)
dend_neuron = (;
        param = SNN.TripodParameter(),
        spike = SNN.PostSpike(At=1, τA=30ms, τabs=10,  up=10), 
        adex= SNN.AdExParameter(Vr=-40mV, b=80pA,a=0),
        dend_syn = SNN.TripodDendSynapse,
        soma_syn = SNN.TripodSomaSynapse,
        )

E = SNNModels.Population(;dend_neuron..., N=1)

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
@profview sim!(model, 50s)
@btime sim!(model, 10s)
#   238.624 ms (1119507 allocations: 20.75 MiB) on DellTower
##

Random.seed!(1234)
dend_neuron = (;
        param = SNN.BallAndStickParameter(),
        spike = SNN.PostSpike(At=1, τA=30ms, τabs=10,  up=10), 
        adex= SNN.AdExParameter(Vr=-40mV, b=80pA,a=0),
        dend_syn = SNN.TripodDendSynapse,
        soma_syn = SNN.TripodSomaSynapse,)

E = SNNModels.Population(;dend_neuron..., N=1)

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
stim_exc1 = Stimulus(poisson_exc, E, :glu, :d, conn=exc_conn, name = "noiseE")
stim_inh2 = Stimulus(poisson_inh, E, :gaba, :d, conn=inh_conn, name = "noiseI")

model = compose(; E, stim_exc1, stim_inh1, stim_exc2, stim_inh2, silent=true)
@profview sim!(model, 50s)
@btime sim!(model, 10s)
#   735.852 ms (799501 allocations: 15.86 MiB) on DellTower
#   238.624 ms (1119507 allocations: 20.75 MiB) on DellTower
#

