using SpikingNeuralNetworks
using Plots
using Random
SNN.@load_units
import SpikingNeuralNetworks: Synapse, Receptor, Glutamatergic, GABAergic, DendNeuronParameter, synapsearray
import SpikingNeuralNetworks: get_time

using BenchmarkTools

Random.seed!(1234)
## Define the neuron model parameters
# Define the synaptic properties for the soma and dendrites
# SomaSynapse has not NMDA and GABAb receptors, they will be assigned to a NullReceptor and skipped at simulation time
SomaSynapse = Synapse(
    AMPA = Receptor(E_rev = 0.0, 
                    τr = 0.26, 
                    τd = 2.0, 
                    g0 = 0.73),
    GABAa = Receptor(E_rev = -70.0, 
                     τr = 0.1, 
                     τd = 15.0, 
                     g0 = 0.38)
)

DendSynapse = Synapse(
    AMPA = Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
    NMDA = Receptor(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.0f0),
    GABAa = Receptor(E_rev = -70.0, τr = 4.8, τd = 29.0, g0 = 0.27),
    GABAb = Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.0006), 
)

NMDA = let
    Mg_mM = 1.0mM
    nmda_b = 3.36   # voltage dependence of nmda channels
    nmda_k = -0.077     # Eyal 2018
    SNN.NMDAVoltageDependency(mg = Mg_mM/mM, b = nmda_b, k = nmda_k)
end

dend_neuron = DendNeuronParameter(
    ds = [160um],
    soma_syn = SomaSynapse,
    dend_syn = DendSynapse,
    NMDA = NMDA,
    physiology = SNN.human_dend
)

E = SNN.SNNModels.BallAndStick(N=1, param = dend_neuron)
poisson_exc = SNN.PoissonStimulusLayer(
    10.2Hz,    # Mean firing rate (Hz) 
    p = 1f0,  # Probability of connecting to a neuron
    μ = 1.0,  # Synaptic strength (nS)
    N = 1000, # Neurons in the Poisson Layer
)

poisson_inh = SNN.PoissonStimulusLayer(
    3Hz,       # Mean firing rate (Hz)
    p = 1f0,   # Probability of connecting to a neuron
    μ = 4.0,   # Synaptic strength (nS)
    N = 1000,  # Neurons in the Poisson Layer
)

# Create the Poisson layers for excitatory and inhibitory inputs
stim_exc = SNN.PoissonLayer(E, :glu, :d, param=poisson_exc, name="noiseE")
stim_inh = SNN.PoissonLayer(E, :gaba, :d, param=poisson_inh, name="noiseI")

model = SNN.merge_models(;E, stim_exc, stim_inh)
SNN.monitor!(E, [:v_s, :v_d, :fire, :g_s, :g_d], sr=1000Hz)

@btime SNN.sim!(model, 10s)
##
SNN.sim!(model, 3s)
p = SNN.vecplot(E, :v_d, sym_id=1, interval=1:2ms:get_time(model), neurons=1)
SNN.vecplot!(p, E, :v_s, sym_id=2, interval=1:2ms:get_time(model), neurons=1, add_spikes=true)
plot!(ylims=:auto)
##