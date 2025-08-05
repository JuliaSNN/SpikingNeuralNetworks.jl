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
SomaSynapse = Synapse(
    AMPA = Receptor(E_rev = 0.0, 
                    τr = 0.26, 
                    τd = 2.0, 
                    g0 = 0.73),
    GABAa = Receptor(E_rev = -70.0, 
                     τr = 0.1, 
                     τd = 15.0, 
                     g0 = 0.38)
    # SomaSynapse has not NMDA and GABAb receptors, 
    # they are assigned to a NullReceptor and skipped at simulation time
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

# We then define the dendritic neuron model. The dendritic neuron holds has the soma and dendritic compartments parameters, and the synaptic properties for both compartments. 
dend_neuron = DendNeuronParameter(
    # adex parameters
    C = 281pF,
    gl = 40nS,
    Vr = -55.6,
    Er = -70.6,
    ΔT = 2,
    Vt = -50.4,
    a = 4,
    b = 80.5pA,
    τw = 144,
    up = 0.1ms,
    τabs = 0.1ms,

    # post-spike adaptation
    postspike = SNN.PostSpike(A= 10.0, τA= 30.0), 

    # synaptic properties
    soma_syn = SomaSynapse,
    dend_syn = DendSynapse,
    NMDA = NMDA,

    # dendrite
    ds = [160um],
    physiology = SNN.human_dend,
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

#
Plots.default(palette = :okabe_ito)
SNN.sim!(model, 3s)
p = SNN.vecplot(E, :v_d, sym_id=1, interval=1:2ms:get_time(model), neurons=1, label="Dendritic Compartment")
SNN.vecplot!(p, E, :v_s, sym_id=2, interval=1:2ms:get_time(model), neurons=1, add_spikes=true, label="Soma Compartment")
plot!(ylims=:auto, legend=:outertop, legendfontsize=12, xlabel="Time (s)", ylabel="Voltage (mV)", title="Ball and Stick Neuron Model")
##
p = SNN.vecplot(E, :g_d, sym_id=1, interval=1:2ms:get_time(model), neurons=1, label="AMPA", factor=DendSynapse.AMPA.gsyn)
SNN.vecplot!(p, E, :g_d, sym_id=2, interval=1:2ms:get_time(model), neurons=1, label="NMDA", factor=DendSynapse.NMDA.gsyn*0.3)
SNN.vecplot!(p, E, :g_d, sym_id=3, interval=1:2ms:get_time(model), neurons=1, label="GABAa", factor=DendSynapse.GABAa.gsyn)
SNN.vecplot!(p,  E, :g_d, sym_id=4, interval=1:2ms:get_time(model), neurons=1, label="GABAb", factor=DendSynapse.GABAb.gsyn)
plot!(legend=:outertop, ylims=:auto)