using DrWatson
using Revise
using Plots
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils

# Define the network
function define_network(N = 800)
    # Number of neurons in the network
    N = N
    # Create dendrites for each neuron
    E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -60mV))
    # Define interneurons 
    I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, El = -50mV))
    # Define synaptic interactions between neurons and interneurons
    E_to_I = SNN.SpikingSynapse(E, I, :he, p = 0.2, μ = 3.0)
    E_to_E = SNN.SpikingSynapse(E, E, :he, p = 0.2, μ = 0.5)#, param = SNN.vSTDPParameter())
    I_to_I = SNN.SpikingSynapse(I, I, :hi, p = 0.2, μ = 1.0)
    I_to_E = SNN.SpikingSynapse(
        I,
        E,
        :hi,
        p = 0.2,
        μ = 10.0,
        param = SNN.iSTDPParameterRate(r = 4Hz),
    )
    norm = SNN.SynapseNormalization(E, [E_to_E], param = SNN.AdditiveNorm(τ = 10ms))

    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I)
    syn = dict2ntuple(@strdict I_to_E E_to_I E_to_E norm I_to_I)
    # Return the network as a tuple
    (pop = pop, syn = syn)
end


network1 = define_network(800)
network2 = define_network(800)
inter = (
    E1_to_I2 = SNN.SpikingSynapse(network1.pop.E, network2.pop.I, :he, p = 0.2, μ = 4.25),
    E2_to_I1 = SNN.SpikingSynapse(network2.pop.E, network1.pop.I, :he, p = 0.2, μ = 4.25),
)

noise = (
    A = SNN.PoissonStimulus(network1.pop.E, :he, param = 4.5kHz, neurons = :ALL, μ = 1.7f0),
    B = SNN.PoissonStimulus(network2.pop.E, :he, param = 4.5kHz, neurons = :ALL, μ = 1.7f0),
)

model = merge_models(noise, inter; n1 = network1, n2 = network2)

## @info "Initializing network"
SNN.monitor([model.pop...], [:fire, :v, :he, :ge])
train!(model = model, duration = 15000ms, pbar = true, dt = 0.125ms)
SNN.raster([model.pop...], [14s, 15s])

train!(model = model, duration = 1000ms, pbar = true, dt = 0.125ms)
SNN.vecplot(model.pop.n1_E, [:v, :he, :ge], r = 0.01s:0.001:1s, dt = 0.125ms)


##
using Statistics
rate1, intervals = SNN.firing_rate(no_noise.pop.network1_E, τ = 10ms)
rate2, intervals = SNN.firing_rate(no_noise.pop.network2_E, τ = 10ms)
r1 = mean(rate1)
r2 = mean(rate2)
cor(r1, r2)
plot(r1, label = "Network 1", xlabel = "Time (s)", ylabel = "Firing rate (Hz)")
plot!(r2, label = "Network 2", title = "Correlation: $(cor(r1, r2))")
plot!(xlims = (100, 500))
##
