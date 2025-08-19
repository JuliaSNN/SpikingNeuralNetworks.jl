using SpikingNeuralNetworks
using Distributions
SNN.@load_units
using ProtoStructs

# The macro @eval is used to define the new neuron model within the SNNModels module. It is equivalent to add a new file in the SNNModels.jl/src/populations directory. We strongly suggest this approach to avoid complications with the module system.
@eval SNN.SNNModels begin

    """
    Define the Poisson refractory stimulus parameters.
    Parameters are used at integration time to compute the equation update.
    All parameters are optional. 
    """
    PoissonRefractoryParameter

    @snn_kw struct PoissonRefractoryParameter <: PoissonStimulusParameter #{R} where {R<:Float32}
    # @proto struct PoissonRefractoryParameter{R = Float32} 
        ΔT::Float32 = 2f0ms  # Absolute refractory period
        N::Int = 100  # Number of neurons
        rate::Float32 = 10Hz
        last_spike::Vector{Float32} = zeros(Float32, N)  # Last spike time for each neuron
        rates::Vector{Float32} = fill(rate, N)  # Firing rate for each neuron
        p::Float32 = 0.1f0  # Fraction of neurons receiving the stimulus
        μ::Float32 = 1f0  # Mean of the weight distribution
        σ::Float32 = 0f0  # Standard deviation of the weight distribution
        active::Vector{Bool} = [true]  # Active neurons
    end

    """
    Generate a Poisson stimulus with an absolute refractory period for a postsynaptic population.
    """
    function stimulate!(
        p::PoissonStimulus,
        param::PoissonRefractoryParameter,
        time::Time,
        dt::Float32,
    )
        @unpack N, randcache, fire, neurons, colptr, W, I, g = p
        @unpack rates, ΔT, last_spike = param
        current_time = get_time(time)
        rand!(randcache)
        @inbounds @simd for j = 1:N
            if (current_time - last_spike[j]) > ΔT && randcache[j] < rates[j] * dt
                fire[j] = true
                last_spike[j] = current_time
                @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                    g[I[s]] += W[s]
                end
            else
                fire[j] = false
            end
        end
    end
    export PoissonRefractoryParameter, stimulate!
end

import SpikingNeuralNetworks: PoissonLayer, PoissonRefractoryParameter, compose, sim!, monitor!, vecplot
# validate_population_model(SNN.Neuron()) # This is only available in SNNModels v1.5.5

neuron_param = SNN.IdentityParam()
neuron = SNN.Identity(; param = neuron_param, N = 1, name = "Identity Neuron")

# Create a withe noise input current 
stim_param = PoissonRefractoryParameter(N=1, ΔT = 20ms, p=1)
stim = PoissonLayer(neuron, :g; param=stim_param)

monitor!(neuron, [:g, :fire], sr = 2kHz)
model = compose(; neuron, stim)
sim!(; model, duration = 100000ms, pbar=true)

vecplot(
    neuron,
    :g,
    neurons=1,
    add_spikes = true,
    ylabel = "Membrane potential (mV)",
    xlims = (0, 1000ms),
    # ylims = (-80, 10),
    c = :black,
)

st = SNN.spiketimes(neuron)[1]
diff(st) |> x-> SNNPlots.histogram(x, bins=100)