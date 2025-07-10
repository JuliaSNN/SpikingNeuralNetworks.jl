
@snn_kw mutable struct AdExMultiTimescaleParameter{
    FT = Float32,
    VFT = Vector{Float32},
    VIT=Vector{Int},
} <: AbstractAdExParameter
    τm::FT = C / gL # Membrane time constant
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -70.6mV # Reset potential
    El::FT = -70.6mV # Resting membrane potential 
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    Vspike::FT = 20mV # Spike potential
    τw::FT = 144ms # Adaptation time constant (Spike-triggered adaptation time scale)
    a::FT = 4nS # Subthreshold adaptation parameter
    b::FT = 80.5pA # Spike-triggered adaptation parameter (amount by which the voltage is increased at each threshold crossing)
    τabs::FT = 1ms # Absolute refractory period

    ## Synapses
    τr::VFT = [1ms, 0.5ms] # Rise time for excitatory synapses
    τd::VFT = [6ms, 2ms] # Decay time for excitatory synapses
    exc_receptors::VIT = [1] # it indices which timescale corresponds to excitatory synapses
    inh_receptors::VIT = [2] # it indices which timescale corresponds to inhibitory synapses

    E_i::FT = -75mV # Reversal potential excitatory synapses 
    E_e::FT = 0mV #Reversal potential excitatory synapses

    gsyn_e::FT = 1.0f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses

    ## Dynamic spike threshold
    At::FT = 10mV # Post spike threshold increase
    τt::FT = 30ms # Adaptive threshold time scale
end





@snn_kw struct AdExMultiTimescale{
    VFT = Vector{Float32},
    MFT = Matrix{Float32},
    VIT = Vector{Int},
    VBT = Vector{Bool},
    VST = Vector{Vector{Float32}}, ## Synapses types 
    AdExT<:AbstractAdExParameter,
} <: AbstractAdEx
    name::String = "AdExMultiTimescale"
    id::String = randstring(12)
    param::AdExT = AdExMultiTimescaleParameter()
    N::Int32 = 100 # Number of neurons
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w::VFT = zeros(N) # Adaptation current
    fire::VBT = zeros(Bool, N) # Store spikes
    θ::VFT = ones(N) * param.Vt # Array with membrane potential thresholds
    ξ_het::VFT = ones(N) # Membrane time constant
    tabs::VIT = ones(N) # Membrane time constant
    I::VFT = zeros(N) # Current

    # synaptic conductance
    syn_curr::VFT = zeros(N)
    g::MFT # = zeros(N, 4)
    h::VST    #! target
    records::Dict = Dict()
end

function synaptic_target(targets::Dict, post::AdExMultiTimescale, sym::Symbol, target::Int)
    g = getfield(post, sym)[target]
    v_post = getfield(post, :v)
    push!(targets, :sym => Symbol(string(sym, target)))
    return g, v_post
end

function AdExMultiTimescale(N::Int; param::AdExMultiTimescaleParameter, kwargs...)
    @assert length(param.τr) == length(param.τd) "Excitatory synapse parameters must have the same length"
    @assert length(param.τr) == length(param.exc_receptors) + length(param.inh_receptors) "There must be the same number of timescale parameters as receptors"
    # Create a new AdExMultiTimescale neuron model with the given parameters.
    return AdExMultiTimescale(
        N = N,
        param = param,
        h = [zeros(N) for n = 1:length(param.τr)],
        g = zeros(Float32, N, length(param.τr));
        kwargs...,
    )
end

# """
# 	[Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
# """
# function integrate!(
#     p::P,
#     param::T,
#     dt::Float32,
# ) where {T<:AbstractAdExParameter,P<:AbstractAdEx}
#     update_synapses!(p, param, dt)
#     synaptic_current!(p, param)
#     update_soma!(p, param, dt)
# end


function update_synapses!(
    p::AdExMultiTimescale,
    param::AdExMultiTimescaleParameter,
    dt::Float32,
)
    @unpack N, g, h = p
    @unpack τr, τd = param

    # Update the conductance from the input spikes (he, hi)
    for n in eachindex(τr)
        τr⁻ = 1/τr[n]
        τd⁻ = 1/τd[n]
        @fastmath @turbo for i ∈ 1:N
            g[i, n] = exp64(-dt * τd⁻) * (g[i, n] + dt * h[n][i])
            h[n][i] = exp64(-dt * τr⁻) * h[n][i]
        end
    end

end

@inline function synaptic_current!(
    p::AdExMultiTimescale,
    param::AdExMultiTimescaleParameter,
)
    @unpack N, g, h, g, v, syn_curr = p
    @unpack τr, τd = param

    fill!(syn_curr, 0.0f0)
    @inbounds for n in eachindex(τr)
        if n ∈ param.exc_receptors
            E_rev = param.E_e
            gsyn = param.gsyn_e
        else
            E_rev = param.E_i
            gsyn = param.gsyn_i
        end
        @simd for i ∈ 1:N
            syn_curr[i] += gsyn * g[i, n] * (v[i] - E_rev)
        end
    end
    return
end

export AdExMultiTimescale, AdExMultiTimescaleParameter, AdExParameterMultiTimescale
