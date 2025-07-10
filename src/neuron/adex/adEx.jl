
abstract type AbstractAdEx <: AbstractGeneralizedIF end

@snn_kw struct AdEx{
    VFT = Vector{Float32},
    VIT = Vector{Int},
    VBT = Vector{Bool},
    AdExT<:AbstractAdExParameter,
} <: AbstractAdEx
    name::String = "AdEx"
    id::String = randstring(12)
    param::AdExT = AdExParameter()
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
    ge::VFT = zeros(N) #
    gi::VFT = zeros(N) # Time-dependent conductance
    he::VFT = zeros(N)
    hi::VFT = zeros(N)
    records::Dict = Dict()
end


@snn_kw struct AdExSynapse{
    VFT = Vector{Float32},
    MFT = Matrix{Float32},
    VIT = Vector{Int},
    VBT = Vector{Bool},
    AdExT<:AbstractAdExParameter,
} <: AbstractAdEx
    name::String = "AdExSynapse"
    id::String = randstring(12)
    param::AdExT = AdExSynapseParameter()
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
    g::MFT = zeros(N, 4)
    h::MFT = zeros(N, 4)
    he::VFT = zeros(N) #! target
    hi::VFT = zeros(N) #! target
    records::Dict = Dict()
end


function synaptic_target(
    targets::Dict,
    post::T,
    sym::Symbol,
    target::Nothing = nothing,
) where {T<:Union{AdEx,AdExSynapse}}
    g = getfield(post, sym)
    v_post = getfield(post, :v)
    push!(targets, :sym => sym)
    return g, v_post
end

AdExSimple = AdEx



"""
	[Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""

function integrate!(
    p::P,
    param::T,
    dt::Float32,
) where {T<:AbstractAdExParameter,P<:AbstractAdEx}
    update_synapses!(p, param, dt)
    synaptic_current!(p, param)
    update_soma!(p, param, dt)

end

function update_synapses!(p::AdExSimple, param::AdExParameter, dt::Float32)
    @unpack N, ge, gi, he, hi = p
    @unpack τde, τre, τdi, τri = param
    @inbounds for i ∈ 1:N
        ge[i] += dt * (-ge[i] / τde + he[i])
        he[i] += dt * (-he[i] / τre)
        gi[i] += dt * (-gi[i] / τdi + hi[i])
        hi[i] += dt * (-hi[i] / τri)
    end
end

function update_synapses!(p::AdExSimple, param::AdExParameterSingleExponential, dt::Float32)
    @unpack N, ge, gi, he, hi = p
    @unpack τe, τi = param
    @fastmath @inbounds for i ∈ 1:N
        ge[i] += dt * (-ge[i] / τe)
        gi[i] += dt * (-gi[i] / τi)
    end
end

function update_synapses!(p::AdExSynapse, param::AdExSynapseParameter, dt::Float32)
    @unpack N, g, h, hi, he = p
    @unpack exc_receptors, inh_receptors, α, syn = param

    # Update the rise_conductance from the input spikes (he, hi)
    @inbounds for n in exc_receptors
        @turbo for i ∈ 1:N
            h[i, n] += he[i] * α[n]
        end
    end
    @inbounds for n in inh_receptors
        @turbo for i ∈ 1:N
            h[i, n] += hi[i] * α[n]
        end
    end
    fill!(hi, 0.0f0)
    fill!(he, 0.0f0)

    for n in eachindex(syn)
        @unpack τr⁻, τd⁻ = syn[n]
        @fastmath @turbo for i ∈ 1:N
            g[i, n] = exp64(-dt * τd⁻) * (g[i, n] + dt * h[i, n])
            h[i, n] = exp64(-dt * τr⁻) * (h[i, n])
        end
    end
end

function update_soma!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:AbstractAdEx,T<:AbstractAdExParameter}
    @unpack N, v, w, fire, θ, I, ξ_het, tabs, syn_curr = p
    @unpack τm, Vt, Vr, El, R, ΔT, τw, a, b, At, τt, τabs = param

    # syn_curr = hasfield(typeof(p), :syn_curr) ? p.syn_curr : zeros(N)
    # if hasfield(typeof(p), :syn_curr)
    #     synaptic_current!(p, param)
    # else
    #     synaptic_current!(p, param, syn_curr)
    # end


    @inbounds for i ∈ 1:N
        # Reset membrane potential after spike
        v[i] = ifelse(fire[i], Vr, v[i])

        # Absolute refractory period
        if tabs[i] > 0
            fire[i] = false
            tabs[i] -= 1
            continue
        end

        # Adaptation current 
        w[i] += dt * (a * (v[i] - El) - w[i]) / τw
        # Membrane potential
        v[i] +=
            dt * (
                -(v[i] - El)  # leakage
                + (ΔT < 0.0f0 ? 0.0f0 : ΔT * exp((v[i] - θ[i]) / ΔT)) # exponential term
                - R * syn_curr[i] # excitatory synapses
                - R * w[i] # adaptation
                + R * I[i] # external current
            ) / (τm * ξ_het[i])
        # Double exponential
        θ[i] += dt * (Vt - θ[i]) / τt

        # Spike
        fire[i] = v[i] >= param.Vspike
        # fire[i] = v[i] > θ[i] + 5.0f0
        v[i] = ifelse(fire[i], 20.0f0, v[i]) # Set membrane potential to spike potential

        # Spike-triggered adaptation
        w[i] = ifelse(fire[i], w[i] + b, w[i])
        θ[i] = ifelse(fire[i], θ[i] + At, θ[i])
        # Absolute refractory period
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i])
        # increase adaptation current
    end
end

@inline function synaptic_current!(p::AdExSynapse, param::AdExSynapseParameter)
    @unpack N, g, h, g, v, syn_curr = p
    @unpack syn, NMDA = param
    @unpack mg, b, k = NMDA
    fill!(syn_curr, 0.0f0)
    @inbounds for r in eachindex(syn)
        @unpack gsyn, E_rev, nmda = syn[r]
        if nmda > 0.0f0
            @simd for i ∈ 1:N
                syn_curr[i] +=
                    gsyn * g[i, r] * (v[i] - E_rev) / (1.0f0 + (mg / b) * exp64(k * (v[i])))
            end
        else
            @simd for i ∈ 1:N
                syn_curr[i] += gsyn * g[i, r] * (v[i] - E_rev)
            end
        end
    end
    return
end

@inline function synaptic_current!(p::AdExSimple, param::T) where {T<:AbstractAdExParameter}
    @unpack gsyn_e, gsyn_i, E_e, E_i = param
    @unpack N, v, ge, gi, syn_curr = p
    @inbounds @simd for i ∈ 1:N
        syn_curr[i] = ge[i] * (v[i] - E_e) * gsyn_e + gi[i] * (v[i] - E_i) * gsyn_i
    end
end

@inline @fastmath function ΔwAdEx(
    v::Float32,
    w::Float32,
    AdEx::T,
)::Float32 where {T<:AbstractAdExParameter}
    return (AdEx.a * (v - AdEx.Er) - w) / AdEx.τw
end


export AdEx,
    AdExParameter,
    AdExParameterSingleExponential,
    AdExParameterGsyn,
    AdExSynapse,
    AdExSynapseParameter,
    AdExNeuron
