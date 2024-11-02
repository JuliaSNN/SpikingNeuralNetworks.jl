C = 281pF        #(pF)
gL = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS

@snn_kw struct AdExParameter{FT = Float32} <: AbstractAdExParameter
    τm::FT = C / gL # Membrane time constant
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -70.6mV # Reset potential
    El::FT = -70.6mV # Resting membrane potential 
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    τw::FT = 144ms # Adaptation time constant (Spike-triggered adaptation time scale)
    a::FT = 4nS # Subthreshold adaptation parameter
    b::FT = 80.5pA # Spike-triggered adaptation parameter (amount by which the voltage is increased at each threshold crossing)
    τabs::FT = 1ms # Absolute refractory period

    ## Synapses
    τre::FT = 1ms # Rise time for excitatory synapses
    τde::FT = 6ms # Decay time for excitatory synapses
    τri::FT = 0.5ms # Rise time for inhibitory synapses
    τdi::FT = 2ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential excitatory synapses 
    E_e::FT = 0mV #Reversal potential excitatory synapses
    gsyn_e::FT = 1.f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses

    ## Dynamic spike threshold
    At::FT = 10mV # Post spike threshold increase
    τt::FT = 30ms # Adaptive threshold time scale
end

function AdExParameterGsyn(;gsyn_i=1., gsyn_e=1., τde=6ms, τre=1ms, τdi=2ms, τri=0.5ms, kwargs...)
    gsyn_e *= norm_synapse(τre, τde) 
    gsyn_i *= norm_synapse(τri, τdi)
    return AdExParameter(
        gsyn_e=Float32(gsyn_e), 
        gsyn_i=Float32(gsyn_i),
        ; kwargs...)
end

@snn_kw struct AdExParameterSingleExponential{FT = Float32} <: AbstractAdExParameter
    τm::FT = C / gL # Membrane time constant
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -70.6mV # Reset potential
    El::FT = -70.6mV # Resting membrane potential 
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    τw::FT = 144ms # Adaptation time constant (Spike-triggered adaptation time scale)
    a::FT = 4nS # Subthreshold adaptation parameter
    b::FT = 80.5pA # Spike-triggered adaptation parameter (amount by which the voltage is increased at each threshold crossing)
    τabs::FT = 1ms # Absolute refractory period

    ## Synapses
    τe::FT = 6ms # Decay time for excitatory synapses
    τi::FT = 0.5ms # Rise time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential excitatory synapses 
    E_e::FT = 0mV #Reversal potential excitatory synapses

    ## Dynamic spike threshold
    At::FT = 10mV # Post spike threshold increase
    τt::FT = 30ms # Adaptive threshold time scale
end

"""
	AdExSoma

An implementation of the Adaptive Exponential Integrate-and-Fire (AdEx) model, adapted for a Tripod neuron.

# Fields
- `C::FT = 281pF`: Membrane capacitance.
- `gl::FT = 40nS`: Leak conductance.
- `R::FT = nS / gl * GΩ`: Total membrane resistance.
- `τm::FT = C / gl`: Membrane time constant.
- `Er::FT = -70.6mV`: Resting potential.
- `Vr::FT = -55.6mV`: Reset potential.
- `Vt::FT = -50.4mV`: Rheobase threshold.
- `ΔT::FT = 2mV`: Slope factor.
- `τw::FT = 144ms`: Adaptation current time constant.
- `a::FT = 4nS`: Subthreshold adaptation conductance.
- `b::FT = 80.5pA`: Spike-triggered adaptation increment.
- `AP_membrane::FT = 10.0f0mV`: After-potential membrane parameter .
- `BAP::FT = 1.0f0mV`: Backpropagating action potential parameter.
- `up::IT = 1ms`, `τabs::IT = 2ms`: Parameters related to spikes.

The types `FT` and `IT` represent Float32 and Int64 respectively.
"""
AdExSoma

@snn_kw struct AdExSoma{FT = Float32,IT = Int64} <: AbstractAdExParameter
    #Membrane parameters
    C::FT = 281pF           # (pF) membrane timescale
    gl::FT = 40nS                # (nS) gl is the leaking conductance,opposite of Rm
    R::FT = nS / gl * GΩ               # (GΩ) total membrane resistance
    τm::FT = C / gl                # (ms) C / gl
    Er::FT = -70.6mV          # (mV) resting potential
    # AdEx model
    Vr::FT = -55.6mV     # (mV) Reset potential of membrane
    Vt::FT = -50.4mV          # (mv) Rheobase threshold
    ΔT::FT = 2mV            # (mV) Threshold sharpness
    # Adaptation parameters
    τw::FT = 144ms          #ms adaptation current relaxing time
    a::FT = 4nS            #nS adaptation current to membrane
    b::FT = 80.5pA         #pA adaptation current increase due to spike
    # After spike timescales and membrane
    AP_membrane::FT = 10.0f0mV
    BAP::FT = 1.0f0mV
    up::IT = 1ms
    τabs::IT = 2ms
end

"""
	PostSpike

A structure defining the parameters of a post-synaptic spike event.

# Fields
- `A::FT`: Amplitude of the Post-Synaptic Potential (PSP).
- `τA::FT`: Time constant of the PSP.

The type `FT` represents Float32.
"""
PostSpike

@snn_kw struct PostSpike{FT<:Float32}
    A::FT
    τA::FT
end


@snn_kw struct AdEx{
    VFT = Vector{Float32},
    VIT = Vector{Int},
    VBT = Vector{Bool},
    AdExT<:AbstractAdExParameter,
} <: AbstractGeneralizedIF
    name::String = "AdEx"
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
    ge::VFT = zeros(N) # Time-dependent conductivity that opens whenever a presynaptic excitatory spike arrives
    gi::VFT = zeros(N) # Time-dependent conductivity that opens whenever a presynaptic inhibitory spike arrives
    he::VFT = zeros(N)
    hi::VFT = zeros(N)
    records::Dict = Dict()
end

"""
	[Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""

function integrate!(p::AdEx, param::T, dt::Float32) where {T<:AbstractAdExParameter}
    update_synapses!(p, param, dt)
    update_soma!(p, param, dt)

end

function update_synapses!(p::AdEx, param::AdExParameter, dt::Float32)
    @unpack N, ge, gi, he, hi = p
    @unpack τde, τre, τdi, τri = param
    @inbounds for i ∈ 1:N
        ge[i] += dt * (- ge[i] / τde + he[i])
        he[i] += dt * (- he[i] / τre)
        gi[i] += dt * (- gi[i] / τdi + hi[i])
        hi[i] += dt * (- hi[i] / τri)
    end
end

function update_synapses!(p::AdEx, param::AdExParameterSingleExponential, dt::Float32)
    @unpack N, ge, gi, he, hi = p
    @unpack τe, τi = param
    @fastmath @inbounds for i ∈ 1:N
        ge[i] += dt * (-ge[i] / τe)
        gi[i] += dt * (-gi[i] / τi)
    end
end

function update_soma!(p::AdEx, param::T, dt::Float32) where {T<:AbstractAdExParameter}
    @unpack N, v, w, fire, θ, I, ge, gi, ξ_het, tabs = p
    @unpack τm, Vt, Vr, El, R, ΔT, τw, a, b, At, τt, E_e, E_i, τabs = param
    @unpack gsyn_e, gsyn_i = param
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
                +
                ΔT * exp((v[i] - θ[i]) / ΔT) # exponential term
                +
                R * ge[i] * (E_e - v[i]) * gsyn_e +
                R * gi[i] * (E_i - v[i]) * gsyn_i +
                - R * w[i] # adaptation
                + R * I[i] # external current
            ) / (τm * ξ_het[i])
        # Double exponential
        θ[i] += dt * (Vt - θ[i]) / τt


        # Spike
        fire[i] = v[i] > θ[i] + 5.0f0
        v[i] = ifelse(fire[i], 10.0f0, v[i]) # Set membrane potential to spike potential

        # Spike-triggered adaptation
        w[i] = ifelse(fire[i], w[i] + b, w[i])
        θ[i] = ifelse(fire[i], θ[i] + At, θ[i])
        # Absolute refractory period
        tabs[i] = ifelse(fire[i], round(Int, τabs/dt), tabs[i])
        # increase adaptation current
    end
end

@inline @fastmath function ΔwAdEx(v::Float32, w::Float32, AdEx::AdExSoma)::Float32
    return (AdEx.a * (v - AdEx.Er) - w) / AdEx.τw
end


export AdEx, AdExSoma, AdExParameter, AdExParameterSingleExponential, AdExParameterGsyn
