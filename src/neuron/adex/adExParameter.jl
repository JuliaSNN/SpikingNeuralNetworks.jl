C = 281pF        #(pF)
gL = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS

@snn_kw mutable struct AdExParameter{FT = Float32} <: AbstractAdExParameter
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
    τre::FT = 1ms # Rise time for excitatory synapses
    τde::FT = 6ms # Decay time for excitatory synapses
    τri::FT = 0.5ms # Rise time for inhibitory synapses
    τdi::FT = 2ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential excitatory synapses 
    E_e::FT = 0mV #Reversal potential excitatory synapses
    gsyn_e::FT = 1.0f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses

    ## Dynamic spike threshold
    At::FT = 10mV # Post spike threshold increase
    τt::FT = 30ms # Adaptive threshold time scale
end

function AdExParameterGsyn(;
    gsyn_i = 1.0,
    gsyn_e = 1.0,
    τde = 6ms,
    τre = 1ms,
    τdi = 2ms,
    τri = 0.5ms,
    kwargs...,
)
    gsyn_e *= norm_synapse(τre, τde)
    gsyn_i *= norm_synapse(τri, τdi)
    return AdExParameter(gsyn_e = Float32(gsyn_e), gsyn_i = Float32(gsyn_i), ; kwargs...)
end

@snn_kw struct AdExParameterSingleExponential{FT = Float32} <: AbstractAdExParameter
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
    τe::FT = 6ms # Decay time for excitatory synapses
    τi::FT = 0.5ms # Rise time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential excitatory synapses 
    E_e::FT = 0mV #Reversal potential excitatory synapses
    gsyn_e::FT = 1.0f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses

    ## Dynamic spike threshold
    At::FT = 10mV # Post spike threshold increase
    τt::FT = 30ms # Adaptive threshold time scale
end

@snn_kw struct AdExSynapseParameter{
    FT = Float32,
    VIT = Vector{Int},
    ST = SynapseArray,
    NMDAT = NMDAVoltageDependency{Float32},
    VFT = Vector{Float32},
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

    ## Dynamic spike threshold
    At::FT = 10mV # Post spike threshold increase
    τt::FT = 30ms # Adaptive threshold time scale

    ## Synapses
    NMDA::NMDAT = SomaNMDA
    exc_receptors::VIT = [1, 2]
    inh_receptors::VIT = [3, 4]
    α::VFT = [syn.α for syn in synapsearray(SomaSynapse)]
    syn::ST = synapsearray(SomaSynapse)
end

function AdExSynapseParam(synapse::Synapse; kwargs...)
    α = [syn.α for syn in synapsearray(synapse)]
    syn = synapsearray(synapse)
    return AdExSynapseParameter(α = α, syn = syn; kwargs...)
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

export AdExParameter,
    AdExParameterSingleExponential, AdExParameterGsyn, AdExSynapseParameter, PostSpike
