@snn_kw struct IFParameter{FT = Float32} <: AbstractIFParameter
    τm::FT = 20ms
    Vt::FT = -50mV
    Vr::FT = -60mV
    El::FT = Vr
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    τre::FT = 1ms # Rise time for excitatory synapses
    τde::FT = 6ms # Decay time for excitatory synapses
    τri::FT = 0.5ms # Rise time for inhibitory synapses
    τdi::FT = 2ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential
    E_e::FT = 0mV # Reversal potential
    τabs::FT = 1ms # Absolute refractory period
    gsyn_e::FT = 1.f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses
    a::FT = 0.0 # Subthreshold adaptation parameter
    b::FT = 0.0 #80.5pA # 'sra' current increment
    τw::FT = 0.0 #144ms # adaptation time constant (~Ca-activated K current inactivation)
end

function IFParameterGsyn(;gsyn_i=1., gsyn_e=1., τde=6ms, τre=1ms, τdi=2ms, τri=0.5ms, kwargs...)
    gsyn_e *= norm_synapse(τre, τde) 
    gsyn_i *= norm_synapse(τri, τdi)
    return IFParameter(
        gsyn_e=Float32(gsyn_e), 
        gsyn_i=Float32(gsyn_i),
        ; kwargs...)
end

@snn_kw struct IFParameterSingleExponential{FT = Float32} <: AbstractIFParameter
    τm::FT = 20ms
    Vt::FT = -50mV
    Vr::FT = -60mV
    El::FT = Vr
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    τe::FT = 6ms # Rise time for excitatory synapses
    τi::FT = 2ms # Rise time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential
    E_e::FT = 0mV # Reversal potential
    τabs::FT = 1ms # Absolute refractory period
    gsyn_e::FT = 1.0 # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0 # Synaptic conductance for inhibitory synapses
end

@snn_kw mutable struct IF{
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    IFT<:AbstractIFParameter,
} <: AbstractGeneralizedIF
    id::String = randstring(12)
    name::String = "IF"
    param::IFT = IFParameter()
    N::Int32 = 100
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    ge::VFT = zeros(N)
    gi::VFT = zeros(N)
    he::VFT = zeros(N)
    hi::VFT = zeros(N)
    tabs::VFT = zeros(N)
    w::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
end

"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""
IF

function integrate!(p::IF, param::T, dt::Float32) where {T<:AbstractIFParameter}
    update_synapses!(p, param, dt)
    update_neuron!(p, param, dt)
    update_spike!(p, param, dt)
end

function update_spike!(p::IF, param::IFParameter, dt::Float32)
    @unpack N, v, w, tabs, fire = p
    @unpack Vt, Vr, τabs, b = param
    @inbounds for i = 1:N
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
        # Adaptation current
        w[i] = ifelse(fire[i], w[i] + b, w[i])
        # Absolute refractory period
        tabs[i] = ifelse(fire[i], round(Int, τabs/dt), tabs[i])
    end
end

function update_neuron!(p::IF, param::IFParameter, dt::Float32)
    @unpack N, v, ge, gi, w, I, tabs, fire = p
    @unpack τm, El, R,  E_i, E_e, τabs, a, b, τw, gsyn_e, gsyn_i = param
    @inbounds for i = 1:N
        if tabs[i] > 0
            fire[i] = false
            tabs[i] -= 1
            continue
        end
        # Adaptation current 
        τw != 0.0 && (w[i] += dt * (a * (v[i] - El) - w[i]) / τw)

        # Membrane potential
        v[i] +=
            dt * (
                -(v[i] - El)  # leakage
                +
                R * ge[i] * (E_e - v[i])* gsyn_e +
                R * gi[i] * (E_i - v[i])* gsyn_i +
                - R * w[i] # adaptation
                + R * I[i] #synaptic term
            ) / τm
        # @show v[i]
    end
end

function update_synapses!(p::IF, param::IFParameter, dt::Float32)
    @unpack N, ge, gi, he, hi = p
    @unpack τde, τre, τdi, τri = param
    @inbounds for i = 1:N
        ge[i] += dt * (-ge[i] / τde + he[i])
        he[i] -= dt * he[i] / τre
        gi[i] += dt * (-gi[i] / τdi + hi[i])
        hi[i] -= dt * hi[i] / τri
    end
end

function update_synapses!(p::IF, param::IFParameterSingleExponential, dt::Float32)
    @unpack N, ge, gi = p
    @unpack τe, τi = param
    @inbounds for i = 1:N
        ge[i] += dt * (-ge[i] / τe)
        gi[i] += dt * (-gi[i] / τi)
    end
end

export IF, IFParameter, IFParameterSingleExponential, IFParameterGsyn
