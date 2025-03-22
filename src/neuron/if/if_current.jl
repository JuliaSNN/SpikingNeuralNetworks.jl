@snn_kw struct IFCurrentParameter{FT = Float32} <: AbstractIFParameter
    τm::FT = 20ms # Membrane time constant
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -60mV # Reset potential
    El::FT = -70mV # Resting membrane potential
    R::FT = nS / gL # 40nS Membrane conductance
    ΔT::FT = 2mV # Slope factor
    τabs::FT = 2ms # Absolute refractory period
    #synapses
    τe::FT = 6ms # Rise time for excitatory synapses
    τi::FT = 2ms # Rise time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential
    E_e::FT = 0mV # Reversal potential
end

@snn_kw struct IFCurrentDeltaParameter{FT = Float32} <: AbstractIFParameter
    τm::FT = 20ms # Membrane time constant
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -60mV # Reset potential
    El::FT = -70mV # Resting membrane potential
    R::FT = nS / gL # 40nS Membrane conductance
    ΔT::FT = 2mV # Slope factor
    τabs::FT = 2ms # Absolute refractory period
    #synapses
end



@snn_kw mutable struct IFCurrent{
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    VIT = Vector{Int32},
    IFT<:AbstractIFParameter,
} <: AbstractGeneralizedIF
    name::String = "IFCurrent"
    id::String = randstring(12)
    param::IFT = IFCurrentParameter()
    N::Int32 = 100
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    ge::VFT = zeros(N)
    gi::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
    tabs::VIT = ones(N) # refractory period
end

"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""
IF

function integrate!(p::IFCurrent, param::T, dt::Float32) where {T<:AbstractIFParameter}
    @unpack N, v, ge, gi, fire, I, records, tabs = p
    @unpack τm, Vt, Vr, El, R, ΔT, τabs = param
    @inbounds for i = 1:N
        # Absolute refractory period
        if tabs[i] > 0
            fire[i] = false
            tabs[i] -= 1
            continue
        end

        v[i] += dt * (
            -(v[i] - El)  # leakage
            + R * (ge[i] - gi[i] + I[i]) #synaptic term
        ) / τm

    end
    update_synapses!(p, param, dt)

    @inbounds for i = 1:N
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i])
    end
end

function update_synapses!(p::IFCurrent, param::IFCurrentParameter, dt::Float32)
    @unpack N, ge, gi = p
    @unpack τe, τi = param
    @inbounds for i = 1:N
        ge[i] += dt * (-ge[i] / τe)
        gi[i] += dt * (-gi[i] / τi)
    end
end

function update_synapses!(p::IFCurrent, param::IFCurrentDeltaParameter, dt::Float32)
    @unpack N, ge, gi = p
    @inbounds for i = 1:N
        ge[i] = 0.0f0
        gi[i] = 0.0f0
    end
end

export IFCurrent, IFCurrentParameter, IFCurrentDeltaParameter
