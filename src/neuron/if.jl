abstract type AbstractGeneralizedIFParameter <: AbstractPopulationParameter end
abstract type AbstractGeneralizedIF <: AbstractPopulation end
abstract type AbstractIFParameter <: AbstractGeneralizedIFParameter end

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
end

@snn_kw mutable struct IF{
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    IFT<:AbstractIFParameter,
} <: AbstractGeneralizedIF
    param::IFT = IFParameter()
    N::Int32 = 100
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    ge::VFT = zeros(N)
    gi::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
    he::Vector{Float64} = zeros(N)
    hi::Vector{Float64} = zeros(N)
    timespikes::Vector{Float64} = zeros(N)
end

"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""
IF

function integrate!(p::IF, param::T, dt::Float32) where {T<:AbstractIFParameter}
    @unpack N, v, ge, gi, fire, I, records, he, hi, timespikes = p
    @unpack τm, Vt, Vr, El, R, ΔT, E_i, E_e, τabs = param
    @inbounds for i = 1:N
        v[i] +=
            dt * (
                -(v[i] - El)  # leakage
                +
                R * ge[i] * (E_e - v[i]) +
                R * gi[i] * (E_i - v[i]) +
                I[i] * R #synaptic term
            ) / τm

    end
    update_synapses!(p, param, dt)

    @inbounds for i = 1:N
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
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
