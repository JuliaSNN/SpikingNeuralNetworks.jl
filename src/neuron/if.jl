abstract type AbstractIFParameter end
@snn_kw struct IFParameter{FT=Float32} <: AbstractIFParameter
    τm::FT = 20ms
    τe::FT = 5ms
    τi::FT = 10ms
    Vt::FT = -50mV
    Vr::FT = -60mV
    El::FT = Vr
end

abstract type AbstractIF end

@snn_kw mutable struct IF{VFT=Vector{Float32},VBT=Vector{Bool}} <: AbstractIF
    param::IFParameter = IFParameter()
    N::Int32 = 100
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    ge::VFT = zeros(N)
    gi::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
end

"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""
IF

function integrate!(p::IF, param::IFParameter, dt::Float32)
    @unpack N, v, ge, gi, fire, I = p
    @unpack τm, τe, τi, Vt, Vr, El = param
    @inbounds for i = 1:N
        v[i] += dt * (ge[i] + gi[i] - (v[i] - El) + I[i]) / τm
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
    end
    @inbounds for i = 1:N
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
    end
end
