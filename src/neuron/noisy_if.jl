@snn_kw struct NoisyIFParameter{FT=Float32} <: AbstractIFParameter
    σ::FT = 0
end

@snn_kw mutable struct NoisyIF{VFT=Vector{Float32}} <: AbstractIF
    param::NoisyIFParameter = NoisyIFParameter()
    N::Int32 = 100
    randncache::VFT = randn(N)
end

"""
Noisy Integrate-And-Fire Neuron
"""
NoisyIF

function integrate!(p::NoisyIF, param::NoisyIFParameter, dt::Float32)
    @unpack N, randncache = p
    @unpack σ = param
    randn!(randncache)
    @inbounds for i = 1:N
        v[i] += dt * (ge[i] + gi[i] - (v[i] - El) + I[i] + σ / √dt * randncache[i]) / τm
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
    end
    @inbounds for i = 1:N
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
    end
end
