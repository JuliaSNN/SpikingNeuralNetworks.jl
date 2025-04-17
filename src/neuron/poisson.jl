@snn_kw struct PoissonParameter{FT = Float32} <: AbstractPopulationParameter
    rate::FT = 1Hz
end

@snn_kw mutable struct Poisson{VFT = Vector{Float32},VBT = Vector{Bool},IT = Int32} <:
                       AbstractPopulation
    id::String = randstring(12)
    name::String = "Poisson"
    param::PoissonParameter = PoissonParameter()
    N::IT = 100
    rate::VFT = fill(param.rate, N)
    # rt::VFT=[-1f0] ## add a variable rate for the population
    randcache::VFT = rand(N)
    fire::VBT = zeros(Bool, N)
    records::Dict = Dict()
end

"""
[Poisson Neuron](https://www.cns.nyu.edu/~david/handouts/poisson.pdf)
"""
Poisson

function integrate!(p::Poisson, param::PoissonParameter, dt::Float32)
    @unpack N, randcache, fire, rate = p
    rand!(randcache)
    @inbounds for i = 1:N
        fire[i] = randcache[i] < rate[i] * dt
    end
end

@snn_kw struct VariablePoissonParameter{VFT}
    β::Float32 = 0.0
    τ::Float32 = 50.0ms
    r0::Float32 = 1kHz
end


@snn_kw struct VariablePoisson{
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    VIT = Vector{Int},
    IT = Int32,
} <: AbstractPopulation
    id::String = randstring(12)
    param::VariablePoissonParameter
    name::String = "VariablePoisson"
    N::IT = 100
    fire::VBT = zeros(Bool, N)
    noise::VFT = zeros(Float32, 1)
    r::VFT = ones(Float32, 1) * param.r0
    ##
    randcache::VFT = rand(N) # random cache
    records::Dict = Dict()
end


function integrate!(p::VariablePoisson, param::VariablePoissonParameter, dt::Float32)
    @unpack N, randcache, fire = p

    ## Inhomogeneous Poisson process
    @unpack r0, β, τ = param
    @unpack noise, r = p
    # Irate::Float32 = r0 * kIE
    R(x::Float32, v0::Float32 = 0.0f0) = x > 0.0f0 ? x : v0

    # Excitatory spike
    re::Float32 = 0.0f0
    cc::Float32 = 0.0f0
    Erate::Float32 = 0.0f0
    rand!(randcache)
    re = rand() - 0.5f0
    cc = 1.0f0 - dt / τ
    i = 1
    noise[i] = (noise[i] - re) * cc + re
    Erate = R(r0 ./ 2 * R(noise[i] * β, 1.0f0) + r[i], 0.0f0)
    r[i] += (r0 - Erate) / 400ms * dt
    @assert Erate >= 0
    # @inbounds @fastmath 
    for j = 1:N # loop on presynaptic neurons
        if randcache[j] < Erate / N * dt
            fire[j] = true
        else
            fire[j] = false
        end
    end
end


export Poisson,
    PoissonParameter, integrate!, VariablePoisson, VariablePoissonParameter, stimulate!
