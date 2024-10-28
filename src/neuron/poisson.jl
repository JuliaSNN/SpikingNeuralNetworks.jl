@snn_kw struct PoissonParameter{FT = Float32}
    rate::FT = 1Hz
end

@snn_kw mutable struct Poisson{VFT = Vector{Float32},VBT = Vector{Bool},IT = Int32} <:AbstractPopulation
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

export Poisson, PoissonParameter
