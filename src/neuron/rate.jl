struct RateParameter
end

@snn_kw mutable struct Rate{VFT=Vector{Float32}}
    param::RateParameter = RateParameter()
    N::Int32 = 100
    x::VFT = 0.5randn(N)
    r::VFT = tanh.(x)
    g::VFT = zeros(N)
    I::VFT = zeros(N)
    records::Dict = Dict()
end

"""
[Rate Neuron](https://neuronaldynamics.epfl.ch/online/Ch15.S3.html)
"""
Rate

function integrate!(p::Rate, param::RateParameter, dt::Float32)
    @unpack N, x, r, g, I = p
    @inbounds for i = 1:N
        x[i] += dt * (-x[i] + g[i] + I[i])
        r[i] = tanh(x[i]) #max(0, x[i])
    end
end
