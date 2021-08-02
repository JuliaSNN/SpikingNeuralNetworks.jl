@snn_kw struct IZParameter{FT=Float32}
    a::FT = 0.01
    b::FT = 0.2
    c::FT = -65
    d::FT = 2
end

@snn_kw mutable struct IZ{VFT=Vector{Float32},VBT=Vector{Bool}}
    param::IZParameter = IZParameter()
    N::Int32 = 100
    v::VFT = fill(-65.0, N)
    u::VFT = param.b * v
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
end

"""
[Izhikevich Neuron](https://www.izhikevich.org/publications/spikes.htm)
"""
IZ

function integrate!(p::IZ, param::IZParameter, dt::Float32)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d = param
    @inbounds for i = 1:N
        v[i] += 0.5f0dt * (0.04f0v[i]^2 + 5f0v[i] + 140f0 - u[i] + I[i])
        v[i] += 0.5f0dt * (0.04f0v[i]^2 + 5f0v[i] + 140f0 - u[i] + I[i])
        u[i] += dt * (a * (b * v[i] - u[i]))
    end

    @inbounds for i = 1:N
        fire[i] = v[i] > 0f0#30f0
        v[i] = ifelse(fire[i], c, v[i])
        u[i] += ifelse(fire[i], d, 0f0)
    end

end
#=
@inbounds for m = 1:N
    vT = v[m]+ 0.5f0dt * (k*(v[m] - vr)*(v[m] - vt)-u[m] + Iext[m])/C;
    v[m+1] = vT + 0.5f0dt  * (k*(v[m] - vr)*(v[m] - vt)-u[m] + Iext[m])/C;
    u[m+1] = u[m] + dt * a*(b*(v[m]-vr)-u[m]);
    u[m+1] = u[m] + dt * a*(b*(v[m+1]-vr)-u[m]);
    if v[m+1]>= vPeak:# % a spike is fired!
        v[m] = vPeak;# % padding the spike amplitude
        v[m+1] = c;# % membrane voltage reset
        u[m+1] = u[m+1] + d;# % recovery variable update
    end
end
=#
