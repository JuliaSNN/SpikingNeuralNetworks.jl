@with_kw struct IZParameter
    a::SNNFloat = 0.01
    b::SNNFloat = 0.2
    c::SNNFloat = -65
    d::SNNFloat = 2

end

@with_kw mutable struct IZ
    param::IZParameter = IZParameter(a,b,c,d)
    N::SNNInt = 1
    v::Vector{SNNFloat} = fill(param.c, N)
    u::Vector{SNNFloat} = param.b * v
    fire::Vector{Bool} = zeros(Bool, N)
    I::Vector{SNNFloat} = zeros(N)
    records::Dict = Dict()
end

function integrate!(p::IZ, param::IZParameter, dt::SNNFloat)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d = param
    @inbounds for i = 1:N
        #v[i] += 0.5f0dt * (0.04f0v[i]^2 + 5f0v[i] + 140f0 - u[i] + I[i])
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
