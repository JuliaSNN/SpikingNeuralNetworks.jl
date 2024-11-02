@snn_kw struct HHParameter{FT = Float32}
    Cm::FT = 1uF * cm^(-2) * 20000um^2
    gl::FT = 5e-5siemens * cm^(-2) * 20000um^2
    El::FT = -65mV
    Ek::FT = -90mV
    En::FT = 50mV
    gn::FT = 100msiemens * cm^(-2) * 20000um^2
    gk::FT = 30msiemens * cm^(-2) * 20000um^2
    Vt::FT = -63mV
    τe::FT = 5ms
    τi::FT = 10ms
    Ee::FT = 0mV
    Ei::FT = -80mV
end

@snn_kw mutable struct HH{VFT = Vector{Float32},VBT = Vector{Bool}} <: AbstractPopulation
    name::String = "HH"
    id::String = randstring(12)
    param::HHParameter = HHParameter()
    N::Int32 = 100
    v::VFT = param.El .+ 5(randn(N) .- 1)
    m::VFT = zeros(N)
    n::VFT = zeros(N)
    h::VFT = ones(N)
    ge::VFT = (1.5randn(N) .+ 4) .* 10nS
    gi::VFT = (12randn(N) .+ 20) .* 10nS
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
end

"""
[Hodgkin–Huxley Neuron](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model)
"""
HH

function integrate!(p::HH, param::HHParameter, dt::Float32)
    @unpack N, v, m, n, h, ge, gi, fire, I = p
    @unpack Cm, gl, El, Ek, En, gn, gk, Vt, τe, τi, Ee, Ei = param
    @inbounds for i = 1:N
        m[i] +=
            dt * (
                0.32f0 * (13.0f0 - v[i] + Vt) /
                (exp((13.0f0 - v[i] + Vt) / 4.0f0) - 1.0f0) * (1.0f0 - m[i]) -
                0.28f0 * (v[i] - Vt - 40.0f0) /
                (exp((v[i] - Vt - 40.0f0) / 5.0f0) - 1.0f0) * m[i]
            )
        n[i] +=
            dt * (
                0.032f0 * (15.0f0 - v[i] + Vt) /
                (exp((15.0f0 - v[i] + Vt) / 5.0f0) - 1.0f0) * (1.0f0 - n[i]) -
                0.5f0 * exp((10.0f0 - v[i] + Vt) / 40.0f0) * n[i]
            )
        h[i] +=
            dt * (
                0.128f0 * exp((17.0f0 - v[i] + Vt) / 18.0f0) * (1.0f0 - h[i]) -
                4.0f0 / (1.0f0 + exp((40.0f0 - v[i] + Vt) / 5.0f0)) * h[i]
            )
        v[i] +=
            dt / Cm * (
                I[i] +
                gl * (El - v[i]) +
                ge[i] * (Ee - v[i]) +
                gi[i] * (Ei - v[i]) +
                gn * m[i]^3 * h[i] * (En - v[i]) +
                gk * n[i]^4 * (Ek - v[i])
            )
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
    end
    @inbounds for i = 1:N
        fire[i] = v[i] > -20.0f0
    end
end

# function HH_spike_count(p::HH, dt = 0.01)
#     neurons = hcat(p.records[:fire]...)
#     spike_count = zeros(size(neurons, 1))
#     for (n, fires) in enumerate(eachrow(neurons))
#         r = length(findall(x -> fires[x] > 0 && fires[x+1] == 0, eachindex(fires[1:end-1])))
#         spike_count[n] = r / 1000
#     end
#     return spike_count
# end
