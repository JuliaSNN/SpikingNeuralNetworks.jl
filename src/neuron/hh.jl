using Statistics

using Plots

unicodeplots()
# @everywhere using PmapProgressmeter
@with_kw struct HHParameter
    Cm::SNNFloat = 1uF * cm^(-2) * 20000um^2
    gl::SNNFloat = 5e-5siemens * cm^(-2) * 20000um^2
    El::SNNFloat = -65mV
    Ek::SNNFloat = -90mV
    En::SNNFloat = 50mV
    gn::SNNFloat = 100msiemens * cm^(-2) * 20000um^2
    gk::SNNFloat = 30msiemens * cm^(-2) * 20000um^2
    Vt::SNNFloat = -63mV
    τe::SNNFloat = 5ms
    τi::SNNFloat = 10ms
    Ee::SNNFloat = 0mV
    Ei::SNNFloat = -80mV
end

@with_kw mutable struct HH
    param::HHParameter = HHParameter()
    N::SNNInt = 100
    v::Vector{SNNFloat} = param.El .+ 5(ones(N) .- 1)
    m::Vector{SNNFloat} = zeros(N)
    n::Vector{SNNFloat} = zeros(N)
    h::Vector{SNNFloat} = ones(N)
    ge::Vector{SNNFloat}  = zeros(N).*10nS # (1.5randn(N) .+ 4) .* 10nS
    gi::Vector{SNNFloat}  = zeros(N).*10nS # (12randn(N) .+ 20) .* 10nS
    fire::Vector{Bool} = zeros(Bool, N)
    I::Vector{SNNFloat} = zeros(N)
    records::Dict = Dict()
end

function integrate!(p::HH, param::HHParameter, dt::SNNFloat)
    @unpack N, v, m, n, h, ge, gi, fire, I = p
    @unpack Cm, gl, El, Ek, En, gn, gk, Vt, τe, τi, Ee, Ei = param
    @inbounds for i = 1:N
        m[i] += dt * (0.32f0 * (13f0 - v[i] + Vt) / (exp((13f0 - v[i] + Vt) / 4f0) - 1f0) * (1f0 - m[i]) -
        0.28f0 * (v[i] - Vt - 40f0) / (exp((v[i] - Vt - 40f0) / 5f0) - 1f0) * m[i])
        n[i] += dt * (0.032f0 * (15f0 - v[i] + Vt) / (exp((15f0 - v[i] + Vt) / 5f0) - 1f0) * (1f0 - n[i]) -
        0.5f0 * exp((10f0 - v[i] + Vt) / 40f0) * n[i])
        h[i] += dt * (0.128f0 * exp((17f0 - v[i] + Vt) / 18f0) * (1f0 - h[i]) -
        4f0 / (1f0 + exp((40f0 - v[i] + Vt) / 5f0)) * h[i])
        v[i] += dt / Cm * ( I[i] + gl * (El - v[i]) + ge[i] * (Ee - v[i]) + gi[i] * (Ei - v[i]) +
        gn * m[i]^3 * h[i] * (En - v[i]) + gk * n[i]^4 * (Ek - v[i]) )
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
    end
    @inbounds for i = 1:N
        fire[i] = v[i] > -20f0
    end
end
