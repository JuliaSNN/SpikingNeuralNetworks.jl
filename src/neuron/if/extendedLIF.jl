@snn_kw struct ExtendedIFParameter{FT = Float32} <: AbstractIFParameter
    Cm::FT = 250pF
    Vt::FT = -40mV
    Vr::FT = -65mV
    El::FT = -70mV
    gl::FT = 10.0nS # ! THIS PARAMETER IS ARBITRARY
    τe::FT = 6ms # Decay time for excitatory synapses
    τi::FT = 20ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential
    E_e::FT = 0mV # Reversal potential
    τabs::FT = 5ms # Absolute refractory period
    α::FT = 0.0 # Dendritic interaction term
end

@snn_kw mutable struct ExtendedIF{
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    IFT<:AbstractIFParameter,
} <: AbstractGeneralizedIF
    id::String = randstring(12)
    name::String = "ExtendedIF"
    param::IFT = ExtendedIFParameter()
    N::Int32 = 100
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    g_Exc::VFT = zeros(N)
    g_PV::VFT = zeros(N)
    g_SST::VFT = zeros(N)
    tabs::VFT = zeros(N)
    w::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
    Δv::VFT = zeros(Float32, N)
    Δv_temp::VFT = zeros(Float32, N)
end

function integrate!(p::ExtendedIF, param::T, dt::Float32) where {T<:AbstractIFParameter}
    update_synapses!(p, param, dt)
    update_neuron!(p, param, dt)
    update_spike!(p, param, dt)
end

function update_spike!(p::ExtendedIF, param::T, dt::Float32) where {T<:AbstractIFParameter}
    @unpack N, v, w, tabs, fire = p
    @unpack Vt, Vr, τabs = param
    @inbounds for i = 1:N
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
        # Absolute refractory period
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i])
    end
end

function update_neuron!(p::ExtendedIF, param::T, dt::Float32) where {T<:AbstractIFParameter}
    @unpack N, v, g_Exc, g_PV, g_SST, w, I, tabs, fire = p
    @unpack El, E_i, E_e, τabs, gl, α = param
    @inbounds for i = 1:N
        if tabs[i] > 0
            fire[i] = false
            tabs[i] -= 1
            continue
        end
        # Membrane potential
        dv =
            (
                gl * (El - v[i]) +
                g_Exc[i] * (E_e - v[i]) +
                g_PV[i] * (E_i - v[i]) +
                g_SST[i] * (E_i - v[i]) +
                -α * g_Exc[i] * g_SST[i] * (E_e - v[i]) +
                + I[i] # synaptic term
                # 0
            ) / param.Cm
        v[i] += dt * dv
    end
    # # Adaptation current
    # if hasfield(typeof(param), :τw) && param.τw > 0.0f0
    #     @unpack a, b, τw = param
    #     @inbounds for i = 1:N
    #         w[i] += dt * (a * (v[i] - El) - w[i]) / τw
    #     end
    # end
end

function update_synapses!(p::ExtendedIF, param::ExtendedIFParameter, dt::Float32)
    @unpack N, g_Exc, g_PV, g_SST = p
    @unpack τe, τi = param
    @inbounds for i = 1:N
        g_Exc[i] += dt * (-g_Exc[i] / τe)
        g_PV[i] += dt * (-g_PV[i] / τi)
        g_SST[i] += dt * (-g_SST[i] / τi)
    end
end

export ExtendedIF, ExtendedIFParameter, update_neuron!, update_synapses!
