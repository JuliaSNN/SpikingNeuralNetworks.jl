@snn_kw struct IFParameter{FT = Float32} <: AbstractIFParameter
    τm::FT = 20ms
    Vt::FT = -50mV # Membrane threshold potential
    Vr::FT = -60mV # Membrane reset potential
    El::FT = -70mV    # Membrane leak potential
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    τre::FT = 1ms # Rise time for excitatory synapses
    τde::FT = 6ms # Decay time for excitatory synapses
    τri::FT = 0.5ms # Rise time for inhibitory synapses
    τdi::FT = 2ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential
    E_e::FT = 0mV # Reversal potential
    τabs::FT = 2ms # Absolute refractory period
    gsyn_e::FT = 1.0f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses
    a::FT = 0.0 # Subthreshold adaptation parameter
    b::FT = 0.0 #80.5pA # 'sra' current increment
    τw::FT = 0.0 #144ms # adaptation time constant (~Ca-activated K current inactivation)
end



function IFParameterGsyn(;
    gsyn_i = 1.0,
    gsyn_e = 1.0,
    τde = 6ms,
    τre = 1ms,
    τdi = 2ms,
    τri = 0.5ms,
    kwargs...,
)
    gsyn_e *= norm_synapse(τre, τde)
    gsyn_i *= norm_synapse(τri, τdi)
    return IFParameter(
        τre = τre,
        τde = τde,
        τri = τri,
        τdi = τdi,
        gsyn_e = Float32(gsyn_e),
        gsyn_i = Float32(gsyn_i),
        ;
        kwargs...,
    )
end

@snn_kw struct IFParameterSingleExponential{FT = Float32} <: AbstractIFParameter
    τm::FT = 20ms
    Vt::FT = -50mV
    Vr::FT = -60mV
    El::FT = Vr
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    τe::FT = 6ms # Rise time for excitatory synapses
    τi::FT = 2ms # Rise time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential
    E_e::FT = 0mV # Reversal potential
    τabs::FT = 2ms # Absolute refractory period
    gsyn_e::FT = 1.0 # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0 # Synaptic conductance for inhibitory synapses
end


@snn_kw mutable struct IF{
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    IFT<:AbstractIFParameter,
} <: AbstractGeneralizedIF
    id::String = randstring(12)
    name::String = "IF"
    param::IFT = IFParameter()
    N::Int32 = 100
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    ge::VFT = zeros(N)
    gi::VFT = zeros(N)
    he::VFT = zeros(N)
    hi::VFT = zeros(N)
    tabs::VFT = zeros(N)
    w::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
    Δv::VFT = zeros(Float32, N)
    Δv_temp::VFT = zeros(Float32, N)
end


"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""
IF

function integrate!(p::IF, param::T, dt::Float32) where {T<:AbstractIFParameter}
    update_synapses!(p, param, dt)
    # Heun_update_neuron!(p, param, dt)
    update_neuron!(p, param, dt)
    update_spike!(p, param, dt)
end

function update_spike!(p::IF, param::T, dt::Float32) where {T<:AbstractIFParameter}
    @unpack N, v, w, tabs, fire = p
    @unpack Vt, Vr, τabs = param
    @inbounds for i = 1:N
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
        # Absolute refractory period
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i])
    end
    # Adaptation current
    # if the adaptation timescale is zero, return
    !(hasfield(typeof(param), :τw) && param.τw > 0.0f0) && (return)
    @unpack b = param
    @inbounds for i = 1:N
        w[i] = ifelse(fire[i], w[i] + b, w[i])
    end
end

function update_neuron!(p::IF, param::T, dt::Float32) where {T<:AbstractIFParameter}
    @unpack N, v, ge, gi, w, I, tabs, fire = p
    @unpack τm, El, R, E_i, E_e, τabs, gsyn_e, gsyn_i = param
    @inbounds for i = 1:N
        if tabs[i] > 0
            fire[i] = false
            tabs[i] -= 1
            continue
        end
        # Membrane potential
        v[i] +=
            dt/τm *
            ( -(v[i] - El) # leakage
                +R *(
                -ge[i] * (v[i] - E_e) * gsyn_e +
                -gi[i] * (v[i] - E_i) * gsyn_i +
                -w[i] # adaptation
                +I[i] #synaptic term
            ))
    end
    # Adaptation current
    # if the adaptation timescale is zero, return
    !(hasfield(typeof(param), :τw) && param.τw > 0.0f0) && (return)
    @unpack a, b, τw = param
    @inbounds for i = 1:N
        (w[i] += dt * (a * (v[i] - El) - w[i]) / τw)
    end

end

function Heun_update_neuron!(p::IF, param::T, dt::Float32) where {T<:AbstractIFParameter}
    function _update_neuron!(
        Δv::Vector{Float32},
        p::IF,
        param::T,
        dt::Float32,
    ) where {T<:AbstractIFParameter}
        @unpack N, v, ge, gi, w, I, tabs, fire = p
        @unpack τm, Vr, El, R, E_i, E_e, τabs, gsyn_e, gsyn_i = param
        @inbounds for i = 1:N
            if tabs[i] > 0
                v[i] = Vr
                fire[i] = false
                tabs[i] -= 1
                continue
            end
            Δv[i] =
                (
                    -(v[i] + Δv[i] * dt - El) / R +# leakage
                    -ge[i] * (v[i] + Δv[i] * dt - E_e) * gsyn_e +
                    -gi[i] * (v[i] + Δv[i] * dt - E_i) * gsyn_i +
                    -w[i] # adaptation
                    +
                    I[i] #synaptic term
                ) * R / τm
        end
    end
    @unpack Δv_temp, Δv = p
    _update_neuron!(Δv, p, param, dt)
    @turbo for i = 1:p.N
        Δv_temp[i] = Δv[i]
    end
    _update_neuron!(Δv, p, param, dt)
    @turbo for i = 1:p.N
        p.v[i] += 0.5f0 * (Δv_temp[i] + Δv[i]) * dt
    end
    !(hasfield(typeof(param), :τw) && param.τw > 0.0f0) && (return)
    @unpack a, b, τw = param
    @inbounds for i = 1:N
        (w[i] += dt * (a * (v[i] - El) - w[i]) / τw)
    end
end


function update_synapses!(p::IF, param::IFParameter, dt::Float32)
    @unpack N, ge, gi, he, hi = p
    @unpack τde, τre, τdi, τri = param
    @inbounds for i = 1:N
        ge[i] += dt * (-ge[i] / τde + he[i])
        he[i] -= dt * he[i] / τre
        gi[i] += dt * (-gi[i] / τdi + hi[i])
        hi[i] -= dt * hi[i] / τri
    end
end

function update_synapses!(p::IF, param::IFParameterSingleExponential, dt::Float32)
    @unpack N, ge, gi = p
    @unpack τe, τi = param
    @inbounds for i = 1:N
        ge[i] += dt * (-ge[i] / τe)
        gi[i] += dt * (-gi[i] / τi)

        ge[i] = clamp(ge[i], 0, 1000pA)
        gi[i] = clamp(gi[i], 0, 1000pA)
    end
end

export IF, IFParameter, IFParameterSingleExponential, IFParameterGsyn
