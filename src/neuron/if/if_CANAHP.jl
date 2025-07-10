

@snn_kw struct IF_CANAHPParameter{
    FT = Float32,
    VIT = Vector{Int},
    ST = SynapseArray,
    NMDAT = NMDAVoltageDependency{Float32},
    VFT = Vector{Float32},
} <: AbstractIFParameter
    ## Neuron parameters
    area::FT = 5e-4cm^2
    C::FT = 1.0uF / cm^2 # Membrane capacitance
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -65mV # Resting membrane potential 
    # Vmean::FT = -57.5mV
    τabs::FT = 3ms # Absolute refractory period

    ## Ionic currents
    gL::FT = 0.05mS/cm^2 # Membrane conductance
    VL::FT = -70mV # Leak potential

    ΔCa::FT = 0.2uM
    Ca0::FT = 0.1uM
    τCa::FT = 100ms

    VAHP::FT = -90mV
    gAHP::FT = 0mS/cm^2
    αAHP::FT = 0.125uM / ms
    βAHP::FT = 0.025uM / ms
    γAHP::FT = 1

    VCAN::FT = 30mV
    gCAN::FT = 0mS / cm^2
    βCAN::FT = 0.025/ms
    αCAN::FT = 0.03125uM / ms
    γCAN::FT = 1

    ## Synapses
    NMDA::NMDAT = NMDA_CANAHP
    exc_receptors::VIT = [1, 2]
    inh_receptors::VIT = [3, 4]
    α::VFT = αs_CANAHP
    syn::ST = synapsearray(Synapse_CANAHP)
end




@snn_kw mutable struct IF_CANAHP{
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    IFT<:AbstractIFParameter,
    VIT = Vector{Int},
    MFT = Matrix{Float32},
} <: AbstractGeneralizedIF
    # Records
    records::Dict = Dict()
    name::String = "IF_CANAHP"
    id::String = randstring(12)

    # Neuron parameters
    param::IF_CANAHPParameter = IF_CANAHPParameter()
    N::Int32 = 100
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    fire::VBT = zeros(Bool, N) # Store spikes
    tabs::VIT = ones(N) # Membrane time constant
    I::VFT = zeros(N) # Current
    # ξ_het::VFT = ones(N) # Membrane time constant

    # Ionic conductances
    pCAN::VFT = zeros(N) # Calcium current
    pAHP::VFT = zeros(N) # Calcium current
    Ca::VFT = zeros(N) # Calcium concentration

    # synaptic conductance
    syn_curr::VFT = zeros(N)
    g::MFT = zeros(N, 4)
    h::MFT = zeros(N, 4)
    he::VFT = zeros(N) #! target
    hi::VFT = zeros(N) #! target
end


"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""
IF

function integrate!(p::IF_CANAHP, param::IF_CANAHPParameter, dt::Float32)
    update_synapses!(p, param, dt)
    update_neuron!(p, param, dt)
    synaptic_current!(p, param)
    update_spike!(p, param, dt)
end

function update_spike!(p::IF_CANAHP, param::IF_CANAHPParameter, dt::Float32)
    @unpack N, v, tabs, fire, Ca = p
    @unpack Vt, Vr, τabs, ΔCa = param
    # @inbounds 
    for i = 1:N
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
        Ca[i] = Ca[i] + ΔCa # Increase in Calcium concentration
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i]) # Absolute refractory period
    end
    # # Adaptation current
    # # if the adaptation timescale is zero, return
    # !(hasfield(typeof(param), :τw) && param.τw > 0.0f0) && (return)
    # @unpack b = param
    # @inbounds for i = 1:N
    #     w[i] = ifelse(fire[i], w[i] + b, w[i])
    # end
end

function update_neuron!(p::IF_CANAHP, param::IF_CANAHPParameter, dt::Float32)
    @unpack N, v, I, tabs, fire, Ca, pCAN, pAHP, syn_curr = p
    @unpack gCAN,
    area,
    gAHP,
    αAHP,
    βAHP,
    γAHP,
    VAHP,
    αCAN,
    βCAN,
    VCAN,
    gL,
    VL,
    Ca0,
    τCa,
    C = param
    # @inbounds 

    for i = 1:N
        if tabs[i] > 0
            fire[i] = false
            tabs[i] -= 1
            continue
        end

        Ca[i] = (Ca0 - Ca[i])/τCa
        pCAN[i] = (1 - pCAN[i])*(αCAN*Ca[i] + βCAN) - βCAN
        pAHP[i] = (1 - pAHP[i])*(αAHP*Ca[i] + βAHP) - βAHP
        I_ionic = let
            I_L = gL * area * (v[i] - VL)  # leakage
            I_CAN = gCAN * area * pCAN[i] * (v[i] - VCAN) # calcium current
            I_AHP = gAHP * area * pAHP[i] * (v[i] - VAHP) # calcium current
            I_L + I_CAN + I_AHP
        end

        # Membrane potential
        v[i] += dt/(C * area) * (- (I_ionic + syn_curr[i]) + I[i])
    end
    # # Adaptation current
    # # if the adaptation timescale is zero, return
    # !(hasfield(typeof(param), :τw) && param.τw > 0.0f0) && (return)
    # @unpack a, b, τw = param
    # @inbounds for i = 1:N
    #     (w[i] += dt * (a * (v[i] - El) - w[i]) / τw)
    # end
end

function update_synapses!(p::IF_CANAHP, param::IF_CANAHPParameter, dt::Float32)
    @unpack N, g, h, hi, he = p
    @unpack exc_receptors, inh_receptors, α, syn = param

    # Update the rise_conductance from the input spikes (he, hi)
    # @turbo 
    for i ∈ 1:N
        g[i, 1] += he[i] * α[1] #AMPA single exponential
        h[i, 2] += he[i] * α[2] #NMDA
        g[i, 3] += hi[i] * α[3] #GABAa single exponential
        h[i, 4] += hi[i] * α[4] #GABAb
    end
    fill!(hi, 0.0f0)
    fill!(he, 0.0f0)
    for n in eachindex(syn)
        @unpack τr⁻, τd⁻ = syn[n]
        # @fastmath @turbo 
        for i ∈ 1:N
            g[i, n] = exp64(-dt * τd⁻) * (g[i, n] + dt * h[i, n])
            h[i, n] = exp64(-dt * τr⁻) * (h[i, n])
        end
    end
end


@inline function synaptic_current!(p::IF_CANAHP, param::IF_CANAHPParameter)
    @unpack N, g, v, syn_curr = p
    @unpack syn, NMDA, area = param
    @unpack mg, b, k = NMDA
    fill!(syn_curr, 0.0f0)
    # @inbounds 
    for r in eachindex(syn)
        @unpack gsyn, E_rev, nmda = syn[r]
        if nmda > 0.0f0
            @simd for i ∈ 1:N
                syn_curr[i] +=
                    area * gsyn * g[i, r] * (v[i] - E_rev) /
                    (1.0f0 + (mg / b) * exp64(k * (v[i])))
            end
        else
            @simd for i ∈ 1:N
                syn_curr[i] += area * gsyn * g[i, r] * (v[i] - E_rev)
            end
        end
    end
    return
end

export IF_CANAHP, IF_CANAHPParameter
