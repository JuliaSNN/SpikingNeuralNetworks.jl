"""
This is a struct representing a spiking neural network model that include two dendrites and a soma based on the adaptive exponential integrate-and-fire model (AdEx)


# Fields 
- `t::VIT` : tracker of simulation index [0] 
- `param::AdExSoma` : Parameters for the AdEx model.
- `N::Int32` : The number of neurons in the network.
- `soma_syn::ST` : Synapses connected to the soma.
- `dend_syn::ST` : Synapses connected to the dendrites.
- `d::VDT`, `d2::VDT` : Dendrite structures.
- `NMDA::NMDAT` : Specifies the properties of NMDA (N-methyl-D-aspartate) receptors.
- `v_s::VFT` : Somatic membrane potential.
- `w_s::VFT` : Adaptation variables for each soma.
- `v_d::VFT` , `v_d::VFT` : Dendritic membrane potential for dendrite 1 and 2.
- `g_s::MFT` , `g_d::MFT`, `g_d2::MFT` : Conductance of somatic and dendritic synapses.
- `h_s::MFT`, `h_d::MFT`, `h_d2::MFT` : Synaptic gating variables.
- `fire::VBT` : Boolean array indicating which neurons have fired.
- `after_spike::VFT` : Post-spike timing.
- `postspike::PST` : Model for post-spike behavior.
- `θ::VFT` : Individual neuron firing thresholds.
- `records::Dict` : A dictionary to store simulation results.
- `Δv::VFT` , `Δv_temp::VFT` : Variables to hold temporary voltage changes.
- `cs::VFT` , `is::VFT` : Temporary variables for currents.
"""
BallAndStick
@snn_kw struct BallAndStick{
    MFT = Matrix{Float32},
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    VDT = Dendrite{Vector{Float32}},
    ST = SynapseArray,
    NMDAT = NMDAVoltageDependency{Float32},
    PST = PostSpike{Float32},
    IT = Int32,
    FT = Float32,
    AdExType = AdExSoma,
} <: AbstractDendriteIF     ## These are compulsory parameters
    N::IT = 100
    soma_syn::ST
    dend_syn::ST
    d::VDT
    NMDA::NMDAT = NMDAVoltageDependency(mg = Mg_mM, b = nmda_b, k = nmda_k)
    ##
    t::VIT = [0]
    param::AdExType = AdExSoma()
    # Membrane potential and adaptation
    v_s::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w_s::VFT = zeros(N)
    v_d::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    # Synapses
    g_s::MFT = zeros(N, 2)
    g_d::MFT = zeros(N, 4)
    h_s::MFT = zeros(N, 2)
    h_d::MFT = zeros(N, 4)
    # Spike model and threshold
    fire::VBT = zeros(Bool, N)
    after_spike::VFT = zeros(Int, N)
    postspike::PST = PostSpike(A = 10, τA = 30ms)
    θ::VFT = ones(N) * param.Vt
    records::Dict = Dict()
    Δv::VFT = zeros(3)
    Δv_temp::VFT = zeros(3)
    cs::VFT = zeros(2)
    is::VFT = zeros(3)
end

function BallAndStick(
    d::Union{Real,Tuple};
    N::Int,
    soma_syn::Synapse,
    dend_syn::Synapse,
    NMDA::NMDAVoltageDependency,
    param = AdExSoma(),
)
    BallAndStick(;
        N = N,
        d = create_dendrite(N, d),
        soma_syn = synapsearray(soma_syn),
        dend_syn = synapsearray(dend_syn),
        NMDA = NMDA,
        param = param,
    )
end


#const dend_receptors::SVector{Symbol,3} = [:AMPA, :NMDA, :GABAa, :GABAb]
# const soma_receptors::Vector{Symbol} = [:AMPA, :GABAa]
function integrate!(p::BallAndStick, param::AdExSoma, dt::Float32)
    @unpack N, v_s, w_s, v_d = p
    @unpack g_s, g_d, h_s, h_d = p
    @unpack fire, θ, after_spike, postspike, Δv, Δv_temp = p
    @unpack Er, up, τabs, BAP, AP_membrane, Vr, Vt, τw, a, b = param
    @unpack dend_syn, soma_syn = p
    @unpack d = p

    # Update all synaptic conductance
    for n in eachindex(dend_syn)
        @unpack τr⁻, τd⁻ = dend_syn[n]
        @fastmath @simd for i ∈ 1:N
            g_d[i, n] = exp32(-dt * τd⁻) * (g_d[i, n] + dt * h_d[i, n])
            h_d[i, n] = exp32(-dt * τr⁻) * (h_d[i, n])
        end
    end
    # for soma
    for n in eachindex(soma_syn)
        @unpack τr⁻, τd⁻ = soma_syn[n]
        @fastmath @simd for i ∈ 1:N
            g_s[i, n] = exp32(-dt * τd⁻) * (g_s[i, n] + dt * h_s[i, n])
            h_s[i, n] = exp32(-dt * τr⁻) * (h_s[i, n])
        end
    end

    # update the neurons
    @inbounds for i ∈ 1:N
        if after_spike[i] > τabs
            v_s[i] = BAP
            ## backpropagation effect
            c1 = (BAP - v_d[i]) * d.gax[i]
            ## apply currents
            v_d[i] += dt * c1 / d.C[i]
        elseif after_spike[i] > 0
            v_s[i] = Vr
            # c1 = (Vr - v_d1[i]) * gax1[i] /1000
            # ## apply currents
            # v_d1[i] += dt * c1 / cd1[i]
        else
            ## Heun integration
            for _i ∈ 1:2
                Δv_temp[_i] = 0.0f0
                Δv[_i] = 0.0f0
            end
            update_ballandstick!(p, Δv, i, param, 0.0f0)
            for _i ∈ 1:2
                Δv_temp[_i] = Δv[_i]
            end
            update_ballandstick!(p, Δv, i, param, dt)
            @fastmath v_s[i] += 0.5 * dt * (Δv_temp[1] + Δv[1])
            @fastmath v_d[i] += 0.5 * dt * (Δv_temp[2] + Δv[2])
            @fastmath w_s[i] += dt * (param.a * (v_s[i] - param.Er) - w_s[i]) / param.τw
        end
    end

    # reset firing
    fire .= false
    @inbounds for i ∈ 1:N
        θ[i] -= dt * (θ[i] - Vt) / postspike.τA
        after_spike[i] -= 1
        if after_spike[i] < 0
            ## spike ?
            if v_s[i] > θ[i] + 10.0f0
                fire[i] = true
                θ[i] += postspike.A
                v_s[i] = AP_membrane
                w_s[i] += b ##  *τw
                after_spike[i] = (up + τabs) / dt
            end
        end
    end
    return
end

function update_ballandstick!(
    p::BallAndStick,
    Δv::Vector{Float32},
    i::Int64,
    param::AdExSoma,
    dt::Float32,
)

    @fastmath @inbounds begin
        @unpack v_d, v_s, w_s, g_s, g_d, θ, d = p
        @unpack soma_syn, dend_syn, NMDA = p
        @unpack is, cs = p
        @unpack mg, b, k = NMDA

        #compute axial currents
        cs[1] = -((v_d[i] + Δv[2] * dt) - (v_s[i] + Δv[1] * dt)) * d.gax[i]

        for _i ∈ 1:2
            is[_i] = 0.0f0
        end
        for r in eachindex(soma_syn)
            @unpack gsyn, E_rev = soma_syn[r]
            is[1] += gsyn * g_s[i, r] * (v_s[i] + Δv[1] * dt - E_rev)
        end
        for r in eachindex(dend_syn)
            @unpack gsyn, E_rev, nmda = dend_syn[r]
            if nmda > 0.0f0
                is[2] +=
                    gsyn * g_d[i, r] * (v_d[i] + Δv[2] * dt - E_rev) /
                    (1.0f0 + (mg / b) * exp32(k * (v_d[i] + Δv[2] * dt)))
            else
                is[2] += gsyn * g_d[i, r] * (v_d[i] + Δv[2] * dt - E_rev)
            end
        end
        @turbo for _i ∈ 1:2
            is[_i] = clamp(is[_i], -1500, 1500)
        end

        # update membrane potential
        @unpack C, gl, Er, ΔT = param
        Δv[1] =
            (
                gl * (
                    (-v_s[i] + Δv[1] * dt + Er) +
                    ΔT * exp32(1 / ΔT * (v_s[i] + Δv[1] * dt - θ[i]))
                ) - w_s[i] - is[1] - cs[1]
            ) / C
        Δv[2] = ((-(v_d[i] + Δv[2] * dt) + Er) * d.gm[i] - is[2] + cs[1]) / d.C[i]
    end

end

export BallAndStick
