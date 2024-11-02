


"""
	AdExSoma

An implementation of the Adaptive Exponential Integrate-and-Fire (AdEx) model, adapted for a Tripod neuron.

# Fields
- `C::FT = 281pF`: Membrane capacitance.
- `gl::FT = 40nS`: Leak conductance.
- `R::FT = nS / gl * GΩ`: Total membrane resistance.
- `τm::FT = C / gl`: Membrane time constant.
- `Er::FT = -70.6mV`: Resting potential.
- `Vr::FT = -55.6mV`: Reset potential.
- `Vt::FT = -50.4mV`: Rheobase threshold.
- `ΔT::FT = 2mV`: Slope factor.
- `τw::FT = 144ms`: Adaptation current time constant.
- `a::FT = 4nS`: Subthreshold adaptation conductance.
- `b::FT = 80.5pA`: Spike-triggered adaptation increment.
- `AP_membrane::FT = 10.0f0mV`: After-potential membrane parameter .
- `BAP::FT = 1.0f0mV`: Backpropagating action potential parameter.
- `up::IT = 1ms`, `τabs::IT = 2ms`: Parameters related to spikes.

The types `FT` and `IT` represent Float32 and Int64 respectively.
"""
AdExSoma

@snn_kw struct AdExSoma{FT = Float32,IT = Int64} <: AbstractGeneralizedIFParameter
    #Membrane parameters
    C::FT = 281pF           # (pF) membrane timescale
    gl::FT = 40nS                # (nS) gl is the leaking conductance,opposite of Rm
    R::FT = nS / gl * GΩ               # (GΩ) total membrane resistance
    τm::FT = C / gl                # (ms) C / gl
    Er::FT = -70.6mV          # (mV) resting potential
    # AdEx model
    Vr::FT = -55.6mV     # (mV) Reset potential of membrane
    Vt::FT = -50.4mV          # (mv) Rheobase threshold
    ΔT::FT = 2mV            # (mV) Threshold sharpness
    # Adaptation parameters
    τw::FT = 144ms          #ms adaptation current relaxing time
    a::FT = 4nS            #nS adaptation current to membrane
    b::FT = 80.5pA         #pA adaptation current increase due to spike
    # After spike timescales and membrane
    AP_membrane::FT = 10.0f0mV
    BAP::FT = 1.0f0mV
    up::IT = 1ms
    τabs::IT = 2ms
end

"""
	PostSpike

A structure defining the parameters of a post-synaptic spike event.

# Fields
- `A::FT`: Amplitude of the Post-Synaptic Potential (PSP).
- `τA::FT`: Time constant of the PSP.

The type `FT` represents Float32.
"""
PostSpike

@snn_kw struct PostSpike{FT<:Float32}
    A::FT
    τA::FT
end

"""
	Dendrite

A structure representing a dendritic compartment within a neuron model.

# Fields
- `Er::FT = -70.6mV`: Resting potential.
- `C::FT = 10pF`: Membrane capacitance.
- `gax::FT = 10nS`: Axial conductance.
- `gm::FT = 1nS`: Membrane conductance.
- `l::FT = 150um`: Length of the dendritic compartment.
- `d::FT = 4um`: Diameter of the dendrite.

The type `FT` represents Float32.
"""
Dendrite

get_dendrites_zeros(N, args...) = (; (Symbol("d$x") => zeros(args...) for x = 1:N)...)
get_dendrites_vr(N, vr, args...) = (; (Symbol("d$x") => fill(vr, args...) for x = 1:N)...)

Nd = 10
N = 4


@snn_kw struct Dendrite{FT = Float32}
    Er::FT = -70.6mV             # (mV) resting potential
    C::FT = 10pF                 # (1/pF) membrane timescale
    gax::FT = 10nS# (nS) axial conductance
    gm::FT = 1nS
    l::FT = 150um# μm distance from next compartment
    d::FT = 4um# μm dendrite diameter
end

"""
This is a struct representing a spiking neural network model that include two dendrites and a soma based on the adaptive exponential integrate-and-fire model (AdEx)

# Fields 
- `t::VIT` : tracker of simulation index [0] 
- `param::AdExSoma` : Parameters for the AdEx model.
- `N::Int32` : The number of neurons in the network.
- `soma_syn::ST` : Synapses connected to the soma.
- `dend_syn::ST` : Synapses connected to the dendrites.
- `d1::VDT`, `d2::VDT` : Dendrite structures.
- `NMDA::NMDAT` : Specifies the properties of NMDA (N-methyl-D-aspartate) receptors.
- `gax1::VFT`, `gax2::VFT` : Axial conductance (reciprocal of axial resistance) for dendrite 1 and 2 respectively.
- `cd1::VFT`, `cd2::VFT` : Capacitance for dendrite 1 and 2.
- `gm1::VFT`, `gm2::VFT` : Membrane conductance for dendrite 1 and 2.
- `v_s::VFT` : Somatic membrane potential.
- `w_s::VFT` : Adaptation variables for each soma.
- `v_d1::VFT` , `v_d2::VFT` : Dendritic membrane potential for dendrite 1 and 2.
- `g_s::MFT` , `g_d1::MFT`, `g_d2::MFT` : Conductance of somatic and dendritic synapses.
- `h_s::MFT`, `h_d1::MFT`, `h_d2::MFT` : Synaptic gating variables.
- `fire::VBT` : Boolean array indicating which neurons have fired.
- `after_spike::VFT` : Post-spike timing.
- `postspike::PST` : Model for post-spike behavior.
- `θ::VFT` : Individual neuron firing thresholds.
- `records::Dict` : A dictionary to store simulation results.
- `Δv::VFT` , `Δv_temp::VFT` : Variables to hold temporary voltage changes.
- `cs::VFT` , `is::VFT` : Temporary variables for currents.
"""
Tripod
@snn_kw struct Multipod{
    VBT = Vector{Bool},
    VIT::Vector{Int},
    MFT, ## Conductance type
    VFT, ## Float type
    VDT, ## Dendrite types 
    ST = SynapseArray,
    NMDAT = NMDAVoltageDependency{Float32},
    PST = PostSpike{Float32},
    IT = Int32,
    FT = Float32,
    AdExType = AdExSoma,
} <: AbstractGeneralizedIF
    name::String = "Multipod"
    param::AdExType = AdExSoma()
    ## These are compulsory parameters
    N::IT = 100
    Nd::IT = 3
    soma_syn::ST
    dend_syn::ST
    d1::VDT
    d2::VDT
    NMDA::NMDAT = NMDAVoltageDependency(mg = Mg_mM, b = nmda_b, k = nmda_k)
    ##
    # dendrites
    gax::VFT = get_dendrites_zeros(Nd, N)
    cd::VFT = get_dendrites_zeros(Nd, N)
    gm::VFT = get_dendrites_zeros(Nd, N)
    ##
    # Membrane potential and adaptation
    v_s::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w_s::VFT = zeros(N)
    v_d::VFT = get_dendrites_zeros(Nd, N)
    # Synapses
    g_s::MFT = zeros(N, 2)
    h_s::MFT = zeros(N, 2)
    g_d::MFT = get_dendrites_zeros(Nd, N, 2)
    h_d::MFT = get_dendrites_zeros(Nd, N, 2)
    # Spike model and threshold
    fire::VBT = zeros(Bool, N)
    after_spike::VFT = zeros(Int, N)
    postspike::PST = PostSpike(A = 10, τA = 30ms)
    θ::VFT = ones(N) * param.Vt
    records::Dict = Dict()
    ## 
    Δv::VFT = zeros(Nd + 1)
    Δv_temp::VFT = zeros(Nd + 1)
    cs::VFT = zeros(Nd)
    is::VFT = zeros(Nd + 1)
end

function MultipodNeurons(;
    N::Int,
    dendrites::Vector,
    soma_syn::Synapse,
    dend_syn::Synapse,
    NMDA::NMDAVoltageDependency,
    param = AdExSoma(),
)::Tripod
    Nd = length(dendrites)
    ds = (; (Symbol("d$nd") => d for d in eachindex(dendrites))...)
    gax = (;
        (
            Symbol("d$nd") => [d.gax for d in dendrites[nd]] for nd in eachindex(dendrites)
        )...
    )
    cd = (;
        (Symbol("d$nd") => [d.cd for d in dendrites[nd]] for nd in eachindex(dendrites))...
    )
    gm = (;
        (Symbol("d$nd") => [d.gm for d in dendrites[nd]] for nd in eachindex(dendrites))...
    )
    return Multipod(
        N = N,
        d1 = d1,
        d2 = d2,
        soma_syn = synapsearray(soma_syn),
        dend_syn = synapsearray(dend_syn),
        NMDA = NMDA,
        gax1 = gax1,
        gax2 = gax2,
        cd1 = cd1,
        cd2 = cd2,
        gm1 = gm1,
        gm2 = gm2,
        param = param,
    )
end

#const dend_receptors::SVector{Symbol,3} = [:AMPA, :NMDA, :GABAa, :GABAb]
# const soma_receptors::Vector{Symbol} = [:AMPA, :GABAa]
const soma_rr = SA[:AMPA, :GABAa]
const dend_rr = SA[:AMPA, :NMDA, :GABAa, :GABAb]

function integrate!(p::Tripod, param::AdExSoma, dt::Float32)
    @unpack N,
    v_s,
    w_s,
    v_d1,
    v_d2,
    g_s,
    g_d1,
    g_d2,
    h_s,
    h_d1,
    h_d2,
    d1,
    d2,
    fire,
    θ,
    after_spike,
    postspike,
    Δv,
    Δv_temp = p
    @unpack Er, up, τabs, BAP, AP_membrane, Vr, Vt, τw, a, b = param
    @unpack dend_syn, soma_syn = p
    @unpack gax1, gax2, gm1, gm2, cd1, cd2 = p

    # Update all synaptic conductance
    for n in eachindex(dend_syn)
        @unpack τr⁻, τd⁻ = dend_syn[n]
        @fastmath @simd for i ∈ 1:N
            g_d1[i, n] = exp32(-dt * τd⁻) * (g_d1[i, n] + dt * h_d1[i, n])
            h_d1[i, n] = exp32(-dt * τr⁻) * (h_d1[i, n])
            g_d2[i, n] = exp32(-dt * τd⁻) * (g_d2[i, n] + dt * h_d2[i, n])
            h_d2[i, n] = exp32(-dt * τr⁻) * (h_d2[i, n])
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
            c1 = (BAP - v_d1[i]) * gax1[i]
            c2 = (BAP - v_d2[i]) * gax2[i]
            ## apply currents
            v_d1[i] += dt * c1 / cd1[i]
            v_d2[i] += dt * c2 / cd2[i]
        elseif after_spike[i] > 0
            v_s[i] = Vr
            # c1 = (Vr - v_d1[i]) * gax1[i] /1000
            # c2 = (Vr - v_d2[i]) * gax2[i] /100
            # ## apply currents
            # v_d1[i] += dt * c1 / cd1[i]
            # v_d2[i] += dt * c2 / cd2[i]
        else
            ## Heun integration
            for _i ∈ 1:3
                Δv_temp[_i] = 0.0f0
                Δv[_i] = 0.0f0
            end
            update_tripod!(p, Δv, i, param, 0.0f0)
            for _i ∈ 1:3
                Δv_temp[_i] = Δv[_i]
            end
            update_tripod!(p, Δv, i, param, dt)
            v_s[i] += 0.5 * dt * (Δv_temp[1] + Δv[1])
            v_d1[i] += 0.5 * dt * (Δv_temp[2] + Δv[2])
            v_d2[i] += 0.5 * dt * (Δv_temp[3] + Δv[3])
            w_s[i] += dt * ΔwAdEx(v_s[i], w_s[i], param)
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

function update_tripod!(
    p::Tripod,
    Δv::Vector{Float32},
    i::Int64,
    param::AdExSoma,
    dt::Float32,
)

    @fastmath @inbounds begin
        @unpack v_d1, v_d2, v_s, w_s, g_s, g_d1, g_d2, θ = p
        @unpack gax1, gax2, gm1, gm2, cd1, cd2 = p
        @unpack d1, d2 = p

        @unpack soma_syn, dend_syn, NMDA = p
        @unpack is, cs = p
        @unpack mg, b, k = NMDA

        #compute axial currents
        cs[1] = -((v_d1[i] + Δv[2] * dt) - (v_s[i] + Δv[1] * dt)) * gax1[i]
        cs[2] = -((v_d2[i] + Δv[3] * dt) - (v_s[i] + Δv[1] * dt)) * gax2[i]

        for _i ∈ 1:3
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
                    gsyn * g_d1[i, r] * (v_d1[i] + Δv[2] * dt - E_rev) /
                    (1.0f0 + (mg / b) * exp32(k * (v_d1[i] + Δv[2] * dt)))
                is[3] +=
                    gsyn * g_d2[i, r] * (v_d2[i] + Δv[3] * dt - E_rev) /
                    (1.0f0 + (mg / b) * exp32(k * (v_d2[i] + Δv[2] * dt)))
            else
                is[2] += gsyn * g_d1[i, r] * (v_d1[i] + Δv[2] * dt - E_rev)
                is[3] += gsyn * g_d2[i, r] * (v_d2[i] + Δv[3] * dt - E_rev)
            end
        end
        for _i ∈ 1:3
            is[_i] = clamp(is[_i], -1500, 1500)
        end
        # @info Δv
        # @info v_d1[i], Δv[2], gm1[i], is[2], cs[1], cd1[i] 

        # update membrane potential
        @unpack C, gl, Er, ΔT = param
        Δv[1] =
            (
                gl * (
                    (-v_s[i] + Δv[1] * dt + Er) +
                    ΔT * exp32(1 / ΔT * (v_s[i] + Δv[1] * dt - θ[i]))
                ) - w_s[i] - is[1] - sum(cs)
            ) / C
        Δv[2] = ((-(v_d1[i] + Δv[2] * dt) + Er) * gm1[i] - is[2] + cs[1]) / cd1[i]
        Δv[3] = ((-(v_d2[i] + Δv[3] * dt) + Er) * gm2[i] - is[3] + cs[2]) / cd2[i]
    end

end


# @inline @fastmath function ΔvAdEx(v::Float32, w::Float32, θ::Float32, axial::Float32, synaptic::Float32, AdEx::AdExSoma)::Float32
#     return 1/ AdEx.C * (
#         AdEx.gl * (
#                 (-v + AdEx.Er) + 
#                 AdEx.ΔT * exp32(1 / AdEx.ΔT * (v - θ))
#                 ) 
#                 - w 
#                 - synaptic 
#                 - axial
#         ) 
# end ## external currents

@inline @fastmath function ΔwAdEx(v::Float32, w::Float32, AdEx::AdExSoma)::Float32
    return (AdEx.a * (v - AdEx.Er) - w) / AdEx.τw
end

export Tripod, TripodPopulation, Dendrite, PostSpike
