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
@snn_kw struct Tripod{
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
} <: AbstractDendriteIF
    id::String = randstring(12)
    name::String = "Tripod"
    ## These are compulsory parameters
    N::IT = 100
    soma_syn::ST = TripodSomaSynapse |> synapsearray
    dend_syn::ST = TripodDendSynapse |> synapsearray
    d1::VDT = create_dendrite(N, 2um)
    d2::VDT = create_dendrite(N, 3um)
    NMDA::NMDAT = NMDAVoltageDependency(mg = Mg_mM, b = nmda_b, k = nmda_k)
    t::VIT = [0]
    param::AdExType = AdExSoma()

    # Membrane potential and adaptation
    v_s::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w_s::VFT = zeros(N)
    v_d1::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    v_d2::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    I::VFT = zeros(N)
    I_d::VFT = zeros(N)

    # Synapses dendrites
    g_d1::MFT = zeros(N, 4)
    g_d2::MFT = zeros(N, 4)
    h_d1::MFT = zeros(N, 4)
    h_d2::MFT = zeros(N, 4)
    hi_d1::VFT = zeros(N) #! target
    hi_d2::VFT = zeros(N) #! target
    he_d1::VFT = zeros(N) #! target
    he_d2::VFT = zeros(N) #! target
    # Receptors properties
    exc_receptors::VIT = [1, 2]
    inh_receptors::VIT = [3, 4]
    α::VFT = [1.0, 1.0, 1.0, 1.0]

    # Synapses soma
    ge_s::VFT = zeros(N)
    gi_s::VFT = zeros(N)
    he_s::VFT = zeros(N) #! target
    hi_s::VFT = zeros(N) #! target

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

function synaptic_target(targets::Dict, post::Tripod, sym::Symbol, target::Symbol)
    sym = Symbol("$(sym)_$target")
    v = Symbol("v_$target")
    g = getfield(post, sym)
    hasfield(typeof(post), v) && (v_post = getfield(post, v))

    push!(targets, :sym => sym)
    push!(targets, :g => post.id)

    return g, v_post
end


function Tripod(
    d1::Union{Real,Tuple},
    d2::Union{Real,Tuple};
    N::Int,
    soma_syn = TripodSomaSynapse,
    dend_syn = TripodDendSynapse,
    NMDA::NMDAVoltageDependency,
    kwargs...,
)
    soma_syn = synapsearray(soma_syn)
    dend_syn = synapsearray(dend_syn)
    d1 = create_dendrite(N, d1)
    d2 = create_dendrite(N, d2)
    Tripod(;
        N = N,
        d1 = d1,
        d2 = d2,
        soma_syn = soma_syn,
        dend_syn = dend_syn,
        NMDA = NMDA,
        α = [syn.α for syn in dend_syn],
        kwargs...,
    )
end

function TripodHet(
    d1::Union{Real,Tuple} = (150um, 400um),
    d2::Union{Real,Tuple} = (150um, 400um);
    kwargs...,
)
    Tripod(d1, d2; kwargs...)
end




#const dend_receptors::SVector{Symbol,3} = [:AMPA, :NMDA, :GABAa, :GABAb]
# const soma_receptors::Vector{Symbol} = [:AMPA, :GABAa]
const soma_rr = SA[:AMPA, :GABAa]
const dend_rr = SA[:AMPA, :NMDA, :GABAa, :GABAb]

function integrate!(p::Tripod, param::AdExSoma, dt::Float32)
    @unpack N, v_s, w_s, v_d1, v_d2 = p
    @unpack fire, θ, after_spike, postspike, Δv, Δv_temp = p
    @unpack Er, up, τabs, BAP, AP_membrane, Vr, Vt, τw, a, b = param
    @unpack dend_syn, soma_syn = p
    @unpack d1, d2 = p

    # update the neurons
    @fastmath @inbounds begin
    # Update all synaptic conductance
        update_synapses!(p, dend_syn, soma_syn, dt)
        # parts = collect(Iterators.partition(1:N, Threads.nthreads()))
        # Threads.@threads :static for part in eachindex(parts)
        for i ∈ 1:N
            # implementation of the absolute refractory period with backpropagation (up) and after spike (τabs)
            if after_spike[i] > (τabs + up - up) / dt # backpropagation
                v_s[i] = BAP
                ## backpropagation effect
                c1 = (BAP - v_d1[i]) * d1.gax[i]
                c2 = (BAP - v_d2[i]) * d2.gax[i]
                ## apply currents
                v_d1[i] += dt * c1 / d1.C[i]
                v_d2[i] += dt * c2 / d2.C[i]
            elseif after_spike[i] > 0 # absolute refractory period
                v_s[i] = Vr
                c1 = (Vr - v_d1[i]) * d1.gax[i]
                c2 = (Vr - v_d2[i]) * d2.gax[i]
                # ## apply currents
                v_d1[i] += dt * c1 / d1.C[i]
                v_d2[i] += dt * c2 / d2.C[i]
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
                w_s[i] += dt * (param.a * (v_s[i] - param.Er) - w_s[i]) / param.τw
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
    end
    return
end

function update_synapses!(
    p::Tripod,
    dend_syn::SynapseArray,
    soma_syn::SynapseArray,
    dt::Float32,
)
    @unpack N, ge_s, g_d1, g_d2, he_s, h_d1, h_d2, hi_s, gi_s = p
    @unpack he_d1, he_d2, hi_d1, hi_d2, exc_receptors, inh_receptors, α = p

    @inbounds @fastmath begin
        for n in exc_receptors
            @turbo for i ∈ 1:N
                h_d1[i, n] += he_d1[i] * α[n]
                h_d2[i, n] += he_d2[i] * α[n]
            end
        end
        for n in inh_receptors
            @turbo for i ∈ 1:N
                h_d1[i, n] += hi_d1[i] * α[n]
                h_d2[i, n] += hi_d2[i] * α[n]
            end
        end
        fill!(he_d1, 0.0f0)
        fill!(he_d2, 0.0f0)
        fill!(hi_d1, 0.0f0)
        fill!(hi_d2, 0.0f0)
        for n in eachindex(dend_syn)
            @unpack τr⁻, τd⁻ = dend_syn[n]
            @turbo for i ∈ 1:N
                g_d1[i, n] = exp64(-dt * τd⁻) * (g_d1[i, n] + dt * h_d1[i, n])
                h_d1[i, n] = exp64(-dt * τr⁻) * (h_d1[i, n])
                g_d2[i, n] = exp64(-dt * τd⁻) * (g_d2[i, n] + dt * h_d2[i, n])
                h_d2[i, n] = exp64(-dt * τr⁻) * (h_d2[i, n])
            end
        end

        @unpack τr⁻, τd⁻ = soma_syn[1]
        @turbo for i ∈ 1:N
            ge_s[i] = exp64(-dt * τd⁻) * (ge_s[i] + dt * he_s[i])
            he_s[i] = exp64(-dt * τr⁻) * (he_s[i])
        end
        @unpack τr⁻, τd⁻ = soma_syn[2]
        @turbo for i ∈ 1:N
            gi_s[i] = exp64(-dt * τd⁻) * (gi_s[i] + dt * hi_s[i])
            hi_s[i] = exp64(-dt * τr⁻) * (hi_s[i])
        end
    end

end

function update_tripod!(
    p::Tripod,
    Δv::Vector{Float32},
    i::Int64,
    param::AdExSoma,
    dt::Float32,
)

    @fastmath @inbounds begin
        @unpack v_d1, v_d2, v_s, I_d, I, w_s, ge_s, gi_s, g_d1, g_d2, θ = p
        @unpack d1, d2 = p
        @unpack soma_syn, dend_syn, NMDA = p
        @unpack is, cs = p
        @unpack mg, b, k = NMDA

        #compute axial currents
        cs[1] = -((v_d1[i] + Δv[2] * dt) - (v_s[i] + Δv[1] * dt)) * d1.gax[i]
        cs[2] = -((v_d2[i] + Δv[3] * dt) - (v_s[i] + Δv[1] * dt)) * d2.gax[i]

        for _i ∈ 1:3
            is[_i] = 0.0f0
        end
        # update synaptic currents soma
        @unpack gsyn, E_rev = soma_syn[1]
        is[1] += gsyn * ge_s[i] * (v_s[i] + Δv[1] * dt - E_rev)
        @unpack gsyn, E_rev = soma_syn[2]
        is[1] += gsyn * gi_s[i] * (v_s[i] + Δv[1] * dt - E_rev)
        # update synaptic currents dendrites
        for r in eachindex(dend_syn)
            @unpack gsyn, E_rev, nmda = dend_syn[r]
            if nmda > 0.0f0
                is[2] +=
                    gsyn * g_d1[i, r] * (v_d1[i] + Δv[2] * dt - E_rev) /
                    (1.0f0 + (mg / b) * exp256(k * (v_d1[i] + Δv[2] * dt)))
                is[3] +=
                    gsyn * g_d2[i, r] * (v_d2[i] + Δv[3] * dt - E_rev) /
                    (1.0f0 + (mg / b) * exp256(k * (v_d2[i] + Δv[2] * dt)))
            else
                is[2] += gsyn * g_d1[i, r] * (v_d1[i] + Δv[2] * dt - E_rev)
                is[3] += gsyn * g_d2[i, r] * (v_d2[i] + Δv[3] * dt - E_rev)
            end
        end
        @turbo for _i ∈ 1:3
            is[_i] = clamp(is[_i], -1500, 1500)
        end

        # update membrane potential
        @unpack C, gl, Er, ΔT = param
        Δv[1] = 1/C * (
                        + gl * (-v_s[i] + Δv[1] * dt + Er) + ΔT * exp64(1 / ΔT * (v_s[i] + Δv[1] * dt - θ[i]))
                        - w_s[i]  # adaptation
                        - is[1]   # synapses
                        - sum(cs) # axial currents
                        + I[i]  # external current
                        ) 

        Δv[2] = ((-(v_d1[i] + Δv[2] * dt) + Er) * d1.gm[i] - is[2] + cs[1] + I_d[i]) / d1.C[i]
        Δv[3] = ((-(v_d2[i] + Δv[3] * dt) + Er) * d2.gm[i] - is[3] + cs[2] + I_d[i]) / d2.C[i]
    end
end


export Tripod
