# Multipod

@snn_kw struct Multipod{
    VBT = Vector{Bool},
    VIT = Vector{Int},
    TFT = Array{Float32,3}, ## Float type
    MFT = Matrix{Float32}, ## Float type
    VFT = Vector{Float32}, ## Float type
    VST = Vector{Vector{Float32}}, ## Synapses types 
    VDT = Vector{Dendrite},
    ST = SynapseArray,
    NMDAT = NMDAVoltageDependency{Float32},
    PST = PostSpike{Float32},
    IT = Int32,
    FT = Float32,
    AdExType = AdExSoma,
} <: AbstractDendriteIF
    id::String = randstring(12)
    name::String = "Multipod"
    ## These are compulsory parameters
    N::IT = 100
    Nd::IT = 3
    soma_syn::ST
    dend_syn::ST
    NMDA::NMDAT
    param::AdExType = AdExSoma()
    dendrites::VDT

    # soma
    v_s::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w_s::VFT = zeros(N)
    # dendrites
    gax::MFT = zeros(Nd, N)
    cd::MFT = zeros(Nd, N)
    gm::MFT = zeros(Nd, N)
    v_d::VST =
        Vector{Vector{Float32}}([param.Vr .+ rand(N) .* (param.Vt - param.Vr) for n = 1:Nd])   #! target

    # Synapses dendrites
    g_d::TFT = zeros(N, Nd, 4)
    h_d::TFT = zeros(N, Nd, 4)
    hi_d::VST = Vector{Vector{Float32}}([zeros(N) for n = 1:Nd])   #! target
    he_d::VST = Vector{Vector{Float32}}([zeros(N) for n = 1:Nd])   #! target

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
    ## 
    Δv::VFT = zeros(Nd + 1)
    Δv_temp::VFT = zeros(Nd + 1)
    cs::VFT = zeros(Nd)
    is::VFT = zeros(Nd + 1)
end


function synaptic_target(targets::Dict, post::Multipod, sym::Symbol, target::Number)
    _sym = Symbol("$(sym)_d")
    _v = Symbol("v_d")
    g = getfield(post, _sym)[target]
    v_post = getfield(post, _v)[target]
    push!(targets, :sym => Symbol(string(_sym, target)))
    return g, v_post
end


function Multipod(ds::R; N, Nd, kwargs...) where {R<:Real}
    dendrites = [ds for _ = 1:Nd]
    return Multipod(dendrites, N = N, Nd = Nd; kwargs...)
end

function Multipod(
    ds::Vector;
    N::Int,
    soma_syn = TripodSomaSynapse,
    dend_syn = TripodDendSynapse,
    NMDA::NMDAVoltageDependency,
    kwargs...,
)

    soma_syn = synapsearray(soma_syn)
    dend_syn = synapsearray(dend_syn)

    Nd = length(ds)

    # if Nd == 2
    #     return Tripod(
    #         ds[1],
    #         ds[2];
    #         N=N,
    #         soma_syn = soma_syn,
    #         dend_syn = dend_syn,
    #         NMDA=NMDA,
    #         kwargs...
    #     )
    # else
    dendrites = [create_dendrite(N, d) for d in ds]
    gax, cd, gm = zeros(Nd, N), zeros(Nd, N), zeros(Nd, N)
    for i in eachindex(dendrites)
        local d = dendrites[i]
        gax[i, :] = d.gax
        cd[i, :] = d.C
        gm[i, :] = d.gm
    end
    return Multipod(
        Nd = Nd,
        N = N,
        dendrites = dendrites,
        soma_syn = synapsearray(soma_syn),
        dend_syn = synapsearray(dend_syn),
        NMDA = EyalNMDA,
        α = [syn.α for syn in dend_syn],
        gax = gax,
        cd = cd,
        gm = gm;
        kwargs...,
    )
    # end
end

function integrate!(p::Multipod, param::AdExSoma, dt::Float32)
    @unpack N, Nd, v_s, w_s, v_d = p
    @unpack fire, θ, after_spike, postspike, Δv, Δv_temp = p
    @unpack Er, up, τabs, BAP, AP_membrane, Vr, Vt, τw, a, b = param
    @unpack gax, cd, gm = p
    @unpack dend_syn, soma_syn = p

    # Update all synaptic conductance
    update_synapses!(p, dend_syn, soma_syn, dt)

    # update the neurons
    @inbounds for i ∈ 1:N
        # implementation of the absolute refractory period with backpropagation (up) and after spike (τabs)
        if after_spike[i] > (τabs + up - up) / dt # backpropagation
            v_s[i] = BAP
            for d in eachindex(v_d)
                v_d[d][i] += dt * (BAP - v_d[d][i]) * gax[d, i] / cd[d, i]
            end
        elseif after_spike[i] > 0 # absolute refractory period
            v_s[i] = Vr
            # for d in eachindex(v_d)
            #     v_d[d][i] += dt * (BAP - v_d[d][i]) * gax[d,i] / cd[d,i]
            # end
        else
            ## Heun integration
            fill!(Δv_temp, 0.0f0)
            fill!(Δv, 0.0f0)
            update_multipod!(p, Δv, i, param, 0.0f0)
            for _i ∈ eachindex(Δv)
                Δv_temp[_i] = Δv[_i]
            end
            update_multipod!(p, Δv, i, param, dt)
            @fastmath v_s[i] += 0.5 * dt * (Δv_temp[1] + Δv[1])
            @fastmath w_s[i] += dt * (param.a * (v_s[i] - param.Er) - w_s[i]) / param.τw
            for d in eachindex(v_d)
                @fastmath v_d[d][i] += 0.5 * dt * (Δv_temp[d+1] + Δv[d+1])
            end
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


#const dend_receptors::SVector{Symbol,3} = [:AMPA, :NMDA, :GABAa, :GABAb]
# const soma_receptors::Vector{Symbol} = [:AMPA, :GABAa]

function update_synapses!(
    p::Multipod,
    dend_syn::SynapseArray,
    soma_syn::SynapseArray,
    dt::Float32,
)
    @unpack N, Nd, ge_s, g_d, he_s, v_d, h_d, hi_s, gi_s = p
    @unpack he_d, hi_d, exc_receptors, inh_receptors, α = p

    @inbounds for n in exc_receptors
        for d in eachindex(v_d)
            @turbo for i ∈ 1:N
                h_d[i, d, n] += he_d[d][i] * α[n]
            end
        end
    end
    @inbounds for n in inh_receptors
        for d in eachindex(v_d)
            @turbo for i ∈ 1:N
                h_d[i, d, n] += hi_d[d][i] * α[n]
            end
        end
    end

    for d = 1:Nd
        fill!(he_d[d], 0.0f0)
        fill!(hi_d[d], 0.0f0)
    end

    for n in eachindex(dend_syn)
        @unpack τr⁻, τd⁻ = dend_syn[n]
        for d in eachindex(v_d)
            @fastmath @turbo for i ∈ 1:N
                g_d[i, d, n] = exp64(-dt * τd⁻) * (g_d[i, d, n] + dt * h_d[i, d, n])
                h_d[i, d, n] = exp64(-dt * τr⁻) * (h_d[i, d, n])
            end
        end
    end

    @unpack τr⁻, τd⁻ = soma_syn[1]
    @fastmath @turbo for i ∈ 1:N
        ge_s[i] = exp64(-dt * τd⁻) * (ge_s[i] + dt * he_s[i])
        he_s[i] = exp64(-dt * τr⁻) * (he_s[i])
    end
    @unpack τr⁻, τd⁻ = soma_syn[2]
    @fastmath @turbo for i ∈ 1:N
        gi_s[i] = exp64(-dt * τd⁻) * (gi_s[i] + dt * hi_s[i])
        hi_s[i] = exp64(-dt * τr⁻) * (hi_s[i])
    end

end

function update_multipod!(
    p::Multipod,
    Δv::Vector{Float32},
    i::Int64,
    param::AdExSoma,
    dt::Float32,
)

    @fastmath @inbounds begin
        @unpack gax, cd, gm, Nd = p
        @unpack soma_syn, dend_syn, NMDA = p
        @unpack v_d, v_s, w_s, ge_s, gi_s, g_d, θ = p
        @unpack is, cs = p
        @unpack mg, b, k = NMDA

        #compute axial currents
        for d in eachindex(v_d)
            cs[d] = -((v_d[d][i] + Δv[d+1] * dt) - (v_s[i] + Δv[1] * dt)) * gax[d, i]
        end

        for _i ∈ eachindex(is)
            is[_i] = 0.0f0
        end
        # update synaptic currents soma
        @unpack gsyn, E_rev = soma_syn[1]
        is[1] += gsyn * ge_s[i] * (v_s[i] + Δv[1] * dt - E_rev)
        @unpack gsyn, E_rev = soma_syn[2]
        is[1] += gsyn * gi_s[i] * (v_s[i] + Δv[1] * dt - E_rev)

        # update synaptic currents dendrites
        for r in eachindex(dend_syn)
            for d in eachindex(v_d)
                @unpack gsyn, E_rev, nmda = dend_syn[r]
                if nmda > 0.0f0
                    is[d+1] +=
                        gsyn * g_d[i, d, r] * (v_d[d][i] + Δv[d+1] * dt - E_rev) / (1.0f0 + (mg / b) * exp256(k * (v_d[d][i] + Δv[d+1] * dt))
                        )
                else
                    is[d+1] += gsyn * g_d[i, d, r] * (v_d[d][i] + Δv[d+1] * dt - E_rev)
                end
            end
        end
        @turbo for _i ∈ eachindex(is)
            is[_i] = clamp(is[_i], -5000, 5000)
        end

        # update membrane potential
        @unpack C, gl, Er, ΔT = param
        Δv[1] =
            (
                gl * (
                    (-v_s[i] + Δv[1] * dt + Er) +
                    ΔT * exp64(1 / ΔT * (v_s[i] + Δv[1] * dt - θ[i]))
                ) - w_s[i] - is[1] - sum(cs)
            ) / C

        for d in eachindex(v_d)
            Δv[d+1] =
                ((-(v_d[d][i] + Δv[d+1] * dt) + Er) * gm[d, i] - is[d+1] + cs[d]) / cd[d, i]
        end
    end
end

export Multipod, MultipodNeurons

# @inline @fastmath function ΔvAdEx(v::Float32, w::Float32, θ::Float32, axial::Float32, synaptic::Float32, AdEx::AdExSoma)::Float32
#     return 1/ AdEx.C * (
#         AdEx.gl * (
#                 (-v + AdEx.Er) + 
#                 AdEx.ΔT * exp64(1 / AdEx.ΔT * (v - θ))
#                 ) 
#                 - w 
#                 - synaptic 
#                 - axial
#         ) 
# end ## external currents
