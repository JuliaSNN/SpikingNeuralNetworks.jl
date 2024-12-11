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
    v_d::VST = Vector{Vector{Float32}}([zeros(N) for n in 1:Nd])   #! target

    # Synapses dendrites
    g_d::TFT = zeros(N, Nd, 4)
    h_d::TFT = zeros(N, Nd, 4)
    hi_d::VST = Vector{Vector{Float32}}([zeros(N) for n in 1:Nd])   #! target
    he_d::VST = Vector{Vector{Float32}}([zeros(N) for n in 1:Nd])   #! target

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


function MultipodNeurons(
    ds::Vector,
    N::Int;
    soma_syn = TripodSomaSynapse,
    dend_syn = TripodDendSynapse,
    NMDA::NMDAVoltageDependency= NMDAVoltageDependency(mg = Mg_mM, b = nmda_b, k = nmda_k), kwargs...)

    soma_syn = synapsearray(soma_syn)
    dend_syn = synapsearray(dend_syn)

    Nd = length(ds)
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
        dendrites= dendrites,
        soma_syn = synapsearray(soma_syn),
        dend_syn = synapsearray(dend_syn),
        NMDA = NMDA,
        α= [syn.α for syn in dend_syn],
        gax = gax,
        cd = cd,
        gm = gm;
        kwargs...
    )
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
        if after_spike[i] > (τabs + up - up)/dt # backpropagation
            v_s[i] = BAP
            ## backpropagation effect
            for d in 1:Nd
                v_d[d][i] += dt * (BAP - v_d[d][i]) * gax[d,i] / cd[d,i]
            end
        elseif after_spike[i] > 0 # absolute refractory period
            v_s[i] = Vr
            # ## apply currents
            for d in 1:Nd
                v_d[d][i] += dt * (BAP - v_d[d][i]) * gax[d,i] / cd[d,i]
            end
        else
            ## Heun integration
            for _i ∈ 1:Nd
                Δv_temp[_i] = 0.0f0
                Δv[_i] = 0.0f0
            end
            update_multipod!(p, Δv, i, param, 0.0f0)
            for _i ∈ 1:Nd
                Δv_temp[_i] = Δv[_i]
            end
            update_multipod!(p, Δv, i, param, dt)
            @fastmath v_s[i] += 0.5 * dt * (Δv_temp[1] + Δv[1])
            @fastmath w_s[i] += dt * (param.a * (v_s[i] - param.Er) - w_s[i]) / param.τw
            for d in 1:Nd
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

function update_synapses!(p::Multipod, dend_syn::SynapseArray, soma_syn::SynapseArray, dt::Float32)
    @unpack N, Nd, ge_s, g_d, he_s, h_d, hi_s, gi_s = p
    @unpack he_d, hi_d, exc_receptors, inh_receptors, α = p

    @inbounds for n in exc_receptors
        for d in 1:Nd
            @turbo for i ∈ 1:N
                h_d[i, d, n] += he_d[d][i] * α[n]
            end
        end
    end
    @inbounds for n in inh_receptors
        for d in 1:Nd
            @turbo  for i ∈ 1:N
                h_d[i, d, n] += hi_d[d][i] * α[n]
            end
        end
    end

    for d in 1:Nd
        fill!(he_d[d], 0.0f0)
        fill!(hi_d[d], 0.0f0)
    end

    for n in eachindex(dend_syn)
        @unpack τr⁻, τd⁻ = dend_syn[n]
        for d in 1:Nd
            @fastmath @turbo for i ∈ 1:N
                g_d[i, d, n] = exp32(-dt * τd⁻) * (g_d[i, d, n] + dt * h_d[i, d, n])
                h_d[i, d, n] = exp32(-dt * τr⁻) * (h_d[i, d, n])
            end
        end
    end

    @unpack τr⁻, τd⁻ = soma_syn[1]
    @fastmath @turbo for i ∈ 1:N
        ge_s[i] = exp32(-dt * τd⁻) * (ge_s[i] + dt * he_s[i])
        he_s[i] = exp32(-dt * τr⁻) * (he_s[i])
    end
    @unpack τr⁻, τd⁻ = soma_syn[2]
    @fastmath @turbo for i ∈ 1:N
        gi_s[i] = exp32(-dt * τd⁻) * (gi_s[i] + dt * hi_s[i])
        hi_s[i] = exp32(-dt * τr⁻) * (hi_s[i])
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
        for d in 1:Nd
            cs[d] = -((v_d[d][i] + Δv[d+1] * dt) - (v_s[i] + Δv[1] * dt)) * gax[d,i]
        end

        for _i ∈ 1:Nd + 1
            is[_i] = 0.0f0
        end
        # update synaptic currents soma
        @unpack gsyn, E_rev = soma_syn[1]
        is[1] += gsyn * ge_s[i] * (v_s[i] + Δv[1] * dt - E_rev)
        @unpack gsyn, E_rev = soma_syn[2]
        is[1] += gsyn * gi_s[i] * (v_s[i] + Δv[1] * dt - E_rev)

        # update synaptic currents dendrites
        for r in eachindex(dend_syn)
            for d in 1:Nd
                @unpack gsyn, E_rev, nmda = dend_syn[r]
                if nmda > 0.0f0
                    is[d+1] +=
                        gsyn * g_d[i, d, r] * (v_d[d][i] + Δv[d+1] * dt - E_rev) /
                        (1.0f0 + (mg / b) * exp32(k * (v_d[d][i] + Δv[d+1] * dt)))
                else
                    is[d+1] += gsyn * g_d[i, d, r] * (v_d[d][i] + Δv[d+1] * dt - E_rev)
                end
            end
        end
        @turbo for _i ∈ 1:Nd+1
            is[_i] = clamp(is[_i], -1500, 1500)
        end

        # update membrane potential
        @unpack C, gl, Er, ΔT = param
        Δv[1] =
            (
                gl * (
                    (-v_s[i] + Δv[1] * dt + Er) +
                    ΔT * exp32(1 / ΔT * (v_s[i] + Δv[1] * dt - θ[i]))
                ) - w_s[i] - is[1] - sum(cs)
            ) / C

        for d in 1:Nd
            Δv[d+1] = ((-(v_d[d][i] + Δv[d+1] * dt) + Er) * gm[d, i] - is[d+1] + cs[d]) / cd[d,i]
        end
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
