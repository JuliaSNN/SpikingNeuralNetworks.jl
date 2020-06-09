@snn_kw struct SpikingSynapseParameter{FT=Float32}
    τpre::FT = 20ms
    τpost::FT = 20ms
    Wmax::FT = 0.01
    ΔApre::FT = 0.01 * Wmax
    ΔApost::FT = -ΔApre * τpre / τpost * 1.05
end

@snn_kw mutable struct SpikingSynapse{MFT=SparseMatrixCSC{Float32},VFT=Vector{Float32},VBT=Vector{Bool}}
    param::SpikingSynapseParameter = SpikingSynapseParameter()
    N::Int # input length
    M::Int # output length
    W::MFT  # synaptic weight
    tpre::VFT = zeros(eltype(VFT), N) # presynaptic spiking time
    tpost::VFT = zeros(eltype(VFT), M) # postsynaptic spiking time
    Apre::VFT = zeros(eltype(VFT), N) # presynaptic trace
    Apost::VFT = zeros(eltype(VFT), M) # postsynaptic trace
    firePre::VBT # presynaptic firing
    firePost::VBT # postsynaptic firing
    g::VFT # postsynaptic conductance
    records::Dict = Dict()
end
vars(ss::SpikingSynapse) = (:W=>ss.W,:g=>ss.g,:Apre=>ss.Apre,:Apost=>ss.Apost)

function SpikingSynapse(pre, post, sym; σ = 0.0, p = 0.0, kwargs...)
    W = σ * sprand(post.N, pre.N, p)
    firePost, firePre = post.fire, pre.fire
    g = getfield(post, sym)
    SpikingSynapse(;@symdict(W, firePre, firePost, g)..., N=pre.N, M=post.N, kwargs...)
end

function forward!(du, u, p::SpikingSynapse, t)
    @unpack firePre, firePost = p
    W, g, Apre, Apost = u.x
    dW, dg, dApre, dApost = du.x
    # FIXME: Use broadcast
    for j in 1:length(firePost)
        dg[j] += firePost[j] * sum(W[j,:])
    end
    # FIXME: Remove me
    #dApre .= 0
    #dApost .= 0
end

function plasticity!(du,u,p::SpikingSynapse,t)
    W, g, Apre, Apost = u.x
    dW, dg, dApre, dApost = du.x
    @unpack tpre, tpost, firePre, firePost = p
    @unpack τpre, τpost, Wmax, ΔApre, ΔApost = p.param

    # Update traces based on spike activity
    #dApre[firePre] .= Apre[firePre] .* exp.(- (t .- tpre[firePre]) ./ τpre)
    #dApost[firePost] .= Apost[firePost] .* exp.(- (t .- tpost[firePost]) ./ τpost)

    # Decay traces
    #dApre[firePre] .-= ΔApre
    #dApost[firePost,:] .-= ΔApost

    # Modify weights
    dW[firePost, firePre] .= Apost[firePost] .+ transpose(Apre[firePre])

    # FIXME: Clamp weights
    #dW .-= dW .+ clamp.(W, zero(Wmax), Wmax)
end

function condition(sn::SpikingSynapse, u)
    W = u.x[1]
    any(W) do w
        (w < 0) || (w > sn.param.Wmax)
    end
end
function affect!(sn::SpikingSynapse, u)
    W = u.x[1]
    Wmax = sn.param.Wmax
    for idx in 1:length(W)
        W[idx] = clamp(W[idx], zero(Wmax), Wmax)
    end
end
