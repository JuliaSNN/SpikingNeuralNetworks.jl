struct IdentityParam <: AbstractPopulationParameter end

@snn_kw mutable struct Identity{VFT = Vector{Float32},VBT = Vector{Bool},IT = Int32} <:
                       AbstractPopulation
    name::String = "identity"
    id::String = randstring(12)
    param::IdentityParam = IdentityParam()
    N::IT = 100
    g::VFT = zeros(N)
    spikecount::VFT = zeros(N)
    h::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    records::Dict = Dict()
end

function integrate!(p::Identity, param::IdentityParam, dt::Float32)
    @unpack g, h, fire, spikecount = p
    for i in eachindex(g)
        h[i] += g[i]
        spikecount[i] = 0.0f0
        if g[i] > 0
            fire[i] = true
            spikecount[i] += Float32(g[i])
        else
            fire[i] = false
        end
        g[i] = 0
    end
end
function synaptic_target(targets::Dict, post::Identity, sym::Symbol, target::Nothing)
    v = :spikecount
    g = getfield(post, :g)
    v_post = getfield(post, v)
    push!(targets, :sym => :g)
    return g, v_post
end



export Identity, IdentityParam
