struct IdentityParam
end

@snn_kw mutable struct Identity{VFT = Vector{Float32},VBT = Vector{Bool},IT = Int32} <:
                       AbstractPopulation
    name::String = "identity"
    id::String = randstring(12)
    param::IdentityParam = IdentityParam()
    N::IT = 100
    g::VFT = zeros(N)
    h::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    records::Dict = Dict()
end

function integrate!(p::Identity, param::IdentityParam, dt::Float32)
    @unpack g,h, fire = p
    for i = eachindex(g)
        h[i]+=g[i]
        if g[i] > 0
            fire[i] = true
        else
            fire[i] = false
        end
        g[i]=0 
    end
end

export Identity, IdentityParam
