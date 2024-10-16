function population_indices(P, type="ˆ")
    n = 1
    indices = Dict{Symbol, Vector{Int}}()
    for k in keys(P)
        !occursin(string(type), string(k)) && continue
        p = getfield(P, k)
        indices[k] = n:(n + p.N - 1)
        n += p.N
    end
    return dict2ntuple(sort(indices))
end

function filter_populations(P, type)
    n = 1
    indices = Dict{Symbol, Any}()
    for k in keys(P)
        !occursin(string(type), string(k)) && continue
        p = getfield(P, k)
        indices[k] = p
    end
    return dict2ntuple(sort(indices))
end