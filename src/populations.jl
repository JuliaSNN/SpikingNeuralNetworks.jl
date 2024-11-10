function population_indices(P, type = "ˆ")
    n = 1
    indices = Dict{Symbol,Vector{Int}}()
    for k in keys(P)
        !occursin(string(type), string(k)) && continue
        p = getfield(P, k)
        indices[k] = n:(n+p.N-1)
        n += p.N
    end
    return dict2ntuple(sort(indices))
end

function filter_populations(P, type)
    indices = Dict{Symbol, AbstractPopulation}()
    for k in keys(P)
        !occursin(string(type), string(k)) && continue
        p = getfield(P, k)
        push!(indices,k => p)
    end
    return dict2ntuple(sort(indices))
end


function subpopulations(stim)
    names = Vector{String}()
    pops = Vector{Int}[]
    my_keys = sort(collect(keys(stim)))
    for key in my_keys
        push!(names, getfield(stim, key).name)
        push!(pops, getfield(stim, key).cells)
    end
    return names, pops
end

export population_indices, filter_populations, subpopulations