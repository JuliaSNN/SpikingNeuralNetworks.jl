"""
    population_indices(P, type = "ˆ")

Given a dictionary `P` containing population names as keys and population objects as values, this function returns a named tuple `indices` that maps each population name to a range of indices. The range represents the indices of the neurons belonging to that population.

# Arguments
- `P`: A dictionary containing population names as keys and population objects as values.
- `type`: A string specifying the type of population to consider. Only population names that contain the specified type will be included in the output. Defaults to "ˆ".

# Returns
A named tuple `indices` where each population name is mapped to a range of indices.
"""
function population_indices(P)
    n = 1
    indices = Dict{Symbol,Vector{Int}}()
    for k in keys(P)
        p = getfield(P, k)
        indices[k] = n:(n+p.N-1)
        n += p.N
    end
    return dict2ntuple(sort(indices))
end

"""
    filter_items(P, regex)

Filter populations in dictionary `P` based on a regular expression `regex`.
Returns a named tuple of populations that match the regex.

# Arguments
- `P`: Container of items.
- `regex`: Regular expression to match population names.

# Returns
A named tuple of populations that match the regex.

# Examples
"""

no_noise(p) = !occursin(string("noise"), string(p.name))

function filter_items(P; condition::Function=no_noise)
    populations = Dict{Symbol, Any}()
    for k in keys(P)
        p = getfield(P, k)
        hasfield(typeof(p), :name) || continue
        condition(p) || continue
        p = getfield(P, k)
        push!(populations,k => p)
    end
    return dict2ntuple(sort(populations, by = x ->getfield(P,x).name))
end



"""
    subpopulations(stim)

Extracts the names and the neuron ids projected from a given set of stimuli.

# Arguments
- `stim`: A dictionary containing stimulus information.

# Returns
- `names`: A vector of strings representing the names of the subpopulations.
- `pops`: A vector of arrays representing the populations of the subpopulations.

# Example
"""
function subpopulations(stim, merge=true)
    # names = Vector{String}()
    # pops = Vector{Int}[]
    populations =  Dict{String, Vector{Int}}()
    my_keys = collect(keys(stim))
    for key in my_keys
        target = merge ? "" : "_$(getfield(stim, key).targets[:sym])"
        name = getfield(stim, key).name*"$target"
        cells = getfield(stim, key).cells
        if haskey(populations, name) 
            populations[name] = vcat(populations[name], cells) |> unique |> collect
        else
            push!(populations, name => cells)
        end
    end
    names = collect(keys(populations))
    pops = collect(values(populations))
    return sort(names, rev=true), pops[sort(1:length(pops), by=x->names[x], rev=true)]
end

export population_indices, filter_populations, subpopulations, filter_items