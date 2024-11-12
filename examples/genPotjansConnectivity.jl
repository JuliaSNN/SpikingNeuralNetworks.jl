using SparseArrays
using ProgressMeter
using Plots

"""
Define Potjans parameters for neuron populations and connection probabilities
"""
function potjans_params(ccu, scale=1.0)
    cumulative = Dict{String, Vector{Int64}}()
    layer_names = ["23E", "23I", "4E", "4I", "5E", "5I", "6E", "6I"]
    
    # Replace static matrix with a regular matrix for `conn_probs`
    conn_probs = Float32[
        0.1009  0.1689 0.0437 0.0818 0.0323 0.0     0.0076 0.0    
        0.1346  0.1371 0.0316 0.0515 0.0755 0.0     0.0042 0.0    
        0.0077  0.0059 0.0497 0.135  0.0067 0.0003  0.0453 0.0    
        0.0691  0.0029 0.0794 0.1597 0.0033 0.0     0.1057 0.0    
        0.1004  0.0622 0.0505 0.0057 0.0831 0.3726  0.0204 0.0    
        0.0548  0.0269 0.0257 0.0022 0.06   0.3158  0.0086 0.0    
        0.0156  0.0066 0.0211 0.0166 0.0572 0.0197  0.0396 0.2252
        0.0364  0.001  0.0034 0.0005 0.0277 0.008   0.0658 0.1443
    ]

    # Calculate cumulative cell counts
    v_old = 1
    for (k, v) in pairs(ccu)
        cumulative[k] = collect(v_old:v + v_old)
        v_old += v
    end
    
    syn_pol = Vector{Int64}(undef, length(ccu))
    for (i, k) in enumerate(keys(ccu))
        syn_pol[i] = occursin("E", k) ? 1 : 0
    end
    
    return cumulative, ccu, layer_names, conn_probs, syn_pol
end

"""
Assign synapse weights selectively, and only update non-zero entries
"""
function index_assignment!(item, w0Weights, g_strengths, Lee, Lie, Lii, Lei)
    (src, tgt, syn0, syn1) = item
    jee, jie, jei, jii = g_strengths
    wig = -20 * 4.5f0

    if syn0 == 1 && syn1 == 1
        w0Weights[src, tgt] = jee
        Lee[src, tgt] = jee
    elseif syn0 == 1 && syn1 == 0
        w0Weights[src, tgt] = jei
        Lei[src, tgt] = jei
    elseif syn0 == 0 && syn1 == 1
        w0Weights[src, tgt] = wig
        Lie[src, tgt] = wig
    elseif syn0 == 0 && syn1 == 0
        w0Weights[src, tgt] = wig
        Lii[src, tgt] = wig
    end
end

"""
Build matrix with memory-efficient computations, filtering and batching
"""
function build_matrix(cumulative, conn_probs, Ncells, g_strengths, syn_pol, batch_size=1000)
    w0Weights = spzeros(Float32, Ncells, Ncells)
    Lee = spzeros(Float32, Ncells, Ncells)
    Lii = spzeros(Float32, Ncells, Ncells)
    Lei = spzeros(Float32, Ncells, Ncells)
    Lie = spzeros(Float32, Ncells, Ncells)
    
    cumvalues = collect(values(cumulative))

    # Process connections in batches
    @inbounds @showprogress for batch in Iterators.partition(1:length(cumvalues), batch_size)
        for i in batch
            (syn0, v) = (syn_pol[i], cumvalues[i])
            for src in v
                for j in batch
                    prob = conn_probs[i, j]
                    if prob > 0
                        (syn1, v1) = (syn_pol[j], cumvalues[j])
                        for tgt in v1
                            if src != tgt && rand() < prob
                                item = (src, tgt, syn0, syn1)
                                index_assignment!(item, w0Weights, g_strengths, Lee, Lie, Lii, Lei)
                            end
                        end
                    end
                end
            end
        end
    end

    Lexc = Lee + Lei
    Linh = Lie + Lii
    
    return w0Weights, Lee, Lie, Lei, Lii, Lexc, Linh
end

"""
Create Potjans weights with modularized params and memory-optimized matrices
"""
function potjans_weights(Ncells, g_strengths, ccu, scale)
    cumulative, ccu, layer_names, conn_probs, syn_pol = potjans_params(ccu, scale)
    build_matrix(cumulative, conn_probs, Ncells, g_strengths, syn_pol)
end

"""
Auxiliary Potjans parameters for neural populations with scaled cell counts
"""
function auxil_potjans_param(scale=1.0)
    ccu = Dict(
        "23E" => trunc(Int32, 20683 * scale), 
        "4E" => trunc(Int32, 21915 * scale),
        "5E" => trunc(Int32, 4850 * scale),  
        "6E" => trunc(Int32, 14395 * scale),
        "6I" => trunc(Int32, 2948 * scale),  
        "23I" => trunc(Int32, 5834 * scale),
        "5I" => trunc(Int32, 1065 * scale),  
        "4I" => trunc(Int32, 5479 * scale)
    )

    Ncells = trunc(Int32, sum(values(ccu)) + 1)
    Ne = trunc(Int32, ccu["23E"] + ccu["4E"] + ccu["5E"] + ccu["6E"])
    Ni = Ncells - Ne
    return Ncells, Ne, Ni, ccu
end
"""
Main function to setup Potjans layer with memory-optimized connectivity
"""
function potjans_layer(scale)
    Ncells, Ne, Ni, ccu = auxil_potjans_param(scale)
    
    # Synaptic strengths for each connection type
    pree = 0.1
    K = round(Int, Ne * pree)
    sqrtK = sqrt(K)
    g = 1.0
    tau_meme = 10.0   # (ms)
    je = 2.0 / sqrtK * tau_meme * g
    ji = 2.0 / sqrtK * tau_meme * g 
    jee = 0.15 * je
    jei = je
    jie = -0.75 * ji
    jii = -ji
    g_strengths = Float32[jee, jie, jei, jii]

    potjans_weights(Ncells, g_strengths, ccu, scale)
end

# Run with a specific scaling factor
scale = 0.01
layer_matrices = potjans_layer(scale)

# Assuming `layer_matrices` is a tuple of matrices, such as:
# layer_matrices = (w0Weights, Lee, Lie, Lei, Lii, Lexc, Linh)
# For example, let's plot the first matrix, `w0Weights`

# Extract the matrix you want to plot
matrix_to_plot = layer_matrices[1]  # Replace 1 with the index of the matrix you need

# Plot the heatmap
heatmap(matrix_to_plot, color=:viridis, xlabel="Source Neurons", ylabel="Target Neurons", title="Connectivity Matrix Heatmap")
