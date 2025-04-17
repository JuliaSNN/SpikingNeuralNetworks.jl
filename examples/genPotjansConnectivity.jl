using Revise
using DrWatson
@quickactivate "SpikingNeuralNetworks"
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter, IFParameter
using Statistics, Random
using Plots
using SparseArrays
using ProgressMeter
using Plots
using SpikingNeuralNetworks

using SGtSNEpi, Random
using Revise
using Colors, LinearAlgebra
#using GLMakie
using Graphs
using JLD2
import StatsBase.mean



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


if !isfile("potjans_wiring.jld")
    # Run with a specific scaling factor
    scale = 0.01
    w0Weights, Lee, Lie, Lei, Lii, Lexc, Linh = potjans_layer(scale)
    @save "potjans_wiring.jld" w0Weights, Lee, Lie, Lei, Lii, Lexc, Linh
else
    @load "potjans_wiring.jld" w0Weights, Lee, Lie, Lei, Lii, Lexc, Linh    

    exc_pop = unique(Lexc.rowval)
    inhib_pop = unique(Linh.rowval)
    exc_pop = Set(Lexc.rowval)
    inhib_pop = Set(Linh.rowval)


    # Find overlapping elements
    overlap = intersect(exc_pop, inhib_pop)

    # Remove overlapping elements from both sets
    Epop_numbers = setdiff(exc_pop, overlap)
    IPop_numbers = setdiff(inhib_pop, overlap)

    overlap = intersect(exc_pop, inhib_pop)

#=
Note I may have to remove these elements from Lee,Lie,Lii,Lexc,Linh

=#

# Ensure they are disjoint (no overlap)
if isempty(overlap)
    println("exc_pop and inhib_pop are unique sets.")
else
    println("exc_pop and inhib_pop have overlapping elements: ", overlap)
end


## Neuron parameters

function initialize_LKD(Epop_numbers,IPop_numbers,w0Weights, Lee, Lie, Lei, Lii;νe = 4.5Hz)
    τm = 20ms
    C = 300SNN.pF # Capacitance
    R = τm / C
    τre = 1ms # Rise time for excitatory synapses
    τde = 6ms # Decay time for excitatory synapses
    τri = 0.5ms # Rise time for inhibitory synapses 
    τdi = 2ms # Decay time for inhibitory synapses

    # Input and synapse paramater
    N = 1000
    # νe = 8.5Hz # Rate of external input to E neurons 
    # νe = 4.5Hz # Rate of external input to E neurons 
    νi = 0#0.025Hz # Rate of external input to I neurons 
    p_in = 0.5 #1.0 # 0.5 
    μ_in_E = 1.78SNN.pF

    μEE = 2.76SNN.pF # Initial E to E synaptic weight
    μIE = 48.7SNN.pF # Initial I to E synaptic weight
    μEI = 1.27SNN.pF # Synaptic weight from E to I
    μII = 16.2SNN.pF # Synaptic weight from I to I

    Random.seed!(28)

    LKD_AdEx_exc = AdExParameter(
        τm = 20ms,
        Vt = -52mV,
        Vr = -60mV,
        El = -70mV,
        R = R,
        ΔT = 2mV,
        a = 4nS,
        b = 0.805SNN.pA,
        τabs = 1ms,
        τw = 150ms,
        τre = τre,
        τde = τde,
        τri = τri,
        τdi = τdi,
        At = 10mV,
        τt = 30ms,
        E_i = -75mV,
        E_e = 0mV,
    ) #  0.000805nA
    LKD_IF_inh = IFParameter(
        τm = 20ms,
        Vt = -52mV,
        Vr = -60mV,
        El = -62mV,
        R = R,
        τre = τre,
        τde = τde,
        τri = τri,
        τdi = τdi,
        E_i = -75mV,
        E_e = 0mV,
    )

    Input_E = SNN.Poisson(; N = N, param = SNN.PoissonParameter(; rate = νe))
    Input_I = SNN.Poisson(; N = N, param = SNN.PoissonParameter(; rate = νi))
    E = SNN.AdEx(; N = size(Lee)[1], param = LKD_AdEx_exc)
    I = SNN.IF(; N = size(Lee)[1], param = LKD_IF_inh)
    EE = SNN.SpikingSynapse(E, E, w = Lee, :ge; μ = μEE, param = SNN.vSTDPParameter())
    EI = SNN.SpikingSynapse(E, I, w = Lei, :ge; μ = μEI, param = SNN.vSTDPParameter())
    IE = SNN.SpikingSynapse(I, E, w = Lie, :gi; μ = μIE, param = SNN.vSTDPParameter())
    II = SNN.SpikingSynapse(I, I, w = Lii, :gi; μ = μII)
    ProjI = SNN.SpikingSynapse(Input_I, I, :ge; μ = μ_in_E, p = p_in)
    ProjE = SNN.SpikingSynapse(Input_E, E, :ge; μ = μ_in_E, p = p_in) # connection from input to E
    P = [E, I, Input_E, Input_I]
    C = [EE, II, EI, IE, ProjE, ProjI]
    return P, C, EE, II, EI, IE, ProjE, ProjI
end

##
P,C, EE, II, EI, IE, ProjE, ProjI = initialize_LKD(Epop_numbers,IPop_numbers,w0Weights, Lee, Lie, Lei, Lii;νe = 4.5Hz)


#
#P, C = initialize_LKD(20Hz)
duration = 15000ms
#pltdur = 500e1
SNN.monitor(P[1:2], [:fire, :v])
SNN.sim!(P, C; duration = duration)

p1 = SNN.raster(P[1])
p2 = SNN.raster(P[2])
display(plot(p1,p2))

matrix_to_plot = w0Weights  # Replace 1 with the index of the matrix you need


# Plot the heatmap
heatmap(matrix_to_plot, color=:viridis, xlabel="Source Neurons", ylabel="Target Neurons", title="Connectivity Matrix Heatmap")

# Assuming `matrix_to_plot` is the matrix you want to visualize
#matrix_to_plot = layer_matrices[1]  # For example, w0Weights
final = Matrix(Lee)+Matrix(Lei)
# Convert the matrix to dense if it's sparse
matrix_dense = Matrix(final)

# Normalize the matrix between 0 and 1
min_val = minimum(matrix_dense)
max_val = maximum(matrix_dense)
normalized_matrix = (matrix_dense .- min_val) ./ (max_val - min_val)

# Plot the normalized heatmap
heatmap(
    normalized_matrix,
    color=:viridis,
    xlabel="Source Neurons",
    ylabel="Target Neurons",
    title="Normalized Connectivity Matrix Heatmap",
    clims=(0, 1)  # Setting color limits for the normalized scale
)

#scale = 0.015
#pot_conn = grab_connectome(scale)
dim = 2
Lx = Vector{Int64}(zeros(size(w0Weights[2,:])))
Lx = convert(Vector{Int64},Lx)

y = sgtsnepi(w0Weights)
cmap_out = distinguishable_colors(
    length(Lx),
    [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

display(SGtSNEpi.show_embedding( y, Lx ,A=w0Weights;edge_alpha=0.15,lwd_in=0.15,lwd_out=0.013,cmap=cmap_out)
#SNN.raster(P[1:2], [1, 1.5] .* pltdur)
#display(SNN.vecplot(P[1], :v, 10))
# E_neuron_spikes = map(sum, E.records[:fire])
# E_neuron_spikes = map(sum, E.records[:fire])
display(histogram(IE.W[:]))
display(histogram(EE.W[:]))
display(histogram(EI.W[:]))

##
function firing_rate(E, I, bin_width, bin_edges)
    # avg spikes at each time step
    E_neuron_spikes = map(sum, E.records[:fire]) ./ E.N
    I_neuron_spikes = map(sum, I.records[:fire]) ./ E.N
    # Count the number of spikes in each bin
    E_bin_count = [sum(E_neuron_spikes[i:(i+bin_width-1)]) for i in bin_edges]
    I_bin_count = [sum(I_neuron_spikes[i:(i+bin_width-1)]) for i in bin_edges]
    return E_bin_count, I_bin_count
end

# frequencies = [1Hz 10Hz 100Hz]
inputs = 0:50:300
E_bin_counts = []
I_bin_counts = []
bin_edges = []
bin_width = 10  # in milliseconds

num_bins = Int(length(EE.records[:fire]) / bin_width)
bin_edges = 1:bin_width:(num_bins*bin_width)

##

##
inputs = 0nA:5e3nA:(50e3)nA
rates = zeros(length(inputs), 2)
for (n, inh_input) in enumerate(inputs)
    @info inh_input



    #
    duration = 500ms
    #P, C = initialize_LKD(20Hz)
    SNN.monitor(P[1:2], [:fire])
    SNN.sim!(P, C; duration = duration)

    duration = 5000ms
    P[2].I .= inh_input
    SNN.sim!(P, C; duration = duration)
    E_bin_count, I_bin_count = firing_rate(P[1], P[2], bin_width, bin_edges)
    rates[n, 1] = mean(E_bin_count[(end-100):end])
    rates[n, 2] = mean(I_bin_count[(end-100):end])
    # push!(E_bin_counts, E_bin_count)
    # push!(I_bin_counts, I_bin_count)
end

plot(
    inputs,
    rates,
    label = ["Excitatory" "Inhibitory"],
    ylabel = "Firing rate (Hz)",
    xlabel = "External input (μA)",
)
##

# Create a new plot or use an existing plot if it exists
plot(
    xlabel = "Time bins",
    size = (800, 800),
    ylabel = "Firing frequency (spikes/$(bin_width) ms)",
    xtickfontsize = 6,
    ytickfontsize = 6,
    yguidefontsize = 6,
    xguidefontsize = 6,
    titlefontsize = 7,
    legend = :bottomright,
    layout = (length(frequencies), 1),
    title = ["νi = $(νi*1000)Hz" for νi in frequencies],
)

# Plot excitatory neurons
plot!(bin_edges, E_bin_counts, label = "Excitatory neurons")
# Plot inhibitory neurons
plot!(bin_edges, I_bin_counts, label = "Inhibitory neurons")


# Optional: If you need them as sorted arrays (not sets)
#exc_pop_array = sort(collect(exc_pop))
#inhib_pop_array = sort(collect(inhib_pop))

#@show exc_pop_array
#@show inhib_pop_array

# Assuming `layer_matrices` is a tuple of matrices, such as:
# layer_matrices = (w0Weights, Lee, Lie, Lei, Lii, Lexc, Linh)
# For example, let's plot the first matrix, `w0Weights`

# Extract the matrix you want to plot
matrix_to_plot = layer_matrices[1]  # Replace 1 with the index of the matrix you need

# Plot the heatmap
heatmap(matrix_to_plot, color=:viridis, xlabel="Source Neurons", ylabel="Target Neurons", title="Connectivity Matrix Heatmap")
