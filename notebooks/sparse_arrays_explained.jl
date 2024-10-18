using SpikingNeuralNetworks
using Statistics
##
w = SNN.sprand(100, 100, 0.2) # Construct a random sparse vector with length post.N, pre.N and density p
w[findall(w .!= 0)] .= 1
rowptr, colptr, I, J, index, W = SNN.dsparse(w) # Get info about the existing connections

println(size(W))
println(size(zero(W)))
println(size(colptr[1]:(colptr[1+1]-1)))
println(size(1:(length(colptr)-1)))

#select a pre-synaptic neuron
j = 100
## check if it spiked or continue
# find the indices of the post-synaptic neurons to which it connects
post_synaptic_indices = colptr[j]:(colptr[j+1]-1)
# retrieve the neurons corresponding to those indices
neurons = I[post_synaptic_indices]
# update the neurons
a = 0
for s in post_synaptic_indices
    a += W[s]
    # synaptic_variable[I[s]] += W[s]
end
a /= length(post_synaptic_indices)
@assert mean(Matrix(w)[neurons, j]) ≈ a

## Normalization first step. Compute the total incoming pre-synaptic weights.
#easy way
W0 = sum(Matrix(w), dims = 2)[:, 1]
#non easy way.
W0_noneasy = Vector{Float32}(undef, 100)
for i = 1:(length(rowptr)-1)
    _post = rowptr[i]:(rowptr[i+1]-1)
    W0_noneasy[i] = sum(W[index[_post]])
end
@assert all(W0 .≈ W0_noneasy)


## Let's manipulate the matrix. Select the post-synaptic neuron with the lowest index associated to the pre-synaptic j and set it to 3.5
j = 4
pre_index_j = colptr[j]:(colptr[j+1]-1)
# get the first
i_index = pre_index_j[1]
W[i_index] = 1.0
Matrix(w)


## Let's manipulate the matrix again. Select the pre-synaptic neuron with the highest index associated to the post-synaptic i and set it to 4.5
i = 3
post_index_i = rowptr[i]:(rowptr[i+1]-1)
j_index = post_index_i[1]
#index() return the index in the vector W corresponding to the index of the row-pointer. 
W[round(Int, index[j_index])] = 6
##

## Scaling: apply the correction element-wise
for i = 1:(length(rowptr)-1)
    _post = rowptr[i]:(rowptr[i+1]-1)
    W[index[_post]] .*= W0[i] / sum(W[index[_post]])
end


# 1 # post-synaptic neuron
# pre_synaptic_indices = rowptr[i]:rowptr[i+1]-1 
# # neurons = I[pre_synaptic_indices]
# for s = pre_synaptic_indices
#     a += W[s]
# end
# a  /=length(pre_synaptic_indices)
