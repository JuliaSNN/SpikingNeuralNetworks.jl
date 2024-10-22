
gtype = SubArray{
    Float32,
    2,
    Matrix{Float32},
    Tuple{Base.Slice{Base.OneTo{Int64}},Vector{Int64}},
    false,
}

@snn_kw mutable struct CompartmentSynapse{
    GT = gtype,
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
} <: AbstractSpikingSynapse
    param::SpikingSynapseParameter = no_STDPParameter()
    plasticity::PlasticityVariables = no_PlasticityVariables()
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    fireI::VBT # postsynaptic firing
    fireJ::VBT # presynaptic firing
    v_post::VFT
    g::GT  # rise conductance
    αs::VFT = []
    receptors::VIT = []
    records::Dict = Dict()
end


function CompartmentSynapse(
    pre,
    post,
    target::Symbol,
    type::Symbol;
    w = nothing,
    p = 0.0,
    μ=1.0,
    σ = 0.0,
    dist=Normal,
    kwargs...,
)
    if isnothing(w)
        w = rand(dist(μ, σ), post.N, pre.N) # Construct a random dense matrix with dimensions post.N x pre.N
        w[[n for n in eachindex(w[:]) if rand() > p]] .= 0
        w[w .< 0] .= 0 
        w = sparse(w)
    else
        w = sparse(w)
    end
    (pre == post) && (w[diagind(w)] .= 0) # remove autapses if pre == post
    rowptr, colptr, I, J, index, W = dsparse(w)
    fireI, fireJ = post.fire, pre.fire
    v_post = getfield(post, Symbol("v_$target"))

    # Get the parameters for post-synaptic cell
    @unpack dend_syn = post
    @unpack soma_syn = post
    if Symbol(type) == :exc
        receptors = target == :s ? [1] : [1, 2]
        g = view(getfield(post, Symbol("h_$target")), :, receptors)
        αs = [post.dend_syn[i].α for i in eachindex(receptors)]
    elseif Symbol(type) == :inh
        receptors = target == :s ? [2] : [3, 4]
        g = view(getfield(post, Symbol("h_$target")), :, receptors)
        αs = [post.dend_syn[i].α for i in eachindex(receptors)]
    else
        throw(ErrorException("Synapse type: $type not implemented"))
    end

    param = haskey(kwargs, :param) ? kwargs[:param] : no_STDPParameter()
    plasticity = get_variables(param, pre.N, post.N)

    CompartmentSynapse(;
        plasticity = plasticity,
        @symdict(rowptr, colptr, I, J, index, receptors, W, g, αs, v_post, fireI, fireJ)...,
        kwargs...,
    )
end

function forward!(c::CompartmentSynapse, param::SpikingSynapseParameter)
    @unpack colptr, I, W, fireJ, g, αs = c
    @inbounds for j ∈ eachindex(fireJ) # loop on presynaptic neurons
        if fireJ[j] # presynaptic fire
            @inbounds @fastmath for a in eachindex(αs)
                @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                    g[I[s], a] += W[s] * αs[a]
                end
            end
        end
    end
end
