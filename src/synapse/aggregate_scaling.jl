@snn_kw struct AggregateScalingParameter{FT = Float32} <: NormParam
    τ::FT = 10ms
    τa::FT
    τe::FT
    Y::Vector{FT}
    Wmin::FT = 0.5pF
    Wmax::FT = 250pF
end

function AggregateScalingParameter(
    N,
    rate = 10Hz;
    τ = 10ms,
    τa = 100ms,
    τe = 100ms,
    Wmin = 0.05,
)
    AggregateScalingParameter(τ, τa, τe, fill(rate, N), Wmin)
end

# AggregateScaling

@snn_kw struct AggregateScaling{
    VFT = Vector{Float32},
    VIT = Vector{Int32},
    VBT = Vector{Bool},
    MFT = Matrix{Float32},
    VST = Vector{<:AbstractSparseSynapse},
} <: AbstractNormalization
    N::Int32 = 0
    id::String = randstring(12)
    param::NormParam = MultiplicativeNorm()
    synapses::VST
    Wt::VFT
    WT::VFT
    fire::VBT
    y::VFT = zeros(Float32, N)
    μ::VFT = zeros(Float32, N)
    targets::Dict = Dict()
    records::Dict = Dict()
end

"""
    SynapseNormalization(N; param, kwargs...)

Constructor function for the SynapseNormalization struct.
- N: The number of synapses.
- param: Normalization parameter, can be either MultiplicativeNorm or AdditiveNorm.
- kwargs: Other optional parameters.
Returns a SynapseNormalization object with the specified parameters.
"""
function AggregateScaling(N, synapses; param::AggregateScalingParameter, kwargs...)
    # Set the target and verify is the same population for all synapses
    targets = Dict()
    posts = [syn.targets[:post] for syn in synapses]
    @assert length(unique(posts)) == 1
    targets[:post] = unique(posts)[1]
    targets[:synapses] = [syn.id for syn in synapses]

    # retrierbr 
    if !isa(N, Int)
        @unpack N = N
    end
    WT = zeros(Float32, N)
    Wt = zeros(Float32, N)
    y = zeros(Float32, N)
    μ = zeros(Float32, N)
    # populate the post-synaptic weight array
    fire = synapses[1].fireI
    for syn in synapses
        @assert isa(syn, AbstractSparseSynapse)
        @unpack rowptr, W, index, fireI = syn
        Is = 1:(length(rowptr)-1)
        @assert length(Is) == N
        for i in eachindex(Is)
            @simd for j ∈ rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons connected to neuron 
                WT[i] += W[index[j]]
            end
        end
    end
    AggregateScaling(; @symdict(param, Wt, WT, y, μ, fire, synapses)..., targets, kwargs...)
end



function forward!(c::AggregateScaling, param::AggregateScalingParameter)
    @unpack y, fire, WT = c
    @unpack Y, τa, τe, Wmax = param

    @inbounds @simd for i in eachindex(fire)
        y[i] -= y[i]/τa
    end
    @inbounds @simd for i in eachindex(fire)
        fire[i] && (y[i] += 1)
    end
    @inbounds @simd for i in eachindex(fire)
        WT[i] += (1-WT[i]/Wmax)*(1 - y[i]/Y[i])/τe
    end


end

"""
    plasticity!(c::SynapseNormalization, param::AdditiveNorm, dt::Float32)

Updates the synaptic weights using additive or multiplicative normalization (operator). This function calculates 
the rate of change `μ` as the difference between initial weight `W0` and the current weight `W1`, 
normalized by `W1`. The weights are updated at intervals specified by time constant `τ`.

# Arguments
- `c`: An instance of SynapseNormalization.
- `param`: An instance of AdditiveNorm.
- `dt`: Simulation time step.
"""
function plasticity!(
    c::AggregateScaling,
    param::AggregateScalingParameter,
    dt::Float32,
    T::Time,
)
    tt = get_step(T)
    @unpack τ = param
    if ((tt) % round(Int, τ / dt)) < dt
        plasticity!(c, param)
    end

end

function plasticity!(c::AggregateScaling, param::AggregateScalingParameter)
    @unpack Wt, WT, μ, synapses, y = c
    @unpack τe, Y, Wmin = param
    fill!(Wt, 0.0f0)
    for syn in synapses
        @unpack rowptr, W, index = syn
        Threads.@threads for i = 1:(length(rowptr)-1) # Iterate over all postsynaptic neuron
            @inbounds @fastmath @simd for j = rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons of i
                Wt[i] += W[index[j]]
            end
        end
    end
    # normalize

    @turbo for i in eachindex(μ)
        μ[i] = (WT[i] - Wmin) / Wt[i] #operator defines additive or multiplicative norm
    end
    # apply
    for syn in synapses
        @unpack rowptr, W, index = syn
        Threads.@threads for i = 1:(length(rowptr)-1) # Iterate over all postsynaptic neuron
            @inbounds @fastmath @simd for j = rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons connected to neuron i
                W[index[j]] = W[index[j]] * μ[i] + Wmin
            end
        end
    end
end

export AggregateScaling, AggregateScalingParameter
