@snn_kw struct CurrentStimulusParameter{VFT}
end


@snn_kw struct CurrentStimulus{FT=Float32, VFT = Vector{Float32}, DT=Distribution{Univariate, Continuous}, VIT = Vector{Int}} <: AbstractStimulus
    param::CurrentStimulusParameter=CurrentStimulusParameter()
    name::String = "Current"
    id::String = randstring(12)
    cells::VIT
    ##
    I_base::FT = 0.0
    I_dist::DT = Normal(0.0, 0.0)
    α::VFT = ones(Float32, length(cells))
    randcache::VFT = rand(length(cells)) # random cache
    I::VFT # target conductance for soma
    records::Dict = Dict()
    targets::Dict = Dict()
end


function CurrentStimulus(post::T; cells=:ALL, I_dist::Distribution{Univariate, Continuous}=Normal(0,0), I_base::R, α::R2=1, kwargs...) where {T <: AbstractPopulation, R<:Real, R2<:Real}
    if cells == :ALL
        cells = 1:post.N
    end 

    α =  isa(α, Number) ? fill(α, length(cells)) : α

    return CurrentStimulus(
        cells=cells,
        I=post.I,
        I_dist = I_dist,
        I_base = I_base,
        α = α;
        kwargs...,
    )
end


# """
#     stimulate!(p::CurrentStimulus, param::CurrentStimulus, time::Time, dt::Float32)

# Generate a Poisson stimulus for a postsynaptic population.
# """
function stimulate!(p, param::CurrentStimulusParameter, time::Time, dt::Float32)
    @unpack I, I_base, cells, randcache, I_dist, α = p
    rand!(I_dist,randcache)
    @inbounds @simd for i in p.cells
        I[i] = (I_base .+ randcache[i]) * α[i] + I[i] * (1 - α[i])
    end
end


export CurrentStimulus, CurrentStimulusParameter, stimulate!