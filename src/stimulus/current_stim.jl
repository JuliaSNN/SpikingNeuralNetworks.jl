@snn_kw struct CurrentStimulusParameter{VFT}
end


@snn_kw struct CurrentStimulus{FT=Float32, VFT = Vector{Float32}, DT=Distribution{Univariate, Continuous}, VIT = Vector{Int}} <: AbstractStimulus
    param::CurrentStimulusParameter=CurrentStimulusParameter()
    name::String = "Current"
    id::String = randstring(12)
    cells::VIT
    ##
    I_base::VFT = zeros(Float32, length(cells))
    I_dist::DT = Normal(0.0, 0.0)
    α::VFT = ones(Float32, length(cells))
    randcache::VFT = rand(length(cells)) # random cache
    I::VFT # target input current
    records::Dict = Dict()
    targets::Dict = Dict()
end


function CurrentStimulus(post::T; cells=:ALL, α::R2=1, I_base = 10pA, kwargs...) where {T <: AbstractPopulation, R<:Real, R2<:Real}
    if cells == :ALL
        cells = 1:post.N
    end 

    I_base = isa(I_base, Number) ? fill(I_base, length(cells)) : I_base
    targets = Dict(:pre => :Current, :g => post.id, :sym=>:soma)
    α =  isa(α, Number) ? fill(α, length(cells)) : α

    return CurrentStimulus(
        cells=cells,
        I=post.I,
        α = α,
        targets=targets;
        I_base=I_base,
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
        I[i] = (I_base[i] .+ randcache[i]) * α[i] + I[i] * (1 - α[i])
    end
end


export CurrentStimulus, CurrentStimulusParameter, stimulate!