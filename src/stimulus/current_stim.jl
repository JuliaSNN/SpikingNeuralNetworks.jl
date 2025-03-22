abstract type CurrentStimulusParameter end

@snn_kw struct CurrentVariableParameter{VFT = Vector{Float32}} <: CurrentStimulusParameter
    variables::Dict{Symbol,VFT} = Dict{Symbol,VFT}()
    func::Function
end

@snn_kw struct CurrentNoiseParameter{VFT = Vector{Float32}} <: CurrentStimulusParameter
    I_base::VFT = zeros(Float32, 0)
    I_dist::Distribution{Univariate,Continuous} = Normal(0.0, 0.0)
    α::VFT = ones(Float32, 0)
end


@snn_kw struct CurrentStimulus{
    FT = Float32,
    VFT = Vector{Float32},
    DT = Distribution{Univariate,Continuous},
    VIT = Vector{Int},
} <: AbstractStimulus
    param::CurrentStimulusParameter = CurrentStimulusParameter()
    name::String = "Current"
    id::String = randstring(12)
    cells::VIT
    ##

    randcache::VFT = rand(length(cells)) # random cache
    I::VFT # target input current
    records::Dict = Dict()
    targets::Dict = Dict()
end



function CurrentStimulus(
    post::T;
    cells = :ALL,
    # α::R = 1f0,
    # I_dist::Distribution = Normal(0.0, 0.0),
    # I_base = 10pA,
    param,
    kwargs...,
) where {T<:AbstractPopulation,R<:Real}
    if cells == :ALL
        cells = 1:post.N
    end

    # if 
    # I_base = isa(I_base, Number) ? fill(I_base, length(cells)) : I_base
    # α = isa(α, Number) ? fill(α, length(cells)) : α
    # @show α
    # param = CurrentNoiseParameter(I_base, I_dist, α)


    targets = Dict(:pre => :Current, :g => post.id, :sym => :soma)
    return CurrentStimulus(
        cells = cells,
        I = post.I,
        targets = targets;
        param = param,
        kwargs...,
    )
end


# """
#     stimulate!(p::CurrentStimulus, param::CurrentStimulus, time::Time, dt::Float32)

# Generate a Poisson stimulus for a postsynaptic population.
# """
function stimulate!(p, param::CurrentNoiseParameter, time::Time, dt::Float32)
    @unpack I, cells, randcache = p
    @unpack I_base, I_dist, α = param
    rand!(I_dist, randcache)
    @inbounds @simd for i in p.cells
        I[i] = (I_base[i] .+ randcache[i]) * α[i] + I[i] * (1 - α[i])
    end
end

function stimulate!(p, param::CurrentVariableParameter, time::Time, dt::Float32)
    @unpack I, cells, randcache = p
    @unpack variables, func = param
    @inbounds @simd for i in p.cells
        I[i] = func(variables, get_time(time), i)
    end
end


function ramping_current(variables::Dict, t::Float32, args...)
    peak = variables[:peak]
    start_time = variables[:start_time]
    peak_time = variables[:peak_time]
    end_time = variables[:end_time]
    if t < start_time || t > end_time
        return 0pA
    end
    if t >= start_time && t <= peak_time
        return peak * (t - start_time) / (peak_time - start_time)
    end
end


export CurrentStimulus,
    CurrentStimulusParameter,
    stimulate!,
    CurrentNoiseParameter,
    CurrentVariableParameter,
    ramping_current
