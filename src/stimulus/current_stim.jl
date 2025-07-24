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

function CurrentNoiseParameter(
    N::Union{Number, AbstractPopulation};
    I_base::Number=0,
    I_dist::Distribution = Normal(0.0, 0.0),
    α::Number = 0.0,
)
    if isa(N, AbstractPopulation)
        N = N.N
    end
    return CurrentNoiseParameter(
        I_base = fill(Float32(I_base), N),
        I_dist = I_dist,
        α = fill(Float32(α), N),
    )
end


@snn_kw struct CurrentStimulus{
    FT = Float32,
    VFT = Vector{Float32},
    DT = Distribution{Univariate,Continuous},
    VIT = Vector{Int},
} <: AbstractStimulus
    param::CurrentStimulusParameter
    name::String = "Current"
    id::String = randstring(12)
    neurons::VIT
    ##

    randcache::VFT = rand(length(neurons)) # random cache
    I::VFT # target input current
    records::Dict = Dict()
    targets::Dict = Dict()
end



function CurrentStimulus(
    post::T, 
    sym::Symbol=:I;
    neurons = :ALL,
    param,
    kwargs...,
) where {T<:AbstractPopulation}
    if neurons == :ALL
        neurons = 1:post.N
    end
    targets =
        Dict(:pre => :Current, :post => post.id, :sym => :soma, :type=>:CurrentStimulus)
    return CurrentStimulus(
        neurons = neurons,
        I = getfield(post, sym),
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
    @unpack I, neurons, randcache = p
    @unpack I_base, I_dist, α = param
    rand!(I_dist, randcache)
    @inbounds @simd for i in p.neurons
        I[i] = (I_base[i]+ randcache[i])*(1-α[i]) + I[i]* (α[i])
    end
end

function stimulate!(p, param::CurrentVariableParameter, time::Time, dt::Float32)
    @unpack I, neurons, randcache = p
    @unpack variables, func = param
    @inbounds @simd for i in p.neurons
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
