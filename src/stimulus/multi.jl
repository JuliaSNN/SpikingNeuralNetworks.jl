# struct MultiParam end
# @snn_kw struct MultiStimulus{VFT = Vector{Float32},VBT = Vector{Bool},VIT = Vector{Int}, IT = Int32, VST} <:  AbstractStimulus
#     id::String = randstring(12)
#     name::String = "Poisson"
#     param::MultiParam = MultiParam()
#     stimuli::Vector{}
#     randcache::VFT = rand(N) # random cache
#     records::Dict = Dict()
#     targets::Dict = Dict()
# end

# function MultiStimulus(post::T, syms::Vector{Symbol}, type::DataType; kwargs...)
#     stimuli = Vector{type}()
#     for sym in syms
#         push!(§)

#     end
#     MultiStimulus(
#         stimuli= stimuli;
#         kwargs...
#     )

# end

# function stimulate!(p::MultiStimulus, param::PoissonStimulusFixed, time::Time, dt::Float32)
#     for s in p.stimuli
#         stimulate!(s, s.p, time, dt)
#     end
# end

# export PoissonStimulus, stimulate!, PSParam, PoissonStimulusParameter
