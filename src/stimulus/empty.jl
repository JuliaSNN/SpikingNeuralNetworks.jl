@snn_kw struct EmptyStimulus <: AbstractStimulus
    param::EmptyParam = EmptyParam()
    records::Dict = Dict()
end

function stimulate!(p::EmptyStimulus, param::EmptyParam, T::Time, dt::Float32) end
