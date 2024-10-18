@snn_kw struct EmptySynapse <: AbstractConnection
    param::EmptyParam = EmptyParam()
    records::Dict = Dict()
end

function forward!(p::EmptySynapse, param::EmptyParam) end
