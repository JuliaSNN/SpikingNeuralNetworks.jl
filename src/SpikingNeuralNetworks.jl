module SpikingNeuralNetworks


using SNNModels
using SNNPlots
using SNNUtils

export SNNModels, SNNPlots, SNNUtils

SNN = SpikingNeuralNetworks
export SNN

"""
porcodio
"""
f

function f()
    @info "SpikingNeuralNetworks.jl is loaded. Use SNNModels, SNNPlots, and SNNUtils for your spiking neural network needs."
    return nothing
end

end
