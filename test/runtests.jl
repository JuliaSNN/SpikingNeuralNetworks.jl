using SpikingNeuralNetworks
using Test

@testset "SpikingNeuralNetworks" begin
    # every exported name of the umbrella module is defined
    @test isempty([n for n in names(SpikingNeuralNetworks) if !isdefined(SpikingNeuralNetworks, n)])
    @test SNN.raster! === SNNPlots.raster!
    @test SNN.modelcopy === SNNModels.modelcopy
end
