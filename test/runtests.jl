using SpikingNeuralNetworks
using Test

@testset "SpikingNeuralNetworks" begin
    # every exported name of the umbrella module is defined
    @test isempty([n for n in names(SpikingNeuralNetworks) if !isdefined(SpikingNeuralNetworks, n)])
    @test SNN.raster! === SNNPlots.raster!
    @test SNN.modelcopy === SNNModels.modelcopy
    # Tripod names of SNNModels 2.1.0 / 2.2.0
    @test SNN.TripodPaper === SNNModels.TripodPaper
    @test :TripodPaper in names(SpikingNeuralNetworks)
    @test SNN.PaperScheme === SNNModels.PaperScheme
    @test SNN.TripodDendSynapseMouse === SNNModels.TripodDendSynapseMouse
    @test SNN.distal_proximal === SNNModels.distal_proximal
    @test SNN.TripodPaper(N = 3) isa SNNModels.Tripod
    # Multipod names of SNNModels 2.3.0
    for name in (:Multipod, :MultipodParameter, :MultipodPaper, :declare_multipod, :set_dendrite_length!)
        @test name in names(SpikingNeuralNetworks)
        @test getfield(SNN, name) === getfield(SNNModels, name)
    end
    @test SNN.Multipod(N = 2, param = SNN.MultipodParameter(ds = [300 * SNN.um, 200 * SNN.um, 100 * SNN.um])) isa SNNModels.Multipod{3}
end
