using Test

# Test the G_axial function
@testset "G_axial function" begin
    @test SNN.G_axial(Ri = 200 * Ω * cm, d = 4um, l = 100um) > 0
    @test SNN.G_axial(Ri = 200 * Ω * cm, d = 4um, l = 200um)  > 0
    @test SNN.G_axial(Ri = 200 * Ω * cm, d = 6um, l = 150um) > 0
end

# Test the G_mem function
@testset "G_mem function" begin
    @test SNN.G_mem(Rd = 38907 * Ω * cm^2, d = 4um, l = 100um) > 0
    @test SNN.G_mem(Rd = 38907 * Ω * cm^2, d = 4um, l = 200um) > 0
    @test SNN.G_mem(Rd = 1700Ω * cm^2, d = 6um, l = 150um) > 0
end

# Test the C_mem function
@testset "C_mem function" begin
    @test SNN.C_mem(Cd = 0.5μF / cm^2, d = 4um, l = 100um) > 0
    @test SNN.C_mem(Cd = 0.5μF / cm^2, d = 4um, l = 200um) > 0
    @test SNN.C_mem(Cd = 1μF / cm^2, d = 6um, l = 150um) > 0
end

# Test the create_dendrite function
@testset "create_dendrite function" begin
    @test create_dendrite(d = 4um, l = 100um, s = "H") == (
        gm = 0.0003252032520325203,
        gax = 0.001591549430918953,
        C = 0.0006283185307179586,
        l = 100um,
        d = 4um,
    )
    @test create_dendrite(d = 4um, l = 200um, s = "M") == (
        gm = 0.0001626016260162608,
        gax = 0.0007957747154594765,
        C = 0.0012566370614359172,
        l = 200um,
        d = 4um,
    )
    @test create_dendrite(d = 6um, l = 150um, s = "H") == (
        gm = 0.0005309734513274337,
        gax = 0.001591549430918953,
        C = 0.001413716694115407,
        l = 150um,
        d = 6um,
    )
end
