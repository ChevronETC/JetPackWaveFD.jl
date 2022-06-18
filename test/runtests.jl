# set random seed to promote repeatability in CI unit tests
using CRC32c, Random, Test, JetPackWaveFD
Random.seed!(101)

@testset "valid srcfield files" begin
    write("wavefield-1.bin", rand(2))
    write("wavefield-2.bin", rand(2))
    m = rand(10)
    c = JetPackWaveFD.crc32c(m)
    write("model.bin", m)
    @test JetPackWaveFD.isvalid_srcfieldfiles(m, c, gethostname(), "wavefield-1.bin", "wavefield-2.bin") == (true, c)
    @test JetPackWaveFD.isvalid_srcfieldfiles(m, c, gethostname(), "wavefield-1.bin") == (true, c)
    rm("wavefield-1.bin")
    @test JetPackWaveFD.isvalid_srcfieldfiles(m, c, gethostname(), "wavefield-1.bin", "wavefield-2.bin") == (false, c)

    _m = copy(m)
    _m[1] += 1
    @test JetPackWaveFD.isvalid_srcfieldfiles(_m, c, gethostname(), "wavefield-2.bin")[1] == false

    @test JetPackWaveFD.isvalid_srcfieldfile(m, gethostname(), "wavefield-2.bin", c) == (true, c)
    rm("wavefield-2.bin")
    rm("model.bin")
end

for filename in (
        "ginsu.jl",
        "jop_prop2DAcoIsoDenQ_DEO2_FDTD.jl",
        "jop_prop2DAcoTTIDenQ_DEO2_FDTD.jl",
        "jop_prop2DAcoVTIDenQ_DEO2_FDTD.jl",
        "jop_prop3DAcoIsoDenQ_DEO2_FDTD.jl",
        "jop_prop3DAcoTTIDenQ_DEO2_FDTD.jl",
        "jop_prop3DAcoVTIDenQ_DEO2_FDTD.jl",
        "jop_sinc_regular.jl",
        "wavefield_separation.jl"
    )
    include(filename)
end
