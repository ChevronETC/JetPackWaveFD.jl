# set random seed to promote repeatability in CI unit tests
using Random
Random.seed!(101)

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
