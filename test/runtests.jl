for filename in (
        "ginsu.jl",
        "jop_prop2DAcoIsoDenQ_DEO2_FDTD.jl",
        "jop_prop2DAcoTTIDenQ_DEO2_FDTD.jl",
        "jop_prop2DAcoVTIDenQ_DEO2_FDTD.jl",
        "jop_prop3DAcoIsoDenQ_DEO2_FDTD.jl",
        "jop_prop3DAcoTTIDenQ_DEO2_FDTD.jl",
        "jop_prop3DAcoVTIDenQ_DEO2_FDTD.jl",
        "jop_sinc_regular.jl"
    )
    include(filename)
end
