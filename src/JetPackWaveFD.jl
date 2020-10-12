module JetPackWaveFD

using CRC32c, Distributed, DistributedJets, DistributedOperations, JetPack, Jets, LinearAlgebra, Printf, Statistics, UUIDs, WaveFD

function isvalid_srcfieldfile(m::AbstractArray, hostname, srcfieldfile, chksum)
    _chksum = crc32c(m)
    validhost = hostname == gethostname()
    validfile = isfile(srcfieldfile)
    validmodel = _chksum == chksum
    (validhost && validfile && validmodel, _chksum)
end

function realizewavelet(wavelet, sz, sx, st, dtmod, ntmod_wav)
    local wavelet_realization
    if isa(wavelet, Array)
        if length(sz) == 1 && ndims(wavelet) == 1
            wavelet_realization = reshape(wavelet, length(wavelet), 1)
        else
            wavelet_realization = wavelet
        end
        if size(wavelet_realization) != (ntmod_wav, length(sx))
            throw(ArgumentError("expected size(wavelet_relaization) = $ntmod_wav - got $(size(wavelet_realization)) -- dtmod=$(dtmod)"))
        end
    else
        wavelet_realization = zeros(Float32, ntmod_wav, length(sx))
        for is = 1:length(sx)
            wavelet_realization[:,is] .= get(wavelet, convert(Array{Float64,1}, st[is] .+ dtmod * [0:ntmod_wav-1;]))
        end
    end
    wavelet_realization
end

@inline function megacells_per_second(nz, nx, nt, time)
    nt > 1 ? nz*nx*nt/1_000_000/time : 0.0
end

@inline function megacells_per_second(nz, ny, nx, nt, time)
    nt > 1 ? nz*ny*nx*nt/1_000_000/time : 0.0
end

include("ginsu.jl")
include("illumination.jl")
include("jop_prop2DAcoIsoDenQ_DEO2_FDTD.jl")
include("jop_prop2DAcoTTIDenQ_DEO2_FDTD.jl")
include("jop_prop2DAcoVTIDenQ_DEO2_FDTD.jl")
include("jop_prop3DAcoIsoDenQ_DEO2_FDTD.jl")
include("jop_prop3DAcoTTIDenQ_DEO2_FDTD.jl")
include("jop_prop3DAcoVTIDenQ_DEO2_FDTD.jl")
include("jop_sinc_regular.jl")

export modelindex, srcillum, srcillum!

end
