# Jets/WaveFD bridge

# 2D source illumination
function srcillum(filename::AbstractString, compressor::WaveFD.Compressor, interior::Bool, ::Type{T}, ginsu::Ginsu, nz::Int, nx::Int, ntrec::Int, nthreads::Int) where {T}
    srcillum!(zeros(T, nz, nx), filename, compressor, ginsu, ntrec, nthreads)
end

function srcillum!(γ::AbstractArray{T,2}, filename::AbstractString, compressor::WaveFD.Compressor, interior::Bool, ginsu::Ginsu, ntrec::Int, nthreads::Int) where {T}
    io = open(filename)
    field_ginsu = zeros(T, size(ginsu, interior=interior))
    for it = 1:ntrec
        WaveFD.compressedread!(io, compressor, it, field_ginsu)
        WaveFD.srcillum_helper(field_ginsu, nthreads)
        super!(γ, ginsu, field_ginsu, accumulate=true, interior=interior)
    end
    close(io)
    γ
end

# 3D source illumination
function srcillum(filename::AbstractString, compressor::WaveFD.Compressor, interior::Bool, ::Type{T}, ginsu::Ginsu, nz::Int, ny::Int, nx::Int, ntrec::Int, nthreads::Int) where {T}
    srcillum!(zeros(T, nz, ny, nx), filename, compressor, interior, ginsu, ntrec, nthreads)
end

function srcillum!(γ::AbstractArray{T,3}, filename::AbstractString, compressor::WaveFD.Compressor, interior::Bool, ginsu::Ginsu, ntrec::Int, nthreads::Int) where {T}
    io = open(filename)
    field_ginsu = zeros(T, size(ginsu, interior=interior))
    for it = 1:ntrec
        WaveFD.compressedread!(io, compressor, it, field_ginsu)
        WaveFD.srcillum_helper(field_ginsu, nthreads)
        super!(γ, ginsu, field_ginsu, accumulate=true, interior=interior)
    end
    close(io)
    γ
end


@doc """
    srcillum(J)
    srcillum(J, m)
    srcillum!(y, J, m)

Compute and return the source illumination for `Jets` operator `J`. The source illumination 
is defined as the sum of squares of the wavefield amplitudes everywhere in the model over 
the time of the finite difference evolution. 

If `J::Jop` is a `Jets` block operator or distributed block operator, the source 
illuminations from all blocks will be accumulated with a simple sum.

`srcillum(J)` creates and returns an array with the source illumination from `J::JopLn`, 
using the current location defined in the `Jet`.

`srcillum(J, m)` creates and returns an array with the source illumination from `J::Jop` 
for the location `m`.

`srcillum!(y, J, m)` zeros the passed array `y` and then accumulates to the source 
illumination from `J::Jop` at the location `m` into `y`.
"""
function srcillum(J::JopLn) 
    s = zeros(eltype(J), size(domain(J))[1:end-1])
    srcillum!(s, J)
end

function srcillum(J::Jop, m::AbstractArray)
    s = zeros(eltype(J), size(domain(J))[1:end-1])
    srcillum!(s, J, m)
end

@doc (@doc srcillum(J))
srcillum!(γ::AbstractArray, J::Jop) = srcillum!(γ, J, jet(J).mₒ)

# composite operators, TODO: the try-catch here is a bit of a kludge
function srcillum!(γ::AbstractArray, A::T, m::AbstractArray) where {D,R,J<:Jet{D,R,typeof(Jets.JetComposite_f!)},T<:Jop{J}}
    for Aᵢ in state(A).ops
        try
            srcillum!(γ, Aᵢ, m)
            break
        catch e
            if length(m) == 0
                rethrow(e)
            end
        end
    end
    γ
end

# block operators
function srcillum!(γ::AbstractArray, A::T, m::AbstractArray) where {D,R,J<:Jet{D,R,typeof(Jets.JetBlock_f!)}, T<:Jop{J}}
    γ .= 0
    for Aᵢ in state(A).ops
        if length(m) == 0
            srcillum!(γ, Aᵢ, jet(Aᵢ).mₒ)
        else
            srcillum!(γ, Aᵢ, m)
        end
    end
    γ
end

# distributed block operators
function srcillum!(γ::AbstractArray, A::T, m::AbstractArray) where {D,R,J<:Jet{D,R,typeof(DistributedJets.JetDBlock_f!)},T<:Jop{J}}
    γ .= 0
    pids = procs(A)
    γ_partials = ArrayFutures(γ, DistributedJets.addmasterpid(pids))

    _srcillum!(γ_partials, A, m) = begin srcillum!(localpart(γ_partials), localpart(A), m); nothing end
    @sync for pid in pids
        @async remotecall_fetch(_srcillum!, pid, γ_partials, A, m)
    end
    reduce!(γ_partials)
end

export srcillum, srcillum!
