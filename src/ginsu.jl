struct Ginsu{N,JA,JB,JC,JD}
    A::JA
    B::JB
    C::JC
    D::JD
    rₒ::NTuple{N,Float64}
    δr::NTuple{N,Float64}
end

"""
    g = Ginsu(r0, dr, nr, sour, recr, padr, ndamp; dims=(:z,:y,:x), stencilhalfwidth=2)

Create a Ginsu object that defines the model aperture for a given shot.  `g` is used to subset earth 
models using the `sub` and `sub!` methods, and provides the inverse operation using the `super`, 
`super!` and `super_accumulate!` methods.  The padding for the earth model subset is determined 
from the source and receiver positions, along with the padding `padr` and damping region `ndamp` 
parameters.

# required parameters[1,2]
* `r0::NTuple{N,Real}` model origin in each model dimension
* `dr::NTuple{N,Real}` model cell widths in each model dimension
* `nr::NTuple{N,Int}` model cell counts in each model dimension
* `sour::NTuple{N,Vector{Float}}` source locations in each model dimension
* `recr::NTuple{N,Vector{Float}}` receiver locations in each model dimension
* `padr::NTuple{N,NTuple{Real,Real}}` padding beyond the survey aperture in each model dimension[3,4]
* `ndamp::NTuple{N,NTuple{Int,Int}}` damping region for absorbing boundaies in each model dimension[3,4]

# named optional parameters
* `dim=(:z,:y,:x)` axis ordering with the default being `z` fast, and `x` slow
* `stencilhalfwidth=0` if there is a free-surface, set `stencilhalfwidth`, padr, and ndamp accordingly. This will add `stencilhalfwidth` cells above the free surface to allow copying the mirrored model to implement the free surface boundary condition.

# type specification
* `g.lextrng` is the logical (1-based) indices for the Ginsu model subset
* `g.lintrng` is the logical (1-based) indices for the Ginsu model subset interior
* `rₒ` is the physical origin of the Ginsu model subset
* `δr` is the grid cell sizes

# Notes
1. `N` is the number of model dimensions
2. `rₒ[i]` is the model origin along the ith model dimension
3. the model padding is the combination of `padr` and `ndamp`
4. `padr[i][1]` is the padding for the dimension start, and `padr[i][2]` for the end.  The same is true for `ndamp`.

## Example (free-surface)
```
o---------------c---x--------------x---c-------o
|               |   |              |   |       |
|               |   |              |   |       |
|               |   |              |   |       |
|               |   |              |   |       |
|               |   x--------------x   |       |
|               |                      |       |
o---------------c----------------------c-------o
```
* `o` denotes the original model
* `c` denotes the Ginsu module subset
* `x` denotes the Ginsu model subset interior
* the region falling in-between the Ginsu model subset and the Ginsu model subset interior is the absorbing boundary region.
"""
function Ginsu(
        r0::NTuple{N,Real},
        dr::NTuple{N,Real},
        nr::NTuple{N,Int64},
        sour::NTuple{N,AbstractArray{Float64,1}},
        recr::NTuple{N,AbstractArray{Float64,1}},
        padr::NTuple{N,Tuple{Real,Real}},
        ndamp::NTuple{N,Tuple{Int,Int}};
        T::DataType = Float32,
        dims::NTuple=(:z,:y,:x),
        stencilhalfwidth::Int=0) where N
    nrec = length(recr[1])
    nsrc = length(sour[1])
    for idim = 2:N
        @assert length(recr[idim]) == nrec
        @assert length(sour[idim]) == nsrc
    end

    lextrng = Array{UnitRange{Int64}}(undef, N)
    lintrng = Array{UnitRange{Int64}}(undef, N)

    for idim = 1:N
        # source and receiver coords for this dimension:
        reci = recr[idim]
        soui = sour[idim]

        if dims[idim] == :z
            lb, ub = r0[idim] - padr[idim][1] - ndamp[idim][1]*dr[idim] - stencilhalfwidth*dr[idim], r0[idim] + (nr[idim]-1+ndamp[idim][2])*dr[idim] + padr[idim][2]
        else
            # midpoints for all sources:
            midi = Array{Float64}(undef, nsrc, nrec)
            for isrc = 1:nsrc
                midi[isrc,:] = (soui[isrc] .+ reci) / 2
            end

            # midpoint float range:
            lb, ub = minimum(midi) - padr[idim][1], maximum(midi) + padr[idim][2]

            # ensure that the range encloses the source:
            lb, ub = min(minimum(soui) - padr[idim][1], lb), max(maximum(soui) + padr[idim][2], ub)

            # ensure that the range encloses all receivers:
            lb, ub = min(minimum(reci) - padr[idim][1], lb), max(maximum(reci) + padr[idim][2], ub)

            # add damping region
            lb, ub = lb - ndamp[idim][1]*dr[idim], ub + ndamp[idim][2]*dr[idim]
        end

        # integer range:
        idx_lb, idx_ub = floor(Int, (lb - r0[idim]) / dr[idim]) + 1, ceil(Int, (ub - r0[idim]) / dr[idim]) + 1

        # ensure lengths are a scalar multiple of 8 (for simd/mjolnir)
        n = idx_ub - idx_lb + 1
        d,r = divrem(n,8)
        if r != 0
            idx_ub = idx_lb + (d+1)*8 - 1
        end

        # interior/exterior ranges
        lextrng[idim] = idx_lb:idx_ub
        lintrng[idim] = (idx_lb+ndamp[idim][1]):(idx_ub-ndamp[idim][2])
    end

    A = JopPad(JetSpace(T,nr), ntuple(i->lintrng[i], N)..., extend=true)
    B = JopPad(JetSpace(T,nr), ntuple(i->lintrng[i], N)..., extend=false)
    C = JopPad(JetSpace(T,nr), ntuple(i->lextrng[i], N)..., extend=true)
    D = JopPad(JetSpace(T,nr), ntuple(i->lextrng[i], N)..., extend=false)

    Ginsu(A, B, C, D, ntuple(i->Float64(r0[i]), N), ntuple(i->Float64(dr[i]), N))
end

function Ginsu(
        r0::NTuple{N,Real},
        dr::NTuple{N,Real},
        nr::NTuple{N,Int64},
        sour::NTuple{N,AbstractArray{Array{Float64,1},1}},
        recr::NTuple{N,AbstractArray{Array{Float64,1},1}},
        padr::NTuple{N,Tuple{Real,Real}},
        ndamp::NTuple{N,Tuple{Int,Int}};
        T = Float32,
        dims::Tuple=(:z,:y,:x),
        stencilhalfwidth::Integer=0) where N
    nshot = length(recr[1])

    ishot = 1
    g = Ginsu(r0, dr, nr, ntuple(idim->sour[idim][ishot], N), ntuple(idim->recr[idim][ishot], N), padr, ndamp, dims=dims, stencilhalfwidth=stencilhalfwidth)

    lextrng_beg = Int[g.lextrng[idim][1] for idim=1:N]
    lextrng_end = Int[g.lextrng[idim][end] for idim=1:N]
    lintrng_beg = Int[g.lintrng[idim][1] for idim=1:N]
    lintrng_end = Int[g.lintrng[idim][end] for idim=1:N]

    for ishot = 2:nshot
        g = Ginsu(r0, dr, nr, ntuple(idim->sour[idim][ishot], N), ntuple(idim->recr[idim][ishot], N), padr, ndamp, dims=dims, stencilhalfwidth=stencilhalfwidth)
        for idim = 1:N
            if g.lextrng[idim][1] < lextrng_beg[idim]
                lextrng_beg[idim] = g.lextrng[idim][1]
                lintrng_beg[idim] = g.lintrng[idim][1]
            end
            if g.lextrng[idim][end] > lextrng_end[idim]
                lextrng_end[idim] = g.lextrng[idim][end]
                lintrng_end[idim] = g.lintrng[idim][end]
            end
        end
    end

    A = JopPad(JetSpace(T,nr), ntuple(i->lintrng_beg[i]:lintrng_end[i], N)..., extend=true)
    B = JopPad(JetSpace(T,nr), ntuple(i->lintrng_beg[i]:lintrng_end[i], N)..., extend=false)
    C = JopPad(JetSpace(T,nr), ntuple(i->lextrng_beg[i]:lextrng_end[i], N)..., extend=true)
    D = JopPad(JetSpace(T,nr), ntuple(i->lextrng_beg[i]:lextrng_end[i], N)..., extend=false)

    Ginsu(A, B, C, D, r0, dr)
end

"""
    g = Ginsu(r0, dr, nr, aperturer, ndamp; T=Float32)

Create a Ginsu object from absolute aperture.

# required parameters[1,2]
* `aperturer::NTuple{Range}` aperture in each dimension
"""
function Ginsu(r0::NTuple{N,Real}, dr::NTuple{N,Real}, nr::NTuple{N,Int}, aperturer::NTuple{N,AbstractRange}, ndamp::NTuple{N,Tuple{Int,Int}}; T=Float32, stencilhalfwidth::Int=0, dims=(:z,:x,:y)) where N
    lextrng = Array{UnitRange{Int64}}(undef, N)
    lintrng = Array{UnitRange{Int64}}(undef, N)

    for idim = 1:N
        if dims[idim] == :z
            lb, ub = aperturer[idim][1] - (stencilhalfwidth+ndamp[idim][1])*dr[idim], aperturer[idim][end] + ndamp[idim][2]*dr[idim]
        else
            lb, ub = aperturer[idim][1] - ndamp[idim][1]*dr[idim], aperturer[idim][end] + ndamp[idim][2]*dr[idim]
        end

        # integer range:
        idx_lb, idx_ub = floor(Int, (lb - r0[idim]) / dr[idim]) + 1, ceil(Int, (ub - r0[idim]) / dr[idim]) + 1

        # ensure lengths are a scalar multiple of 8 (for simd)
        n = idx_ub - idx_lb + 1
        d,r = divrem(n,8)
        if r != 0
            idx_ub = idx_lb + (d+1)*8 - 1
        end

        lextrng[idim] = idx_lb:idx_ub
        lintrng[idim] = (idx_lb+ndamp[idim][1]):(idx_ub-ndamp[idim][2])
    end

    A = JopPad(JetSpace(T,nr), ntuple(i->lintrng[i], N)..., extend=true)
    B = JopPad(JetSpace(T,nr), ntuple(i->lintrng[i], N)..., extend=false)
    C = JopPad(JetSpace(T,nr), ntuple(i->lextrng[i], N)..., extend=true)
    D = JopPad(JetSpace(T,nr), ntuple(i->lextrng[i], N)..., extend=false)

    Ginsu(A, B, C, D, r0, dr)
end

function _op(ginsu::Ginsu; extend=false, interior=false)
    interior  && extend  && return ginsu.A
    interior  && !extend && return ginsu.B
    !interior && extend  && return ginsu.C
    ginsu.D
end

Base.copy(ginsu::Ginsu) = Ginsu(ginsu.A, ginsu.B, ginsu.C, ginsu.D, ginsu.rₒ, ginsu.δr)

"""
    sub!(mg, ginsu, m; extend=true, interior=false)

Store the subset of `m` corresponding to `ginsu` in `mg`.
"""
sub!(prop_ginsu::AbstractArray, ginsu::Ginsu, prop::AbstractArray; extend=true, interior=false) = mul!(prop_ginsu, _op(ginsu;extend=extend,interior=interior), prop)

"""
    mg = sub(ginsu, m; extend=true, interior=false)

Store the subset of `m` corresponding to `ginsu` in `mg`.  New memory is allocated
for `mg`.  `interior=false` includes the sponge region.
"""
sub(ginsu::Ginsu, prop::AbstractArray; extend=true, interior=false) = _op(ginsu,extend=extend,interior=interior)*prop

"""
    super!(m, ginsu, mg; interior=false, accumulate=false)

Store the superset of `mg` corresponding to `ginsu` in `m`. `interior=false` includes the sponge
region.
"""
function super!(prop::AbstractArray, ginsu::Ginsu, prop_ginsu::AbstractArray; interior=false, accumulate=false)
    A = _op(ginsu, extend=false, interior=interior)
    state!(A, (accumulate=accumulate,))
    mul!(prop, A', prop_ginsu)
    prop
end

"""
    mi = interior(ginsu, mg)

Get a view of the Ginsu'd subset `mg` corresponding to its interior region which
excludes the sponge.
"""
function interior(ginsu::Ginsu{N}, prop_ginsu::AbstractArray{T,N}) where {T,N}
    rng = interior(ginsu)
    view(prop_ginsu, rng...)
end

"""
    I = interior(ginsu)

Get the index range corresponding to the interior region which excludes
the sponge.
"""
function interior(ginsu::Ginsu{N}) where {N}
    AI = _op(ginsu, extend=false, interior=true)
    AE = _op(ginsu, extend=false, interior=false)
    ntuple(i->begin
            strt = state(AI).pad[i][1] - state(AE).pad[i][1] + 1
            stop = strt + length(state(AI).pad[i]) - 1
            strt:stop
        end, N)::NTuple{N,UnitRange{Int}}
end

Base.size(gn::Ginsu, i::Int; interior=false) = length(state(_op(gn; interior=interior)).pad[i])
Base.size(gn::Ginsu{N}; interior=false) where {N} = ntuple(i->size(gn, i; interior=interior), Val{N}())

lextents(gn::Ginsu, i::Int; interior=false) = state(_op(gn; interior=interior)).pad[i]
lextents(gn::Ginsu{N}; interior=false) where {N} = ntuple(i->lextents(gn, i; interior=interior), Val{N}())

pextents(gn::Ginsu, i::Int; interior=false) = begin l=lextents(gn,i;interior=interior); (gn.rₒ[i] + (l[1]-1)*gn.δr[i]):(gn.δr[i]):(gn.rₒ[i] + (l[end]-1)*gn.δr[i]) end
pextents(gn::Ginsu{N}; interior=false) where {N} = ntuple(i->pextents(gn,i;interior=interior), Val{N}())

origin(gn::Ginsu, i::Int; interior=false) = gn.rₒ[i] + (first(lextents(gn,i;interior=interior))-1)*gn.δr[i]
origin(gn::Ginsu{N}; interior=false) where {N} = ntuple(i->origin(gn, i;interior=interior)[1], Val{N}())

export Ginsu, lextents, pextents, origin, interior, sub, sub!, super!
