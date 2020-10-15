function JetProp2DAcoIsoDenQ_DEO2_FDTD(;
        b::Array{Float32,2} = ones(Float32,0,0),
        srcfieldfile = joinpath(tempdir(), "field-$(uuid4()).bin"),
        comptype = nothing,
        compscale = 1e-2,
        isinterior = false,
        nz_subcube = 32,
        nx_subcube = 32,
        sz = 0.0,
        sx = 0.0,
        st = 0.0,
        interpmethod = :hicks,
        rz = [0.0],
        rx = [0.0],
        z0 = 0.0,
        x0 = 0.0,
        dz = 10.0,
        dx = 10.0,
        freqQ = 5.0,
        qMin = 0.1,
        qInterior = 100.0,
        padz = 0.0,
        padx = 0.0,
        dtmod,
        dtrec,
        ntrec,
        nbz_cache = 512,
        nbx_cache = 8,
        nbz_inject = 16,
        nbx_inject = 16,
        nsponge = 50,
        wavelet = WaveletCausalRicker(f=5.0),
        freesurface = false,
        nthreads = Sys.CPU_THREADS,
        reportinterval = 500)
    # dtrec must be integer multiple of dtmod
    @assert abs(dtrec - dtmod*round(dtrec/dtmod,RoundNearest)) < eps(Float32)

    # source location (sz,sx,st)
    sz,sx,st = map(val->Float64[val...], (sz,sx,st))

    # domain and range
    dom = JetSpace(Float32, size(b))
    rng = JetSpace(Float32, ntrec, length(rz))

    # type conversions
    z0, x0, padz, padx = map(val->Float64(val), (z0, x0, padz, padx)) # used for computing integer grid locations for source injection
    rz, rx = map(val->convert(Array{Float64}, val), (rz, rx))
    dz, dx, dtmod, dtrec, freqQ, qMin, qInterior = map(val->Float32(val), (dz, dx, dtmod, dtrec, freqQ, qMin, qInterior)) # used in F.D. calculations
    @assert length(rz) == length(rx)
    if isa(wavelet, Array) == true
        wavelet = convert(Array{Float32}, wavelet)
    end

    # ginsu view of earth model... ginsu is aware of the sponge
    nsponge_top = freesurface ? 0 : nsponge
    padz_top = freesurface ? 0.0 : padz

    ginsu = Ginsu((z0,x0), (dz,dx), size(b), (sz,sx), (rz,rx), ((padz_top,padz),(padx,padx)), ((nsponge_top,nsponge),(nsponge,nsponge)), T=Float32)

    # if srcfieldfile is specified make sure its containing folder exists
    if srcfieldfile != ""
        srcfieldpath = split(srcfieldfile, "/")[1:end-1]
        if length(srcfieldpath) > 0
            srcfieldpath = join(srcfieldpath, "/")
            mkpath(srcfieldpath)
        end
    end

    # compression for nonlinear source wavefields
    C = WaveFD.comptype(comptype, Float32)[1]
    compressor = Dict{String,WaveFD.Compressor{Float32,Float32,C,2}}()
    for wavefield_active in ["P","DP"]
        compressor[wavefield_active] = WaveFD.Compressor(Float32, Float32, C, size(ginsu,interior=isinterior), (nz_subcube,nx_subcube), compscale, ntrec, isinterior)
    end

    Jet(
        dom = dom,
        rng = rng,
        f! = JopProp2DAcoIsoDenQ_DEO2_FDTD_f!,
        df! = JopProp2DAcoIsoDenQ_DEO2_FDTD_df!,
        df′! = JopProp2DAcoIsoDenQ_DEO2_FDTD_df′!,
        s = (
            b = b,
            srcfieldfile = srcfieldfile,
            srcfieldhost = Ref(""),
            chksum = Ref(zero(UInt32)),
            compressor = compressor,
            isinterior = isinterior,
            sz = sz,
            sx = sx,
            st = st,
            interpmethod = interpmethod,
            rz = rz,
            rx = rx,
            z0 = z0,
            x0 = x0,
            dz = dz,
            dx = dx,
            freqQ = freqQ,
            qMin = qMin,
            qInterior = qInterior,
            ginsu = ginsu,
            nsponge = nsponge,
            dtmod = dtmod,
            dtrec = dtrec,
            ntrec = ntrec,
            nbz_cache = nbz_cache,
            nbx_cache = nbx_cache,
            nbz_inject = nbz_inject,
            nbx_inject = nbx_inject,
            wavelet = wavelet,
            freesurface = freesurface,
            nthreads = nthreads,
            reportinterval = reportinterval,
            stats = Dict{String,Float64}("MCells/s"=>0.0, "%io"=>0.0, "%inject/extract"=>0.0, "%imaging"=>0.0)))
end

@doc """
    JopNlProp2DAcoIsoDenQ_DEO2_FDTD(; kwargs...)
    JopLnProp2DAcoIsoDenQ_DEO2_FDTD(; v, kwargs...) 

Create a `Jets` nonlinear or linearized operator for 2D visco-acoustic, isotropic, 
variable density modeling.

# Model Parameters
This propagator operates with two model parameters, as shown in the table below.

| Parameter | Description |
|:---:|:---|
| `v` | P wave velocity |
| `b` | buoyancy (reciprocal density) |

Velocity `v` is an **active** parameter and can be inverted for using the Jacobian machinery, 
and buoyancy `b` is a **passive** parameter that is constant.

# Examples

## Model and acquisition geometry setup
1. load modules Jets, WaveFD, and JetPackWaveFD
1. set up the model discretization, coordinate size and spacing
1. set up the acquisition geometry, including the time discretization, locations for source and receivers, and the source wavelet
1. create constant buoyancy array
```
using Jets, WaveFD, JetPackWaveFD
nz,nx = 200, 100                      # spatial discretization size
dz,dx = 20.0, 20.0                    # spatial discretization sampling
ntrec = 1101                          # number of temporal samples in recorded data
dtrec = 0.0100                        # sample rate for recorded data
dtmod = 0.0025                        # sample rate for modeled data
wavelet = WaveletCausalRicker(f=5.0)  # type of wavelet to use (source signature) 
sz = dz                               # source Z location 
sx = dx*(nx/2)                        # source X location 
rx = dx*[0:0.5:nx-1;];                # array of receiver X locations
rz = 2*dz*ones(length(0:0.5:nx-1));   # array of receiver Z locations
b = ones(Float32, nz, nx);            # buoyancy model (reciprocal density)
```

## Construct and apply the nonlinear operator 
1. create the nonlinear operator `F`
1. create the constant velocity model m₀ 
1. perform nonlinear forward modeling with constant velocity model `m₀` and return the resulting modeled data in `d`
```
F = JopNlProp2DAcoIsoDenQ_DEO2_FDTD(; b = b, ntrec = ntrec, 
    dz = dz, dx = dx, dtrec = dtrec, dtmod = dtmod, wavelet = wavelet, 
    sz = sz, sx = sx, rz = rz, rx = rx)
m₀ = 1500 .* ones(domain(F));
d = F*m₀;              # forward nonlinear op
```

## Construct and apply the linearized Jacobian operator (method 1)
1. create the nonlinear operator `F` directly by construction of a jet. 
1. create the constant velocity model m₀ 
1. create the Jacobian operator `J` by linearization of `F` at point `m₀` 
1. create a random model perturbation vector `δm`.
1. perform linearized forward (Born) modeling on the model perturbation vector `δm` and 
    return the resulting data perturbation in `δd`.
1. perform linearized adjoint (Born) migration on the data perturbation vector `δd` and 
    return the resulting model perturbation in `δm`. 

Note that the Jacobian operators `J` and `J'` require the serialized nonlinear forward wavefield, 
and this is generated automatically whenever required. If you watch the logging to standard out
from this example, you will first see the finite difference evolution for the nonlinear forward, 
followed by the linearized forward, and finally the linearized adjoint.
```
F = JopNlProp2DAcoIsoDenQ_DEO2_FDTD(; b = b, ntrec = ntrec, 
    dz = dz, dx = dx, dtrec = dtrec, dtmod = dtmod, wavelet = wavelet, 
    sz = sz, sx = sx, rz = rz, rx = rx)
m₀ = 1500 .* ones(domain(F));
J = jacobian(F, m₀)
δm = rand(domain(J));
δd = J*δm;             # forward linearized op
δm = J'*δd;            # adjoint linearized op
```

## Construct and apply the linearized Jacobian operator (method 2)
1. create the constant velocity model m₀ 
1. create the Jacobian operator `J` at point `m₀` directly by construction of a jet.
1. create a random model perturbation vector `δm`.
1. perform linearized forward (Born) modeling on the model perturbation vector `δm` and 
    return the resulting data perturbation in `δd`.
1. perform linearized adjoint (Born) migration on the data perturbation vector `δd` and 
    return the resulting model perturbation in `δm`. 
```
m₀ = 1500 * ones(Float32, nz, nx);
J = JopLnProp2DAcoIsoDenQ_DEO2_FDTD(; v = m₀, b = b, ntrec = ntrec, 
    dz = dz, dx = dx, dtrec = dtrec, dtmod = dtmod, wavelet = wavelet, 
    sz = sz, sx = sx, rz = rz, rx = rx)
δm = rand(domain(J));
δd = J*δm;             # forward linearized op
δm = J'*δd;            # adjoint linearized op
```

# Required Parameters
* `v` the point at which the jet is linearized. Note this argument is required in the constuctor for  
    `JopLnProp2DAcoIsoDenQ_DEO2_FDTD` but not `JopNlProp2DAcoIsoDenQ_DEO2_FDTD`. This constuctor is shown  
    in the last example above.
* `dtmod` The sample rate for the modeled data. You can establish a lower bound for the modeling sample rate  
    with the following expression: `dt = 0.75 * 0.38 * max(dx,dz) / maximum(v)`. Note that the usual  
    Courant–Friedrichs–Lewy condition (CFL condition) for stability in finite difference modeling is  
    modified *heuristically* to include the impact of the visco-acoustic implementation of this operator,  
    requiring a 25% smaller smaple rate.
* `dtrec` The number of time samples in the recorded data. Note requirement that `dtrec` 
    be an even multiple of `dtmod`.
* `ntrec` The number of time samples in the recorded data. Note that the number of samples in the modeled  
    data is determined from `ntrec`, `dtrec`, and `dtmod`.

# Optional Parameters
Defaults for arguments are shown inside square brackets.

* `b [ones(Float32,0,0)]` the buoyancy (reciprocal density) array.
* `srcfieldfile [joinpath(tempdir(), "field-$(uuid4()).bin")]` the full path to a scratch file used for 
    the serializationof the compressed nonlinear source wavefield. 
* `comptype [nothing]` the type of compression to use for the serialization of
    the nonlinear source wavefield. The type of compression must be one of:
    * `nothing` - no compression.
    * `Float32` - if wavefield is `Float64`, then do a simple conversion to Float32.
    * `UInt32` - compression using CvxCompress (windowing + 2D wavelet transform + 
        thresholding + quantization + run-length-encoding).
* `compscale [1e-2]` determines the thresholding for the compression of the nonlinear source 
    wavefield prior to serialization. Smaller values mean more aggressive thresholding.
* `nz_subcube, nx_subcube [32]` The Z and X sizes of windows used for compression 
    of the nonlinear source wavefield with `CvxCompress`. Note the requirement
    `[8 <= n*_subcube <=256]`.
    * `isinterior [false]` boolean flag that indicates how the nonlinear source wavefield is
    saved. For large models, operation will be faster with `isinterior = true`, but 
    the linearization correctness test may fail.
    * `true` the entire model including absorbing boundaries is serialized and deserialized
    * `false` the interior part of the model excluding absorbing boundaries is serialized 
        and deserialized
* `sz [0.0]` Array of source Z coordinate. Note that if multiple sources are provided, 
    they will will be injected simultaneously during finite difference evolution. 
* `sx [0.0]` Array of source X coordinate.
* `st [0.0]` Array of source delay times.
* `interpmethod [:hicks]` Type of physical interpolation for sources and receivers. For locations 
    that are not on the physical grid coordinates, interpolation is used either to inject or extract
    information. `interpmethod` must be one of:
    * `:hicks` Hicks 2D sinc interpolation (up to 8x8 nonzero points per location)
    * `:linear` bilinear interpolation (up to 2x2 nonzero points per location)
* `rz [[0.0]]` 2D array of receiver Z coordinates
* `rx [[0.0]]` 2D array of receiver Z coordinates
* `z0 [0.0]` Origin of physical coordinates for the Z dimension.
* `x0 [0.0]` Origin of physical coordinates for the X dimension.
* `dz [10.0]` Spacing of physical coordinates in the Z dimension.
* `dx [10.0]` Spacing of physical coordinates in the X dimension.
* `freqQ [5.0]` The center frequency for the Maxwell body approximation to dissipation only attenuation.
    Please see `JetPackWaveFD` package documentation for more information concerning the attenuation model.   
* `qMin [0.1]` The minimum value for Qp at the boundary of the model used in our Maxwell body approximation
    to dissipation only attenuation. This is not a physically meaningful value for Qp, as we use the 
    attenuation to implement absorbing boundary conditions and eliminate outgoing waves on the 
    boundaries of the computational domain. 
    Please see `JetPackWaveFD` package documentation for more information concerning the attenuation model.   
* `qInterior [100.0]` the value for Qp in the interior of the model used in our Maxwell body approximation
    to dissipation only attenuation. This is the value for Qp away from the absorbing boundaries and is 
    a physically meaningful value.
    Please see `JetPackWaveFD` package documentation for more information concerning the attenuation model.   
* `padz, padx [0.0], [0.0]` - apply extra padding to the survey determined aperture in `Ginsu`.
    Please see `Ginsu` for more information. 
* `nbz_cache, nbx_cache [512], [8]` The size of cache blocks in the Z and X dimensions. 
    In general the cache block in the Z (fast) dimension should be ≥ the entire size of that dimension, 
    and the cache block size in the slower dimension is generally small in order to allow the entire
    block to fit in cache. 
* `nbz_inject, nbx_inject = [16], [16]` The number of blocks in the Z and X dimensions for threading 
    the wavefield injection.
* `nsponge [50]` The number of grid cells to use for the absorbing boundary. For high fidelity modeling
    this should be > 60 grid points, but can be significantly smaller for some use cases like low frequency 
    full waveform inversion. 
* `wavelet [WaveletCausalRicker(f=5.0)]` The source wavelet, can be specified as either a Wavelet type 
    or an array.
* `freesurface [false]` Determines if a free surface (`true`) or absorbing (`false`) top boundary condition
    is applied.
* `nthreads [Sys.CPU_THREADS]` The number of threads to use for OpenMP parallelization of the modeling.
* `reportinterval [500]` The interval at which information about the propagtion is logged.

See also: `Ginsu`, `WaveletSine`, `WaveletRicker`, `WaveletMinPhaseRicker`, `WaveletDerivRicker`, 
    `WaveletCausalRicker`, `WaveletOrmsby`, `WaveletMinPhaseOrmsby` 
"""
JopNlProp2DAcoIsoDenQ_DEO2_FDTD(;kwargs...) = JopNl(JetProp2DAcoIsoDenQ_DEO2_FDTD(;kwargs...))

@doc (@doc JopNlProp2DAcoIsoDenQ_DEO2_FDTD)
JopLnProp2DAcoIsoDenQ_DEO2_FDTD(; v, kwargs...) = JopLn(JetProp2DAcoIsoDenQ_DEO2_FDTD(;kwargs...), v)

export JopNlProp2DAcoIsoDenQ_DEO2_FDTD
export JopLnProp2DAcoIsoDenQ_DEO2_FDTD

function JopProp2DAcoIsoDenQ_DEO2_FDTD_nonlinearforward!(d::AbstractArray, m::AbstractArray; kwargs...)
    # make propagator
    nz_ginsu,nx_ginsu = size(kwargs[:ginsu])
    z0_ginsu,x0_ginsu = origin(kwargs[:ginsu])
    p = Prop2DAcoIsoDenQ_DEO2_FDTD(
        nz = nz_ginsu, 
        nx = nx_ginsu,
        nbz = kwargs[:nbz_cache], 
        nbx = kwargs[:nbx_cache],
        dz = kwargs[:dz], 
        dx = kwargs[:dx], 
        dt = kwargs[:dtmod],
        nsponge = kwargs[:nsponge],
        nthreads = kwargs[:nthreads],
        freesurface = kwargs[:freesurface],
        freqQ = kwargs[:freqQ], 
        qMin = kwargs[:qMin], 
        qInterior = kwargs[:qInterior])

    # init arrays
    m_ginsu,b_ginsu = WaveFD.V(p),WaveFD.B(p)
    pcur,pold,pspace = WaveFD.PCur(p),WaveFD.POld(p),WaveFD.PSpace(p)

    # ginsu'd earth model
    sub!(m_ginsu, kwargs[:ginsu], m, extend=true)
    sub!(b_ginsu, kwargs[:ginsu], kwargs[:b], extend=true)
    ginsu_interior_range = interior(kwargs[:ginsu])

    it0, ntmod_wav = WaveFD.default_ntmod(kwargs[:dtrec], kwargs[:dtmod], kwargs[:st], kwargs[:ntrec])

    # source wavelet for injection, one for each source location
    wavelet_realization = realizewavelet(kwargs[:wavelet], kwargs[:sz], kwargs[:sx], kwargs[:st], kwargs[:dtmod], ntmod_wav)

    # Get source and receiver interpolation coefficients
    local iz_sou, ix_sou, c_sou
    if kwargs[:interpmethod] == :hicks
        iz_sou, ix_sou, c_sou = WaveFD.hickscoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:sz], kwargs[:sx])
    else
        iz_sou, ix_sou, c_sou = WaveFD.linearcoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:sz], kwargs[:sx])
    end
    blks_sou = WaveFD.source_blocking(nz_ginsu, nx_ginsu, kwargs[:nbz_inject], kwargs[:nbx_inject], iz_sou, ix_sou, c_sou)

    c_sou_scaled = Array{Array{Float32,2},1}(undef, length(c_sou))
    for i = 1:length(c_sou_scaled)
        c_sou_scaled[i] = similar(c_sou[i])
        for ix = 1:size(c_sou_scaled[i], 2), iz = 1:size(c_sou_scaled[i], 1)
            jz = iz_sou[i][iz]
            jx = ix_sou[i][ix]
            c_sou_scaled[i][iz,ix] = c_sou[i][iz,ix] * kwargs[:dtmod]^2 * m_ginsu[jz,jx]^2 / b_ginsu[jz,jx]
        end
    end
    blks_sou_scaled = WaveFD.source_blocking(nz_ginsu, nx_ginsu, kwargs[:nbz_inject], kwargs[:nbx_inject], iz_sou, ix_sou, c_sou_scaled)

    local iz_rec, ix_rec, c_rec
    if length(d) > 0
        if kwargs[:interpmethod] == :hicks
            iz_rec, ix_rec, c_rec = WaveFD.hickscoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:rz], kwargs[:rx])
        else
            iz_rec, ix_rec, c_rec = WaveFD.linearcoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:rz], kwargs[:rx])
        end
    end

    # disk-file for source-field serialization
    iofield = Dict{String,IOStream}()
    if kwargs[:srcfieldfile] != ""
        for wavefield_active in ["P","DP"]
            filename = "$(kwargs[:srcfieldfile])-$(wavefield_active)"
            isfile(filename) && rm(filename)
            iofield[wavefield_active] = open(filename, "w")
            open(kwargs[:compressor][wavefield_active])
        end
    end

    itskip = round(Int, kwargs[:dtrec]/kwargs[:dtmod])
    time1 = time()
    cumtime_io, cumtime_ex = 0.0, 0.0
    kwargs[:reportinterval] == 0 || @info "nonlinear forward on $(gethostname()), srcfieldfile=$(kwargs[:srcfieldfile])"

    set_zero_subnormals(true)
    for it = 1:ntmod_wav
        if kwargs[:reportinterval] != 0 && (it % kwargs[:reportinterval] == 0 || it == ntmod_wav)
            JopProp2DAcoIsoDenQ_DEO2_FDTD_write_history_nl(kwargs[:ginsu], it, ntmod_wav, time()-time1, cumtime_io, cumtime_ex, pcur, d)
        end

        # propagate and wavefield swap
        WaveFD.propagateforward!(p)
        pcur,pold = pold,pcur

        # inject source wavelet
        cumtime_ex += @elapsed begin
            WaveFD.injectdata!(pcur, blks_sou_scaled, wavelet_realization, it)
            WaveFD.injectdata!(pspace, blks_sou, wavelet_realization, it)
        end

        if it >= it0 && rem(it-1,itskip) == 0
            if length(d) > 0
                # extract receiver data
                WaveFD.extractdata!(d, pold, div(it-1,itskip)+1, iz_rec, ix_rec, c_rec)
            end

            # scale spatial derivatives by v^2/b to make them temporal derivatives
            WaveFD.scale_spatial_derivatives!(p)

            if kwargs[:srcfieldfile] != ""
                cumtime_io += @elapsed begin
                    if kwargs[:isinterior]
                        WaveFD.compressedwrite(iofield["P"], kwargs[:compressor]["P"], div(it-1,itskip)+1, pold, ginsu_interior_range)
                        WaveFD.compressedwrite(iofield["DP"], kwargs[:compressor]["DP"], div(it-1,itskip)+1, pspace, ginsu_interior_range)
                    else
                        WaveFD.compressedwrite(iofield["P"], kwargs[:compressor]["P"], div(it-1,itskip)+1, pold)
                        WaveFD.compressedwrite(iofield["DP"], kwargs[:compressor]["DP"], div(it-1,itskip)+1, pspace)
                    end
                end
            end
        end
    end
    set_zero_subnormals(false)
    JopProp2DAcoIsoDenQ_DEO2_FDTD_stats(kwargs[:stats], kwargs[:ginsu], ntmod_wav, time()-time1, cumtime_io, cumtime_ex)

    if kwargs[:srcfieldfile] != ""
        for wavefield_active in ["P","DP"]
            close(iofield[wavefield_active])
            close(kwargs[:compressor][wavefield_active])
        end
    end

    free(p)

    nothing
end

function JopProp2DAcoIsoDenQ_DEO2_FDTD_f!(d::AbstractArray, m::AbstractArray{Float32}; kwargs...)
    d .= 0
    isvalid, _chksum = isvalid_srcfieldfile(m, kwargs[:srcfieldhost][], kwargs[:srcfieldfile]*"-P", kwargs[:chksum][])
    if !isvalid
        JopProp2DAcoIsoDenQ_DEO2_FDTD_nonlinearforward!(d, m; kwargs...)
        kwargs[:chksum][] = _chksum
        kwargs[:srcfieldhost][] = gethostname()
    else
        field = Array{Float32}(undef,size(kwargs[:ginsu], interior=kwargs[:isinterior]))
        local iz, ix, c
        if kwargs[:interpmethod] == :hicks
            iz, ix, c = WaveFD.hickscoeffs(kwargs[:dz], kwargs[:dx], 
                origin(kwargs[:ginsu], interior=kwargs[:isinterior])..., 
                size(kwargs[:ginsu], interior=kwargs[:isinterior])..., kwargs[:rz], kwargs[:rx])
        else
            iz, ix, c = WaveFD.linearcoeffs(kwargs[:dz], kwargs[:dx], 
                origin(kwargs[:ginsu], interior=kwargs[:isinterior])..., 
                size(kwargs[:ginsu], interior=kwargs[:isinterior])..., kwargs[:rz], kwargs[:rx])
        end
        iofield = open("$(kwargs[:srcfieldfile])-P")
        open(kwargs[:compressor]["P"])
        for it = 1:kwargs[:ntrec]
            WaveFD.compressedread!(iofield, kwargs[:compressor]["P"], it, field)
            WaveFD.extractdata!(d, field, it, iz, ix, c)
        end
        close(iofield)
        close(kwargs[:compressor]["P"])
    end
    d
end

function JopProp2DAcoIsoDenQ_DEO2_FDTD_df!(δd::AbstractArray, δm::AbstractArray; kwargs...)
    nz_ginsu,nx_ginsu=size(kwargs[:ginsu])
    z0_ginsu,x0_ginsu=origin(kwargs[:ginsu])
    p = Prop2DAcoIsoDenQ_DEO2_FDTD(
        nz = nz_ginsu,
        nx = nx_ginsu,
        nbz = kwargs[:nbz_cache],
        nbx = kwargs[:nbx_cache],
        dz = kwargs[:dz],
        dx = kwargs[:dx],
        dt = kwargs[:dtmod],
        nsponge = kwargs[:nsponge],
        nthreads = kwargs[:nthreads],
        freesurface = kwargs[:freesurface],
        freqQ = kwargs[:freqQ],
        qMin = kwargs[:qMin],
        qInterior = kwargs[:qInterior])

    # init arrays
    v_ginsu,b_ginsu = WaveFD.V(p),WaveFD.B(p)
    pcur,pold = WaveFD.PCur(p),WaveFD.POld(p)

    # ginsu'd earth model
    sub!(v_ginsu, kwargs[:ginsu], kwargs[:mₒ], extend=true)
    sub!(b_ginsu, kwargs[:ginsu], kwargs[:b], extend=true)

    δm_ginsu = sub(kwargs[:ginsu], δm, extend=false)

    ginsu_interior_range = interior(kwargs[:ginsu])

    # pre-compute receiver interpolation coefficients
    local iz, ix, c
    if kwargs[:interpmethod] == :hicks
        iz, ix, c = WaveFD.hickscoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:rz], kwargs[:rx])
    else
        iz, ix, c = WaveFD.linearcoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:rz], kwargs[:rx])
    end
    # if necessary, re-run the nonlinear forward
    isvalid, _chksum = isvalid_srcfieldfile(kwargs[:mₒ], kwargs[:srcfieldhost][], kwargs[:srcfieldfile]*"-P", kwargs[:chksum][])
    if !isvalid
        JopProp2DAcoIsoDenQ_DEO2_FDTD_nonlinearforward!(Array{Float32}(undef,0,0), kwargs[:mₒ]; kwargs...)
        kwargs[:srcfieldhost][] = gethostname()
        kwargs[:chksum][] = _chksum
    end

    # local disk file for source wavefield deserialization
    iofield = open("$(kwargs[:srcfieldfile])-DP")
    open(kwargs[:compressor]["DP"])
    DP = zeros(Float32, nz_ginsu, nx_ginsu)

    ntmod = WaveFD.default_ntmod(kwargs[:dtrec], kwargs[:dtmod], kwargs[:ntrec])

    δdinterp = zeros(Float32, ntmod, size(δd, 2))

    itskip = round(Int, kwargs[:dtrec]/kwargs[:dtmod])
    kwargs[:reportinterval] == 0 || @info "linear forward on $(gethostname()), srcfieldfile=$(kwargs[:srcfieldfile])"
    time1 = time()
    cumtime_io, cumtime_ex, cumtime_im = 0.0, 0.0, 0.0

    set_zero_subnormals(true)
    for it = 1:ntmod
        if kwargs[:reportinterval] != 0 && (it % kwargs[:reportinterval] == 0 || it == ntmod)
            JopProp2DAcoIsoDenQ_DEO2_FDTD_write_history_ln(kwargs[:ginsu], it, ntmod, time()-time1, cumtime_io, cumtime_ex, cumtime_im, pcur, δdinterp, "forward")
        end

        # propagate and swap wavefields
        WaveFD.propagateforward!(p)
        pcur,pold = pold,pcur

        if rem(it-1,itskip) == 0
            # read source field from disk
            cumtime_io += @elapsed if kwargs[:isinterior]
                WaveFD.compressedread!(iofield, kwargs[:compressor]["DP"], div(it-1,itskip)+1, DP, ginsu_interior_range)
            else
                WaveFD.compressedread!(iofield, kwargs[:compressor]["DP"], div(it-1,itskip)+1, DP)
            end
            # born injection
            cumtime_im += @elapsed WaveFD.forwardBornInjection!(p, δm_ginsu, DP)
        end

        # extract data at receivers
        cumtime_ex += @elapsed WaveFD.extractdata!(δdinterp, pold, it, iz, ix, c)
    end
    JopProp2DAcoIsoDenQ_DEO2_FDTD_stats(kwargs[:stats], kwargs[:ginsu], ntmod, time()-time1, cumtime_io, cumtime_ex, cumtime_im)
    set_zero_subnormals(false)

    δd .= 0
    WaveFD.interpforward!(WaveFD.interpfilters(kwargs[:dtmod], kwargs[:dtrec], 0, WaveFD.LangC(), kwargs[:nthreads]), δd, δdinterp)

    close(iofield)
    close(kwargs[:compressor]["DP"])

    free(p)

    δd
end

function JopProp2DAcoIsoDenQ_DEO2_FDTD_df′!(δm::AbstractArray, δd::AbstractArray; kwargs...)
    nz_ginsu,nx_ginsu = size(kwargs[:ginsu])
    z0_ginsu,x0_ginsu = origin(kwargs[:ginsu])
    p = Prop2DAcoIsoDenQ_DEO2_FDTD(
        nz = nz_ginsu,
        nx = nx_ginsu,
        nbz = kwargs[:nbz_cache],
        nbx = kwargs[:nbx_cache],
        dz = kwargs[:dz],
        dx = kwargs[:dx],
        dt = kwargs[:dtmod],
        nsponge = kwargs[:nsponge],
        nthreads = kwargs[:nthreads],
        freesurface = kwargs[:freesurface],
        freqQ = kwargs[:freqQ], 
        qMin = kwargs[:qMin], 
        qInterior = kwargs[:qInterior])

    # init arrays
    m_ginsu,b_ginsu = WaveFD.V(p),WaveFD.B(p)
    pcur,pold,pspace = WaveFD.PCur(p),WaveFD.POld(p),WaveFD.PSpace(p)

    # ginsu'd earth model
    sub!(m_ginsu, kwargs[:ginsu], kwargs[:mₒ], extend=true)
    sub!(b_ginsu, kwargs[:ginsu], kwargs[:b], extend=true)

    δm_ginsu = zeros(Float32, nz_ginsu, nx_ginsu)

    ginsu_interior_range = interior(kwargs[:ginsu])

    # Get receiver interpolation coefficients
    local iz, ix, c
    if kwargs[:interpmethod] == :hicks
        iz, ix, c = WaveFD.hickscoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:rz], kwargs[:rx])
    else
        iz, ix, c = WaveFD.linearcoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:rz], kwargs[:rx])
    end
    for i = 1:length(c)
        for jx = 1:size(c[i], 2), jz = 1:size(c[i], 1)
            kz = iz[i][jz]
            kx = ix[i][jx]
            c[i][jz,jx] *= kwargs[:dtmod]^2 * m_ginsu[kz,kx]^2 / b_ginsu[kz,kx]
        end
    end
    blks = WaveFD.source_blocking(nz_ginsu, nx_ginsu, kwargs[:nbz_inject], kwargs[:nbx_inject], iz, ix, c)

    ntmod = WaveFD.default_ntmod(kwargs[:dtrec], kwargs[:dtmod], kwargs[:ntrec])

    # if necessary, re-run the nonlinear forward
    isvalid, _chksum = isvalid_srcfieldfile(kwargs[:mₒ], kwargs[:srcfieldhost][], kwargs[:srcfieldfile]*"-P", kwargs[:chksum][])
    if !isvalid
        JopProp2DAcoIsoDenQ_DEO2_FDTD_nonlinearforward!(Array{Float32}(undef,0,0), kwargs[:mₒ]; kwargs...)
        kwargs[:srcfieldhost][] = gethostname()
        kwargs[:chksum][] = _chksum
    end

    # local disk file for source wavefield deserialization
    iofield = open(kwargs[:srcfieldfile]*"-DP")
    open(kwargs[:compressor]["DP"])
    DP = zeros(Float32, nz_ginsu, nx_ginsu)

    # adjoint interpolation of receiver wavefield
    δdinterp = zeros(Float32, ntmod, size(δd, 2))
    WaveFD.interpadjoint!(WaveFD.interpfilters(kwargs[:dtmod], kwargs[:dtrec], 0, WaveFD.LangC(), kwargs[:nthreads]), δdinterp, δd)

    itskip = round(Int, kwargs[:dtrec]/kwargs[:dtmod])
    kwargs[:reportinterval] == 0 || @info "linear adjoint on $(gethostname()), srcfieldfile=$(kwargs[:srcfieldfile])"
    time1 = time()
    cumtime_io, cumtime_ex, cumtime_im = 0.0, 0.0, 0.0

    set_zero_subnormals(true)
    for it = ntmod:-1:1
        if kwargs[:reportinterval] != 0 && (it % kwargs[:reportinterval] == 0 || it == ntmod)
            JopProp2DAcoIsoDenQ_DEO2_FDTD_write_history_ln(kwargs[:ginsu], it, ntmod, time()-time1, cumtime_io, cumtime_ex, cumtime_im, pcur, δdinterp, "adjoint")
        end

        # propagate and wavefield swap
        WaveFD.propagateadjoint!(p)
        pcur,pold = pold,pcur

        # inject receiver data
        cumtime_ex += @elapsed WaveFD.injectdata!(pcur, blks, δdinterp, it)

        if rem(it-1,itskip) == 0
            # read source field from disk
            cumtime_io += @elapsed if kwargs[:isinterior]
                WaveFD.compressedread!(iofield, kwargs[:compressor]["DP"], div(it-1,itskip)+1, DP, ginsu_interior_range)
            else
                WaveFD.compressedread!(iofield, kwargs[:compressor]["DP"], div(it-1,itskip)+1, DP)
            end

            # born accumulation
            cumtime_im += @elapsed WaveFD.adjointBornAccumulation!(p, δm_ginsu, DP)
        end
    end
    JopProp2DAcoIsoDenQ_DEO2_FDTD_stats(kwargs[:stats], kwargs[:ginsu], ntmod, time()-time1, cumtime_io, cumtime_ex, cumtime_im)
    set_zero_subnormals(false)

    # undo ginsu
    super!(δm, kwargs[:ginsu], δm_ginsu, accumulate=false)

    close(iofield)
    close(kwargs[:compressor]["DP"])

    free(p)

    δm
end

function srcillum!(γ, A::T, m::AbstractArray{Float32}) where {D,R,J<:Jet{D,R,typeof(JopProp2DAcoIsoDenQ_DEO2_FDTD_f!)},T<:Jop{J}}
    s = state(A)
    isvalid, _chksum = isvalid_srcfieldfile(m, s.srcfieldhost[], s.srcfieldfile*"-P", s.chksum[])
    if !isvalid
        JopProp2DAcoIsoDenQ_DEO2_FDTD_nonlinearforward!(Array{Float32}(undef,0,0), m; s...)
        s.chksum[] = _chksum
        s.srcfieldhost[] = gethostname()
    end
    open(s.compressor["P"])
    I = srcillum!(γ, s.srcfieldfile*"-P", s.compressor["P"], s.isinterior, s.ginsu, s.ntrec, s.nthreads)
    close(s.compressor["P"])
    I
end

function JopProp2DAcoIsoDenQ_DEO2_FDTD_stats(stats, ginsu, it, cumtime_total, cumtime_io, cumtime_ex, cumtime_im=0.0)
    stats["MCells/s"] = megacells_per_second(size(ginsu)..., it-1, cumtime_total)
    stats["%io"] = cumtime_total > 0 ? cumtime_io/cumtime_total*100.0 : 0.0
    stats["%inject/extract"] = cumtime_total > 0 ? cumtime_ex/cumtime_total*100.0 : 0.0
    stats["%imaging"] = cumtime_total > 0 ? cumtime_im/cumtime_total*100.0 : 0.0
end

Jets.perfstat(J::T) where {D,R,T<:Jet{D,R,typeof(JopProp2DAcoIsoDenQ_DEO2_FDTD_f!)}} = state(J).stats

@inline function JopProp2DAcoIsoDenQ_DEO2_FDTD_write_history_ln(ginsu, it, ntmod, cumtime_total, cumtime_io, cumtime_ex, cumtime_im, pcur, d::AbstractArray{T}, mode) where {T}
    itd = occursin("adjoint", mode) ? ntmod-it+1 : it
    rmsp = sqrt(norm(pcur)^2 / length(pcur))
    rmsd = d != nothing ? sqrt(norm(d)^2 / length(d)) : zero(T)
    @info @sprintf("PropLn2DAcoIsoDenQ_DEO2_FDTD, %s, time step %5d of %5d ; %7.2f MCells/s (IO=%5.2f%%, EX=%5.2f%%, IM=%5.2f%%) -- rms d,p; %10.4e %10.4e", mode, itd, ntmod,
                    megacells_per_second(size(ginsu)..., itd-1, cumtime_total),
                    cumtime_total > 0 ? cumtime_io/cumtime_total*100.0 : 0.0,
                    cumtime_total > 0 ? cumtime_ex/cumtime_total*100.0 : 0.0,
                    cumtime_total > 0 ? cumtime_im/cumtime_total*100.0 : 0.0, rmsd, rmsp)
end

@inline function JopProp2DAcoIsoDenQ_DEO2_FDTD_write_history_nl(ginsu, it, ntmod, cumtime_total, cumtime_io, cumtime_ex, pcur, d::AbstractArray{T}) where {T}
    rmsp = sqrt(norm(pcur)^2 / length(pcur))
    rmsd = length(d) > 0 ? sqrt(norm(d)^2 / length(d)) : zero(T)
    @info @sprintf("Prop2DAcoIsoDenQ_DEO2_FDTD, nonlinear forward, time step %5d of %5d ; %7.2f MCells/s (IO=%5.2f%%, EX=%5.2f%%) -- rms d,p; %10.4e %10.4e", it, ntmod,
                    megacells_per_second(size(ginsu)..., it-1, cumtime_total),
                    cumtime_total > 0 ? cumtime_io/cumtime_total*100.0 : 0.0,
                    cumtime_total > 0 ? cumtime_ex/cumtime_total*100.0 : 0.0, rmsd, rmsp)
end

function Base.close(jet::Jet{D,R,typeof(JopProp2DAcoIsoDenQ_DEO2_FDTD_f!)}) where {D,R}
    rm(string(state(jet).srcfieldfile, "-P"), force=true)
    rm(string(state(jet).srcfieldfile, "-DP"), force=true)
end
