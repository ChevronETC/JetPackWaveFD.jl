using Formatting, FFTW, Jets, JetPackWaveFD, LinearAlgebra, SpecialFunctions, Statistics, Test, WaveFD

function make_op(modeltype, interpmethod, fs; v₀=1500, ϵ₀=0.2, η₀=0.4, comptype = Float32)
    nsponge = 10
    pad = 20
    nx = 50+2*pad
    ny = 40+2*pad
    nz = 30+2*pad
    nt = 101
    dx,dy,dz,dt = 25.0,25.0,25.0,0.002
    xmin,ymin,zmin,tmin = -pad*dx,-pad*dy,-pad*dz,0.0
    xmax,ymax,zmax,tmax = xmin+dx*(nx-pad-1),ymin+dy*(ny-pad-1),zmin+dz*(nz-pad-1),tmin+dt*(nt-1)
    sx = dx*div(nx,2)
    sy = dy*div(ny,2)
    sz = dz
    rz = [dz for iy = 1+pad:ny-pad, ix = 1+pad:nx-pad][:]
    ry = [(iy-1)*dy for iy = 1+pad:ny-pad, ix=1+pad:nx-pad][:]
    rx = [(ix-1)*dx for iy = 1+pad:ny-pad, ix=1+pad:nx-pad][:]

    wavelet = WaveletCausalRicker(f=5.0)

    local ϵ,η
    if modeltype  == "v"
        ϵ = zeros(Float32,nz,ny,nx)
        η = zeros(Float32,nz,ny,nx)
    else
        ϵ = Float32[]
        η = Float32[]
    end
    
    b = ones(Float32,nz,ny,nx)

    F = JopNlProp3DAcoVTIDenQ_DEO2_FDTD(b = b, f = 0.85, ϵ = ϵ, η = η, 
        z0 = zmin, y0 = ymin, x0 = xmin, dz = dz, dy = dy, dx = dx, 
        sz = sz, sy = sy, sx = sx, rz = rz, ry = ry, rx = rx, 
        dtrec = dt, dtmod = dt, ntrec = nt, nsponge = nsponge, comptype = comptype, compscale = 1e-4,
        freqQ = 5.0, qMin = 0.1, qInterior = 100.0, wavelet = wavelet, freesurface = fs, 
        reportinterval = 0, interpmethod = interpmethod, isinterior = false)

    m₀ = zeros(domain(F))
    m₀[:,:,:,modelindex(F,"v")] .= v₀
    if modeltype == "vϵη"
        m₀[:,:,:,modelindex(F,"ϵ")] .= ϵ₀
        m₀[:,:,:,modelindex(F,"η")] .= η₀
    end

    m₀, F
end

@testset "JopProp3DAcoVTIDenQ_DEO2_FDTD -- close" begin
    m₀, F = make_op("vϵη", :hicks, false) 
    d = F * m₀

    @test isfile(state(F).srcfieldfile*"-pspace")
    @test isfile(state(F).srcfieldfile*"-mspace")
    close(F)
    @test !(isfile(state(F).srcfieldfile*"-pspace"))
    @test !(isfile(state(F).srcfieldfile*"-mspace"))
    d = F * m₀
    @test isfile(state(F).srcfieldfile*"-pspace")
    @test isfile(state(F).srcfieldfile*"-mspace")
    close(F)
    @test !(isfile(state(F).srcfieldfile*"-pspace"))
    @test !(isfile(state(F).srcfieldfile*"-mspace"))

    close(F)
end

@testset "JopProp3DAcoVTIDenQ_DEO2_FDTD -- perfstat" begin
    m₀, F = make_op("vϵη", :hicks, false) 
    d = F * m₀

    s = Jets.perfstat(F)
    @test isa(s["MCells/s"], Float64)
    @test isa(s["%io"], Float64)
    @test isa(s["%inject/extract"], Float64)
    @test s["%imaging"] ≈ 0.0

    J = jacobian!(F, m₀)
    d = J * m₀
    s = Jets.perfstat(J)
    @test isa(s["MCells/s"], Float64)
    @test isa(s["%io"], Float64)
    @test isa(s["%inject/extract"], Float64)
    @test isa(s["%imaging"], Float64)

    m = J' * d
    s = Jets.perfstat(J)
    @test isa(s["MCells/s"], Float64)
    @test isa(s["%io"], Float64)
    @test isa(s["%inject/extract"], Float64)
    @test isa(s["%imaging"], Float64)

    close(F)
end

@testset "JopProp3DAcoVTIDenQ_DEO2_FDTD -- linearization, model type is $modeltype, interpmethod=$interpmethod, fs=$fs" for modeltype in ("vϵη","v"), interpmethod in (:linear,:hicks), fs in (true,false)
    m₀, F = make_op(modeltype, interpmethod, fs) 
    d = F * m₀

    mmask = zeros(domain(F))
    mmask[:,:,:,modelindex(F,"v")] .= 1
    if modeltype == "vϵη"
        mmask[:,:,:,modelindex(F,"ϵ")] .= 0.001
        mmask[:,:,:,modelindex(F,"η")] .= 0.001
    end

    μobs, μexp = linearization_test(F, m₀,
        mmask = mmask,
        μ = 100*sqrt.([1.0,1.0/2,1.0/4,1.0/8,1.0/16,1.0/32,1.0/64,1.0/128,1.0/256,1.0/512]))

    @show μobs, μexp
    @show minimum(abs, μobs - μexp)
    @test minimum(abs, μobs - μexp) < 0.01

    close(F)
end

@testset "JopProp3DAcoVTIDenQ_DEO2_FDTD -- dot product, model type is $modeltype, interpmethod=$interpmethod, fs=$fs" for modeltype in ("vϵη","v"), interpmethod in (:linear, :hicks), fs in (true,false)
    m₀, F = make_op(modeltype, interpmethod, fs) 
    J = jacobian!(F, m₀)
    m = zeros(domain(J))
    m[:,:,:,modelindex(F,"v")] .= 100 * (-1 .+ 2*rand(Float32,size(m)[1:3]))
    if modeltype == "vϵη"
        m[:,:,:,modelindex(F,"ϵ")] .= rand(Float32,size(m)[1:3]...)
        m[:,:,:,modelindex(F,"η")] .= rand(Float32,size(m)[1:3]...)
    end
    d = -1 .+ 2*rand(range(J))

    lhs, rhs = dot_product_test(J, m, d)
    err = abs(lhs - rhs) / abs(lhs + rhs)
    @show lhs,rhs,(lhs-rhs)/(lhs+rhs)
    @test lhs ≈ rhs rtol=1e-4
    @test err < 1e-4

    close(F)
end

@testset "JopProp3DAcoVTIDenQ_DEO2_FDTD -- linearity, model type is $modeltype, interpmethod=$interpmethod, fs=$fs" for modeltype in ("vϵη","v"), interpmethod in (:linear,:hicks), fs in (true,false)
    m₀, F = make_op(modeltype, interpmethod, fs) 
    J = jacobian!(F, m₀)
    lhs,rhs = linearity_test(J)
    @test lhs ≈ rhs

    close(F)
end

# note the compression is exercised on the second pass of F * m₀
@testset "JopProp2DAcoVTIDenQ_DEO2_FDTD -- serialization, modeltype=$modeltype, C=$C" for modeltype in ("vϵη", "v"), C in (Float32,UInt32)
    m₀, F = make_op(modeltype, :hicks, false, comptype = C) 
    d₁ = F * m₀
    d₂ = F * m₀
    @test d₁ ≈ d₂

    close(F)
end

#=
@testset "JopProp3DAcoVTIDenQ_DEO2_FDTD -- analytic, direct, for model type $modeltype, interpmethod=$interpmethod" for modeltype in ("vϵη","v"), interpmethod in (:linear,:hicks)
    z,y,x,dz,dy,dx,dtrec,dtmod,tmax,sz,sy,sx,c = 2000.0,500.0,2500.0,40.0,40.0,40.0,0.016,0.004,2.0,1000.0,250.0,20.0,1500.0
    nz,ny,nx,nt = round(Int,z/dz)+1,round(Int,y/dy)+1,round(Int,x/dx)+1,round(Int,tmax/dtrec)+1

    fmax = c/6/max(dx,dy,dz)
    fdef = WaveletCausalRicker(f=fmax/2)

    function analytic_result()
        # non-negative frequency axis
        nt_pad = nextprod([2,3,5,7],4*nt)
        nw = div(nt_pad,2)+1
        w = range(0, stop=pi/dtrec, length=nw)

        # wavelet
        f = WaveFD.get(fdef, dtrec*collect(0:nt_pad-1))
        F = rfft(f)

        # Morse and Feshbach, 1953, p. 891
        u_ana = zeros(Float32,nt,nz,ny,nx)
        U = zeros(Complex{Float64},nw)
        for ix = 1:nx, iy = 1:ny, iz = 1:nz
            z = (iz-1)*dz
            y = (iy-1)*dy
            x = (ix-1)*dx
            r = sqrt((z-sz)^2+(y-sy)^2+(x-sx)^2)
            for iw = 2:nw
                if r*w[iw] > eps(Float32)
                    U[iw] = exp(-im*w[iw]/c*r)*F[iw]
                end
            end
            u_ana[:,iz,iy,ix] = irfft(U,nt_pad)[1:nt]
            printfmt("{:.2f}", ((ix-1)*ny*nz+(iy-1)*nz+iz-1)/(nz*ny*nx)*100))
        end
        u_ana
    end
    @info "computing analytic result (be patient, can take several minutes)..."
    u_ana = analytic_result()
    @info "...done."

    local ϵ,η
    if modeltype == "v"
        ϵ = zeros(Float32,nz,ny,nx)
        η = zeros(Float32,nz,ny,nx)
    else
        ϵ = Float32[]
        η = Float32[]
    end

    rx = (dx*[0:nx-1;] * ones(ny)')[:]
    ry = (ones(nx) * (dy*[0:ny-1;])')[:]
    rz = 2*dz*ones(length(rx))

    # Finite difference
    M = JopNlProp3DAcoVTIDenQ_DEO2_FDTD(
        comptype = UInt32,
        compscale = 1e-3,
        srcfieldfile = "$_scratch/srcfield-$(randstring()).bin",
        b = (1.0f0/2.0f0)*ones(Float32,nz,ny,nx),
        f = 0.75,
        ϵ = ϵ,
        η = η,
        sz = sz,
        sy = sy,
        sx = sx,
        rz = rz,
        ry = ry,
        rx = rx,
        interpmethod = interpmethod,
        dz = dz,
        dy = dy,
        dx = dx,
        dtrec = dtrec,
        dtmod = dtmod,
        ntrec = nt,
        wavelet = fdef,
        nsponge = 200,
        freqQ = 5.0,
        qMin = 0.1,
        qInterior = 100.0,
        freesurface = false,
        reportinterval = 250)

    m = zeros(domain(M))
    m[:,:,:,modelindex(M,"v")] .= c
    @info "computing numerical result (be patient)"
    d_tmp = M*m
    @info "...done."

    # Read FD field back from disk
    u_fd = zeros(Float32,state(M).ntrec,nz,ny,nx)
    u_snap = zeros(Float32,nz,ny,nx)
    u_snap_ginsu = zeros(Float32,size(state(M).ginsu, interior=false)...)
    io = open("$(state(M).srcfieldfile)-pold")

    open(state(M).compressor["pold"])
    for i = 1:state(M).ntrec
        WaveFD.compressedread!(io, state(M).compressor["pold"], i, u_snap_ginsu)
        super!(u_snap,state(M).ginsu,u_snap_ginsu,interior=false)
        u_fd[i,:,:,:] = u_snap[:,:,:]
    end
    close(state(M).compressor["pold"])

    # compute Pearson correlation (https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient)
    xx = ones(nz,ny,nx)
    for ix = 1:nx, iy = 1:ny, iz = 1:nz
        xx[iz,iy,ix] = cor(u_fd[:,iz,iy,ix],u_ana[:,iz,iy,ix])
    end
    xx_mean = mean(xx)
    @test xx_mean > .9

    close(M)
    @info "be patient, the garbage collection is probably deleting large scratch-files"
end
=#