using Formatting, FFTW, Jets, JetPackWaveFD, LinearAlgebra, SpecialFunctions, Statistics, Test, WaveFD

function make_op(interpmethod, fs; comptype = Float32)
    nsponge = 10
    pad = 20
    nx = 60+2*pad
    nz = 50+2*pad
    nt = 101
    dx,dz,dt = 25.0,25.0,0.002
    xmin,zmin,tmin = -pad*dx,-pad*dz,0.0
    xmax,zmax,tmax = xmin+dx*(nx-pad-1),zmin+dz*(nz-pad-1),tmin+dt*(nt-1)
    sx = dx*div(nx,2)
    sz = dz
    rx = dx*[pad:nx-pad-1;]
    rz = 2*dz*ones(length(rx))

    wavelet = WaveletCausalRicker(f=5.0)

    b = ones(Float32,nz,nx)
    v = 1500 .* ones(Float32,nz,nx)

    F = JopNlProp2DAcoIsoDenQ_DEO2_FDTD(b = b, 
        sx = sx, sz = sz, rx = rx, rz = rz, dx = dx, dz = dz, z0 = zmin, x0 = xmin, 
        dtrec = dt, dtmod = dt, ntrec = nt, nsponge = nsponge, comptype = comptype, compscale = 1e-4,
        freqQ = 5.0, qMin = 0.1, qInterior = 100.0, wavelet = wavelet, freesurface = fs, 
        reportinterval = 0, interpmethod = interpmethod, isinterior = false)

    v,F
end

@testset "JopProp2DAcoIsoDenQ_DEO2_FDTD -- close" begin
    m₀, F = make_op(:hicks, false)
    d = F * m₀
    @test isfile(state(F).srcfieldfile*"-P")
    @test isfile(state(F).srcfieldfile*"-DP")
    close(F)
    @test !(isfile(state(F).srcfieldfile*"-P"))
    @test !(isfile(state(F).srcfieldfile*"-DP"))
    d = F * m₀
    @test isfile(state(F).srcfieldfile*"-P")
    @test isfile(state(F).srcfieldfile*"-DP")
    close(F)
    @test !(isfile(state(F).srcfieldfile*"-P"))
    @test !(isfile(state(F).srcfieldfile*"-DP"))

    close(F)
end

@testset "JopProp2DAcoIsoDenQ_DEO2_FDTD -- perfstat" begin
    m₀, F = make_op(:hicks, false)
    
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

@testset "JopProp2DAcoIsoDenQ_DEO2_FDTD -- linearization, interpmethod=$interpmethod, fs=$fs" for interpmethod in (:hicks,:linear), fs in (false,true)
    m₀, F = make_op(interpmethod, fs)

    μobs, μexp = linearization_test(F,m₀,
        μ = 100*sqrt.([1.0,1.0/2.0,1.0/4.0,1.0/8.0,1.0/16.0,1.0/32.0,1.0/64.0,1.0/128.0,1.0/256.0,1.0/512.0]))

    @show μobs, μexp
    @show minimum(abs, μobs - μexp)
    @test minimum(abs, μobs - μexp) < 0.01

    close(F)
end

@testset "JopProp2DAcoIsoDenQ_DEO2_FDTD -- dot product, interpmethod=$interpmethod, fs=$fs" for interpmethod in (:hicks,:linear), fs in (true,false)
    m₀, F = make_op(interpmethod, fs)
    J = jacobian!(F, m₀)
    m = -1 .+ 2*rand(domain(J))
    d = -1 .+ 2*rand(range(J))

    lhs, rhs = dot_product_test(J, m, d)
    err = abs(lhs - rhs) / abs(lhs + rhs)
    @show lhs,rhs,(lhs-rhs)/(lhs+rhs)
    @test lhs ≈ rhs rtol=1e-4
    @test err < 1e-4

    close(F)
end

@testset "JopProp2DAcoIsoDenQ_DEO2_FDTD -- linearity, interpmethod=$interpmethod, fs=$fs" for interpmethod in (:linear,:hicks), fs in (true,false)
    m₀, F = make_op(interpmethod, fs)
    J = jacobian!(F, m₀) 
    lhs,rhs = linearity_test(J)
    @test lhs ≈ rhs

    close(F)
end

# note the compression is exercised on the second pass of F * m₀
@testset "JopProp2DAcoIsoDenQ_DEO2_FDTD -- serialization, C=$C" for C in (Float32, UInt32)
    m₀, F = make_op(:hicks, false, comptype = C)
    d₁ = F * m₀
    d₂ = F * m₀
    @test d₁ ≈ d₂

    close(F)
end

#=
@testset "JopProp2DAcoIsoDenQ_DEO2_FDTD -- analytic, direct, interpmethod=$interpmethod" for interpmethod in (:hicks, :linear)
    z,x,dz,dx,dtrec,dtmod,t,sz,sx,c = 0.5,1.0,0.02,0.02,0.004,0.0004,1.0,0.25,0.02,1.5
    nz,nx,ntrec = round(Int,z/dz)+1,round(Int,x/dx)+1,round(Int,t/dtrec)+1

    fmax = c/6/max(dx,dz)
    fdef = WaveletCausalRicker(f=fmax/2);

    function analytic_result()
        # non-negative frequency axis
        nt_pad = nextprod([2,3,5,7],4*ntrec)
        nw = div(nt_pad,2)+1
        w = range(0, stop=pi/dtrec, length=nw)

        # wavelet
        f = WaveFD.get(fdef, dtrec*collect(0:nt_pad-1))
        F = rfft(f)

        # Morse and Feshbach, 1953, p. 891
        u_ana = zeros(Float32,ntrec,nz,nx)
        U = zeros(Complex{Float64},nw)
        for ix = 1:nx, iz = 1:nz
            z = (iz-1)*dz
            x = (ix-1)*dx
            r = sqrt((z-sz)^2+(x-sx)^2)
            for iw = 2:nw
                if r*w[iw] > eps(Float32)
                    U[iw] = -im*pi*hankelh2(0,w[iw]/c*r)*F[iw]
                end
            end
            u_ana[:,iz,ix] = irfft(U,nt_pad)[1:ntrec]
            printfmt("{.2f}\r", ((ix-1)*nz+iz-1)/((nz-1)*(nx-1))*100))
        end
        u_ana
    end
    write(stdout, "computing analytic result (be patient, can take several minutes)...\n")
    u_ana = analytic_result()
    write(stdout, "...done.\n")

    # Finite difference
    M = JopNlProp2DAcoIsoDenQ_DEO2_FDTD(
        b = (1.0f0/2.0f0)*ones(Float32,nz,nx),
        sx = sx,
        sz = sz,
        rz = sz*ones(nx),
        rx = dx*collect(0:nx-1),
        dx = dx,
        dz = dz,
        dtrec = dtrec,
        dtmod = dtmod,
        ntrec = ntrec,
        wavelet = fdef,
        nsponge = 200,
        freqQ = 5.0,
        qMin = 0.1,
        qInterior = 100.0,
        freesurface = false,
        reportinterval = 0,
        isinterior = true)

    vp = c*ones(nz,nx)
    write(stdout, "computing numerical result (be patient)...\n")
    d_tmp = M*vp
    write(stdout, "...done.\n")

    # Read FD field back from disk
    u_fd = zeros(Float32,ntrec,nz,nx)
    u_snap = zeros(Float32,nz,nx)
    u_snap_ginsu = zeros(Float32,size(state(M).ginsu,interior=true)...)
    io = open("$(state(M).srcfieldfile)-P")
    for it = 1:state(M).ntrec
        read!(io,u_snap_ginsu)
        super!(u_snap,state(M).ginsu,u_snap_ginsu,interior=true)
        u_fd[it,:,:] = u_snap[:,:]
    end
    close(io)

    # compute Pearson correlation (https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient)
    xx = zeros(nz,nx)
    for ix = 1:nx, iz = 1:nz
        xx[iz,ix] = cor(u_fd[:,iz,ix],u_ana[:,iz,ix])
    end
    xx_mean = mean(xx)
    @test xx_mean > .9

    # measure geometric spreading
    amp_ana = zeros(nz,nx)
    amp_fd = zeros(nz,nx)
    for ix = 1:nx, iz = 1:nz
        amp_ana[iz,ix] = maximum(u_ana[:,iz,ix])
        amp_fd[iz,ix] = maximum(u_fd[:,iz,ix])
    end

    # don't trust the values at the injection point
    szi = round(Int,sz/dz)+1
    sxi = round(Int,sx/dx)+1
    amp_fd[szi,sxi] = 0.0
    amp_ana[szi,sxi] = 0.0

    # we only care about relative values
    amp_fd /= maximum(amp_fd)
    amp_ana /= maximum(amp_ana)
    @test cor(amp_fd[:],amp_ana[:]) > .99

    close(M)
end
=#