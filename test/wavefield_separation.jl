#=
These unit tests for the 3 types of imaging condition are a little bit heuristic. We use two 
half-space models with the upper half-plane velocity greater in the model we compute the 
gradients in, so that we expect all positive gradient.

Tests:

1. Ensure gradients from different imaging conditions are different
    a. test: norm(g_STD .- g_FWI) > ϵ
    b. test: norm(g_FWI .- g_RTM) > ϵ

2. For FWI, we expect longer wavelengths, so a larger sum after maximum value normalization
    a. normalize to maximum value unity
    b. test: sum(sign(g_FWI)) > sum(sign(g_STD)) 

3. For RTM, we test the correlation of an image with that image after high-pass filter. 
   With the RTM imaging condition, there should be less low wavenumber energy, hence this 
   correlation should be larger for RTM IC.
   a. model in water velocity and subtract to remove direct wave
   b. test: cc(g_RTM,highpass * g_RTM) > cc(g_STD,highpass * g_STD)

Note these tests are skipped currently as serialized wavefields for the imaging condition 
unit tests are too large for github CI.
=#
using WaveFD, JetPackWaveFD, JetPack, Jets, Random, LinearAlgebra, Test

fpeak = 20
dtmod,dtrec = 0.001,0.005
ntrec,nsponge,reportinterval = 251,40,0
z,y,x,δz,δy,δx = 600.0,700.0,800.0,10.0,10.0,10.0
nz,ny,nx = round(Int,z/δz)+1,round(Int,y/δy)+1,round(Int,x/δx)+1

function makeF(; dim::Int=2, physics::String="ISO", imgcondition::String="standard", RTM_weight=0.25)
    if dim == 2
        sz = δz * 1;
        sx = δx * div(nx,2);
        rz = δz .* ones(Float32, nx);
        rx = δx .* Float32[0:(nx-1);];
    else
        sz = δz * 1;
        sy = δy * div(ny,2);
        sx = δx * div(nx,2);
        rz = δz .* ones(Float32, nx*ny);
        ry = (δx .* ones(Float32, nx) * Float32[0:ny-1;]')[:];
        rx = (δx .* Float32[0:nx-1;] * ones(Float32, ny)')[:];
    end

    if dim == 2 && physics == "ISO"
        b = 0.001f0 .* ones(Float32,nz,nx)

        JopNlProp2DAcoIsoDenQ_DEO2_FDTD(; b = b, 
            comptype = UInt32, compscale = 1e-1,
            sz = sz, sx = sx, rz = rz, rx = rx, dz = δz, dx = δx,
            dtmod = dtmod, dtrec = dtrec, ntrec = ntrec, nsponge = nsponge,
            wavelet = WaveletCausalRicker(f=fpeak),
            imgcondition = imgcondition, RTM_weight = RTM_weight, reportinterval = reportinterval)

    elseif dim == 2 && physics == "VTI"
        b = 0.001f0 .* ones(Float32,nz,nx)
        ϵ = 0.1f0 .* ones(Float32,nz,nx)
        η = 0.2f0 .* ones(Float32,nz,nx)

        JopNlProp2DAcoVTIDenQ_DEO2_FDTD(; b = b, ϵ = ϵ, η = η,  
            comptype = UInt32, compscale = 1e-1,
            sz = sz, sx = sx, rz = rz, rx = rx, dz = δz, dx = δx,
            dtmod = dtmod, dtrec = dtrec, ntrec = ntrec, nsponge = nsponge,
            wavelet = WaveletCausalRicker(f=fpeak),
            imgcondition = imgcondition, RTM_weight = RTM_weight, reportinterval = reportinterval)

    elseif dim == 2 && physics == "TTI"
        b = 0.001f0 .* ones(Float32,nz,nx)
        ϵ = 0.1f0 .* ones(Float32,nz,nx)
        η = 0.2f0 .* ones(Float32,nz,nx)
        θ = Float32(π/8) .* ones(Float32,nz,nx)

        JopNlProp2DAcoTTIDenQ_DEO2_FDTD(; b = b, ϵ = ϵ, η = η, θ = θ, 
            comptype = UInt32, compscale = 1e-1,
            sz = sz, sx = sx, rz = rz, rx = rx, dz = δz, dx = δx,
            dtmod = dtmod, dtrec = dtrec, ntrec = ntrec, nsponge = nsponge,
            wavelet = WaveletCausalRicker(f=fpeak),
            imgcondition = imgcondition, RTM_weight = RTM_weight, reportinterval = reportinterval)

    elseif dim == 3 && physics == "ISO"
        b = 0.001f0 .* ones(Float32,nz,ny,nx)

        JopNlProp3DAcoIsoDenQ_DEO2_FDTD(; b = b,  
            comptype = UInt32, compscale = 1e-1,
            sz = sz, sy = sy, sx = sx, rz = rz, ry = ry, rx = rx, dz = δz, dy = δy, dx = δx,
            dtmod = dtmod, dtrec = dtrec, ntrec = ntrec, nsponge = nsponge,
            wavelet = WaveletCausalRicker(f=fpeak),
            imgcondition = imgcondition, RTM_weight = RTM_weight, reportinterval = reportinterval)

    elseif dim == 3 && physics == "VTI"
        b = 0.001f0 .* ones(Float32,nz,ny,nx)
        ϵ = 0.1f0 .* ones(Float32,nz,ny,nx)
        η = 0.2f0 .* ones(Float32,nz,ny,nx)

        JopNlProp3DAcoVTIDenQ_DEO2_FDTD(; b = b, ϵ = ϵ, η = η,  
            comptype = UInt32, compscale = 1e-1,
            sz = sz, sy = sy, sx = sx, rz = rz, ry = ry, rx = rx, dz = δz, dy = δy, dx = δx,
            dtmod = dtmod, dtrec = dtrec, ntrec = ntrec, nsponge = nsponge,
            wavelet = WaveletCausalRicker(f=fpeak),
            imgcondition = imgcondition, RTM_weight = RTM_weight, reportinterval = reportinterval)

    elseif dim == 3 && physics == "TTI"
        b = 0.001f0 .* ones(Float32,nz,ny,nx)
        ϵ = 0.1f0 .* ones(Float32,nz,ny,nx)
        η = 0.2f0 .* ones(Float32,nz,ny,nx)
        θ = Float32(π/8) .* ones(Float32,nz,ny,nx)
        ϕ = Float32(π/6) .* ones(Float32,nz,ny,nx)

        JopNlProp3DAcoTTIDenQ_DEO2_FDTD(; b = b, ϵ = ϵ, η = η, θ = θ, ϕ = ϕ, 
            comptype = UInt32, compscale = 1e-1,
            sz = sz, sy = sy, sx = sx, rz = rz, ry = ry, rx = rx, dz = δz, dy = δy, dx = δx,
            dtmod = dtmod, dtrec = dtrec, ntrec = ntrec, nsponge = nsponge,
            wavelet = WaveletCausalRicker(f=fpeak),
            imgcondition = imgcondition, RTM_weight = RTM_weight, reportinterval = reportinterval)
    end
end

@testset "ImgCondition" begin
    @testset "Imaging Condition 2D tests, $(physics)" for physics in ("ISO", "VTI", "TTI")
        write(stdout, "\n")
        write(stdout, "Imaging conditions 2D, physics=$(physics)\n")

        opStd = makeF(;dim=2, physics=physics, imgcondition="standard");
        opFwi = makeF(;dim=2, physics=physics, imgcondition="FWI");
        opRtm = makeF(;dim=2, physics=physics, imgcondition="RTM");
        opMix = makeF(;dim=2, physics=physics, imgcondition="MIX", RTM_weight=0.25);

        vw = 1500 .* ones(domain(opStd)); # water velocity model
        va = 1500 .* ones(domain(opStd));
        vb = 1505 .* ones(domain(opStd));
        va[div(nz,2):end,:,:] .= 2500;
        vb[div(nz,2):end,:,:] .= 2500;

        d1a = opStd * va;
        d1w = opStd * vw;

        # standard FWI
        d1b = opStd * vb;
        r1  = d1b .- d1a;
        J1 = jacobian!(opStd,vb);
        g1 = J1' * r1;

        # wavefield separation FWI
        d2b = opFwi * vb;
        r2  = d2b .- d1a;
        J2 = jacobian!(opFwi,vb);
        g2 = J2' * r2;

        # standard RTM
        d3b = opRtm * vb;
        r3 = d1a .- d1w
        J3 = jacobian!(opStd,vb);
        g3 = J3' * r3;

        # wavefield separation RTM
        d4b = opRtm * vb;
        r4 = d1a .- d1w
        J4 = jacobian!(opRtm,vb);
        g4 = J4' * r4;

        # wavefield seperation mix
        d5b = opStd * vb
        r5 = d5b .- d1a
        J5 = jacobian!(opMix,vb)
        g5 = J5' * r5

        close(opStd)
        close(opFwi)
        close(opRtm)
        close(opMix)
        
        # remove pixels near source to eliminate very large amplitudes
        nzero = 5
        g1[1:nzero,:,:] .= 0
        g2[1:nzero,:,:] .= 0
        g3[1:nzero,:,:] .= 0
        g4[1:nzero,:,:] .= 0
        g5[1:nzero,:,:] .= 0

        # FWI gradients differ
        @test norm(g1 .- g2) / norm(g1) > eps(Float32) 

        # RTM gradients differ
        @test norm(g3 .- g4) / norm(g3) > eps(Float32) 

        # MIX gradients differ
        @test norm(g1 .- g5) / norm(g5) > eps(Float32)

        # FWI gradient longer wavelength (more positive) than standard gradient
        @show sum(sign.(g1)), sum(sign.(g2))
        @test sum(sign.(g2)) > sum(sign.(g1))

        # cc(g_RTM,highpass,g_RTM) > cc(g_STD,highpass,g_STD)
        op = JopLaplacian(JetSpace(Float32,nz,nx)) ∘ JopReshape(domain(opStd), JetSpace(Float32,nz,nx))
        h3 = op * g3;
        h4 = op * g4;
        cc3 = dot(abs.(g3[:]),abs.(h3[:])) / sqrt(dot(abs.(g3[:]),abs.(g3[:])) * dot(abs.(h3[:]),abs.(h3[:])))
        cc4 = dot(abs.(g4[:]),abs.(h4[:])) / sqrt(dot(abs.(g4[:]),abs.(g4[:])) * dot(abs.(h4[:]),abs.(h4[:])))
        @show cc3, cc4
        @test cc4 > cc3
    end

    @testset "Imaging Condition 3D tests, $(physics)" for physics in ("ISO", "VTI", "TTI")
        write(stdout, "\n")
        write(stdout, "Imaging conditions 3D, physics=$(physics)\n")

        opStd = makeF(;dim=3, physics=physics, imgcondition="standard");
        opFwi = makeF(;dim=3, physics=physics, imgcondition="FWI");
        opRtm = makeF(;dim=3, physics=physics, imgcondition="RTM");
        opMix = makeF(;dim=3, physics=physics, imgcondition="MIX", RTM_weight=0.25);

        vw = 1500 .* ones(domain(opStd)); # water velocity model
        va = 1500 .* ones(domain(opStd));
        vb = 1505 .* ones(domain(opStd));
        va[div(nz,2):end,:,:,:] .= 2500;
        vb[div(nz,2):end,:,:,:] .= 2500;

        d1a = opStd * va;
        d1w = opStd * vw;

        # standard FWI
        d1b = opStd * vb;
        r1  = d1b .- d1a;
        J1 = jacobian!(opStd,vb);
        g1 = J1' * r1;

        # wavefield separation FWI
        d2b = opFwi * vb;
        r2  = d2b .- d1a;
        J2 = jacobian!(opFwi,vb);
        g2 = J2' * r2;

        # standard RTM
        d3b = opRtm * vb;
        r3 = d1a .- d1w
        J3 = jacobian!(opStd,vb);
        g3 = J3' * r3;

        # wavefield separation RTM
        d4b = opRtm * vb;
        r4 = d1a .- d1w
        J4 = jacobian!(opRtm,vb);
        g4 = J4' * r4;

        # wavefield seperation mix
        d5b = opStd * vb
        r5 = d5b .- d1a
        J5 = jacobian!(opMix,vb)
        g5 = J5' * r5

        close(opStd)
        close(opFwi)
        close(opRtm)
        close(opMix)

        # remove pixels near source to eliminate very large amplitudes there
        nzero = 5
        g1[1:nzero,:,:,:] .= 0
        g2[1:nzero,:,:,:] .= 0
        g3[1:nzero,:,:,:] .= 0
        g4[1:nzero,:,:,:] .= 0

       # FWI gradients differ
       @test norm(g1 .- g2) / norm(g1) > eps(Float32) 

       # RTM gradients differ
       @test norm(g3 .- g4) / norm(g3) > eps(Float32) 

       # MIX gradients differ
       @test norm(g1 .- g5) / norm(g5) > eps(Float32)

       # FWI gradient longer wavelength (more positive) than standard gradient
       @show sum(sign.(g1)), sum(sign.(g2))
       @test sum(sign.(g2)) > sum(sign.(g1))

       # cc(g_RTM,highpass,g_RTM) > cc(g_STD,highpass,g_STD)
       op = JopLaplacian(JetSpace(Float32,nz,ny,nx)) ∘ JopReshape(domain(opStd), JetSpace(Float32,nz,ny,nx))
       h3 = op * g3;
       h4 = op * g4;
       cc3 = dot(abs.(g3[:]),abs.(h3[:])) / sqrt(dot(abs.(g3[:]),abs.(g3[:])) * dot(abs.(h3[:]),abs.(h3[:])))
       cc4 = dot(abs.(g4[:]),abs.(h4[:])) / sqrt(dot(abs.(g4[:]),abs.(g4[:])) * dot(abs.(h4[:]),abs.(h4[:])))
       @show cc3, cc4
       @test cc4 > cc3
    end
end