using Revise
using PyPlot
using Formatting, FFTW, Jets, JetPackWaveFD, LinearAlgebra, SpecialFunctions, Statistics, Test, WaveFD

modeltypes = (WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD_Model_V, WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD_Model_B, WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD_Model_VB)

npad = 50;
nx = 300;
nz = 300;
nt = 1501

dx,dz,dt = 25.0,25.0,0.002
xmin,zmin,tmin = 0.0,0.0,0.0
xmax,zmax,tmax = xmin+dx*(nx-1),zmin+dz*(nz-1),tmin+dt*(nt-1)
sx = dx * div(nx,2)
sz = 1 * dz
rx = dx*[npad:nx-npad-1;]
rz = 2 * dz .* ones(length(rx))

function make_op(interpmethod, modeltype, fs, v, b; comptype = Float32)
    wavelet = WaveletCausalRicker(f=5.0)

    local m
    if modeltype == WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD_Model_V
        kwargs = (b = reshape(b, nz, nx), )
        m = reshape(v, nz, nx, 1)
    elseif modeltype == WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD_Model_VB
        kwargs = (nz = nz, nx = nx)
        m = zeros(Float32, nz, nx, 2)
        m[:,:,1] .= reshape(v, nz, nx)
        m[:,:,2] .= reshape(b, nz, nx)
    elseif modeltype == WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD_Model_B
        kwargs = (v = reshape(v, nz, nx), )
        m = reshape(b, nz, nx, 1)
    end

    F = JopNlProp2DAcoIsoDenQ_DEO2_FDTD(; kwargs..., 
        z0 = zmin, x0 = xmin, dx = dx, dz = dz, sx = sx, sz = sz, rx = rx, rz = rz,
        dtrec = dt, dtmod = dt, ntrec = nt, nsponge = npad, comptype = comptype, compscale = 1e-4,
        freqQ = 5.0, qMin = 0.1, qInterior = 100.0, wavelet = wavelet, freesurface = fs, 
        reportinterval = 0, interpmethod = interpmethod, isinterior = false)

    m,F
end

boxsize = 10

kz0 = div(nz,10)
kz1 = kz0 - div(boxsize,2)
kz2 = kz0 + div(boxsize,2)

vx0 = 2 * div(nx,5)
vx1 = vx0 - div(boxsize,2)
vx2 = vx0 + div(boxsize,2)

bx0 = 3 * div(nx,5)
bx1 = bx0 - div(boxsize,2)
bx2 = bx0 + div(boxsize,2)

b1 =    1 .* ones(Float32, nz, nx, 1);
b2 =    1 .* ones(Float32, nz, nx, 1); 
v1 = 1500 .* ones(Float32, nz, nx, 1);
v2 = 1500 .* ones(Float32, nz, nx, 1); 

v2[kz1:kz2,vx1:vx2] .*= 1.25;
b2[kz1:kz2,bx1:bx2] .*= 1.50;

v3 = v2 .- v1
b3 = b2 .- b1

m1, F1 = make_op(:linear, WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD_Model_V, false, v1, b1);
m2, F2 = make_op(:linear, WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD_Model_V, false, v2, b1);

m3, F3 = make_op(:linear, WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD_Model_B, false, v1, b1);
m4, F4 = make_op(:linear, WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD_Model_B, false, v1, b2);

m5, F5 = make_op(:linear, WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD_Model_VB, false, v1, b1);
m6, F6 = make_op(:linear, WaveFD.Prop2DAcoIsoDenQ_DEO2_FDTD_Model_VB, false, v2, b2);

@show size(m1)
@show size(m3)
@show size(m5)

@show extrema(m1[:,:,1]), extrema(m2[:,:,1])
@show extrema(m3[:,:,1]), extrema(m4[:,:,1])
@show extrema(m5[:,:,1]), extrema(m6[:,:,1])

local d1,d2,d3,d4,d5,d6

tv = @elapsed begin
    @show "velocity only -- v1" 
    d1 = F1 * m1;
    @show "velocity only -- v2" 
    d2 = F2 * m2;
    @show "velocity only -- jacobian" 
    J1 = jacobian!(F1,m1);
    δd2 = J1 * (m2 .- m1);
end
@show tv

tb = @elapsed begin
    @show "buoyancy only -- b1" 
    d3 = F3 * m3;
    @show "buoyancy only -- b2" 
    d4 = F4 * m4;
    @show "buoyancy only -- jacobian" 
    J3 = jacobian!(F3,m3);
    δd3 = J3 * (m4 .- m3);
end
@show tb

tvb = @elapsed begin
    @show "velocity + buoyancy -- vb1" 
    d5 = F5 * m5;
    @show "velocity + buoyancy -- vb2" 
    d6 = F6 * m6;
    @show "velocity + buoyancy -- jacobian" 
    J5 = jacobian!(F5,m5);
    δd5 = J5 * (m6 .- m5);
end
@show tvb

vmax = maximum(abs,v2); vmin = - vmax 
bmax = maximum(abs,b2); bmin = - bmax 
dmax = 0.25 * maximum(abs,d2); dmin = - dmax 

close("all"); 

shape = (nz, nx)

figure(1,figsize=(18,12))
subplot(2,3,1); imshow(reshape(v1, shape), cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax); title("2D Velocity Background v1"); colorbar();
subplot(2,3,2); imshow(reshape(v2, shape), cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax); title("2D Velocity Perturbed v2"); colorbar();
subplot(2,3,3); imshow(reshape(v3, shape), cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax); title("2D Velocity Difference (v2-v1)"); colorbar();
subplot(2,3,4); imshow(reshape(b1, shape), cmap="seismic", aspect="auto", vmin=bmin, vmax=bmax); title("2D Buoyancy Background b1"); colorbar();
subplot(2,3,5); imshow(reshape(b2, shape), cmap="seismic", aspect="auto", vmin=bmin, vmax=bmax); title("2D Buoyancy Perturbed b2"); colorbar();
subplot(2,3,6); imshow(reshape(b3, shape), cmap="seismic", aspect="auto", vmin=bmin, vmax=bmax); title("2D Buoyancy Difference (b2-b1)"); colorbar();
tight_layout()
display(gcf())
savefig("figure.2D.model.png")

scale = 10
@show scale

figure(3,figsize=(18,12))
subplot(2,2,1); imshow(d1, cmap="seismic", aspect="auto", vmin=dmin, vmax=dmax); title("2D Reference F*v1"); colorbar();
subplot(2,2,2); imshow(d2, cmap="seismic", aspect="auto", vmin=dmin, vmax=dmax); title("2D Velocity Only F*v2"); colorbar();
subplot(2,2,3); imshow(d4, cmap="seismic", aspect="auto", vmin=dmin, vmax=dmax); title("2D Buoyancy Only F*b2"); colorbar();
subplot(2,2,4); imshow(d6, cmap="seismic", aspect="auto", vmin=dmin, vmax=dmax); title("2D Velocity + Buoyancy F*vb2"); colorbar();
tight_layout()
display(gcf())
savefig("figure.2D.data.nonlinear.png")

figure(4,figsize=(18,12))
subplot(2,2,2); imshow(scale .* (d2 .- d1), cmap="seismic", aspect="auto", vmin=dmin, vmax=dmax); title("2D Velocity Only F*v2 - F*v1 (($(scale)x)"); colorbar();
subplot(2,2,3); imshow(scale .* (d4 .- d3), cmap="seismic", aspect="auto", vmin=dmin, vmax=dmax); title("2D Buoyancy Only F*b2 - F*b1 (($(scale)x)"); colorbar();
subplot(2,2,4); imshow(scale .* (d6 .- d5), cmap="seismic", aspect="auto", vmin=dmin, vmax=dmax); title("2D Velocity + Buoyancy F*vb2 - F*vb1 (($(scale)x)"); colorbar();
tight_layout()
display(gcf())
savefig("figure.2D.data.difference.png")

figure(5,figsize=(18,12))
subplot(2,2,2); imshow(scale .* (δd2), cmap="seismic", aspect="auto", vmin=dmin, vmax=dmax); title("2D Velocity Only J*(v2-v1) ($(scale)x)"); colorbar();
subplot(2,2,3); imshow(scale .* (δd3), cmap="seismic", aspect="auto", vmin=dmin, vmax=dmax); title("2D Buoyancy Only J*(b2-b1) (($(scale)x)"); colorbar();
subplot(2,2,4); imshow(scale .* (δd5), cmap="seismic", aspect="auto", vmin=dmin, vmax=dmax); title("2D Velocity + Buoyancy J*(vb2-vb1) (($(scale)x)"); colorbar();
tight_layout()
display(gcf())
savefig("figure.2D.data.linear.png")

close(F1)
close(F2)
close(F3)
close(F4)
close(F5)
close(F6)
