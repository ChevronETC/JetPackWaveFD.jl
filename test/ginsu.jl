using JetPackWaveFD, Random, Test

function test_ginsu_sub(g::Ginsu{2}, x, xsub, interior, extend)
    N1 = size(x,1)
    N2 = size(x,2)

    I1 = lextents(g, interior=interior)[1]
    I2 = lextents(g, interior=interior)[2]

    for i2 = 1:N2, i1 = 1:N1
        if in(i1, I1) && in(i2, I2)
            @test x[i1,i2] ≈ xsub[i1-I1[1]+1,i2-I2[1]+1]
        end
    end
end

function test_ginsu_sub(g::Ginsu{3}, x, xsub, interior, extend)
    N1 = size(x,1)
    N2 = size(x,2)
    N3 = size(x,3)

    I1 = lextents(g, interior=interior)[1]
    I2 = lextents(g, interior=interior)[2]
    I3 = lextents(g, interior=interior)[3]

    for i3 = 1:N3, i2 = 1:N2, i1 = 1:N1
        if in(i1, I1) && in(i2, I2) && in(i3, I3)
            @test x[i1,i2,i3] ≈ xsub[i1-I1[1]+1,i2-I2[1]+1,i3-I3[1]+1]
        end
    end
end

function test_ginsu_super(g::Ginsu{2}, x, xsuper, interior)
    N1 = size(x,1)
    N2 = size(x,2)

    I1 = lextents(g, interior=interior)[1]
    I2 = lextents(g, interior=interior)[2]

    for i2 = 1:N2, i1 = 1:N1
        if in(i1,I1) && in(i2,I2)
            @test x[i1,i2] ≈ xsuper[i1,i2]
        else
            @test xsuper[i1,i2] ≈ 0
        end
    end
end

function test_ginsu_super(g::Ginsu{3}, x, xsuper, interior)
    N1 = size(x,1)
    N2 = size(x,2)
    N3 = size(x,3)

    I1 = lextents(g, interior=interior)[1]
    I2 = lextents(g, interior=interior)[2]
    I3 = lextents(g, interior=interior)[3]

    for i3 = 1:N3, i2 = 1:N2, i1 = 1:N1
        if in(i1,I1) && in(i2,I2) && in(i3,I3)
            @test x[i1,i2,i3] ≈ xsuper[i1,i2,i3]
        else
            @test xsuper[i1,i2,i3] ≈ 0
        end
    end
end

@testset "Ginsu 2D" begin
    nz, nx = 100, 200
    dz, dx = 10.0, 20.0
    z0, x0 = 0.0, 0.0

    sz, sx = [10.0], [2000.0]
    rz, rx = [0.0, 0.0], [0.0, 1990.0]
    padz, padx = 100.0, 200.0
    ndamp = 10

    ginsu = Ginsu((z0,x0), (dz,dx), (nz,nx), (sz,sx), (rz,rx), ((padz,padz),(padx,padx)), ((0,0),(ndamp,ndamp)), dims=(:z,:x), T=Float64)
    @test lextents(ginsu, interior=false) == (-9:110,-19:124)
    @test lextents(ginsu, interior=true) == (-9:110,-9:114)

    map(i->@test(pextents(ginsu, interior=false)[i] ≈ (-100.0:10.0:1090.0,-400.0:20.0:2460.0)[i]), 1:2)
    map(i->@test(pextents(ginsu, interior=true)[i] ≈ (-100.0:10.0:1090.0,-200.0:20.0:2260.0)[i]), 1:2)

    for interior in (false, true), extend in (false, true)
        x = rand(nz,nx)
        z = zeros(nz,nx)

        y = sub(ginsu, x, interior=interior, extend=extend)
        test_ginsu_sub(ginsu, x, y, interior, extend)

        super!(z, ginsu, y, interior=interior, accumulate=false)
        test_ginsu_super(ginsu, x, z, interior)

        super!(z, ginsu, y, interior=interior, accumulate=false)
        test_ginsu_super(ginsu, x, z, interior)

        super!(z, ginsu, y, interior=interior, accumulate=true)
        test_ginsu_super(ginsu, 2x, z, interior)
    end
end

@testset "Ginsu 3D" begin
    nz, ny, nx = 100, 30, 200
    dz, dy, dx = 10.0, 10.0, 20.0
    z0, y0, x0 = 0.0, 0.0, 0.0

    sz, sy, sx = [10.0], [150.0], [2000.0]
    rz, ry, rx = [0.0, 0.0], [0.0,270.0], [0.0, 1990.0]
    padz, pady, padx = 100.0, 50.0, 200.0
    ndamp = 10

    ginsu = Ginsu((z0,y0,x0), (dz,dy,dx), (nz,ny,nx), (sz,sy,sx), (rz,ry,rx), ((padz,padz),(pady,pady),(padx,padx)), ((0,0),(ndamp,ndamp),(ndamp,ndamp)), dims=(:z,:y,:x), T=Float64)
    @test lextents(ginsu, interior=false) == (-9:110,-14:49,-19:124)
    @test lextents(ginsu, interior=true) == (-9:110,-4:39,-9:114)

    map(i->@test(pextents(ginsu, interior=false)[i] ≈ (-100.0:10.0:1090.0,-150.0:10.0:480.0,-400.0:20.0:2460.0)[i]), 1:2)
    map(i->@test(pextents(ginsu, interior=true)[i] ≈ (-100.0:10.0:1090.0,-50.0:10.0:380.0,-200.0:20.0:2260.0)[i]), 1:2)

    for interior in (false, true), extend in (false, true)
        x = rand(nz,ny,nx)
        z = zeros(nz,ny,nx)

        y = sub(ginsu, x, interior=interior, extend=extend)
        test_ginsu_sub(ginsu, x, y, interior, extend)

        super!(z, ginsu, y, interior=interior, accumulate=false)
        test_ginsu_super(ginsu, x, z, interior)

        super!(z, ginsu, y, interior=interior, accumulate=false)
        test_ginsu_super(ginsu, x, z, interior)

        super!(z, ginsu, y, interior=interior, accumulate=true)
        test_ginsu_super(ginsu, 2x, z, interior)
    end
end

@testset "Ginsu dot product" begin
    rz = 750.0*ones(size(75:125))
    rx = 10.0*collect(75:125)
    ginsu = Ginsu((0.0,0.0), (10.0,10.0), (201,201), ([1000.0],[1000.0]), (rz,rx), ((500.0,500.0),(500.0,500.0)), ((50,50),(50,50)), T=Float64)
    m = rand(201,201)
    d = rand(size(ginsu)...)
    ds = sub(ginsu, m, extend=false)
    ms = zeros(201,201)
    super!(ms, ginsu, d)
    lhs = sum(d .* ds)
    rhs = sum(m .* ms)
    err = (lhs - rhs) / (lhs + rhs)
    @test isapprox(err, 0.0, atol=100*eps(Float32))

    rz = 10.0*ones(size(1:201))
    rx = 10.0*collect(1:201)
    ginsu = Ginsu((0.0,0.0), (10.0,10.0), (201,201), ([10.0],[1000.0]), (rz,rx), ((500.0,500.0),(500.0,500.0)), ((50,50),(50,50)), T=Float64)
    m = rand(201,201)
    d = rand(size(ginsu)...)
    ds = sub(ginsu, m, extend=false)
    ms = zeros(201,201)
    super!(ms, ginsu, d)
    lhs = sum(d .* ds)
    rhs = sum(m .* ms)
    @test lhs ≈ rhs rtol=1e-5
end

@testset "Ginsu, absolute constructor" begin
    ginsu = Ginsu((0.0,0.0), (10.0,10.0), (201,201), (-100.0:200.0,-50.0:150.0), ((50,50),(50,50)))
    @test lextents(ginsu, interior=false) == (-59:76,-54:73)
    @test lextents(ginsu, interior=true) == (-9:26,-4:23)

    map(i->@test(pextents(ginsu, interior=false)[i] ≈ (-600.0:10.0:750.0,-550.0:10.0:720.0)[i]), 1:2)
    map(i->@test(pextents(ginsu, interior=true)[i] ≈ (-100.0:10.0:250.0,-50.0:10.0:220.0)[i]), 1:2)
end
