using JetPackWaveFD, Jets, Test

tmax = 9.0
dtdom,dtrng = 0.004,0.0005
n1dom,n1rng,n2 = round(Int,tmax/dtdom)+1,round(Int,tmax/dtrng)+1,10

@testset "JopSincRegular linearity test, T=$(T)" for T in (Float32,Float64)
    A = JopSincRegular(JetSpace(T,n1dom,n2), JetSpace(T,n1rng,n2), dtdom, dtrng)
    lhs, rhs = linearity_test(A)
    @test lhs ≈ rhs
    lhs, rhs = linearity_test(A')
    @test lhs ≈ rhs
end

@testset "JotOpFilter dot product test, T=$(T)" for T in (Float32,Float64)
    A = JopSincRegular(JetSpace(T,n1dom,n2), JetSpace(T,n1rng,n2), dtdom, dtrng)
    lhs,rhs = dot_product_test(A, -1 .+ 2 .* rand(domain(A)), -1 .+ 2 .* rand(range(A)))
    @test lhs ≈ rhs
end
