#using TensorToolbox
include("/home/lana/.julia/v0.6/TensorToolbox/src/tttensor.jl")
using Test

println("\n\n****Testing tttensor.jl")

N=3
G=TensorCell(N)
G[1]=rand(1,4,3)
G[2]=rand(3,6,4)
G[3]=rand(4,3,1)

println("\n...Test core tensors G of sizes: ")
[println(size(G[n])) for n=1:N]

T=tttensor(G)

println("\n...Test size of tttensor T: ", size(T))
println("\n...Test ndims of tttensor T: ", ndims(T))
println("\n...Test ttrank of tttensor T: ", ttrank(T))

X=rand(5,4,3,2)

println("\n...Test tensor X of size: ", size(X))
println("\n...Testing ttsvd.")

T=ttsvd(X)

println("Results:")
println("Type of output T: ", typeof(T))
@test isa(T,tttensor)
err=vecnorm(full(T) - X)
println("\n\n...Testing function full, i.e. contracted product (conprod): vecnorm(full(T)-X) = ", err)
@test err ≈ 0 atol=1e-10
