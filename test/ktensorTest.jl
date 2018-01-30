using TensorToolbox
using Base.Test

Ax = rand(4,2)
Bx = rand(3,2)
Cx = rand(2,2)
X = ktensor([Ax,Bx,Cx])
Ay = rand(4,3)
By = rand(3,3)
Cy = rand(2,3)
lambda=[2,3,1]
Y=ktensor(lambda,[Ay,By,Cy])

println("\n**Test ktensors X and Y of size: ", size(X))
println()

#TODO: println("...Testing function full.")

println("...Testing functions plus and minus.")
Z=X-Y
@test vecnorm(full(Z)-(full(X)-full(Y))) ≈ 0 atol=1e-12
Z=X+Y
@test vecnorm(full(Z)-(full(X)+full(Y))) ≈ 0 atol=1e-12

println("...Testing function innerprod.")
@test vecnorm(innerprod(X,Y)-innerprod(full(X),full(Y))) ≈ 0 atol=1e-12
Z=rand(4,3,2)
@test vecnorm(innerprod(X,Z)-innerprod(full(X),Z)) ≈ 0 atol=1e-12

println("...Testing function vecnorm.")
n=vecnorm(X)
@test abs(n-vecnorm(full(X)))≈ 0 atol=1e-12


println("...Testing functions arrange and arrange!.")
Z=arrange(X)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
Z=arrange(X,2)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
Z=arrange(X,[2,1])
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
Z=deepcopy(X)
arrange!(X)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
arrange!(X,2)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
arrange!(X,[2,1])
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12

println("...Testing functions normalize and normalize!.")
Z=normalize(X)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
Z=normalize(X,2)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
Z=normalize(X,3,normtype=1)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
Z=normalize(X,"sort")
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
Z=normalize(X,factor=2)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
Z=deepcopy(X)
normalize!(X)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
normalize!(X,2)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
normalize!(X,3,normtype=1)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
normalize!(X,"sort")
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
normalize!(X,factor=2)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12

