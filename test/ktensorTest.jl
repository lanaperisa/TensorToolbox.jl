println("\n\n****Testing ktensor.jl")

Ax = rand(4,2)
Bx = rand(3,2)
Cx = rand(2,2)
Xlambda=[0.5,2]
X = ktensor(Xlambda,[Ax,Bx,Cx])
Ay = rand(4,3)
By = rand(3,3)
Cy = rand(2,3)
Ylambda=[2,3,1]
Y=ktensor(Ylambda,[Ay,By,Cy])

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
Xcopy=deepcopy(X)
arrange!(X)
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-12
arrange!(X,2)
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-12
arrange!(X,[2,1])
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-12

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
Xcopy=deepcopy(X)
normalize!(X)
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-12
normalize!(X,2)
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-12
normalize!(X,3,normtype=1)
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-12
normalize!(X,"sort")
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-12
normalize!(X,factor=2)
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-12

println("...Testing functions fixsigns, fixsigns! and isequal.")
Xcopy=deepcopy(X)
X.fmat[1][:,1]=-X.fmat[1][:,1]
X.fmat[2][:,1]=-X.fmat[2][:,1]
Z=fixsigns(X)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-12
fixsigns!(X)
@test X==Xcopy #≈ 0 atol=1e-12

println("...Testing functions ttm and ttv.")
M=MatrixCell(2)
M[1]=rand(4,3)
M[2]=rand(4,2)
mode=[2,3]
Z=ttm(X,M,mode)
W=ttm(full(X),M,mode)
@test vecnorm(W-full(Z)) ≈ 0 atol=1e-12
v=VectorCell(2)
v[1]=rand(3)
v[2]=rand(2)
mode=[2,3]
Z=ttv(X,v,mode)
W=ttv(full(X),v,mode)
@test vecnorm(W-full(Z)) ≈ 0 atol=1e-12
