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

println("\n...Testing function ndims.")
println("X is of order: ",ndims(X))

println("\n...Testing function ttensor and full.")
T=ttensor(X)
@test vecnorm(full(T)-full(X)) ≈ 0 atol=1e-10

println("...Testing scalar multiplication.")
err=vecnorm(full(3*X) - 3*full(X))
println("vecnorm(full(3*X) - 3*full(X)) = ",err)
@test vecnorm(full(3*X) - 3*full(X)) ≈ 0 atol=1e-10

println("...Testing functions plus and minus.")
Z=X+Y
err=vecnorm(full(Z)-(full(X)+full(Y)))
println("vecnorm(full(Z)-(full(X)+full(Y))) = ",err)
@test err ≈ 0 atol=1e-10
Z=X-Y
err=vecnorm(full(Z)-(full(X)-full(Y)))
println("vecnorm(full(Z)-(full(X)-full(Y))) = ",err)
@test err ≈ 0 atol=1e-10


println("...Testing function innerprod.")
err=vecnorm(innerprod(X,Y)-innerprod(full(X),full(Y)))
println("vecnorm(innerprod(X,Y)-innerprod(full(X),full(Y))) = ",err)
@test err  ≈ 0 atol=1e-10
Z=rand(4,3,2)
@test vecnorm(innerprod(X,Z)-innerprod(full(X),Z)) ≈ 0 atol=1e-10

println("...Testing function vecnorm.")
n=vecnorm(X)
@test abs(n-vecnorm(full(X)))≈ 0 atol=1e-10


println("...Testing functions arrange and arrange!.")
Z=arrange(X)
err=vecnorm(full(X)-full(Z))
println("vecnorm(full(X)-full(Z)) = ",err)
@test err ≈ 0 atol=1e-10
Z=arrange(X,2)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-10
Z=arrange(X,[2,1])
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-10
Xcopy=deepcopy(X)
arrange!(X)
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10
arrange!(X,2)
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10
arrange!(X,[2,1])
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10

println("...Testing functions normalize and normalize!.")
Z=normalize(X)
err=vecnorm(full(X)-full(Z))
println("vecnorm(full(X)-full(Z)) = ",err)
@test err ≈ 0 atol=1e-10
Z=normalize(X,2)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-10
Z=normalize(X,3,normtype=1)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-10
Z=normalize(X,"sort")
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-10
Z=normalize(X,factor=2)
@test vecnorm(full(X)-full(Z)) ≈ 0 atol=1e-10
Xcopy=deepcopy(X)
normalize!(X)
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10
normalize!(X,2)
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10
normalize!(X,3,normtype=1)
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10
normalize!(X,"sort")
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10
normalize!(X,factor=2)
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10

println("...Testing functions redistribute and redistribute!.")
n=2
Z=redistribute(X,n)
err=vecnorm(full(X)-full(Z))
println("vecnorm(full(X)-full(Z)) = ",err)
@test err ≈ 0 atol=1e-10
Xcopy=deepcopy(X)
redistribute!(X,n)
@test vecnorm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10

println("...Testing functions fixsigns, fixsigns! and isequal.")
Xcopy=deepcopy(X)
X.fmat[1][:,1]=-X.fmat[1][:,1]
X.fmat[2][:,1]=-X.fmat[2][:,1]
Z=fixsigns(X)
err=vecnorm(full(X)-full(Z))
println("vecnorm(full(X)-full(Z)) = ",err)
@test err ≈ 0 atol=1e-10
fixsigns!(X)
@test X==Xcopy #≈ 0 atol=1e-10

println("...Testing functions ttm and ttv.")
M=MatrixCell(2)
M[1]=rand(4,3)
M[2]=rand(4,2)
mode=[2,3]
Z=ttm(X,M,mode)
W=ttm(full(X),M,mode)
@test vecnorm(W-full(Z)) ≈ 0 atol=1e-10
v=VectorCell(2)
v[1]=rand(3)
v[2]=rand(2)
mode=[2,3]
Z=ttv(X,v,mode)
W=ttv(full(X),v,mode)
@test vecnorm(W-full(Z)) ≈ 0 atol=1e-10

println("\n...Testing function tocell.")
M=tocell(X)
println("Is the output MatrixCell: ",isa(M,MatrixCell))
@test isa(M,MatrixCell)

n=2;
println("\n...Testing function tenmat by mode $n.")
Xn=tenmat(X,n)
@test tenmat(full(X),n) == Xn

println("\n...Testing function mttkrp.")
X=randktensor([5,4,3],3)
n=1
M1=rand(5,5);
M2=rand(4,5);
M3=rand(3,5);
M=[M1,M2,M3]
println("Multiplying mode-$n matricized tensor X by Khatri-Rao product of matrices.")
Z=mttkrp(X,M,n)
err = vecnorm(Z-tenmat(X,n)*khatrirao(M3,M2))
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-10
