#using TensorToolbox, Test, LinearAlgebra

println("\n\n****Testing ktensor.jl")

Isz=5;J=4;K=3;
a=rand(Isz);
b=rand(J);
c=rand(K);
X=zeros(Isz,J,K);
for i=1:Isz,j=1:J,k=1:K
    X[i,j,k]=a[i]*b[j]*c[k];
end
println("\n**Test rank-1 tensor X of size: ", size(X))
println("...Testing cp_als.")
Z=cp_als(X,1)
err=norm(full(Z)-X)
println("norm(full(Xcp)-X) = ",err)
@test err ≈ 0 atol=1e-10

#Ax = rand(4,2)
#Bx = rand(3,2)
#Cx = rand(2,2)
#Xlambda=[0.5,2]
#X = ktensor(Xlambda,[Ax,Bx,Cx])
X=randktensor([4,3,2],2)
Ay = rand(4,3)
By = rand(3,3)
Cy = rand(2,3)
Ylambda=[2,3,1]
Y=ktensor(Ylambda,[Ay,By,Cy])

println("\n**Test ktensors X and Y of size: ", size(X))
println()

println("...Testing function ndims.")
println("X is of order: ",ndims(X))

println("...Testing function ttensor and full.")
T=ttensor(X)
@test norm(full(T)-full(X)) ≈ 0 atol=1e-10

println("...Testing scalar multiplication.")
err=norm(full(3*X) - 3*full(X))
println("norm(full(3*X) - 3*full(X)) = ",err)
@test norm(full(3*X) - 3*full(X)) ≈ 0 atol=1e-10

println("...Testing functions plus and minus.")
Z=X+Y
err=norm(full(Z)-(full(X)+full(Y)))
println("norm(full(Z)-(full(X)+full(Y))) = ",err)
@test err ≈ 0 atol=1e-10
Z=X-Y
err=norm(full(Z)-(full(X)-full(Y)))
println("norm(full(Z)-(full(X)-full(Y))) = ",err)
@test err ≈ 0 atol=1e-10


println("...Testing function innerprod.")
err=norm(innerprod(X,Y)-innerprod(full(X),full(Y)))
println("norm(innerprod(X,Y)-innerprod(full(X),full(Y))) = ",err)
@test err  ≈ 0 atol=1e-10
Z=rand(4,3,2)
@test norm(innerprod(X,Z)-innerprod(full(X),Z)) ≈ 0 atol=1e-10

println("...Testing function norm.")
@test abs(norm(X)-norm(full(X)))≈ 0 atol=1e-10


println("...Testing functions arrange and arrange!.")
Z=arrange(X)
err=norm(full(X)-full(Z))
println("norm(full(X)-full(Z)) = ",err)
@test err ≈ 0 atol=1e-10
Z=arrange(X,2)
@test norm(full(X)-full(Z)) ≈ 0 atol=1e-10
Z=arrange(X,[2,1])
@test norm(full(X)-full(Z)) ≈ 0 atol=1e-10
Xcopy=deepcopy(X)
arrange!(X)
@test norm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10
arrange!(X,2)
@test norm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10
arrange!(X,[2,1])
@test norm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10

println("...Testing functions normalize and normalize!.")
Z=normalize(X)
err=norm(full(X)-full(Z))
println("norm(full(X)-full(Z)) = ",err)
@test err ≈ 0 atol=1e-10
Z=normalize(X,2)
@test norm(full(X)-full(Z)) ≈ 0 atol=1e-10
Z=normalize(X,3,normtype=1)
@test norm(full(X)-full(Z)) ≈ 0 atol=1e-10
Z=normalize(X,"sort")
@test norm(full(X)-full(Z)) ≈ 0 atol=1e-10
Z=normalize(X,factor=2)
@test norm(full(X)-full(Z)) ≈ 0 atol=1e-10
Xcopy=deepcopy(X)
normalize!(X)
@test norm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10
normalize!(X,2)
@test norm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10
normalize!(X,3,normtype=1)
@test norm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10
normalize!(X,"sort")
@test norm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10
normalize!(X,factor=2)
@test norm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10

println("...Testing functions redistribute and redistribute!.")
mode=2
Z=redistribute(X,mode)
err=norm(full(X)-full(Z))
println("norm(full(X)-full(Z)) = ",err)
@test err ≈ 0 atol=1e-10
Xcopy=deepcopy(X)
redistribute!(X,mode)
@test norm(full(X)-full(Xcopy)) ≈ 0 atol=1e-10

println("...Testing functions fixsigns, fixsigns! and isequal.")
Xcopy=deepcopy(X)
X.fmat[1][:,1]=-X.fmat[1][:,1]
X.fmat[2][:,1]=-X.fmat[2][:,1]
Z=fixsigns(X)
err=norm(full(X)-full(Z))
println("norm(full(X)-full(Z)) = ",err)
@test err ≈ 0 atol=1e-10
fixsigns!(X)
@test norm(X-Xcopy) ≈ 0 atol=1e-7

println("...Testing functions ttm and ttv.")
M=MatrixCell(undef,2)
M[1]=rand(4,3)
M[2]=rand(4,2)
mode=[2,3]
Z=ttm(X,M,mode)
W=ttm(full(X),M,mode)
@test norm(W-full(Z)) ≈ 0 atol=1e-10
v=VectorCell(undef,2)
v[1]=rand(3)
v[2]=rand(2)
mode=[2,3]
Z=ttv(X,v,mode)
W=ttv(full(X),v,mode)
@test norm(W-full(Z)) ≈ 0 atol=1e-10

println("\n...Testing function tocell.")
M=tocell(X)
println("Is the output MatrixCell: ",isa(M,MatrixCell))
@test isa(M,MatrixCell)

mode=2;
println("\n...Testing function tenmat by mode $mode.")
Xn=tenmat(X,mode)
@test tenmat(full(X),mode) == Xn

println("\n...Testing function mttkrp.")
X=randktensor([5,4,3],3)
mode=1
M1=rand(5,5);
M2=rand(4,5);
M3=rand(3,5);
M=[M1,M2,M3]
println("Multiplying mode-$mode matricized tensor X by Khatri-Rao product of matrices.")
Z=mttkrp(X,M,mode)
err = norm(Z-tenmat(X,mode)*khatrirao(M3,M2))
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-10
