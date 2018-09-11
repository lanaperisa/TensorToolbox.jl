#using TensorToolbox, Test, LinearAlgebra

println("\n\n**** Testing tensor.jl")

X=rand(20,10,50,5)
N=ndims(X)
println("\n**Test tensor X of size: ", size(X))

println("\n...Testing functions matten and tenmat (by mode).")
for n=1:N
  Xn=tenmat(X,n);
  M=setdiff(1:N,n);
  println("Size of $n-mode matricization: ", [size(Xn)...])
  @test size(Xn) == (size(X,n),prod([size(X,M[k]) for k=1:length(M)]))
  println("Check if it folds back correctly: ",matten(Xn,n,[size(X)...]) == X)
  @test matten(Xn,n,[size(X)...]) == X
end

println("\n...Testing functions matten and tenmat (by rows and columns).")
R=[2,1];C=[4,3];
Xmat=tenmat(X,row=R,col=C);
println("Size of R=$R and C=$C matricization: ", [size(Xmat)...])
println("Check if it folds back correctly: ",matten(Xmat,R,C,[size(X)...]) == X)
@test matten(Xmat,R,C,[size(X)...]) == X

println("\n...Testing function ttm.")
M=MatrixCell(undef,N)
for n=1:N
  M[n]=rand(5,size(X,n))
end
println("Created $N matrices with 5 rows and appropriate number of columns.")
Xprod=ttm(X,M)
println("Size of tensor Y=ttm(X,M): ",size(Xprod))
err=norm(tenmat(Xprod,1) - M[1]*tenmat(X,1)*kron(M[end:-1:2])')
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-10

println("\n...Testing function ttv.")
Xk=reshape(collect(1:24),(3,4,2))
mode=2
v=collect(1:4)
println("Multiplying a tensor X by a vector v in mode $mode.")
Xprod=ttv(Xk,v,mode)
println("Size of tensor Y=ttv(X,v): ",size(Xprod))
res=[70 190;80 200;90 210]
@test Xprod==res

println("\n...Testing function krontm.")
X=rand(5,4,3)
Y=rand(2,5,4)
println("Created two tensors X and Y of order ",ndims(X)," and sizes ",size(X)," and ",size(Y),".")
mode=3
M1=rand(20,10)
M2=rand(20,20)
M3=rand(20,12)
println("Multiplying tkron(X,Y) by random matrix in mode $mode.")
Z=krontm(X,Y,M3,mode)
err= norm(Z-ttm(tkron(X,Y),M3,mode))
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-10
mode=[3,2]
M=[M3,M2]
println("Multiplying tkron(X,Y) by random matrices in modes $mode.")
Z=krontm(X,Y,M,mode)
err = norm(Z-ttm(tkron(X,Y),M,mode))
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-10
M=[M1,M2,M3]
println("Multiplying tkron(X,Y) by random matrices in all modes.")
Z=krontm(X,Y,M)
err = norm(Z-ttm(tkron(X,Y),M))
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-10

println("\n...Testing function mkrontv.")
v=rand(240)
mode=1
println("Multiplying mode-$mode matricized tkron(X,Y) by a random vector.")
Z=mkrontv(X,Y,v,mode)
err = norm(Z-tenmat(tkron(X,Y),mode)*v)
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-10
v=rand(10)
Z=mkrontv(X,Y,v,mode,'t')
err = norm(Z-tenmat(tkron(X,Y),mode)'*v)
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-10

println("\n...Testing function mttkrp.")
X=rand(5,4,3)
mode=1
A1=rand(2,5);
A2=rand(4,5);
A3=rand(3,5);
A=[A1,A2,A3]
println("Multiplying mode-$mode matricized tensor X by Khatri-Rao product of matrices.")
Z=mttkrp(X,A,mode)
err = norm(Z-tenmat(X,mode)*khatrirao(A3,A2))
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-10

println("\n...Testing function dropdims.")
X=rand(5,4,1,3,6,1)
Xsq=dropdims(X)
println("Tensor X of size :",size(X)," squeezed to size :",size(Xsq),".")

println("\n...Testing contraction (contract).")
X=rand(5,4,3);
Y=rand(3,4,2);
Z=contract(X,Y)
for i1=1:5, i2=1:4, j2=1:4,j3=1:2
    s=0;
    for i=1:3
        s+=X[i1,i2,i]*Y[i,j2,j3];
    end
    @test Z[i1,i2,j2,j3] ≈ s atol=1e-10
end
X=rand(5,4,3,2);
Y=rand(5,7,3,6);
Z=contract(X,[1,3],Y,[1,3])
err=norm(Z-reshape(tenmat(X,row=[1,3])'*tenmat(Y,row=[1,3]),(4,2,7,6)))
println("Error: ",err)
@test err ≈ 0 atol=1e-10
