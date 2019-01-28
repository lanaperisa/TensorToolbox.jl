#using TensorToolbox, Test, LinearAlgebra

println("\n\n****Testing TTtensor.jl")

N=3
G=CoreCell(undef,N)
G[1]=rand(1,4,3)
G[2]=rand(3,6,4)
G[3]=rand(4,3,1)

println("\n...Test core tensors G of sizes: ")
[println(size(G[n])) for n=1:N]

T=TTtensor(G)

println("\n...Test size of TTtensor T: ", size(T))
println("\n...Test ndims of TTtensor T: ", ndims(T))
println("\n...Test TTsvd of TTtensor T: ", TTrank(T))

X=randTTtensor([6,5,4,3],[4,3,2]);
Y=randTTtensor([6,5,4,3],[3,2,2]);
Z=X+Y
err=norm(full(Z)-(full(X)+full(Y)))
println("\n...Test addition: ")
@test err ≈ 0 atol=1e-10

z=innerprod(X,Y)
err=norm(z-innerprod(full(X),full(Y)))
println("\n...Test dot product: ")
@test err ≈ 0 atol=1e-10



X=rand(5,4,3,2)

println("\n...Test tensor X of size: ", size(X))
println("\n...Testing TTsvd.")

T=TTsvd(X)

println("Results:")
println("Type of output T: ", typeof(T))
@test isa(T,TTtensor)
err=norm(full(T) - X)
println("\n...Testing function full, i.e. contracted product (conprod): norm(full(T)-X) = ", err)
@test err ≈ 0 atol=1e-10

println("\n...Testing TTsvd with requested rank.")

G=CoreCell(undef,4);
I=[6,5,6,5]
R=[4,4,4]
G[1]=reshape(rand(I[1],R[1]),(1,I[1],R[1]))
G[2]=reshape(rand(R[1],I[2],R[2]),(R[1],I[2],R[2]))
G[3]=reshape(rand(R[2],I[3],R[3]),(R[2],I[3],R[3]))
G[4]=reshape(rand(R[3],I[4]),(R[3],I[4],1))
T=TTtensor(G)
X=full(T)
println("Created tensor x of size ",size(X),", with TTrank = ",TTrank(T), ".")
println("T1=TTsvd(X,reqrank=$R)")
T1=TTsvd(X,reqrank=R)
err=norm(X-full(T1))
println("norm(X-full(T1)) = ",err)
@test err ≈ 0 atol=1e-8

println("\n...Testing recompression.")

S=TTsvd(T)
err=norm(full(S) - X)
println("S=TTsvd(T): norm(full(S)-X) = ", err)
@test err ≈ 0 atol=1e-10

println("\n...Testing reorthogonalization - function reorth.")

I=[6,5,4,3]
R=[4,3,2]
N=length(R)
T=randTTtensor(I,R)
println("Test TT-tensor T of size: ", size(T)," with flags T.lorth = ",T.lorth," and T.rorth = ",T.rorth,".")
println("\nPerforming (default) left orthogonalization: Tl=reorth(T).")
Tl=reorth(T)
println("Flags: Tl.lorth = ",Tl.lorth,", Tl.rorth = ",Tl.rorth,".")
@test Tl.lorth == true
println("\nPerforming right orthogonalization: Tr=reorth(T,\"right\").")
Tr=reorth(T,"right")
println("Flags: Tr.lorth = ",Tr.lorth,", Tr.rorth = ",Tr.rorth,".")

Gr=Tr.cores;
for n=1:N-1
    Gn=tenmat(Gr[n+1],1);
    @show norm(Diagonal(ones(size(Gn,1)))- Gn*Gn')
end

@test Tr.rorth == true

println("\n...Testing reorthogonalization with overwriting - function reorth!.")
println("\nPerforming (default) left orthogonalization: reorth!(T)")
reorth!(T)
println("Flags: T.lorth = ",T.lorth,", T.rorth = ",T.rorth,".")
@test T.lorth == true
println("\nPerforming right orthogonalization: reorth!(T,\"right\")")
reorth!(T,"right")
println("Flags: T.lorth = ",T.lorth,", T.rorth = ",T.rorth,".")
@test T.lorth == true && T.rorth==true


println("\n...Testing TTsvd on TT-tensors.")
epsilon=1e-8
println("\nFixed precision problem with ϵ = ",epsilon,".")
G=CoreCell(undef,4);
I=[6,5,6,5]
R=[4,4,4]
r=2
G[1]=reshape(rand(I[1],r)*rand(r,R[1]),(1,I[1],R[1]))
G[2]=reshape(rand(R[1]*I[2],r)*rand(r,R[2]),(R[1],I[2],R[2]))
G[3]=reshape(rand(R[2]*I[3],r)*rand(r,R[3]),(R[2],I[3],R[3]))
G[4]=reshape(rand(R[3],r)*rand(r,I[4]),(R[3],I[4],1))
T=TTtensor(G)
println("Created TT-tensor T of size ",size(T),", with TTrank = ",TTrank(T), ", but its actual TT-rank is (2,2,2)")
println("T1=TTsvd(T,ϵ)")
T1=TTsvd(T)
rnk=TTrank(T1)
println("TTrank(T1) = ",rnk)
@test rnk==(2,2,2)
println("norm(full(T)-full(T1)) = ",norm(full(T)-full(T1)))
@test norm(full(T)-full(T1)) ≈ 0 atol=epsilon


R=[2,2,2]
println("\nFixed rank problem with reqrank = ",R,".")
T=TTsvd(rand(6,5,4,3))
println("Created TT-tensor T of size ",size(T),", with TTrank = ",TTrank(T),".")
println("T1=TTsvd(T,reqrank=$R)")
T1=TTsvd(T,reqrank=R)
rnk=TTrank(T1)
println("TTrank(T1) = ",rnk)
@test rnk==(2,2,2)


println("\n... Testing contraction with vectors - function TTtv.")
X=randTTtensor([6,5,4,3,2],[3,3,3,3])
G=X.cores
N=ndims(X)
Isz=[size(X)...]
J=1
L=N-J+1
u=VectorCell(undef,L)
for i=1:L
    u[i]=randn(Isz[J+i-1])
end
Xf=full(X)
let
sz=copy(Isz)
for i=1:L
    sz[i]=1
    global Xf=reshape(tenmat(Xf,J+i-1)'*u[i],tuple(sz...))
end
end
@test TTtv(G,u) ≈ dropdims(Xf) atol=1e-12
J=2;
L=N-J+1;
v=VectorCell(undef,L)
[v[i]=randn(Isz[J+i-1]) for i=1:L]
@test norm(TTtv(X,v,J) - dropdims(contract(G[1],TTtv(G[2:end],v)))) ≈ 0 atol=1e-12
for J=3:N
    L=N-J+1;
    v=VectorCell(undef,L)
    [v[i]=randn(Isz[J+i-1]) for i=1:L]
    T=TTtv(X,v,J)
    Gtest=CoreCell(undef,J-1)
    [Gtest[n]=copy(G[n]) for n=1:J-2]
    Gtest[J-1]=contract(G[J-1],TTtv(G[J:end],v))
    @test norm(full(T)-contract(Gtest)) ≈ 0 atol=1e-12
end

println("\n... Testing transformation of ktensor to TTtensor - function kten2TT.")
Isz=[5,4,3,2]
R=3
T=randktensor(Isz,R)
println("Test ktensor T of size ",size(T),", and k-rank = $R, transofrming to TTtensor Ttt.")
Ttt=kten2TT(T)
err=norm(full(T)-full(Ttt))
println("norm(full(T)-full(Ttt)) = ", err)
@test err ≈ 0 atol=1e-8

println("\n... Testing contraction of full tensor to TTtensor - function contract.")
N=5
Isz=repeat([5],N,1)
R=repeat([3],N-1,1)
T=randTTtensor(Isz,R)
Xsz=[Isz[1:N-1]...;2]
X=rand(Xsz...)
Tfull=full(T)
T1=contract(Tfull,collect(1:N-1),X,collect(1:N-1),[2,1])
Y=contract(T,X,1,4)
@test norm(T1-full(Y)) ≈ 0 atol=1e-12

Xsz=[Isz[1:N-2]...;2]
X=rand(Xsz...)
T2=contract(Tfull,collect(2:N-1),X,collect(1:N-2),[1,3,2])
Y=contract(T,X,2,3)
@test norm(T2-full(Y)) ≈ 0 atol=1e-12

println("\n... Testing element-wise product of two TTtensor - function ewprod.")
N=4;
Isz=[8,7,6,5];
Rx=[7,6,5];
Ry=[6,5,4];
X=randTTtensor(Isz,Rx);
Y=randTTtensor(Isz,Ry);
Zfull=full(X).*full(Y);
Z=ewprod(X,Y)
@test norm(Zfull-full(Z)) ≈ 0 atol=1e-8
