#using TensorToolbox, Test, LinearAlgebra

println("\n\n****Testing ttensor.jl")

X=rand(20,10,50,5)

println("\n...Test tensor X of size: ", size(X))
println("\n...Testing hosvd.")
println("\nCreating exact decomposition with rank = size(X):")
T=hosvd(X)
println("Results:")
println("Type of output T: ", typeof(T))
@test isa(T,ttensor)
println("Core tensor size: ", coresize(T))
@test coresize(T) == size(X)
print("Factor matrices sizes: ")
for n=1:ndims(T)
	print(size(T.fmat[n])," ")
  @test size(T.fmat[n]) == (size(X,n),size(X,n))
end

err=norm(full(T) - X)
println("\n\n...Testing function full, i.e. n-mode multiplication (ttm): norm(full(T)-X) = ", err)
@test err ≈ 0 atol=1e-10

A=MatrixCell(undef,ndims(T))
for n=1:ndims(T)
  A[n]=rand(rand(1:10),size(T,n))
end
err=norm(full(ttm(T,A)) - ttm(full(T),A))
println("\n...Testing ttm for ttensor T and array of matrices A : norm(full(ttm(T,A))-ttm(full(T),A)) = ", err)
@test err ≈ 0 atol=1e-10

R=[5,5,5,5]
println("\n...Testing hosvd with smaller multilinear rank: ", R)
T=hosvd(X,reqrank=R)
println("Results:")
println("Type of output T: ", typeof(T))
@test isa(T,ttensor)
println("Core tensor size: ", coresize(T))
@test coresize(T) == tuple(R...)
print("Factor matrices sizes: ")
for n=1:ndims(T)
  print(size(T.fmat[n])," ")
  @test size(T.fmat[n]) == (size(X,n),R[n])
end
println("\n\n...Testing size of ttensor T : ", size(T))
@test size(T) == size(X)
println("\n...Testing ndims of ttensor T : ", ndims(T))
@test ndims(T) == ndims(X)
println("\n...Testing nrank of ttensor T for mode 1: ", nrank(T,1))
@test nrank(T,1) == R[1]
println("\n...Testing mrank of ttensor T: ", mrank(T))
@test mrank(T) == tuple(R...)

println("\n...Testing functions matten and tenmat (by mode).")
for n=1:ndims(T)
  Tn=tenmat(T,n)
  @test norm(Tn-tenmat(full(T),n)) ≈ 0 atol=1e-10
end

R=[5,5,5];
T=hosvd(rand(60,50,40),reqrank=R);
sz = size(T);
X=full(T);
println("\n\n...Testing hosvd for tensor with noise.")
println("For ttensor T, X=full(T) tensor of size ",size(X)," and rank ",R," , N noise tensor and S=hosvd(X+N,",R,").")
rsz = tuple([[sz...]' 1 1]...)
Rdm = reshape(randn(rsz), sz);
N=1e-3 * norm(X) * Rdm / norm(Rdm);
Y = X + N;
S=hosvd(Y,reqrank=R)
err=norm(T-S)
noise=norm(N)
println("Error( norm(T-S) ): ",err,". Noise norm: ",noise,".")

f(x,y,z)=1/(x+y+z)
dx=dy=dz=[n*0.1 for n=1:20]
X=[ f(x,y,z) for x=dx, y=dy, z=dz ]

println("\n\n...Testing hosvd with eps_abs=1e-5 on function defined tensor X of size ", size(X), " and multlinear rank", mrank(X))
T=hosvd(X,eps_abs=1e-5)
println("Results:")
println("Type of output T: ", typeof(T))
@test isa(T,ttensor)
println("Core tensor size: ",  coresize(T))
print("Factor matrices sizes: ")
for fmat in T.fmat
	print(size(fmat)," ")
end
@test norm(full(T)-X) ≈ 0 atol=1e-5

println("\n\n...Testing hosvd with eps_rel=1e-5 on function defined tensor X of size ", size(X))
T=hosvd(float(X),eps_rel=1e-5)
println("Results:")
println("Type of output T: ", typeof(T))
@test isa(T,ttensor)
println("Core tensor size: ", coresize(T))
print("Factor matrices sizes: ")
for fmat in T.fmat
	print(size(fmat)," ")
end
println("\n\n...Testing if factor matrices of ttensor T are orthogonal: ", T.isorth)
@test T.isorth

R=[3,3,3]
println("\n\n...Recompress Tucker tensor T to smaller rank: ", R)
S=hosvd(T,reqrank=R)
println("Results:")
println("Type of output T: ", typeof(S))
@test isa(S,ttensor)
println("Core tensor size: ", coresize(S))
@test coresize(S) == tuple(R...)
print("Factor matrices sizes: ")
for n=1:ndims(S)
  print(size(S.fmat[n])," ")
  @test size(S.fmat[n]) == (size(T,n),R[n])
end

S=randttensor([4,5,2],[3,3,3])
println("\n\n...Testing orthogonal flag and reorthogonalization for random ttensor S of size ", size(S),".")
println("Has orthogonal factor matrices: ", S.isorth)
println("After reorthogonalization: ", reorth!(S).isorth)
@test S.isorth

println("\n\n...Testing norm of ttensor T.")
err=abs(norm(T) - norm(full(T)))
println("|norm(T) - norm(full(T))| = ", err)
@test err ≈ 0 atol=1e-10

println("\n\n...Testing scalar multiplication 3*T.")
err=norm(full(3*T) - 3*full(T))
println("norm(full(3*T) - 3*full(T)) = ", err )
@test err ≈ 0 atol=1e-10

X=randttensor([6,8,2,5,4],[4,3,2,2,3]);
Y=randttensor([6,8,2,5,4],[3,6,3,4,3]);
println("\n\n...Creating two random ttensors X and Y of size ", size(X),".")
println("\n...Testing addition.")
Z=X+Y;
F=full(X)+full(Y);
err=norm(full(Z) - F)
println("norm(full(X+Y) - (full(X)+full(Y))) = ",err )
@test err ≈ 0 atol=1e-10

println("\n\n...Testing inner product.")
Z=innerprod(X,Y)
err=abs(Z - innerprod(full(X),full(Y)))
println("|innerprod(X,Y) - innerprod(full(X),full(Y))| = ",err )
@test err ≈ 0 atol=1e-10

println("\n\n...Testing Hadamard product.")
err=norm(full(ewprod(X,Y)) - full(X).*full(Y))
println("norm(full(ewprod(X,Y)) - full(X).*full(Y)) = ", err)
@test  err ≈ 0 atol=1e-10


println("\n\n...Testing singular values of matricizations of Tucker Tensor.")
R=[3,3,3]
T=randttensor([10,9,8],R)
println("\nSingular values of matricizations of random Tucker tensor of size ", size(T), ", rank ",R," and norm ",norm(T),".")
for n=1:ndims(T)
  sv = msvdvals(T,n)
  Tn=tenmat(T,n);
  println("Mode-$n singular values error: ",norm(sv-svd!(tenmat(full(T),n)).S[1:length(sv)]))
  @test norm(sv-svd!(tenmat(full(T),n)).S[1:length(sv)]) ≈ 0 atol=1e-10
end

R=[5,5,5,5]
T=randttensor([20,20,20,20],R);
println("\nSingular values of matricizations of random Tucker tensor of size ", size(T), ", rank ",R," and norm ",norm(T),".")
for n=1:ndims(T)
  sv = msvdvals(T,n)
  Tn=tenmat(T,n)
  println("Mode-$n singular values error: ",norm(sv-svd!(tenmat(full(T),n)).S[1:length(sv)]))
  @test norm(sv-svd!(tenmat(full(T),n)).S[1:length(sv)]) ≈ 0 atol=1e-10
end

println("\n...Testing function mhadtv.")
X=randttensor([5,4,3],[3,3,3])
Y=randttensor([5,4,3],[2,2,2])
println("\n\n...Creating two random ttensors X and Y of size ", size(X),".")
v=rand(12)
mode=1
println("Multiplying mode-$mode matricized ewprod(X,Y) by a random vector.")
Z=mhadtv(X,Y,v,mode,'n')
Hn=tenmat(ewprod(X,Y),mode)
err = norm(Z-Hn*v)
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-10
v=rand(5)
Z=mhadtv(X,Y,v,mode,'t')
err = norm(Z-Hn'*v)
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-10
v=rand(5)
Z=mhadtv(X,Y,v,mode,'b')
err = norm(Z-Hn*Hn'*v)
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-10

println("\n...Testing function mttkrp.")
X=randttensor([5,4,3],[3,3,3])
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

println("\n...Testing cp_als on ttensor.")
I=5;J=4;K=3;
a=rand(I);
b=rand(J);
c=rand(K);
X=zeros(I,J,K);
for i=1:I,j=1:J,k=1:K
    X[i,j,k]=a[i]*b[j]*c[k];
end
T=hosvd(X)
println("Test rank-1 ttensor T of size: ", size(T))
Z=cp_als(T,1)
err=norm(full(Z)-full(T))
println("norm(full(Tcp)-T) = ",err)
@test err ≈ 0 atol=1e-10
