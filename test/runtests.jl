using TensorToolbox
using Base.Test

X=rand(20,10,50,5);

println("\n...Test tensor X of size: ", size(X))
println("\n...Testing matten and tenmat.")
for n=1:ndims(X)
  Xn=tenmat(X,n);
  N=setdiff(1:ndims(X),n);
  println("Size of $n-mode matricization: ", [size(Xn)...])
  @test size(Xn) == (size(X,n),prod([size(X,N[k]) for k=1:length(N)]))
  println("Check if it folds back correctly: ",matten(Xn,n,[size(X)...]) == X)
  @test matten(Xn,n,[size(X)...]) == X
end

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

println("\n\n...Testing function full, i.e. n-mode multiplication (ttm): norm(full(T)-X) = ", norm(full(T) - X))
@test_approx_eq_eps norm(full(T)-X) 0 1e-12

A=MatrixCell(ndims(T))
for n=1:ndims(T)
  A[n]=rand(rand(1:10),size(T,n))
end
println("\n...Testing ttm for ttensor T and array of matrices A : norm(full(ttm(T,A))-ttm(full(T),A)) = ", norm(full(ttm(T,A)) - ttm(full(T),A)))
@test_approx_eq_eps norm(full(ttm(T,A)) - ttm(full(T),A)) 0 1e-10

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
S=hosvd(Y,reqrank=R);
err=norm(T-S)
noise=norm(N)
println("Error( norm(T-S) ): ",err,". Noise norm: ",noise,".")

f(x,y,z)=1/(x+y+z)
dx=dy=dz=[n*0.1 for n=1:20]
X=Float64[ f(x,y,z) for x=dx, y=dy, z=dz ]

println("\n\n...Testing hosvd with eps_abs=1e-5 on function defined tensor X of size ", size(X))
T=hosvd(X,eps_abs=1e-5)
println("Results:")
println("Type of output T: ", typeof(T))
@test isa(T,ttensor)
println("Core tensor size: ",  coresize(T))
print("Factor matrices sizes: ")
for A in T.fmat
	print(size(A)," ")
end
@test_approx_eq_eps norm(full(T)-X) 0 1e-5

println("\n\n...Testing hosvd with eps_rel=1e-5 on function defined tensor X of size ", size(X))
T=hosvd(float(X),eps_rel=1e-5)
println("Results:")
println("Type of output T: ", typeof(T))
@test isa(T,ttensor)
println("Core tensor size: ", coresize(T))
print("Factor matrices sizes: ")
for A in T.fmat
	print(size(A)," ")
end
println("\n\n...Testing if factor matrices of ttensor T are orthogonal: ", T.isorth)
@test T.isorth

R=[3,3,3]
println("\n\n...Recompress Tucer tensor T to smaller rank: ", R)
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
println("|norm(T) - norm(full(T))| = ", abs(norm(T) - norm(full(T))))
@test_approx_eq_eps abs(norm(T) - norm(full(T))) 0 1e-12

println("\n\n...Testing scalar multiplication 3*T.")
println("norm(full(3*T) - 3*full(T)) = ",  norm(full(3*T) - 3*full(T)))
@test_approx_eq_eps norm(full(3*T) - 3*full(T)) 0 1e-12

X=randttensor([6,8,2,5,4],[4,3,2,2,3]);
Y=randttensor([6,8,2,5,4],[3,6,3,4,3]);
println("\n\n...Creating two random ttensors X and Y of size ", size(X),".")
println("\n...Testing addition.")
Z=X+Y;
F=full(X)+full(Y);
println("norm(full(X+Y) - (full(X)+full(Y))) = ", norm(full(Z) - F))
@test_approx_eq_eps norm(full(Z) - F) 0 1e-12

println("\n\n...Testing inner product.")
Z=innerprod(X,Y)
println("|innerprod(X,Y) - innerprod(full(X),full(Y))| = ", abs(Z - innerprod(full(X),full(Y))))
@test_approx_eq_eps abs(Z - innerprod(full(X),full(Y))) 0 1e-10

println("\n\n...Testing Hadamard product.")
println("norm(full(X.*Y) - full(X).*full(Y)) = ", norm(full(X.*Y) - full(X).*full(Y)))
@test_approx_eq_eps  norm(full(X.*Y) - full(X).*full(Y)) 0 1e-10


println("\n\n...Testing singular values of matricizations of Tucker Tensor.")
R=[3,3,3]
T=randttensor([10,9,8],R);
println("\nSingular values of matricizations of random Tucker tensor of size ", size(T), ", rank ",R," and norm ",norm(T),".")
for n=1:ndims(T)
  sv = msvdvals(T,n)
  Tn=tenmat(T,n);
  println("Mode-$n singular values error: ",norm(sv-svdfact!(tenmat(full(T),n))[:S][1:length(sv)]))
  @test_approx_eq_eps norm(sv-svdfact!(tenmat(full(T),n))[:S][1:length(sv)]) 0 1e-10
end

R=[5,5,5,5]
T=randttensor([20,20,20,20],R);
println("\nSingular values of matricizations of random Tucker tensor of size ", size(T), ", rank ",R," and norm ",norm(T),".")
for n=1:ndims(T)
  sv = msvdvals(T,n)
  Tn=tenmat(T,n);
  println("Mode-$n singular values error: ",norm(sv-svdfact!(tenmat(full(T),n))[:S][1:length(sv)]))
  @test_approx_eq_eps norm(sv-svdfact!(tenmat(full(T),n))[:S][1:length(sv)]) 0 1e-10
end

println("\n\n")
