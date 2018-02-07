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

err=vecnorm(full(T) - X)
println("\n\n...Testing function full, i.e. n-mode multiplication (ttm): vecnorm(full(T)-X) = ", err)
@test err ≈ 0 atol=1e-12

A=MatrixCell(ndims(T))
for n=1:ndims(T)
  A[n]=rand(rand(1:10),size(T,n))
end
err=vecnorm(full(ttm(T,A)) - ttm(full(T),A))
println("\n...Testing ttm for ttensor T and array of matrices A : vecnorm(full(ttm(T,A))-ttm(full(T),A)) = ", err)
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
  @test vecnorm(Tn-tenmat(full(T),n)) ≈ 0 atol=1e-12
end

R=[5,5,5];
T=hosvd(rand(60,50,40),reqrank=R);
sz = size(T);
X=full(T);
println("\n\n...Testing hosvd for tensor with noise.")
println("For ttensor T, X=full(T) tensor of size ",size(X)," and rank ",R," , N noise tensor and S=hosvd(X+N,",R,").")
rsz = tuple([[sz...]' 1 1]...)
Rdm = reshape(randn(rsz), sz);
N=1e-3 * vecnorm(X) * Rdm / vecnorm(Rdm);
Y = X + N;
S=hosvd(Y,reqrank=R);
err=vecnorm(T-S)
noise=vecnorm(N)
println("Error( vecnorm(T-S) ): ",err,". Noise vecnorm: ",noise,".")

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
for A in T.fmat
	print(size(A)," ")
end
@test vecnorm(full(T)-X) ≈ 0 atol=1e-5

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

println("\n\n...Testing vecnorm of ttensor T.")
err=abs(vecnorm(T) - vecnorm(full(T)))
println("|vecnorm(T) - vecnorm(full(T))| = ", err)
@test err ≈ 0 atol=1e-12

println("\n\n...Testing scalar multiplication 3*T.")
err=vecnorm(full(3*T) - 3*full(T))
println("vecnorm(full(3*T) - 3*full(T)) = ", err )
@test err ≈ 0 atol=1e-12

X=randttensor([6,8,2,5,4],[4,3,2,2,3]);
Y=randttensor([6,8,2,5,4],[3,6,3,4,3]);
println("\n\n...Creating two random ttensors X and Y of size ", size(X),".")
println("\n...Testing addition.")
Z=X+Y;
F=full(X)+full(Y);
err=vecnorm(full(Z) - F)
println("vecnorm(full(X+Y) - (full(X)+full(Y))) = ",err )
@test err ≈ 0 atol=1e-12

println("\n\n...Testing inner product.")
Z=innerprod(X,Y)
err=abs(Z - innerprod(full(X),full(Y)))
println("|innerprod(X,Y) - innerprod(full(X),full(Y))| = ",err )
@test err ≈ 0 atol=1e-10

println("\n\n...Testing Hadamard product.")
err=vecnorm(full(X.*Y) - full(X).*full(Y))
println("vecnorm(full(X.*Y) - full(X).*full(Y)) = ", err)
@test  err ≈ 0 atol=1e-10


println("\n\n...Testing singular values of matricizations of Tucker Tensor.")
R=[3,3,3]
T=randttensor([10,9,8],R);
println("\nSingular values of matricizations of random Tucker tensor of size ", size(T), ", rank ",R," and vecnorm ",vecnorm(T),".")
for n=1:ndims(T)
  sv = msvdvals(T,n)
  Tn=tenmat(T,n);
  println("Mode-$n singular values error: ",norm(sv-svdfact!(tenmat(full(T),n))[:S][1:length(sv)]))
  @test norm(sv-svdfact!(tenmat(full(T),n))[:S][1:length(sv)]) ≈ 0 atol=1e-10
end

R=[5,5,5,5]
T=randttensor([20,20,20,20],R);
println("\nSingular values of matricizations of random Tucker tensor of size ", size(T), ", rank ",R," and vecnorm ",vecnorm(T),".")
for n=1:ndims(T)
  sv = msvdvals(T,n)
  Tn=tenmat(T,n);
  println("Mode-$n singular values error: ",norm(sv-svdfact!(tenmat(full(T),n))[:S][1:length(sv)]))
  @test norm(sv-svdfact!(tenmat(full(T),n))[:S][1:length(sv)]) ≈ 0 atol=1e-10
end

println("\n...Testing function mhadtv.")
X=randttensor([5,4,3],[3,3,3])
Y=randttensor([5,4,3],[2,2,2])
println("\n\n...Creating two random ttensors X and Y of size ", size(X),".")
v=rand(12)
n=1
println("Multiplying mode-$n matricized had(X,Y) by a random vector.")
Z=mhadtv(X,Y,v,n,'n')
Hn=tenmat(X.*Y,n)
err = vecnorm(Z-Hn*v)
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-12
v=rand(5)
Z=mhadtv(X,Y,v,n,'t')
err = vecnorm(Z-Hn'*v)
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-12
v=rand(5)
Z=mhadtv(X,Y,v,n,'b')
err = vecnorm(Z-Hn*Hn'*v)
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-12

println("\n...Testing function mttkrp.")
X=randttensor([5,4,3],[3,3,3])
n=1
M1=rand(5,5);
M2=rand(4,5);
M3=rand(3,5);
M=[M1,M2,M3]
println("Multiplying mode-$n matricized tensor X by Khatri-Rao product of matrices.")
Z=mttkrp(X,M,n)
err = vecnorm(Z-tenmat(X,n)*khatrirao(M3,M2))
println("Multiplication error: ",err)
@test err ≈ 0 atol=1e-12
