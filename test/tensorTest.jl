println("\n\n****Testing tensor.jl")

X=rand(20,10,50,5)

println("\n...Test tensor X of size: ", size(X))
println("\n...Testing matten and tenmat by mode.")
for n=1:ndims(X)
  Xn=tenmat(X,n);
  N=setdiff(1:ndims(X),n);
  println("Size of $n-mode matricization: ", [size(Xn)...])
  @test size(Xn) == (size(X,n),prod([size(X,N[k]) for k=1:length(N)]))
  println("Check if it folds back correctly: ",matten(Xn,n,[size(X)...]) == X)
  @test matten(Xn,n,[size(X)...]) == X
end
println("\n...Testing matten and tenmat by rows and columns.")
R=[2,1];C=[4,3];
Xmat=tenmat(X,R=R,C=C);
@test matten(Xmat,R,C,[size(X)...]) == X
