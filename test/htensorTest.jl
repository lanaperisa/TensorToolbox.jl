#using TensorToolbox, Test, LinearAlgebra

T1=dimtree([8,9,5,6,7])
T2=dimtree([2,4,6,8,9])
T3=dimtree(7)

X=float(reshape(collect(1:32),(2,2,2,2,2)))
println("\n\nTesting truncation of tensor X of size ",size(X)," to htensor.")
H1=htrunc(X,T1);
err=norm(full(H1)-X)
println("norm(full(H)-X) = ",err)
@test err ≈ 0 atol=1e-10

H2=htrunc(X,T2)
err=norm(full(H2)-X)
println("norm(full(H)-X) = ",err)
@test err ≈ 0 atol=1e-10

Y=float(reshape(collect(1:2^7),(2,2,2,2,2,2,2)));
H3=htrunc(Y,T3);
@test norm(full(H3)-Y) ≈ 0 atol=1e-10

X=rand(5,4,3,2,4,2);
H=htrunc(X,dimtree(ndims(X)),maxrank=20);
@test norm(full(H)-X) ≈ 0 atol=1e-10

println("\n\n...Testing size of htensor H : ", size(H))
@test size(H) == size(X)
println("\n...Testing ndims of htensor H : ", ndims(H))
@test ndims(H) == ndims(X)


A=MatrixCell(undef,ndims(H))
for n=1:ndims(H)
  A[n]=rand(rand(1:10),size(H,n))
end
err=norm(full(ttm(H,A)) - ttm(full(H),A))
println("\n...Testing ttm for ttensor H and array of matrices A : norm(full(ttm(H,A))-ttm(full(H),A)) = ", err)
@test err ≈ 0 atol=1e-10

println("\n\n...Testing norm of htensor H.")
err=abs(norm(H) - norm(full(H)))
println("|norm(H) - norm(full(H))| = ", err)
@test err ≈ 0 atol=1e-10

println("\n\n...Testing scalar multiplication 3*H.")
err=norm(full(3*H) - 3*full(H))
println("norm(full(3*H) - 3*full(H)) = ", err )
@test err ≈ 0 atol=1e-10

println("\n\nTesting functions plus and minus for htensors H1 and H2.")
T=dimtree(3)
H1=htrunc(rand(5,4,3),T)
H2=htrunc(rand(5,4,3),T)
H=H1+H2
F=full(H1)+full(H2);
err=norm(full(H) - F)
println("norm(full(H1+H2) - (full(H1)+full(H2))) = ",err )
@test err ≈ 0 atol=1e-10
H=H1-H2
F=full(H1)-full(H2);
err=norm(full(H) - F)
println("norm(full(H1+H2) - (full(H1)+full(H2))) = ",err )
@test err ≈ 0 atol=1e-10

println("\n\n...Testing inner product.")
H=innerprod(H1,H2)
err=abs(H - innerprod(full(H1),full(H2)))
println("|innerprod(H1,H2) - innerprod(full(H1),full(H2))| = ",err )
@test err ≈ 0 atol=1e-10

println("\n\nTesting function dropdims for squeezed tensor X and squeezed htensor Y.")
X=rand(1,2,4,3,2)
H=htrunc(X)
Y=dropdims(H)
err = norm(full(Y)-dropdims(X,dims=1))
println("norm(full(Y)-dropdims(X,dims=1)) = ",err)
@test err ≈ 0 atol=1e-10
X=rand(2,1,4,3,2)
H=htrunc(X)
Y=dropdims(H)
err = norm(full(Y)-dropdims(X,dims=2))
println("norm(full(Y)-dropdims(X,dims=2)) = ",err)
@test err ≈ 0 atol=1e-10
X=rand(2,4,1,3,2)
H=htrunc(X)
Y=dropdims(H)
err = norm(full(Y)-dropdims(X,dims=3))
println("norm(full(Y)-dropdims(X,dims=3)) = ",err)
@test err ≈ 0 atol=1e-10
X=rand(2,4,3,1,2)
H=htrunc(X)
Y=dropdims(H)
err = norm(full(Y)-dropdims(X,dims=4))
println("norm(full(Y)-dropdims(X,dims=4)) = ",err)
@test err ≈ 0 atol=1e-10
X=rand(2,4,3,2,1)
H=htrunc(X)
Y=dropdims(H)
err = norm(full(Y)-dropdims(X,dims=5))
println("norm(full(Y)-dropdims(X,dims=5)) = ",err)
@test err ≈ 0 atol=1e-10

H=randhtensor([4,5,2])
println("\n\n...Testing reorthogonalization for random htensor H of size ", size(H),".")
println("Has orthogonal factor matrices: ", H.isorth)
println("After reorthogonalization: ", reorth!(H).isorth)
@test H.isorth

H=htrunc(rand(5,5,5),dimtree(3),maxrank=2);
println("Testing hrank: ",hrank(H))
@test hrank(H)==[1, 2, 2, 2, 2]
