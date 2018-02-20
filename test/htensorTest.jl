using TensorToolbox, TimeIt, Base.Test

M1=[2 3;4 5;6 7;8 9;0 0;0 0;0 0;0 0;0 0];
M2=[2 3;0 0;4 5;0 0;6 7;0 0;8 9;0 0;0 0];
M3=[2 3;4 5;6 7;8 9;10 11;0 0;12 13;0 0;0 0;0 0;0 0;0 0;0 0];

T1=dimtree(M1)
T2=dimtree(M2)
T3=dimtree(M3)

X=float(reshape(collect(1:32),(2,2,2,2,2)))
H1=htdecomp(X,T1);
@test vecnorm(full(H1)-X) ≈ 0 atol=1e-10

H2=htdecomp(X,dimtree(T2))
@test vecnorm(full(H2)-X) ≈ 0 atol=1e-10

Y=float(reshape(collect(1:2^7),(2,2,2,2,2,2,2)));
H3=htdecomp(Y,dimtree(M3));
@test vecnorm(full(H3)-Y) ≈ 0 atol=1e-10

X=rand(5,4,3,2,4,2,7);
H=htdecomp(X,create_dimtree(X));
@test vecnorm(full(H)-X) ≈ 0 atol=1e-10
