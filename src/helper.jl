using Base.Cartesian

import Base.reshape

#export indicesmat2vec, indicesmat, shiftsmat
export khatrirao, krontkron, krontv, krtv, tkrtv, lanczos, randsvd
export MatrixCell, VectorCell

typealias MatrixCell Array{Matrix,1}
typealias VectorCell Array{Vector,1}

#Extended Base.reshape to work with p defined as vector
function reshape{T<:Number,D<:Integer}(A::Array{T},p::Vector{D})
    reshape(A,tuple(p...))
end

#Transforms matrix of multi-indices into a vector of linear indices
function indicesmat2vec{D<:Integer}(I::Matrix{D},sz::Tuple)
	mult = [1 cumprod([sz[1:end-1]...])']
	(I - 1) * mult' + 1
end

#Creates a matrix of all multi-indices of a tensor shifted by a vector - each row of a matrix is one multi-index
@generated function indicesmat{T<:Number,D<:Integer,N}(A::Array{T,N},shift::Vector{D})
  quote
	I=zeros(D,0,N)
	@nloops $N i A begin
	   	ind = [(@ntuple $N i)...]
   		I=vcat(I,(ind+shift)')
	end
	I
  end
end

#Creates a matrix of all shifts for element-wise multiplication with block sizes defined in blsize - each row of a matrix is one shift vector
@generated function shiftsmat{T<:Number,D<:Integer,N}(A::Array{T,N},blsize::Vector{D})
  quote
    I=zeros(D,0,N)
    @nloops $N i A begin
        in = ([(@ntuple $N i)...]-ones([size(A)...])).*blsize
        I=vcat(I,in') #'
    end
    I
    end
end

@doc """ Khatri-Rao product of matrices. """ ->
#if t='t' computes transpose Khatri-Rao
function khatrirao(M::MatrixCell,t='n')
    N=length(M)
    if t== 't'
      I=[size(M[n],1) for n=1:N]
      @assert(any(map(Bool,I-size(M[1],1)))==0,"Matrices must have the same number of rows.")
      [M[n]=M[n]' for n=1:N]
      khatrirao(M)'
    else
      K=[size(M[n],2) for n=1:N]
      @assert(any(map(Bool,K-size(M[1],2)))==0,"Matrices must have the same number of columns.")
      J,K=size(M[end])
      X=reshape(M[end],J, 1, K)
      for n=N-1:-1:1
        I=size(M[n],1)
        Y=reshape(M[n],1,I,K)
        X=reshape(Y.*X,I*J,1,K)
        J=I*J
      end
      reshape(X,size(X,1),K)
    end
end
khatrirao{T<:Number}(M::Array{Matrix{T}},t='n')=khatrirao(MatrixCell(M))
function khatrirao{T1<:Number,T2<:Number}(M1::Matrix{T1}, M2::Matrix{T2},t='n')
  M=MatrixCell(2);
  M[1]=M1;M[2]=M2;
  khatrirao(M,t)
end

@doc """Kronecker times Kronecer. Kronecker product of matrices multiplied by kronecker product of vectors. """ ->
function krontkron(A::MatrixCell,v::VectorCell,t='n')
  if t=='t'
    A=vec(A')
  end
  @assert(length(A)==length(v),"There must be the same number of matrices and vectors.")
  N=length(A);
  @assert(!any(map(Bool,[size(A[n],2)-length(v[n]) for n=1:N])),"Dimension mismatch.")
  w=A[1]*v[1];
  [w=kron(w,A[n]*v[n]) for n=2:N];
  w
end
krontkron{T1<:Number,T2<:Number}(A::Array{Matrix{T1}},v::Array{Vector{T2}},t='n')=krontkron(MatrixCell(A),VectorCell(v),t)
krontkron{T<:Number}(A::Array{Matrix{T}},v::VectorCell,t='n')=krontkron(MatrixCell(A),v,t)
krontkron{T<:Number}(A::MatrixCell,v::Array{Vector{T}},t='n')=krontkron(A,VectorCell(v),t)

@doc """Kronecker product times vector or matrix. """
function krontv{T1<:Number,T2<:Number,T3<:Number}(A::Matrix{T1},B::Matrix{T2},v::Vector{T3})
  m,n=size(A);
  p,q=size(B);
  @assert(length(v)==q*n, "Dimensions mismatch.")
  if n*q*p+m*n*p <= m*q*p+q*n*m
    vec(B*reshape(v,q,n)*A');
  else
    vec(B*(reshape(v,q,n)*A'));
  end
end
function krontv{T1<:Number,T2<:Number,T3<:Number}(A::Matrix{T1},B::Matrix{T2},M::Matrix{T3})
  if sort(collect(size(vec(M))))[1]==1
        return krontv(A,B,vec(M));
  end
  m,n=size(A);
  p,q=size(B);
  @assert(size(M,1)==q*n, "Dimensions mismatch.");
  Mprod=zeros(m*p,size(M,2))
  for j=1:size(M,2)
    Mprod[:,j]=krontv(A,B,M[:,j])
  end
  Mprod
end

@doc """Khatri-Rao product times a vector or a matrix. """
function krtv{T1<:Number,T2<:Number,T3<:Number}(A::Matrix{T1},B::Matrix{T2},v::Vector{T3})
  @assert(size(A,2)==size(B,2),"Dimension mismatch.")
  m,n=size(A);
  p=size(B,1);
  @assert(length(v)==n, "Dimensions mismatch.")
  if n<=p
    vec(B*(v.*A'));
  else
    vec((B.*v')*A');
  end
end
function krtv{T1<:Number,T2<:Number,T3<:Number}(A::Matrix{T1},B::Matrix{T2},M::Matrix{T3})
  @assert(size(A,2)==size(B,2),"Dimension mismatch.")
  if sort(collect(size(vec(M))))[1]==1
    return krtv(A,B,vec(M));
  end
  m,n=size(A);
  p=size(B,1);
  @assert(size(M,1)==n, "Dimensions mismatch.");
  Mprod=zeros(m*p,size(M,2))
  for j=1:size(M,2)
    Mprod[:,j]=krtv(A,B,M[:,j])
  end
  Mprod
end

@doc """Transpose Khatri-Rao product times vector of matrix. """
function tkrtv{T1<:Number,T2<:Number,T3<:Number}(A::Matrix{T1},B::Matrix{T2},v::Vector{T3})
  @assert(size(A,1)==size(B,1),"Dimension mismatch.")
  m,n=size(A);
  p=size(B,2);
  @assert(length(v)==n*p, "Dimensions mismatch.")
  if n<=p
    sum((B*reshape(v,p,n)).*A,2);
  else
    sum(B.*(A*reshape(v,p,n)'),2);
  end
end
function tkrtv{T1<:Number,T2<:Number,T3<:Number}(A::Matrix{T1},B::Matrix{T2},M::Matrix{T3})
  @assert(size(A,1)==size(B,1),"Dimension mismatch.")
  if sort(collect(size(vec(M))))[1]==1
    return tkrtv(A,B,vec(M));
  end
  m,n=size(A);
  p=size(B,2);
  @assert(size(M,1)==n*p, "Dimensions mismatch.");
  Mprod=zeros(m,size(M,2))
  for j=1:size(M,2)
    Mprod[:,j]=tkrtv(A,B,M[:,j])
  end
  Mprod
end


@doc """ Lanczos tridiagonalization algorithm - returns left singular vectors and singular values of a matrix. """ ->
function lanczos{N<:Number}(A::Matrix{N};tol=1e-8,maxit=1000,reqrank=0)
  m,n=size(A)
  K=min(m,maxit);
  if reqrank != 0
      K=min(reqrank+10,K);
  end
  α=zeros(K)
  β=zeros(K)
  v=randn(m);
  q=v/norm(v);
  Q=zeros(m,1)
  Q[:,1]=q;
  k=0; #needed if stopping criterion is met
  for k=1:K
    r=A*(A'*Q[:,k])
    α[k]=dot(r,Q[:,k])
    r=r-α[k]*Q[:,k] #without orthogonalization: r=r-α[k]*Q[:,k]-β[k-1]*Q[:,k-1]
    [r=r-Q*(Q'*r) for i=1:3]
    β[k]=norm(r)
    if β[k] < tol
      break
    end
    if k!=K
      Q=[Q r/β[k]]
    end
  end
  T=SymTridiagonal(α[1:k], β[1:k-1])
  E=eigfact(T,tol,Inf);
  U=E[:vectors][:,end:-1:1];
  S=sqrt(abs(E[:values][end:-1:1]));
  if reqrank!=0
    U=Q*U[:,1:reqrank];
    S=S[1:reqrank];
  else
    I=find(x-> x>tol ? true : false,S)
    U=Q*U[:,I];
    S=S[I];
  end
  U,S
end

@doc """ Randomized SVD algorithm - returns left singular vectors and singular values of a matrix. """ ->
function randsvd{N<:Number}(A::Matrix{N};tol=1e-8,maxit=1000,reqrank=0,r=10)
  #r... oversampling parameter
  m,n=size(A)
  if reqrank!=0
    Y=A*(A'*randn(m,reqrank+r));
    Q=qrfact(Y)[:Q];
  else
    maxit=min(m,n,maxit);
    rangetol=tol*sqrt(pi/2)/10; #assures ||A-Q*Q'*A||<=tol
    Y=A*(A'*randn(m,r));
    j=0;
    Q=zeros(m,0);
    maxcolnorm=maximum([norm(Y[:,i]) for i=1:r]);
    while maxcolnorm > rangetol && j<maxit
      j+=1;
      p=Q'*Y[:,j];
      Y[:,j]-=Q*p;
      q=Y[:,j]/norm(Y[:,j]);
      Q=[Q q];
      w=A*(A'*randn(m));
      p=Q'*w;
      Y=[Y w-Q*p]; #Y[:,j+r]=w-Q*p;
      p=q'*Y[:,j+1:j+r-1]
      Y[:,j+1:j+r-1]-=q*p;
      maxcolnorm=maximum([norm(Y[:,i]) for i=j+1:j+r]);
    end
  end
  B=A'*Q;
  B=Symmetric(B'*B);
  E=eigfact(B,tol,Inf);
  U=E[:vectors][:,end:-1:1];
  S=sqrt(abs(E[:values][end:-1:1]));
  if reqrank != 0
    U=Q*U[:,1:reqrank];
    S=S[1:reqrank];
  else
    I=find(x-> x>tol ? true : false,S)
    U=Q*U[:,I];
    S=S[I];
  end
  U,S
end
