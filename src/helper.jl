#export check_vector_input
export colspace, eye, ewprod, khatrirao, krontkron, kron, krontv, krtv, tkrtv, lanczos, lanczos_tridiag, randsvd
export VectorCell, MatrixCell, TensorCell

"""
    VectorCell(N)

Cell of vectors of length N.
"""
const VectorCell = Array{Vector,1}
"""
    MatrixCell(N)

Cell of matrices of length N.
"""
const MatrixCell = Array{Matrix,1}
"""
    TensorCell(N)

Cell of multidimensional arrays of length N.
"""
const TensorCell = Array{Array,1}

function ewprod(I1::CartesianIndex,I2::CartesianIndex)
  N=length(I1)
  @assert(length(I2)==N,"Dimension mismatch.")
  prod=zeros(Int,N)
  [prod[n]=I1[n]*I2[n] for n=1:N]
  CartesianIndex{N}(tuple(prod...))
end

"""
    check_vector_input(input,dim,default_value)

Check whether input vector is of appropriate size or if input is number create vector out of it.
"""
function check_vector_input(input,dim::Integer,default_value::Number)
  if length(input)>0
     if isa(input,Number)
        input=repeat([input],dim)
    else
      @assert(dim==length(input),"Dimensions mismatch.")
    end
  else #fixed-precision problem
    input=repeat([default_value],dim)
  end
  input
end

"""

    colspace(X; <keyword arguments>)

Column space basis.
## Arguments:
- `X`: Matrix.
- `method` ∈ {"svd","lanczos","randsvd"} Method for SVD. Default: "svd".
- `maxrank::Integer`: Maximal rank. Optional.
- `atol::Number`: Drop singular values below atol.  Default: 1e-8.
- `rtol::Number`: Drop singular values below rtol*sigma_1. Optional.
- `p::Integer`: Oversampling parameter used by lanczos and randsvd methods. Defaul p=10.
"""
function colspace(X::Matrix{T};method="svd",maxrank=0,atol=1e-8,rtol=0,p=10) where T<:Number
  if method == "lanczos"
    U,S=lanczos(X,tol=atol,maxrank=maxrank,p=p)
  elseif method == "randsvd"
    U,S=randsvd(X,tol=atol,maxrank=maxrank,p=p)
  else
    U,S,V=svd(X)
  end
  if maxrank!=0 && size(U,2)>maxrank
    U=U[:,1:maxrank]
    if maxrank<length(S)
      S=S[1:maxrank]
    end
  end
  rtol != 0 ? tol=rtol*S[1] : tol=atol
  K=findall(x-> x>tol ? true : false,S)
  U[:,K]
end

"""
    eye(n::Integer)

Identity matrix of size nxn.
"""
function eye(n::Integer)
  Matrix(1.0I, n, n)
end

"""
    khatrirao(M,t='n')
    khatrirao(M,n,t='n')

Khatri-Rao product of matrices from M:  M₁⊙ ⋯ ⊙ Mₙ. Optionally skip nth matrix.
If t='t', compute transpose Khatri-Rao product: M₁⊙ᵀ ⋯ ⊙ᵀ Mₙ.
"""
function khatrirao(M::MatrixCell,t='n')
    N=length(M)
    if t== 't'
      sz=[size(M[n],1) for n=1:N]
      @assert(any(map(Bool,sz .-size(M[1],1)))==0,"Matrices must have the same number of rows.")
      [M[n]=M[n]' for n=1:N]
      khatrirao(M)'
    else
      K=[size(M[n],2) for n=1:N]
      @assert(any(map(Bool,K .-size(M[1],2)))==0,"Matrices must have the same number of columns.")
      J,K=size(M[end])
      X=reshape(M[end],J, 1, K)
      for n=N-1:-1:1
        sz=size(M[n],1)
        Y=reshape(M[n],1,sz,K)
        X=reshape(Y.*X,sz*J,1,K)
        J=sz*J
      end
      reshape(X,size(X,1),K)
    end
end
khatrirao(M::Array{Matrix{T}},t='n') where T<:Number =khatrirao(MatrixCell(M),t)
function khatrirao(M1::Matrix{T1}, M2::Matrix{T2},t='n') where {T1<:Number,T2<:Number}
  M=MatrixCell(undef,2);
  M[1]=M1;M[2]=M2;
  khatrirao(M,t)
end
function khatrirao(M1::MatrixCell, M2::MatrixCell,t='n')
    @assert(length(M1)==length(M2),"Matrix cells must be of same length.")
    N=length(M1)
    M=MatrixCell(undef,N)
    for n=1:N
      M[n]=khatrirao(M1[n],M2[n],t)
    end
    M
end
#Skip nth matrix from M
function khatrirao(M::MatrixCell,n::Integer,t='n')
    khatrirao(deleteat!(M,n),t)
end
khatrirao(M::Array{Matrix{T}},n::Integer,t='n') where T<:Number=khatrirao(MatrixCell(M),n,t)


#Extension of Base.kron to work with MatrixCell
function kron(M::MatrixCell,t='n')
  N=length(M)
  t=='n' ? K=M[1] : K=M[1]'
  for n=2:N
      t=='n' ? K=kron(K,M[n]) : K=kron(K,M[n]')
  end
  K
end
#If array of matrices isn't defined as MatrixCell, but as M=[M1,M2,...,Mn]:
kron(M::Array{Matrix{T}},t='n') where T<:Number =kron(MatrixCell(M),t)

"""
    krontkron(A,v,t='n')

Kronecker product of matrices multiplied by Kronecker product of vectors.
"""
function krontkron(A::MatrixCell,v::VectorCell,t='n')
  if t=='t'
      [A[n]=A[n]' for n=1:length(A)]
  end
  @assert(length(A)==length(v),"There must be the same number of matrices and vectors.")
  N=length(A)
  @assert(!any(map(Bool,[size(A[n],2)-length(v[n]) for n=1:N])),"Dimension mismatch.")
  w=A[1]*v[1]
  [w=kron(w,A[n]*v[n]) for n=2:N]
  w
end
krontkron(A::Array{Matrix{T1}},v::Array{Vector{T2}},t='n') where {T1<:Number,T2<:Number}=krontkron(MatrixCell(A),VectorCell(v),t)
krontkron(A::Array{Matrix{T}},v::VectorCell,t='n') where T<:Number=krontkron(MatrixCell(A),v,t)
krontkron(A::MatrixCell,v::Array{Vector{T}},t='n') where T<:Number=krontkron(A,VectorCell(v),t)

"""
    krontv(A,B,v)

Kronecker product times vector: (A ⊗ B)v.
If v is a matrix, multiply column by column.
"""
function krontv(A::Matrix{T1},B::Matrix{T2},v::Vector{T3}) where {T1<:Number,T2<:Number,T3<:Number}
  m,n=size(A);
  p,q=size(B);
  @assert(length(v)==q*n, "Dimensions mismatch.")
  if n*q*p+m*n*p <= m*q*p+q*n*m
    vec(B*reshape(v,q,n)*A');
  else
    vec(B*(reshape(v,q,n)*A'));
  end
end
function krontv(A::Matrix{T1},B::Matrix{T2},M::Matrix{T3}) where {T1<:Number,T2<:Number,T3<:Number}
  if sort(collect(size(vec(M))))[1]==1
        return krontv(A,B,vec(M))
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

"""
    krtv(A,B,v)

Khatri-Rao product times vector: (A ⊙ B)v.
If v is a matrix, multiply column by column.
"""
function krtv(A::Matrix{T1},B::Matrix{T2},v::Vector{T3}) where {T1<:Number,T2<:Number,T3<:Number}
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
function krtv(A::Matrix{T1},B::Matrix{T2},M::Matrix{T3}) where {T1<:Number,T2<:Number,T3<:Number}
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

"""
   tkrtv(A,B,v)

Transpose Khatri-Rao product times vector: (A ⊙ᵀ B)v.
If v is a matrix, multiply column by column.
"""
function tkrtv(A::Matrix{T1},B::Matrix{T2},v::Vector{T3}) where {T1<:Number,T2<:Number,T3<:Number}
  @assert(size(A,1)==size(B,1),"Dimension mismatch.")
  m,n=size(A)
  p=size(B,2)
  @assert(length(v)==n*p, "Dimensions mismatch.")
  if n<=p
    sum((B*reshape(v,p,n)).*A,dims=2)
  else
    sum(B.*(A*reshape(v,p,n)'),dims=2)
  end
end
function tkrtv(A::Matrix{T1},B::Matrix{T2},M::Matrix{T3}) where {T1<:Number,T2<:Number,T3<:Number}
  @assert(size(A,1)==size(B,1),"Dimension mismatch.")
  if sort(collect(size(vec(M))))[1]==1
    return tkrtv(A,B,vec(M))
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


"""
    lanczos(A; <keyword arguments>)

Lanczos based SVD - computes left singular vectors and singular values of a matrix.

## Arguments:
- `A::Matrix`
- `tol`: Tolerance - discard singular values below tol. Default: 1e-8.
- `maxit`: Maximal number of iterations. Default: 1000.
- `reqrank`: Number of singular values and singular vectors to compute. Optional.
- `p`: Oversampling parameter. Default: p=10.
"""
function lanczos(A::Matrix{N};tol=1e-8,maxit=1000,reqrank=0,p=10) where N<:Number
  Q,T=lanczos_tridiag(A,tol=tol,maxit=maxit,reqrank=reqrank,p=p)
  E=eigen(T,tol,Inf);
  U=E.vectors[:,end:-1:1];
  S=sqrt.(abs.(E.values[end:-1:1]));
  if reqrank!=0
    U=Q*U[:,1:reqrank];
    S=S[1:reqrank];
  else
    K=findall(x-> x>tol ? true : false,S)
    U=Q*U[:,K];
    S=S[K];
  end
  U,S
end

"""
    lanczos_tridiag(A; <keyword arguments>)

Lanczos tridiagonalization algorithm - returns orthonormal Q and symmetric tridiagonal T such that A≈Q*T*Q'.

## Arguments:
- `A::Matrix`
- `tol`: Tolerance. Default: 1e-8.
- `maxit`: Maximal number of iterations. Default: 1000.
- `reqrank`: Number of singular values and singular vectors to compute. Optional.
- `p`: Oversampling parameter. Default: p=10.
"""
function lanczos_tridiag(A::Matrix{N};tol=1e-8,maxit=1000,reqrank=0,p=10) where N<:Number
  m,n=size(A)
  K=min(m,maxit);
  if reqrank != 0
      K=min(reqrank+p,K);
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
      K=k
      break
    end
    if k!=K
      Q=[Q r/β[k]]
    end
  end
  T=SymTridiagonal(α[1:K], β[1:K-1])
  Q,T
  end

function randrange(A::Matrix{T},gram=true,t='t';tol=1e-8,maxit=1000,reqrank=0,r=10,p=10) where T<:Number
  m,n=size(A)
  if reqrank!=0
    if t=='t'
      Y=A'*randn(m,reqrank+p)
      if gram
        Y=A*Y
      end
    elseif t=='n'
      Y=A*randn(n,reqrank+p)
      if gram
        Y=A'*Y
      end
    else
      error("Wrong input.")
    end
    Q=qr(Y).Q;
  else
    maxit=min(m,n,maxit);
    rangetol=tol*sqrt.(pi/2)/10; #assures ||A-Q*Q'*A||<=tol
    if t=='t'
      Y=A'*randn(m,r)
      if gram
        Y=A*Y
        Q=zeros(m,0)
      else
        Q=zeros(n,0)
      end
    elseif t=='n'
      Y=A*randn(n,r)
      if gram
        Y=A'*Y
        Q=zeros(n,0)
      else
        Q=zeros(m,0)
      end
    end

    j=0
    maxcolnorm=maximum([norm(Y[:,i]) for i=1:r]);
    while maxcolnorm > rangetol && j<maxit
      j+=1;
      v=Q'*Y[:,j];
      Y[:,j]-=Q*v;
      q=Y[:,j]/norm(Y[:,j]);
      Q=[Q q];
      if t=='t'
        w=A'*randn(m)
        if gram
          w=A*w
        end
      elseif t=='n'
        w=A*randn(n)
        if gram
          w=A'*w
        end
      end
      v=Q'*w;
      Y=[Y w-Q*v]; #Y[:,j+r]=w-Q*v;
      v=q'*Y[:,j+1:j+r-1]
      Y[:,j+1:j+r-1]-=q*v;
      maxcolnorm=maximum([norm(Y[:,i]) for i=j+1:j+r]);
    end
  end
  Q
end

"""
   randsvd(A; <keyword arguments>)

Randomized SVD algorithm - returns left singular vectors and singular values of a matrix.

## Arguments:
- `A::Matrix`
- `tol`: Tolerance - discard singular values below tol. Default: 1e-8.
- `maxit`: Maximal number of iterations. Default: 1000.
- `reqrank`: Number of singular values and singular vectors to compute. Optional.
- `r`: Number of samples for stopping criterion. Default: r=10.
- `p`: Oversampling parameter. Default: p=10.
"""
function randsvd(A::Matrix{T},svdvecs="left";tol=1e-8,maxit=1000,reqrank=0,r=10,p=10) where T<:Number
  m,n=size(A)
  if svdvecs=="left" || svdvecs=="right"
    svdvecs=="left" ? t ='t' : t='n'
    Q=randrange(A,true,t,tol=tol,maxit=maxit,reqrank=reqrank,r=r,p=p)
    svdvecs=="left" ? B=A'*Q : B=A*Q
    B=Symmetric(B'*B)
    E=eigen(B,tol,Inf)
    U=E.vectors[:,end:-1:1];
    S=sqrt.(abs.(E.values[end:-1:1]))
    if reqrank != 0
      if size(U,2)<reqrank
	      warn("Requested rank exceeds the actual rank. Try changing tolerance.");
        U=Q*U;
      else
	      U=Q*U[:,1:reqrank]
      	S=S[1:reqrank]
      end
    else
      K=findall(x-> x>tol ? true : false,S)
      U=Q*U[:,K]
      S=S[K]
    end
    U,S
  elseif svdvecs=="both"
    Q=randrange(A,false,'n',tol=tol,maxit=maxit,reqrank=reqrank,r=r,p=p)
    B=transpose(Q)*A
    U,S,V=svd(B)
    if reqrank != 0
      if size(U,2)<reqrank
	      warn("Requested rank exceeds the actual rank. Try changing tolerance.");
        U=Q*U;
      else
	      U=Q*U[:,1:reqrank]
      	S=S[1:reqrank]
        V=V[:,1:reqrank]
      end
    else
      K=findall(x-> x>tol ? true : false,S)
      U=Q*U[:,K]
      S=S[K]
      V=V[:,K]
    end
    U,S,V
  end
end
