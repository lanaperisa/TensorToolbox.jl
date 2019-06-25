#export check_vector_input
export colspace, eye, ewprod, khatrirao, krontkron, kron, krontv, krtv, tkrtv, perfect_shuffle, lanczos, lanczos_tridiag, randsvd
export VectorCell, MatrixCell, TensorCell, CoreCell

"""
    VectorCell(undef,N)

Cell of vectors of length N.
"""
const VectorCell = Array{AbstractVector{<:Number},1}
"""
    MatrixCell(undef,N)

Cell of matrices of length N.
"""
const MatrixCell = Array{AbstractMatrix{<:Number},1}
"""
    TensorCell(undef,N)

Cell of multidimensional arrays of length N.
"""
const TensorCell = Array{AbstractArray{<:Number},1}
export CoreCell

"""
    CoreCell(undef,N)

Cell of 3D tensors of length N. Suitable for the cores of TTtensor.
"""
const CoreCell = Array{AbstractArray{<:Number,3},1}

# """
#     ewprod(I1::CartesianIndex,I2::CartesianIndex)
#
# Element-wise product of two inputs of type CartesianIndex.
# """
function ewprod(I1::CartesianIndex,I2::CartesianIndex)
  N=length(I1)
  @assert(length(I2)==N,"Dimension mismatch.")
  prod=zeros(Int,N)
  [prod[n]=I1[n]*I2[n] for n=1:N]
  CartesianIndex{N}(tuple(prod...))
end

"""
    check_vector_input(input,dim,default_value)

Check whether the input vector is of appropriate size or if input is number create vector out of it.
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
function colspace(X::AbstractMatrix{<:Number};method="svd",maxrank=0,atol=1e-8,rtol=0,p=10)
  if method == "lanczos"
    U,S=lanczos(X,tol=atol,reqrank=maxrank,p=p)
  elseif method == "randsvd"
    U,S=randsvd(X,tol=atol,reqrank=maxrank,p=p)
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
    eye(n::Integer[,full=1])

Identity matrix of size nxn. If full=0, returns type Diagonal.
"""
function eye(n::Integer,full=1)
  full==1 ? Matrix(1.0I, n, n) : Diagonal(ones(n))
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
khatrirao(M::Array{Matrix{<:Number}},t='n')=khatrirao(MatrixCell(M),t)
function khatrirao(M1::AbstractMatrix{<:Number}, M2::AbstractMatrix{<:Number},t='n')
  M=MatrixCell(undef,2)
  M[1]=M1;M[2]=M2
  khatrirao(M,t)
end
function khatrirao(M1::MatrixCell, M2::MatrixCell,t='n')
    @assert(length(M1)==length(M2),"Matrix cells must be of the same length.")
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
khatrirao(M::Array{Matrix{<:Number}},n::Integer,t='n') =khatrirao(MatrixCell(M),n,t)


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
kron(M::Array{AbstractMatrix{<:Number}},t='n')=kron(MatrixCell(M),t)

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
#If arrays are not defined as VectorCell/MatrixCell, but as [M1,M2,...,Mn]:
krontkron(A::Array{Matrix{<:Number}},v::Array{Vector{<:Number}},t='n')=krontkron(MatrixCell(A),VectorCell(v),t)
krontkron(A::Array{Matrix{<:Number}},v::VectorCell,t='n')=krontkron(MatrixCell(A),v,t)
krontkron(A::MatrixCell,v::Array{Vector{<:Number}},t='n')=krontkron(A,VectorCell(v),t)

"""
    krontv(A,B,v)

Kronecker product times vector: (A ⊗ B)v.
If v is a matrix, multiply column by column.
"""
function krontv(A::AbstractMatrix{<:Number},B::AbstractMatrix{<:Number},v::AbstractVector{<:Number})
  m,n=size(A);
  p,q=size(B);
  @assert(length(v)==q*n, "Dimensions mismatch.")
  if n*q*p+m*n*p <= m*q*p+q*n*m
    vec(B*reshape(v,q,n)*A');
  else
    vec(B*(reshape(v,q,n)*A'));
  end
end
function krontv(A::AbstractMatrix{<:Number},B::AbstractMatrix{<:Number},M::AbstractMatrix{<:Number})
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
function krtv(A::AbstractMatrix{<:Number},B::AbstractMatrix{<:Number},v::AbstractVector{<:Number})
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
function krtv(A::AbstractMatrix{<:Number},B::AbstractMatrix{<:Number},M::AbstractMatrix{<:Number})
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
function tkrtv(A::AbstractMatrix{<:Number},B::AbstractMatrix{<:Number},v::AbstractVector{<:Number})
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
function tkrtv(A::AbstractMatrix{<:Number},B::AbstractMatrix{<:Number},M::AbstractMatrix{<:Number})
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
    perfect_shuffle(piles_nmbr,piles_len,piles_block=0)

Perfect shuffle vector.
"""
function perfect_shuffle(piles_nmbr::Integer,piles_len::Integer,piles_block=[])
    if piles_block==[]
        p=zeros(Int,piles_nmbr*piles_len)
        let
            i=1
            for l=1:piles_len, n=1:piles_nmbr
                p[i]=(n-1)*piles_len+l
                i+=1
            end
        end
    else
        @assert(length(piles_block)==piles_nmbr,"Dimension mismatch.")
        #p=zeros(Int,piles_nmbr*piles_len*piles_block)
        p=zeros(Int,1,0)
        for l=1:piles_len, n=1:piles_nmbr
            p=[p sum(piles_block[1:n-1])*piles_len .+ collect((l-1)*piles_block[n]+1:l*piles_block[n])...]
        end
    end
    p[:]
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
function lanczos(A::AbstractMatrix{<:Number};tol=1e-8,maxit=1000,reqrank=0,p=10)
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
function lanczos_tridiag(A::AbstractMatrix{<:Number};tol=1e-8,maxit=1000,reqrank=0,p=10)
  m,n=size(A)
  K=min(m,maxit);
  if reqrank != 0
      K=min(reqrank+p,K);
  end
  isreal(A) ? α=zeros(K) : α=zeros(Complex,K)
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
  isreal(α) ? T=SymTridiagonal(α[1:K], β[1:K-1]) : T=SymTridiagonal(real(α[1:K]), β[1:K-1]) +im*Diagonal(imag(α[1:K])) #T=Hermitian(Matrix(SymTridiagonal(real(α[1:K]), β[1:K-1]) +im*Diagonal(imag(α[1:K]))))
  Q,T
  end

function randrange(A::AbstractMatrix{<:Number},gram=true,t='t';tol=1e-8,maxit=1000,reqrank=0,r=10,p=10)
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
    Q=Matrix(qr(Y).Q)
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
function randsvd(A::AbstractMatrix{<:Number},svdvecs="left";tol=1e-8,maxit=1000,reqrank=0,r=10,p=10)
  m,n=size(A)
  if svdvecs=="left" || svdvecs=="right"
    svdvecs=="left" ? t ='t' : t='n'
    Q=randrange(A,true,t,tol=tol,maxit=maxit,reqrank=reqrank,r=r,p=p)
    svdvecs=="left" ? B=A'*Q : B=A*Q
    if isreal(B)
      B=Symmetric(B'*B)
    else
      B=Hermitian(B'*B)
    end
    E=eigen(B,tol,Inf)
    U=E.vectors[:,end:-1:1];
    S=sqrt.(abs.(E.values[end:-1:1]))
    if reqrank != 0
      if size(U,2)<reqrank
	      @warn "Requested rank exceeds the actual rank. Try changing tolerance."
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
	      @warn "Requested rank exceeds the actual rank. Try changing tolerance."
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
