#Tensors in Tucker format + functions

export ttensor, randttensor
export coresize, full, had, hadcten, hosvd, hosvd1, hosvd2, hosvd3, hosvd4, innerprod, isequal, lanczos, mhadtm, mhadtv, minus, mrank
export msvdvals, mtimes, mttkrp, ndims, norm, nrank, nvecs, permutedims, plus, randrange, randsvd, reorth, reorth!, size, tenmat, ttm, ttv, uminus

import Base: size, ndims, +, -, *, .*, ==, full, isequal, permutedims

type ttensor{T<:Number}
	cten::Array{T}
	fmat::MatrixCell
	isorth::Bool
	function ttensor(cten::Array{T},fmat::MatrixCell,isorth::Bool)
		for A in fmat
			if norm(A'*A-eye(size(A,2)))>(size(A,1)^2)*eps()
				isorth=false
			end
		end
		new(cten,fmat,isorth)
	end
end
ttensor{T}(cten::Array{T},fmat::MatrixCell,isorth::Bool)=ttensor{T}(cten,fmat,isorth)
ttensor{T}(cten::Array{T},fmat::MatrixCell)=ttensor{T}(cten,fmat,true)
ttensor{T}(cten::Array{T},mat::Matrix)=ttensor{T}(cten,collect(mat),true)
@doc """ Tucker tensor is defined by its core tensor and factor matrices. """ ->
function ttensor{T}(cten::Array{T},mat...)
  fmat=MatrixCell(0)
  for M in mat
			push!(fmat,M)
	end
	ttensor{T}(cten,fmat,true)
end
#If array of matrices isn't defined as MatrixCell, but as M=[M1,M2,...,Mn]:
ttensor{T,T1<:Number}(cten::Array{T},fmat::Array{Matrix{T1}},isorth::Bool)=ttensor{T}(cten,MatrixCell(fmat),isorth)
ttensor{T,T1<:Number}(cten::Array{T},fmat::Array{Matrix{T1}})=ttensor{T}(cten,MatrixCell(fmat),true)

@doc """ Creates random Tucker tensor. """ ->
#I...size, R...rank
function randttensor{D<:Integer}(I::Vector{D},R::Vector{D})
  @assert(size(I)==size(R),"Size and rank should be of same length.")
  cten=randn(tuple(R...)) #create radnom core tensor
  fmat=Matrix[randn(I[n],R[n]) for n=1:length(I)] #create random factor matrices
  ttensor(cten,fmat)
end
randttensor(I::Number,R::Number,N::Integer)=randttensor(repmat([I],N),repmat([R],N));
#For input defined as tuples or nx1 matrices - ranttensor(([I,I,I],[R,R,R]))
function randttensor(arg...)
  randttensor([arg[1]...],[arg[2]...])
end

@doc """ Returns dimension of core of Tucker tensor. """ ->
function coresize{T<:Number}(X::ttensor{T})
  size(X.cten)
end

@doc """ Makes full tensor out of a Tucker tensor. """ ->
function full{T<:Number}(X::ttensor{T})
  ttm(X.cten,X.fmat)
end

@doc """ Hadamard multiplication of two ttensors. """ ->
function had{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2})
  @assert(size(X1) == size(X2))
	fmat=MatrixCell(ndims(X1)) #initilize factor matrix
  n=1
  for (A1,A2) in zip(X1.fmat,X2.fmat)
      fmat[n]=khatrirao(A1,A2,'t')
      n+=1
	 end
  cten=tkron(X1.cten,X2.cten) #Kronecker product of core tensors
  ttensor(cten,fmat)
end
.*{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2}) = had(X1,X2)

@doc """ Calculates core tensor of Hadamard product of two Tucker tensors for given factor matrices. """ ->
function hadcten{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2},fmat::MatrixCell)
  N=ndims(X1);
  C=MatrixCell(N)
  for n=1:N
    C[n]=fmat[n]'*khatrirao(X1.fmat[n],X2.fmat[n],'t');
  end
  cten=krontm(X1.cten,X2.cten,C)
end
hadcten{T1<:Number,T2<:Number,T3<:Number}(X1::ttensor{T1},X2::ttensor{T2},fmat::Array{Matrix{T3}})=hadcten(X1,X2,MatrixCell(fmat))

@doc """ HOSVD for Tucker tensor. """ ->
function hosvd{T<:Number}(X::ttensor{T};method="lapack",reqrank=[],eps_abs=1e-5,eps_rel=0)
	F=hosvd(X.cten,method=method,reqrank=reqrank,eps_abs=eps_abs,eps_rel=eps_rel)
  fmat=MatrixCell(ndims(X))
  [fmat[n]=X.fmat[n]*F.fmat[n] for n=1:ndims(X)]
  reorth(ttensor(F.cten,fmat))
end

@doc """ HOSVD1 algorithm for getting Tucker representation of Hadamard product of two Tucker tensors. """ ->
function hosvd1{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2};method="randsvd",reqrank=[],eps_abs=1e-8,eps_rel=0)
  Xprod=full(X1).*full(X2);
  hosvd(Xprod,method=method,reqrank=reqrank,eps_abs=eps_abs,eps_rel=eps_rel)
end

@doc """ HOSVD2 algorithm for getting Tucker representation of Hadamard product of two Tucker tensors. """ ->
function hosvd2{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2};method="randsvd",reqrank=[],eps_abs=1e-8,eps_rel=0)
@assert(size(X1) == size(X2))
  N=ndims(X1)
	Ahad=MatrixCell(N) #initialize factor matrices
  Q=MatrixCell(N);R=MatrixCell(N);
  n=1
  for (A1,A2) in zip(X1.fmat,X2.fmat)
    Ahad[n]=khatrirao(A1,A2,'t')
    Q[n],R[n]=qr(Ahad[n])
    n+=1
	 end
  X=hosvd(krontm(X1.cten,X2.cten,R),method=method,reqrank=reqrank,eps_abs=1e-8,eps_rel=0);
  cten=X.cten;
  fmat=MatrixCell(N)
  [fmat[n]=Q[n]*X.fmat[n] for n=1:N];
  ttensor(cten,fmat)
end

@doc """ HOSVD3 algorithm for getting Tucker representation of Hadamard product of two Tucker tensors. """ ->
function hosvd3{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2};reqrank=[],method="lanczos",variant='B',eps_abs=1e-8,eps_rel=0,p=10)
 	@assert(size(X1) == size(X2))
  N=ndims(X1)
	Ahad=MatrixCell(N) #initilize factor matrices
  if method != "lanczos" && method != "randsvd"
    error("Incorect method name.")
  end
  if length(reqrank) == 0
	  reqrank=zeros(N);
  elseif isa(reqrank,Int)
    reqrank=repmat([reqrank],N)
  end
  @assert(N==length(reqrank),"Dimensions mismatch.")
  for n=1:N
    if method=="lanczos"
      Ahad[n],S=lanczos(X1,X2,n,variant=variant,reqrank=reqrank[n],tol=eps_abs,p=p)
    elseif method=="randsvd"
      Ahad[n],S=randsvd(X1,X2,n,variant=variant,reqrank=reqrank[n],tol=eps_abs,p=p)
    end
    if reqrank[n] == 0
      eps_rel != 0 ?  tol=eps_rel*S[1] : tol=eps_abs;
      I=find(x-> x>tol ? true : false,S)
      Ahad[n]=Ahad[n][:,I];
    end
  end
  core=hadcten(X1,X2,Ahad)
  ttensor(core,Ahad)
end

@doc """ HOSVD4 algorithm for getting Tucker representation of Hadamard product of two Tucker tensors. """ ->
function hosvd4{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2};reqrank=[],method="lapack",tol=1e-8,p=10)
  @assert(size(X1) == size(X2))
  N=ndims(X1)
  if length(reqrank) == 0
	  reqrank=repmat([0],N)
  elseif isa(reqrank,Int)
    reqrank=repmat([reqrank],N)
  end
  @assert(N==length(reqrank),"Dimensions mismatch.")
  Q=MatrixCell(N) #range approximation of of tenmat(X1.*X2,n)
  KR=MatrixCell(N); #transpose Khatri-Rao product of X1.fmat and X2.fmat
  fmat=MatrixCell(N);
  [KR[n]=khatrirao(X1.fmat[n],X2.fmat[n],'t') for n=1:N]
  for n=1:N
    Q[n]=randrange(X1.cten,X2.cten,KR,n,reqrank=reqrank[n],tol=tol,p=p);
  end
  [fmat[n]=Q[n]'*KR[n] for n=1:N]
  H=krontm(X1.cten,X2.cten,fmat)
  Htucker=hosvd(H,reqrank=reqrank,method=method,eps_abs=tol)
  [fmat[n]=Q[n]*Htucker.fmat[n] for n=1:N]
  ttensor(Htucker.cten,fmat)
end

@doc """ Inner product of two Tucker tensors. """ ->
function innerprod{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2})
	@assert size(X1) == size(X2)
	if prod(size(X1.cten)) > prod(size(X2.cten))
		innerprod(X2,X1)
	else
    N=ndims(X1)
    fmat=MatrixCell(N)
    [fmat[n]=X1.fmat[n]'*X2.fmat[n] for n=1:N]
		innerprod(X1.cten,ttm(X2.cten,fmat))
	end
end

@doc """ True if Tucker tensors have equal core tensors and factor matrices, false otherwise. """ ->
function isequal{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2})
  if (X1.cten == X2.cten) && (X1.fmat == X2.fmat)
    true
  else
    false
  end
end
=={T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2})=isequal(X1,X2)

@doc """ Lanczos tridiagonalization algorithm for finding left singular vectors and singular values of matrix ZnZn',
         where Zn is n-mode matricization of Z=X1.*X2 and X1 and X2 two Tucker tensors.""" ->
function lanczos{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2},mode::Integer;variant='B',tol=1e-8,maxit=1000,reqrank=0,p=10)
  #p...oversampling parameter
  @assert(size(X1)==size(X2),"Dimensions mismatch")
  I=size(X1)
  m=I[mode]
  n=prod(deleteat!(copy([I...]),mode))
  K=min(m,maxit);
  if reqrank!=0
    K=min(K,reqrank+p);
  end
  α=zeros(K)
  β=zeros(K)
  v=randn(m);
  q=v/norm(v);
  Q=zeros(m,1)
  Q[:,1]=q;
  k=0; #needed if stopping criterion is met
  for k=1:K
    r=mhadtv(X1,X2,Q[:,k],mode,variant=variant)
    α[k]=dot(r,Q[:,k])
    r=r-α[k]*Q[:,k]
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
    if reqrank > size(U,2)
      warn("Required rank for mode $mode exceeds actual rank, the resulting rank will be ",size(U,2),". Try changing eps_abs.");
    else
      U=U[:,1:reqrank];
      S=S[1:reqrank];
    end
  end
  U=Q*U;
  U,S
end

@doc """ Matricized Hadamard product of two Tucker tensors times vector. """ ->
#For Xₙ=tenmat(had(X1,X2),n), where X1 and X2 are ttensors and v is a vector,
  # for t='b', returns Xₙ*Xₙ'*v.
  # for t='n', returns Xₙ*v.
  # for t='t', returns Xₙ'*v.
function mhadtv{T1<:Number,T2<:Number,T3<:Number}(X1::ttensor{T1},X2::ttensor{T2},v::Vector{T3},n::Integer,t='b';variant='B')
  @assert(size(X1)==size(X2),"Dimensions mismatch")
  I=size(X1)
  r1=coresize(X1)
  r2=coresize(X2)
  R=[r1...].*[r2...]
  N=setdiff(1:ndims(X1),n) #all indices but n

  #X1=G₁ ×₁ A₁ ×₂ ... ×ₗ Aₗ
  #X2=G₂ ×₁ B₁ ×₂ ... ×ₗ Bₗ

  if t=='t'
    @assert(length(v) == I[n],"Vector v is of inappropriate size.")
    #U=broadcast(.*,v,X1.fmat[n]);  #U=diagm(v)*Aₙ
    w1=krtv(X1.fmat[n]',X2.fmat[n]',v);
    #W1=mkrontv(X1.cten,X2.cten,vec(X2.fmat[n]'*U),n,'t') #W1=tenmat(G₁ ⨂ G₂,n)'*vec(Bₙ'*U)
    W1=mkrontv(X1.cten,X2.cten,w1,n,'t')
    for k in N
      W1=reshape(W1,R[k],round(Int,prod(size(W1))/R[k]))
      W2=tkrtv(X1.fmat[k],X2.fmat[k],W1) #vec(W2)=(Aₖ ⨀' Bₖ)*vec(W1)
      #W2=zeros(I[k],size(W1,2))
      #for i=1:size(W1,2)
      #  W3=reshape(W1[:,i],r2[k],r1[k])
      #  W2[:,i]=sum((X2.fmat[k]*W3).*X1.fmat[k],2)  #diag(Bₖ*P₁*Aₖ')=sum((Bₖ*P₁).*Aₖ,2)
      #end
      W1=W2'
    end
    vec(W1)
  elseif t=='n'
    @assert(length(v) == prod(deleteat!(copy(collect(I)),n)),"Vector v is of inappropriate size.")
    W1=v
    for k in N
      W1=reshape(W1,I[k],round(Int,prod(size(W1))/I[k]))
      #W2=zeros(R[k],size(W1,2))
      W2=krtv(X1.fmat[k]',X2.fmat[k]',W1) #W2=(Aₖ' ⨀ Bₖ')*W1
      #for i=1:size(W1,2)
      #  U=broadcast(.*,W1[:,i],X1.fmat[k]);  #vec(U)=diagm(W1[:,i])*Aₖ
      #  W2[:,i]=vec(X2.fmat[k]'*U);
      #end
      W1=W2'
    end
    W2=mkrontv(X1.cten,X2.cten,vec(W1),n) #W1=tenmat(G₁ ⨂ G₂),n)*vec(W2)
    #vec(sum((X2.fmat[n]*reshape(W2,size(X2.fmat[n],2),size(X1.fmat[n],2))).*X1.fmat[n],2)) #(Aₖ' ⨀ Bₖ')'*vec(W2)
    tkrtv(X1.fmat[n],X2.fmat[n],W2)
  elseif t=='b'
    @assert(length(v) == I[n],"Vector v is of inappropriate size.")
    if variant == 'A'    #use when prod(I[N])-prod(R[N]) < 0
      mhadtv(X1,X2,mhadtv(X1,X2,v,n,'t'),n,'n')
    elseif variant == 'B'
      #U=broadcast(.*,v,X1.fmat[n]);  #U=diagm(v)*Aₙ
      w1=krtv(X1.fmat[n]',X2.fmat[n]',v);
      #W1=mkrontv(X1.cten,X2.cten,vec(X2.fmat[n]'*U),n,'t') #W1=tenmat(G₁ ⨂ G₂,n)'*vec(Bₙ'*U)
      W1=mkrontv(X1.cten,X2.cten,w1,n,'t')
      for k in N
        W1=reshape(W1,R[k],round(Int,prod(size(W1))/R[k]))
        #W2=zeros(R[k],size(W1,2))
        W2=tkrtv(X1.fmat[k],X2.fmat[k],W1)
        W1=krtv(X1.fmat[k]',X2.fmat[k]',W2)'
        #for i=1:size(W1,2)
        #   W3=reshape(W1[:,i],r2[k],r1[k])
        #   w=sum((X2.fmat[k]*W3).*X1.fmat[k],2)  #diag(Bₖ*P₁*Aₖ')=sum((Bₖ*P₁).*Aₖ,2)
        #   U=broadcast(.*,w,X1.fmat[k]);
        #   W2[:,i]=vec(X2.fmat[k]'*U);
        #end
        ##vec(W2)=(Aₖ' ⨀ Bₖ')(Aₖ' ⨀ Bₖ')'*vec(W1)
        #W1=W2'
      end
      W2=mkrontv(X1.cten,X2.cten,vec(W1),n) #W2=tenmat(G₁ ⨂ G₂),n)*vec(W1)
      #vec(sum((X2.fmat[n]*reshape(W2,size(X2.fmat[n],2),size(X1.fmat[n],2))).*X1.fmat[n],2)) #(Aₖ' ⨀ Bₖ')'*vec(W2)
      tkrtv(X1.fmat[n],X2.fmat[n],W2)
    else
      error("Variant should be either 'A' or 'B'.")
    end
  end
end
#Computes Xₙ*M for t='n', Xₙ'*M for t='t' and Xₙ*Xₙ'*M for t='b', for Xₙ=tenmat(hadamard(X1,X2),n).
function mhadtv{T1<:Number,T2<:Number,T3<:Number}(X1::ttensor{T1},X2::ttensor{T2},M::Matrix{T3},n::Integer,t='b';variant='B')
  @assert(size(X1)==size(X2),"Dimensions mismatch")
    if sort(collect(size(vec(M))))[1]==1
        return mhadtv(X1,X2,vec(M),n,t,variant=variant);
  end
  I=size(X1)
  In=I[n]
  Im=prod(deleteat!(copy([I...]),n))
  if t=='n'
    @assert(size(M,1) == Im, "Dimensions mismatch")
    Mprod=zeros(In,size(M,2))
  elseif t=='t'
    @assert(size(M,1) == In, "Dimensions mismatch")
    Mprod=zeros(Im,size(M,2))
  elseif t=='b'
    @assert(size(M,1) == In, "Dimensions mismatch")
    Mprod=zeros(In,size(M,2))
  end
  [Mprod[:,j]=  mhadtv(X1,X2,M[:,j],n,t,variant=variant) for j=1:size(M,2)]
  Mprod
end

function mhadtm{T1<:Number,T2<:Number,T3<:Number}(X1::ttensor{T1},X2::ttensor{T2},M::Matrix{T3},n::Integer,t='b';variant='B')
  warn("Function mhadtm is depricated. Use mhadtv.")
  mhadtm(X1,X2,M,n,t,variant=variant)
end

function minus{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2})
  X1+(-1)*X2
end
-{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2})=minus(X1,X2)

function mrank{T<:Number}(X::ttensor{T})
  ntuple(n->nrank(X,n),ndims(X))
end

@doc """ Singular values of n-mode matricization of Tucker Tensor. """ ->
function msvdvals{T<:Number}(X::ttensor{T},n::Integer)
  if X.isorth != true
    reorth!(X)
  end
  Gn=tenmat(X.cten,n)
  svdvals(Gn)
end

@doc """ Scalar times ttensor. """ ->
#Multiplies core tensor by the scalar
function mtimes{T<:Number}(α::Number,X::ttensor{T})
	ttensor(α*X.cten,X.fmat);
end
*{T1<:Number,T2<:Number}(α::T1,X::ttensor{T2})=mtimes(α,X)
*{T1<:Number,T2<:Number}(X::ttensor{T1},α::T2)=*(α,X)

@doc """ Matricized Tucker tensor times Khatri-Rao product. """->
function mttkrp{T<:Number}(X::ttensor{T},M::MatrixCell,n::Integer)
  N=ndims(X)
  @assert(length(M) == N,"Wrong number of matrices")
  modes=setdiff(1:N,n)
  I=[size(X)...]
  K=size(M[modes[1]],2)
  @assert(!any(map(Bool,[size(M[m],2)-K for m in modes])),"Matrices must have the same number of columns")
  @assert(!any(map(Bool,[size(M[m],1)-I[m] for m in modes])),"Matrices are of wrong size")
  Y=mttkrp(X.cten,vec(X.fmat').*M,n)
  X.fmat[n]*Y
end
mttkrp{T1<:Number,T2<:Number}(X::ttensor{T1},M::Array{Matrix{T2}},n::Integer)=mttkrp(X,MatrixCell(M),n)

@doc """ Number of modes of ttensor. """ ->
function ndims{T<:Number}(X::ttensor{T})
	ndims(X.cten)
end

@doc """ Norm of Tucker tensor. """ ->
function norm{T<:Number}(X::ttensor{T})
	if prod(size(X)) > prod(size(X.cten))
		if X.isorth == true
			norm(X.cten)
		else
			R=MatrixCell(ndims(X))
			for n=1:ndims(X)
				R[n]=qrfact(X.fmat[n])[:R]
			end
			norm(ttm(X.cten,R))
		end
	else
		norm(full(X))
	end
end

function nrank{T<:Number}(X::ttensor{T},n::Integer)
  rank(X.fmat[n])
end

@doc """ Computes eigenvalues of Xₙ*Xₙ', where Xₙ is the n-mode matricization of a Tucker tensor. """ ->
#Computes the r leading eigenvalues of Xₙ*Xₙ', where Xₙ is the mode-n matricization of X.
function nvecs{T<:Number}(X::ttensor{T},n::Integer,r=0)
  if r==0
    r=size(X,n)
  end
  N=ndims(X)
  V=MatrixCell(N)
  for i=1:N
    if i==n
      V[i]=X.fmat[i];
    else
      V[i]=X.fmat[i]'*X.fmat[i];
    end
  end
  H=ttm(X.cten,V)
  Hn=tenmat(H,n);
  Gn=tenmat(X.cten,n);
  Y=Symmetric(Hn*Gn'*X.fmat[n]');
  sort(eigvals(Y),rev=true)[1:r]
end

function permutedims{T<:Number,D<:Integer}(X::ttensor{T},perm::Vector{D})
  @assert(collect(1:ndims(X))==sort(perm),"Invalid permutation")
  cten=permutedims(X.cten,perm)
  fmat=X.fmat[perm]
  ttensor(cten,fmat)
end

function plus{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2})
	@assert size(X1) == size(X2)
	fmat=Matrix[[X1.fmat[n] X2.fmat[n]] for n=1:ndims(X1)] #concatenate factor matrices
	coresize=tuple([size(X1.cten)...]+[size(X2.cten)...]...)
	cten=zeros(coresize) #initialize core tensor
	I1=indicesmat(X1.cten,zeros([size(X1.cten)...]))
	I2=indicesmat(X2.cten,[size(X1.cten)...])
	idx1=indicesmat2vec(I1,size(cten))
	idx2=indicesmat2vec(I2,size(cten))
	cten[idx1]=vec(X1.cten) #first diagonal block
	cten[idx2]=vec(X2.cten) #second diagonal block
	ttensor(cten,fmat)
end
+{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2})=plus(X1,X2)

@doc """ Randomized range approximation for matrix Zn, where Zn is n-mode matricization of Z=X1.*X2 and X1 and X2 two Tucker tensors.
        As input accepts core tensors and transpose Khatri-Rao product of factor matrices of X1 and X2. """ ->
function randrange{T1<:Number,T2<:Number,N}(C1::Array{T1,N},C2::Array{T2,N},KR::MatrixCell,mode::Integer;tol=1e-8,maxit=1000,reqrank=0,p=10,r=10)
  #p... oversampling parameter
  #r... balances computational cost and reliability
  I=zeros(Int,N)
  [I[n]= size(KR[n],1) for n=1:N]
  m=I[mode]
  n=prod(deleteat!(copy([I...]),mode))
  remmodes=setdiff(1:N,mode);
  y=VectorCell(N-1);
  if reqrank!=0
    Y=zeros(m,reqrank+p);
    for i=1:reqrank+p
      [y[k]=randn(size(KR[remmodes[k]],1)) for k=1:N-1]
      w=krontkron(reverse(KR[remmodes]),reverse(y),'t')
      Y[:,i]=KR[mode]*mkrontv(C1,C2,w,mode)
    end
    Q=full(qrfact(Y)[:Q]);
  else
    maxit=min(m,n,maxit);
    rangetol=tol*sqrt(pi/2)/10;
    Y=zeros(m,r);
    for i=1:r
      [y[k]=randn(size(KR[remmodes[k]],1)) for k=1:N-1]
      w=krontkron(reverse(KR[remmodes]),reverse(y),'t')
      Y[:,i]=KR[mode]*mkrontv(C1,C2,w,mode);
    end
    j=0;
    Q=zeros(m,0);
    maxcolnorm=maximum([norm(Y[:,i]) for i=1:r]);
    while maxcolnorm > rangetol && j<maxit
      j+=1;
      p=Q'*Y[:,j];
      Y[:,j]-=Q*p;
      q=Y[:,j]/norm(Y[:,j]);
      Q=[Q q];
      [y[k]=randn(size(KR[remmodes[k]],1)) for k=1:N-1]
      w=krontkron(reverse(KR[remmodes]),reverse(y),'t')
      w=KR[mode]*mkrontv(C1,C2,w,mode);
      p=Q'*w;
      Y=[Y w-Q*p]; #Y[:,j+r]=w-Q*p;
      p=q'*Y[:,j+1:j+r-1]
      Y[:,j+1:j+r-1]-=q*p;
      maxcolnorm=maximum([norm(Y[:,i]) for i=j+1:j+r]);
    end
  end
  Q
end
randrange{T1<:Number,T2<:Number,T3<:Number,N}(C1::Array{T1,N},C2::Array{T2,N},KR::Array{Matrix{T3}},mode::Integer;tol=1e-8,maxit=1000,reqrank=0,p=10,r=10)=randrange(C1,C2,MatrixCell(KR),mode,tol,maxit,reqrank,p,r)

function randrange{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2},mode::Integer;tol=1e-8,maxit=1000,reqrank=0,p=10,r=10)
  @assert(size(X1) == size(X2))
  N=ndims(X1)
  KR=MatrixCell(N); #transpose Khatri-Rao product of X1.fmat and X2.fmat
  [KR[n]=khatrirao(X1.fmat[n],X2.fmat[n],'t') for n=1:N]
  randrange(X1.cten,X2.cten,KR,mode,tol=tol,maxit=maxit,reqrank=reqrank,p=p,r=r)
end

@doc """ Randomized SVD algorithm for finding left singular vectors and singular values of matrix ZnZn',
         where Zn is n-mode matricization of Z=X1.*X2 and X1 and X2 two Tucker tensors.""" ->
function randsvd{T1<:Number,T2<:Number}(X1::ttensor{T1},X2::ttensor{T2},mode::Integer;variant='B',tol=1e-8,maxit=1000,reqrank=0,p=10,r=10)
  #p... oversampling parameter
  #r... balances computational cost and reliability
  @assert(size(X1)==size(X2),"Dimensions mismatch")
  I=size(X1)
  m=I[mode]
  n=prod(deleteat!(copy([I...]),mode))
  if reqrank!=0
     Y=mhadtv(X1,X2,randn(m,reqrank+p),mode,variant=variant);  #Y=A*(A'*randn(m,reqrank+p));
     Q=full(qrfact(Y)[:Q]);
  else
    maxit=min(m,n,maxit);
    rangetol=tol*sqrt(pi/2)/10;
    Y=mhadtv(X1,X2,randn(m,r),mode,variant=variant);  #Y=A*(A'*randn(m,r));
    j=0;
    Q=zeros(m,0);
    maxcolnorm=maximum([norm(Y[:,i]) for i=1:r]);
    while maxcolnorm > rangetol && j<maxit
      j+=1;
      p=Q'*Y[:,j];
      Y[:,j]-=Q*p;
      q=Y[:,j]/norm(Y[:,j]);
      Q=[Q q];
      w=mhadtv(X1,X2,randn(m),mode,variant=variant); #w=A*(A'*randn(m));
      p=Q'*w;
      Y=[Y w-Q*p]; #Y[:,j+r]=w-Q*p;
      p=q'*Y[:,j+1:j+r-1]
      Y[:,j+1:j+r-1]-=q*p;
      maxcolnorm=maximum([norm(Y[:,i]) for i=j+1:j+r]);
    end
  end
  B=mhadtv(X1,X2,Q,mode,'t');#B=A'*Q;
  B=Symmetric(B'*B);
  #or (faster for small rank):
  #B=mhadtv(X1,X2,Q,mode,'n');
  #B=Symmetric(Q'*B);
  E=eigfact(B,tol,Inf);
  U=E[:vectors][:,end:-1:1];
  S=sqrt(abs(E[:values][end:-1:1]));
  if reqrank != 0
    if reqrank > size(U,2)
      warn("Required rank for mode $mode exceeds actual rank, the resulting rank will be smaller.")
    else
      U=U[:,1:reqrank];
      S=S[1:reqrank];
    end
  end
  U=Q*U;
  U,S
end

@doc """ Orthogonalize factor matrices of a Tucker tensor. """ ->
function reorth{T<:Number}(X::ttensor{T})
  N=ndims(X)
	if X.isorth == true
		X
	else
		Q=MatrixCell(N)
		R=MatrixCell(N)
    n=1;
		for A in X.fmat
			Qt,Rt=qr(A)
			Q[n]=Qt
			R[n]=Rt
      n+=1
		end
		ttensor(ttm(X.cten,R),Q)
	end
end

function reorth!{T<:Number}(X::ttensor{T})
	if X.isorth != true
		for n=1:ndims(X)
			Q,R=qr(X.fmat[n])
			X.fmat[n]=Q
			X.cten=ttm(X.cten,R,n)
		end
    X.isorth=true;
  end
  X
end

function size{T<:Number}(X::ttensor{T})
	tuple([size(X.fmat[n],1) for n=1:ndims(X)]...)
end
#n-th dimension of X
function size{T<:Number}(X::ttensor{T},n::Integer)
  size(X.fmat[n],1)
end

@doc """ n-mode matricization of Tucker tensor. """ ->
tenmat{T<:Number}(X::ttensor{T},n::Integer)=tenmat(full(X),n)

@doc """ Tucker tensor times matrices (n-mode product). """ ->
#Multiplies ttensor X with matrices from array M by modes; t='t' transposes matrices
function ttm{T<:Number,D<:Integer}(X::ttensor{T},M::MatrixCell,modes::Vector{D},t='n')
  if t=='t'
	 M=vec(M')
	end
	@assert(length(modes)<=length(M),"Too few matrices")
	@assert(length(M)<=ndims(X),"Too many matrices")
  if length(modes)<length(M)
    M=M[modes]
  end
  fmat=copy(X.fmat)
  for n=1:length(modes)
    fmat[modes[n]]=M[n]*X.fmat[modes[n]]
  end
    ttensor(X.cten,fmat)
end
ttm{T<:Number,D<:Integer}(X::ttensor{T},M::MatrixCell,modes::Range{D},t='n')=ttm(X,M,collect(modes),t)
ttm{T1<:Number,T2<:Number}(X::ttensor{T1},M::Matrix{T2},n::Integer,t='n')=ttm(X,[M],[n],t)
ttm{T<:Number}(X::ttensor{T},M::MatrixCell,t::Char)=ttm(X,M,1:length(M),t)
ttm{T<:Number}(X::ttensor{T},M::MatrixCell)=ttm(X,M,1:length(M))
function ttm{T<:Number}(X::ttensor{T},M::MatrixCell,n::Integer,t='n')
	if n>0
		ttm(X,M[n],n,t)
	else
		modes=setdiff(1:ndims(X),-n)
		ttm(X,M,modes,t)
	end
end
#If array of matrices isn't defined as MatrixCell, but as M=[M1,M2,...,Mn]:
ttm{T1<:Number,T2<:Number,D<:Integer}(X::ttensor{T1},M::Array{Matrix{T2}},modes::Vector{D},t='n')=ttm(X,MatrixCell(M),modes,t)
ttm{T1<:Number,T2<:Number,D<:Integer}(X::ttensor{T1},M::Array{Matrix{T2}},modes::Range{D},t='n')=ttm(X,MatrixCell(M),modes,t)
ttm{T1<:Number,T2<:Number}(X::ttensor{T1},M::Array{Matrix{T2}},t::Char)=ttm(X,MatrixCell(M),t)
ttm{T1<:Number,T2<:Number}(X::ttensor{T1},M::Array{Matrix{T2}})=ttm(X,MatrixCell(M))
ttm{T1<:Number,T2<:Number}(X::ttensor{T1},M::Array{Matrix{T2}},n::Integer,t='n')=ttm(X,MatrixCell(M),n,t)

@doc """ Tucker tensor times vectors (n-mode product). """ ->
function ttv{T<:Number,D<:Integer}(X::ttensor{T},V::VectorCell,modes::Vector{D})
  N=ndims(X)
  remmodes=setdiff(1:N,modes)
  fmat=VectorCell(N)
  if length(modes) < length(V)
    V=V[modes]
  end
  for n=1:length(modes)
    fmat[modes[n]]=X.fmat[modes[n]]'*V[n]
  end
  cten=ttv(X.cten,fmat,modes)
  if isempty(remmodes)
    cten
  else
    ttensor(cten,X.fmat[remmodes])
  end
end
ttv{T1<:Number,T2<:Number}(X::ttensor{T1},v::Vector{T2},n::Integer)=ttv(X,Vector[v],[n])
ttv{T<:Number,D<:Integer}(X::ttensor{T},V::VectorCell,modes::Range{D})=ttv(X,V,collect(modes))
ttv{T<:Number}(X::ttensor{T},V::VectorCell)=ttv(X,V,1:length(V))
function ttv{T<:Number}(X::ttensor{T},V::VectorCell,n::Integer)
	if n>0
		ttm(X,V[n],n)
	else
		modes=setdiff(1:ndims(X),-n)
		ttm(X,A,modes)
	end
end
#If array of vectors isn't defined as VectorCell, but as V=[v1,v2,...,vn]:
ttv{T1<:Number,T2<:Number,D<:Integer}(X::ttensor{T1},V::Array{Vector{T2}},modes::Vector{D})=ttv(X,VectorCell(V),modes)
ttv{T1<:Number,T2<:Number,D<:Integer}(X::ttensor{T1},V::Array{Vector{T2}},modes::Range{D})=ttv(X,VectorCell(V),modes)
ttv{T1<:Number,T2<:Number}(X::ttensor{T1},V::Array{Vector{T2}})=ttv(X,VectorCell(V))
ttv{T1<:Number,T2<:Number}(X::ttensor{T1},V::Array{Vector{T2}},n::Integer)=ttv(X,VectorCell(V),n)

uminus{T<:Number}(X::ttensor{T})=mtimes(-1,X)
-{T<:Number}(X::ttensor{T})=uminus(X)
