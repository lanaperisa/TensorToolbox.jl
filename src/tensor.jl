export hosvd, innerprod, krontm, matten, mkrontm, mkrontv, mrank, mttkrp, nrank, sthosvd, tenmat, tkron, ttm, ttt, ttv

@doc """ Higher-order singular value decomposition. """ ->
#methods={"lapack","lanczos","randsvd"}
#If reqrank not defined there are two options:
# - drop singular values below eps_abs
# - drop singular values below eps_rel*sigma_1
function hosvd{T<:Number,N}(X::Array{T,N};method="lapack",reqrank=[],eps_abs=[],eps_rel=[],p=10)
	fmat=MatrixCell(N)

  reqrank=check_vector_input(reqrank,N,0);
  eps_abs=check_vector_input(eps_abs,N,1e-8);
  eps_rel=check_vector_input(eps_rel,N,0);

	for n=1:N
    Xn=float(tenmat(X,n))
    if method == "lapack"
      fmat[n],S=LAPACK.gesvd!('A','N',Xn)
    elseif method == "lanczos"
      fmat[n],S=lanczos(Xn,tol=eps_abs[n],reqrank=reqrank[n],p=p)
    elseif method == "randsvd"
      fmat[n],S=randsvd(Xn,tol=eps_abs[n],reqrank=reqrank[n],p=p)
    else
      fmat[n],S,V=svd(Xn)
    end
    if reqrank[n]!=0 && size(fmat[n],2)>reqrank[n]
      fmat[n]=fmat[n][:,1:reqrank[n]];
    else
      eps_rel[n] != 0 ? tol=eps_rel[n]*S[1] : tol=eps_abs[n];
      I=find(x-> x>tol ? true : false,S)
      fmat[n]=fmat[n][:,I]
    end
	end
  ttensor(ttm(X,fmat,'t'),fmat)
end

function innerprod{T1<:Number,T2<:Number}(X1::Array{T1},X2::Array{T2})
	@assert(size(X1) == size(X2),"Dimension mismatch")
	sum(X1.*conj(X2))
end

@doc """ Kronecker product of two tensors times matrices (n-mode product). """ ->
#Multiplies tensor tkron(X1,X2) with matrices from array M by modes; t='t' transposes matrices
function krontm{T1<:Number,T2<:Number,D<:Integer,N}(X1::Array{T1,N},X2::Array{T2,N},M::MatrixCell,modes::Vector{D},t='n')
  if t=='t'
    M=vec(M')
	end
	@assert(length(modes)<=length(M),"Dimension mismatch.")
	@assert(length(M)<=N,"Too many matrices.")
  I=[size(X1)...].*[size(X2)...]
  R=copy(I)
  if length(modes) == length(M)
    @assert(sort(modes) == collect(1:length(M)),"Badly defined vector of modes")
    n=indmax(R-Int[size(M[i],1) for i=1:length(M)]) #mode for largest possible dimension reduction
    ind=setdiff(1:length(M),n);
    Xn=zeros(size(M[n],1),prod([size(M[m],1) for m in ind]));
    for i=1:size(M[n],1)
      W1=reshape(M[n][i,:],size(X2,n),size(X1,n))
      w=mkrontv(X1,X2,vec(W1),n,'t')
      W2=M[ind[1]]*reshape(w,size(M[ind[1]],2),round(Int,length(w)/size(M[ind[1]],2)))
      W2=W2';
      for m in ind[2:end]
        W2=(M[m]*reshape(W2,size(M[m],2),round(Int,prod(size(W2))/size(M[m],2))))'
      end
       Xn[i,:]=vec(W2);
    end
    R=round(Int,[size(M[i],1) for i=1:length(M)]);
    X=matten(Xn,n,R)
  else
    M=M[modes] #discard matrices not needed for multiplication
    for n=1:length(modes)
      R[modes[n]]=size(M[n],1) #vector of rₖ
    end
    #Order of multiplication - if tkron(X₁,X₂) is i₁ × i₂ × ... × iₙ and Mₖ are rₖ × iₖ, sort by largest possible dimension reduction iₖ-rₖ
    p=sortperm(I[modes]-R[modes],rev=true)
    M=M[p]
    modes=modes[p]
    @assert(I[modes[1]] == size(M[1],2),"Dimensions mismatch")
    Xn=mkrontv(X1,X2,M[1]',modes[1],'t')';
    I[modes[1]]=size(M[1],1)
    X=matten(Xn,modes[1],I)
    for n=2:length(M)
	     @assert(I[modes[n]] == size(M[n],2),"Dimensions mismatch")
       Xn=tenmat(X,modes[n])
	     I[modes[n]]=size(M[n],1)
	     X=matten(M[n]*Xn,modes[n],I)
	  end
  end
  X
end
krontm{T1<:Number,T2<:Number,T3<:Number}(X1::Array{T1},X2::Array{T2},M::Matrix{T3},n::Integer,t='n')=krontm(X1,X2,collect(M),[n],t)
krontm{T1<:Number,T2<:Number}(X1::Array{T1},X2::Array{T2},M::MatrixCell,t::Char)=krontm(X1,X2,M,1:length(M),t)
krontm{T1<:Number,T2<:Number}(X1::Array{T1},X2::Array{T2},M::MatrixCell)=krontm(X1,X2,M,1:length(M))
krontm{T1<:Number,T2<:Number,D<:Integer}(X1::Array{T1},X2::Array{T2},M::MatrixCell,modes::Range{D},t::Char)=krontm(X1,X2,M,collect(modes),t)
krontm{T1<:Number,T2<:Number,D<:Integer}(X1::Array{T1},X2::Array{T2},M::MatrixCell,modes::Range{D})=krontm(X1,X2,M,collect(modes))
function krontm{T1<:Number,T2<:Number,N}(X1::Array{T1,N},X2::Array{T2,N},M::MatrixCell,n::Integer,t='n')
 	if n>0
 		krontm(X1,X2,M[n],n,t)
 	else
 		modes=setdiff(1:N,-n)
 		krontm(X1,X2,M,modes,t)
 	end
end
#If array of matrices isn't defined as MatrixCell, but as M=[M1,M2,...,Mn]:
krontm{T1<:Number,T2<:Number,T3<:Number,D<:Integer,N}(X1::Array{T1,N},X2::Array{T2,N},M::Array{Matrix{T3}},modes::Vector{D},t='n')=krontm(X1,X2,MatrixCell(M),modes,t)
krontm{T1<:Number,T2<:Number,T3<:Number}(X1::Array{T1},X2::Array{T2},M::Array{Matrix{T3}},t::Char)=krontm(X1,X2,MatrixCell(M),t)
krontm{T1<:Number,T2<:Number,T3<:Number}(X1::Array{T1},X2::Array{T2},M::Array{Matrix{T3}})=krontm(X1,X2,MatrixCell(M))
krontm{T1<:Number,T2<:Number,T3<:Number,D<:Integer}(X1::Array{T1},X2::Array{T2},M::Array{Matrix{T3}},modes::Range{D},t::Char)=krontm(X1,X2,MatrixCell{M},modes,t)
krontm{T1<:Number,T2<:Number,T3<:Number,D<:Integer}(X1::Array{T1},X2::Array{T2},M::Array{Matrix{T3}},modes::Range{D})=krontm(X1,X2,MatrixCell{M},modes)
krontm{T1<:Number,T2<:Number,T3<:Number,N}(X1::Array{T1,N},X2::Array{T2,N},M::Array{Matrix{T3}},n::Integer,t='n')=krontm(X1,X2,MatrixCell{M},n,t)


@doc """ Folds matrix into tensor. """ ->
#Folds matrix into tensor of dimension dims by mode n
function matten{T<:Number,D<:Integer}(A::Matrix{T},n::Integer,dims::Vector{D})
	@assert(dims[n]==size(A,1),"Dimensions mismatch")
	m = setdiff(1:length(dims), n)
	@assert prod(dims[m])==size(A,2)
	X = reshape(A,[dims[n];dims[m]]...)
	permutedims(X,invperm([n;m]))
end

@doc """ Matricized Kronecker product of tensors times vector. """ ->
#for t='n' calculates tenmat(tkron(X1,X2),n)*v
#for t='t' calculates tenmat(tkron(X1,X2),n)'*v
function mkrontv{T1<:Number,T2<:Number,T3<:Number,N}(X1::Array{T1,N},X2::Array{T2,N},v::Vector{T3},n::Integer,t='n')
  I1=size(X1)
  I2=size(X2)
  kronsize=tuple(([I1...].*[I2...])...);
  ind=setdiff(1:N,n) #all indices but n
  X1n=tenmat(X1,n);
  X2n=tenmat(X2,n);
  perfect_shuffle=round(Int,[ [2*k-1 for k=1:N-1]; [2*k for k=1:N-1] ]);
  if t=='n'
    @assert(length(v) == prod(kronsize[ind]),"Vector is of inapropriate size.")
    tenshape=vec([[I2[ind]...] [I1[ind]...]]');
    vperm=permutedims(reshape(v,tenshape...),perfect_shuffle);
    vec(X2n*reshape(vperm,size(X2n,2),size(X1n,2))*X1n')
  elseif t=='t'
    @assert(length(v) == kronsize[n],"Vector is of inapropriate size.")
    if I1[n]*prod(I2)+prod(I1)*prod(I2[ind]) > I2[n]*prod(I1)+prod(I2)*prod(I1[ind])
      W=(X2n'*reshape(v,size(X2n,1),size(X1n,1)))*X1n
    else
      W=X2n'*(reshape(v,size(X2n,1),size(X1n,1))*X1n)
    end
    tenshape=[[I2[ind]...];[I1[ind]...]];
    vec(permutedims(reshape(W,tenshape...),invperm(perfect_shuffle)))
    end
end

#Matricized Kronecker product times matrix - column by column.
function mkrontv{T1<:Number,T2<:Number,T3<:Number,N}(X1::Array{T1,N},X2::Array{T2,N},M::Matrix{T3},n::Integer,t='n')
  if sort(collect(size(vec(M))))[1]==1
    return mkrontv(A,B,vec(M));
  end
  I1=size(X1);
  I2=size(X2);
  kronsize=([I1...].*[I2...]);
  ind=setdiff(1:N,n) #all indices but n
  if t=='n'
    Mprod=zeros(kronsize[n],size(M,2))
  else
    Mprod=zeros(prod(kronsize[ind]),size(M,2))
  end
  for i=1:size(M,2)
    Mprod[:,i]=mkrontv(X1,X2,M[:,i],n,t);
  end
  Mprod
end
function mkrontm{T1<:Number,T2<:Number,T3<:Number,N}(X1::Array{T1,N},X2::Array{T2,N},M::Matrix{T3},n::Integer,t='n')
  warn("Function mkrontm is depricated. Use mkrontv.")
  mkrontv(X1,X2,M,n,t)
end

@doc """Multilinear rank of a tensor. """->
function mrank{T<:Number,N}(X::Array{T,N})
   ntuple(n->nrank(X,n),N)
end
function mrank{T<:Number,N}(X::Array{T,N},tol::Number)
   ntuple(n->nrank(X,n,tol),N)
end

@doc """ Matricized tensor times Khatri-Rao product. """ ->
function mttkrp{T<:Number,N}(X::Array{T,N},M::MatrixCell,n::Integer)
  modes=setdiff(1:N,n)
  I=[size(X)...]
  K=size(M[modes[1]],2)
  @assert(!any(map(Bool,[size(M[m],2)-K for m in modes])),"Matrices must have the same number of columns")
  @assert(!any(map(Bool,[size(M[m],1)-I[m] for m in modes])),"Matrices are of wrong size")
  Xn=tenmat(X,n)
  Xn*khatrirao(reverse(M[modes]))
end
mttkrp{T1<:Number,T2<:Number,N}(X::Array{T1,N},M::Array{Matrix{T2}},n::Integer)=mttkrp(X,MatrixCell(M),n)

@doc """n-rank of a tensor. """->
function nrank{T<:Number}(X::Array{T},n::Integer)
  rank(tenmat(X,n))
end
function nrank{T<:Number}(X::Array{T},n::Integer,tol::Number)
  rank(tenmat(X,n),tol)
end

@doc """ Sequentially truncated HOSVD. """ ->
function sthosvd{T<:Number,D<:Integer,N}(X::Array{T,N},R::Vector{D},p::Vector{D})
	@assert(N==length(R)==length(p),"Dimensions mismatch")
	I=[size(X)...]
	fmat=MatrixCell(N)
	for n=1:N
		fmat[n]=zeros(I[n],R[n])
	end
	for n in p
		Xn=tenmat(X,n)
		U,S,V=svd(Xn)
		fmat[n]=U[:,1:R[n]]
		Xn=diagm(S[1:R[n]])*V'[1:R[n],:]
		I[n]=R[n]
		X=matten(Xn,n,I)
	end
	ttensor(X,fmat)
end

@doc """ Tensor matricization. """ ->
#n-mode matricization
function tenmat{T<:Number,N}(X::Array{T,N},n::Integer)
	@assert(n<=ndims(X),"Mode exceedes number of dimensions")
	I=size(X)
	m=setdiff(1:N,n)
	reshape(permutedims(X,[n;m]),I[n],prod(I[m]))
end

@doc """ Kronecker product of two tensors - direct generalization of Kronecker product of matrices. """ ->
function tkron{T1<:Number,T2<:Number,N1,N2}(X1::Array{T1,N1},X2::Array{T2,N2})
  if N1<3 && N2<3
    kron(X1,X2)
  else
    core1sz=[size(X1)...]
	  core2sz=[size(X2)...]
	  S=shiftsmat(X1,core2sz) #matrix of shifts for indices of blocks of Xk
	  Xk=zeros(tuple(core1sz.*core2sz...)) #initalize solution
	  for i=1:prod(core1sz)
		  I=indicesmat(X2,[S[i,:]...]) #matrix of shifted indices of X2
		  idx=indicesmat2vec(I,size(Xk))
		  Xk[idx]=X1[i]*vec(X2)
	  end
    Xk
  end
end

@doc """ Tensor times matrices (n-mode product). """ ->
#Multiplies tensor X with matrices from array M by modes; t='t' transposes matrices.
function ttm{T<:Number,D<:Integer,N}(X::Array{T,N},M::MatrixCell,modes::Vector{D},t='n')
  if t=='t'
    M=vec(M')
	end
	@assert(length(modes)<=length(M),"Too few matrices.")
	@assert(length(M)<=N,"Too many matrices.")
  I=[size(X)...]
  if length(modes) < length(M)
    M=M[modes] #discard matrices not needed for multiplication
  end
  R=copy(I)
  for n=1:length(modes)
    R[modes[n]]=size(M[n],1) #vector of rₖ
  end
  #Order of multiplication - if X is i₁ × i₂ × ... × iₙ and Mₖ is rₖ × iₖ, sort by largest possible dimension reduction iₖ-rₖ
  p=sortperm(I[modes]-R[modes],rev=true)
  M=M[p]
  modes=modes[p]
  for n=1:length(M)
	 @assert(I[modes[n]] == size(M[n],2),"Dimensions mismatch")
   Xn=tenmat(X,modes[n])
	 I[modes[n]]=size(M[n],1)
	 X=matten(M[n]*Xn,modes[n],I)
	end
  X
end
ttm{T1<:Number,T2<:Number}(X::Array{T1},M::Matrix{T2},n::Integer,t='n')=ttm(X,Matrix[M],[n],t)
ttm{T<:Number}(X::Array{T},M::MatrixCell,t::Char)=ttm(X,M,1:length(M),t)
ttm{T<:Number}(X::Array{T},M::MatrixCell)=ttm(X,M,1:length(M))
ttm{T<:Number,D<:Integer}(X::Array{T},M::MatrixCell,R::Range{D},t::Char)=ttm(X,M,collect(R),t)
ttm{T<:Number,D<:Integer}(X::Array{T},M::MatrixCell,R::Range{D})=ttm(X,M,collect(R))
function ttm{T<:Number,N}(X::Array{T,N},M::MatrixCell,n::Integer,t='n')
	if n>0
		ttm(X,M[n],n,t)
	else
		modes=setdiff(1:N,-n)
		ttm(X,M,modes,t)
	end
end
#If array of matrices isn't defined as MatrixCell, but as M=[M1,M2,...,Mn]:
ttm{T1<:Number,T2<:Number,D<:Integer,N}(X::Array{T1,N},M::Array{Matrix{T2}},modes::Vector{D},t='n')=ttm(X,MatrixCell(M),modes,t)
ttm{T1<:Number,T2<:Number}(X::Array{T1},M::Array{Matrix{T2}},t::Char)=ttm(X,MatrixCell(M),t)
ttm{T1<:Number,T2<:Number}(X::Array{T1},M::Array{Matrix{T2}})=ttm(X,MatrixCell(M))
ttm{T1<:Number,T2<:Number,D<:Integer}(X::Array{T1},M::Array{Matrix{T2}},R::Range{D},t::Char)=ttm(X,MatrixCell{M},R,t)
ttm{T1<:Number,T2<:Number,D<:Integer}(X::Array{T1},M::Array{Matrix{T2}},R::Range{D})=ttm(X,MatrixCell(M),R)
ttm{T1<:Number,T2<:Number,N}(X::Array{T1,N},M::Array{Matrix{T2}},n::Integer,t='n')=ttm(X,MatrixCell(M),n,t)

@doc """ Outer product of two tensors. """ ->
function ttt{T1<:Number,T2<:Number}(X1::Array{T1},X2::Array{T2})
  sz=tuple([[size(X1)...];[size(X2)...]]...);
  Xprod=vec(X1)*vec(X2)';
  reshape(Xprod,sz)
end

@doc """ Tensor times vectors (n-mode product). """ ->
function ttv{T<:Number,D<:Integer,N}(X::Array{T,N},V::VectorCell,modes::Vector{D})
  remmodes=setdiff(1:N,modes)'
  if N > 1
    X=permutedims(X,[remmodes modes'])
  end
  I=size(X)
  if length(modes) < length(V)
    V=V[modes]
  end
  M=N
  for n=length(modes):-1:1
    X=reshape(X,prod(I[1:M-1]),I[M])
    X=X*V[n]
    M-=1
  end
  if M>0
    X=reshape(X,I[1:M])
  end
  X
end
ttv{T1<:Number,T2<:Number,N}(X::Array{T1,N},v::Vector{T2},n::Integer)=ttv(X,Vector[v],[n])
ttv{T<:Number,N}(X::Array{T,N},V::VectorCell)=ttv(X,V,collect(1:length(V)))
function ttv{T<:Number,N}(X::Array{T,N},V::VectorCell,n::Integer)
	if n>0
		ttv(X,V[n],n)
	else
		modes=setdiff(1:N,-n)
		ttv(X,V,modes)
	end
end
#If array of vectors isn't defined as VectorCell, but as V=[v1,v2,...,vn]:
ttv{T1<:Number,T2<:Number,D<:Integer,N}(X::Array{T1,N},V::Array{Vector{T2}},modes::Vector{D})=ttv(X,VectorCell(V),modes)
ttv{T1<:Number,T2<:Number,N}(X::Array{T1,N},V::Array{Vector{T2}})=ttv(X,VectorCell(V))
ttv{T1<:Number,T2<:Number,N}(X::Array{T1,N},V::Array{Vector{T2}},n::Integer)=ttv(X,VectorCell(V),n)
