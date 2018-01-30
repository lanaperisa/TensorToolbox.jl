#Tensors in Kruskal (CP) format + functions

export ktensor, arrange, arrange!, display, extract, innerprod, isequal, full, minus, mtimes, mttkrp, ncomponents, ndims
export normalize, normalize!, permutedims, plus, redistribute, redistribute!, size, tenmat, ttensor, vecnorm, uminus

type ktensor{T<:Number}
	lambda::Vector{T}
	fmat::MatrixCell
	isorth::Bool
	function ktensor{T}(lambda::Vector{T},fmat::MatrixCell,isorth::Bool) where T<:Number
    @assert(length(unique([size(fmat[n],2) for n=1:length(fmat)]))==1,"Factor matrices must have same number of columns.")
    @assert(length(lambda)==size(fmat[1],2),"Dimension mismatch.")
		for A in fmat
			if norm(A'*A-eye(size(A,2)))>(size(A,1)^2)*eps()
				isorth=false
			end
		end
		new(lambda,fmat,isorth)
	end
end
ktensor{T}(lambda::Vector{T},fmat::MatrixCell,isorth::Bool)=ktensor{T}(lambda,fmat,isorth)
ktensor{T}(lambda::Vector{T},fmat::MatrixCell)=ktensor{T}(lambda,fmat,true)
ktensor{T}(lambda::Vector{T},mat::Matrix)=ktensor{T}(lambda,collect(mat),true)
ktensor(fmat::MatrixCell)=ktensor(ones(size(fmat[1],2)),fmat)
@doc """ Kruskal tensor is defined by a vector of weights and factor matrices. """ ->
function ktensor{T}(lambda::Vector{T},mat...)
  fmat=MatrixCell(0)
  for M in mat
			push!(fmat,M)
	end
	ktensor{T}(lambda,fmat,true)
end
#If array of matrices isn't defined as MatrixCell, but as M=[M1,M2,...,Mn]:
ktensor{T,T1<:Number}(lambda::Vector{T},fmat::Array{Matrix{T1}},isorth::Bool)=ktensor{T}(lambda,MatrixCell(fmat),isorth)
ktensor{T,T1<:Number}(lambda::Vector{T},fmat::Array{Matrix{T1}})=ktensor{T}(lambda,MatrixCell(fmat),true)
ktensor{T<:Number}(fmat::Vector{Matrix{T}})=ktensor(MatrixCell(fmat))

@doc """Arranges the rank-1 components of a ktensor.""" ->
function arrange{T<:Number}(X::ktensor{T},mode=-1)
    N=ndims(X)
    Xnorm=normalize(X) #Ensure that matrices are normalized
    p=sortperm(Xnorm.lambda,rev=true) #Sort
    lambda=Xnorm.lambda[p]
    fmat=MatrixCell(N)
    [fmat[n]=Xnorm.fmat[n][:,p] for n=1:N]
    #Absorb the weight into one factor, if requested
    if mode>0
        fmat[mode]=fmat[mode].*repmat(lambda',size(X,mode),1)
        lambda = ones(lambda)
    end
    ktensor(lambda,fmat)
end
function arrange{T<:Number,D<:Integer}(X::ktensor{T},perm::Vector{D})
    N=ndims(X)
    @assert(collect(1:ncomponents(X))==sort(perm),"Invalid permutation.")
    lambda = X.lambda[perm]
    fmat=MatrixCell(N)
    [fmat[n]=X.fmat[n][:,perm] for n=1:N]
    ktensor(lambda,fmat)
end

function arrange!{T<:Number}(X::ktensor{T},mode=-1)
    N=ndims(X)
    normalize!(X) #Ensure that matrices are normalized
    p=sortperm(X.lambda,rev=true) #Sort
    X.lambda=X.lambda[p]
    [X.fmat[n]=X.fmat[n][:,p] for n=1:N]
    #Absorb the weight into one factor, if requested
    if mode>0
        X.fmat[mode]=X.fmat[mode].*repmat(X.lambda',size(X,mode),1)
        X.lambda = ones(X.lambda)
    end
    X
end
function arrange!{T<:Number,D<:Integer}(X::ktensor{T},perm::Vector{D})
    N=ndims(X)
    @assert(collect(1:ncomponents(X))==sort(perm),"Invalid permutation.")
    X.lambda = X.lambda[perm]
    [X.fmat[n]=X.fmat[n][:,perm] for n=1:N]
    X
end


#@doc """Displays a ktensor.""" ->
function display{T<:Number}(X::ktensor{T},name="ktensor")
    println("Kruskal tensor of size ",size(X),":\n")
    println("$name.lambda: ")
    show(STDOUT, "text/plain", X.lambda)
    for n=1:ndims(X)
        println("\n\n$name.fmat[$n]:")
        show(STDOUT, "text/plain", X.fmat[n])
    end
end

@doc """Creates a new ktensor with only the specified factors."""->
function extract{T<:Number,D<:Integer}(X::ktensor{T},factors::Vector{D})
    N=ndims(X)
    lambda = X.lambda[factors]
    fmat = MatrixCell(N)
    [fmat[n]=X.fmat[n][:,factors] for n=1:N]
    ktensor(lambda, fmat)
end
extract{T<:Number}(X::ktensor{T},factors::Integer)=extract(X,[factors])

#@doc """ Makes full tensor out of a ktensor. """ ->
function full{T<:Number}(X::ktensor{T})
  reshape(Array(X.lambda'*khatrirao(X.fmat[end:-1:1])'),size(X)...)
end

#@doc """ Inner product of two ktensors. """ ->
function innerprod{T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})
  @assert(size(X1)==size(X2),"Dimension mismatch.")
  inpr=X1.lambda*X2.lambda'
  for n=1:ndims(X1)
    inpr=inpr.*(X1.fmat[n]'*X2.fmat[n])
  end
  sum(inpr[:])
end
function innerprod{T<:Number}(X1::ktensor{T},X2)
  @assert(isa(X2,ttensor) || isa(X2,Array), "Inner product not available for type $(typeof(X2)).")
  @assert(size(X1)==size(X2),"Dimension mismatch.")
  inpr=0
  for r=1:ncomponents(X1)
    inpr+=X1.lambda[r]*ttv(X2,[X1.fmat[n][:,r] for n=1:ndims(X1)])
  end
  inpr
end
innerprod{T<:Number}(X1,X2::ktensor{T})=innerprod(X2,X1)

#@doc """ Checks wheater two tensors have equal components. """ ->
function isequal{T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})
  if (X1.lambda == X2.lambda) && (X1.fmat == X2.fmat)
    true
  else
    false
  end
end
=={T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})=isequal(X1,X2)

#@doc """ Matricized tensor times Khatri-Rao product. """->
function mttkrp{T<:Number}(X::ktensor{T},M::MatrixCell,n::Integer)
  N=ndims(X)
  @assert(length(M) == N,"Wrong number of matrices")
  n==1 ? R=size(M[2],2) : R=size(M[1],2)
  modes=setdiff(1:N,n)
  W=repmat(X.lambda,1,R) #matrix of weights
  for m in modes
    W=W.*(X.fmat[m]'*M[m])
  end
  X.fmat[n]*W
end
mttkrp{T1<:Number,T2<:Number}(X::ktensor{T1},M::Array{Matrix{T2}},n::Integer)=mttkrp(X,MatrixCell(M),n)

function minus{T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})
  X1+(-1)*X2
end
-{T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})= minus(X1,X2)

#@doc """ Scalar times tensor. """ ->
function mtimes{T<:Number}(α::Number,X::ktensor{T})
	ktensor(α*X.lambda,X.fmat);
end
*{T1<:Number,T2<:Number}(α::T1,X::ktensor{T2})=mtimes(α,X)
*{T1<:Number,T2<:Number}(X::ktensor{T1},α::T2)=*(α,X)

@doc """Number of components of a ktensor."""->
function ncomponents{T<:Number}(X::ktensor{T})
  length(X.lambda)
end

#@doc """ Number of modes of a ktensor. """ ->
function ndims{T<:Number}(X::ktensor{T})
  length(X.fmat)
end

@doc """Normalize columns of factor matrices of a ktensor. Also ensures that lambda is positive."""->
#absorbing the excess weight into lambda or (mode)th factor matrix
#if mode=0, equally divides the weights across the factor matrices and all lambda values are 1
function normalize{T<:Number}(X::ktensor{T},mode=-1;normtype=2,factor=-1)
  if mode=="sort"
    mode=-2
  end
  N=ndims(X)
  lambda=deepcopy(X.lambda)
  fmat=deepcopy(X.fmat)
  if factor>0
    @assert(0<factor<=N,"Factor exceeds dimension.")
    for r = 1:ncomponents(X)
      nr = norm(fmat[factor][:,r],normtype);
      lambda[r]=lambda[r]*nr
      if nr > 0
         fmat[factor][:,r]=fmat[factor][:,r]/nr
      end
    end
  end
  for r=1:ncomponents(X)
    for n=1:N
      nr=norm(fmat[n][:,r],normtype)
      lambda[r]=lambda[r]*nr
      if nr>0
        fmat[n][:,r]=fmat[n][:,r]/nr
      end
    end
  end
  #Check that all the lambda values are positive
  negind = find(lambda .< 0)
  for i in negind
    lambda[i] = -1 * lambda[i]
    fmat[1][:,i] = -1 * fmat[1][:,i]
  end
  if mode == 0
    #D = diagm(nthroot(X.lambda,ndims(X)))
    #[fmat[n]=X.fmat[n]*D for n=1:N]
    d=(lambda').^(1/N)
   [fmat[n]=fmat[n].*repmat(d,size(X,n),1) for n=1:N]
    lambda = ones(lambda)
  elseif mode > 0
    #fmat[mode] = fmat[mode]*diagm(lambda)
    fmat[mode] = fmat[mode].*repmat(lambda',size(X,mode),1)
    lambda = ones(lambda)
  elseif mode==-2
    p = sortperm(lambda,rev=true)
    return arrange(ktensor(lambda,fmat),p)
  end
  ktensor(lambda,fmat)
end

function normalize!{T<:Number}(X::ktensor{T},mode=-1;normtype=2,factor=-1)
  if mode=="sort"
    mode=-2
  end
  N=ndims(X)
  if factor>0
    @assert(0<factor<=N,"Factor exceeds dimension.")
    for r = 1:ncomponents(X)
      nr = norm(X.fmat[factor][:,r],normtype);
      X.lambda[r]=X.lambda[r]*nr
      if nr > 0
         X.fmat[factor][:,r]=X.fmat[factor][:,r]/nr
      end
    end
  end
  for r=1:ncomponents(X)
    for n=1:N
      nr=norm(X.fmat[n][:,r],normtype)
      X.lambda[r]=X.lambda[r]*nr
      if nr>0
        X.fmat[n][:,r]=X.fmat[n][:,r]/nr
      end
    end
  end
  #Check that all the lambda values are positive
  negind = find(X.lambda .< 0)
  for i in negind
    X.fmat[1][:,i] = -1 * X.fmat[1][:,i]
    X.lambda[i] = -1 * X.lambda[i]
  end
  if mode == 0
     d=(X.lambda').^(1/N)
     [X.fmat[n]=X.fmat[n].*repmat(d,size(X,n),1) for n=1:N]
     X.lambda = ones(X.lambda)
  elseif mode > 0
    X.fmat[mode] = X.fmat[mode].*repmat(X.lambda',size(X,mode),1)
    X.lambda = ones(X.lambda)
  elseif mode==-2
    p = sortperm(X.lambda,rev=true)
    X = arrange(X,p)
  end
   X
end

@doc """Permute dimensions of a ktensor."""->
function permutedims{T<:Number,D<:Integer}(X::ktensor{T},perm::Vector{D})
    N=ndims(X)
    @assert(collect(1:N)==sort(perm),"Invalid permutation.")
    ktensor(X.lambda,X.fmat[perm])
end

function plus{T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})
	@assert(size(X1) == size(X2),"Dimension mismatch.")
  lambda=[X1.lambda; X2.lambda]
	fmat=Matrix[[X1.fmat[n] X2.fmat[n]] for n=1:ndims(X1)] #concatenate factor matrices
	ktensor(lambda,fmat)
end
+{T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})=plus(X1,X2)

@doc """Distribute lambda values to a specified mode.."""->
function redistribute{T<:Number}(X::ktensor{T},mode::Integer)
    lambda=deepcopy(X.lambda)
    fmat=deepcopy(X.fmat)
    for r = 1:ncomponents(X)
        fmat[mode][:,r] = fmat[mode][:,r] * lambda[r];
        lambda[r] = 1;
    end
    ktensor(lambda,fmat)
end
function redistribute!{T<:Number}(X::ktensor{T},mode::Integer)
    for r = 1:ncomponents(X)
        X.fmat[mode][:,r] = X.fmat[mode][:,r] * X.lambda[r];
        X.lambda[r] = 1;
    end
    X
end


function Base.show{T<:Number}(io::IO,X::ktensor{T})
    display(X)
end

function size{T<:Number}(X::ktensor{T})
  tuple([size(X.fmat[n],1) for n=1:length(X.fmat)]...)
end
#n-th dimension of X
function size{T<:Number}(X::ktensor{T},n::Integer)
  size(X.fmat[n],1)
end

#@doc """ n-mode matricization of a tensor. """ ->
tenmat{T<:Number}(X::ktensor{T},n::Integer)=tenmat(full(X),n)

@doc """Create ttensor out of ktensor."""->
function ttensor{T<:Number}(X::ktensor{T})
    Xred=redistribute(X,1)
    ttensor(neye(ndims(X)),Xred.fmat)
end

#@doc """ Frobenius norm of a ktensor. """ ->
function vecnorm{T<:Number}(X::ktensor{T})
  nrm=X.lambda*X.lambda'
  for n=1:ndims(X)
    nrm=nrm.*(X.fmat[n]'*X.fmat[n])
  end
  sqrt(abs(sum(nrm[:])))
end

uminus{T<:Number}(X::ktensor{T})=mtimes(-1,X)
-{T<:Number}(X::ktensor{T})=uminus(X)
