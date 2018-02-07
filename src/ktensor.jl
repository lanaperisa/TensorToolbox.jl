#Tensors in Kruskal (CP) format + functions

export ktensor, randktensor, arrange, arrange!, cp_als, display, extract, innerprod, isequal, fixsigns, fixsigns!, full, minus, mtimes, mttkrp, ncomponents, ndims
export normalize, normalize!, nvecs, permutedims, plus, redistribute, redistribute!, size, tenmat, tocell, ttensor, ttm, ttv, uminus, vecnorm

"""
    ktensor(fmat)
    ktensor(lambda,fmat)

Tensor in Kruskal format defined by its vector of weights lambda and factor matrices. Default lambda: vector of ones.
For ktensor X, X.isorth=true if factor matrices are othonormal.
"""
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
"""
    randktensor(I::Vector,R::Integer)
    randktensor(I::Integer,R::Integer,N::Integer)

Creates random Kruskal tensor of size I with R components, or of order N and size I × ⋯ × I.
"""
function randktensor{D<:Integer}(I::Vector{D},R::Integer)
  fmat=Matrix[randn(I[n],R) for n=1:length(I)] #create random factor matrices
  ktensor(fmat)
end
randktensor(I::Number,R::Number,N::Integer)=randktensor(repmat([I],N),R)
#For input defined as tuples or nx1 matrices - ranttensor(([I,I,I],[R,R,R]))

"""
    arrange(X[,n::Integer])
    arrange(X[,P::Vector])

Arrange the rank-1 components of a ktensor: normalize the columns of the factor matrices and then sort the ktensor components by magnitude, greatest to least.

## Arguments:
- `n`: Absorb the weights into the nth factor matrix instead of lambda.
- `P`: Rearrange the components of X according to the permutation P. P should be a permutation of 1 to ncomponents(X).
"""
function arrange{T<:Number}(X::ktensor{T},mode=-1)
    N=ndims(X)
    Xnorm=normalize(X) #Ensure that matrices are normalized
    p=sortperm(Xnorm.lambda,rev=true) #Sort
    lambda=Xnorm.lambda[p]
    fmat=MatrixCell(N)
    [fmat[n]=Xnorm.fmat[n][:,p] for n=1:N]
    #Absorb the weight into one factor, if requested
    if mode>0
        fmat[mode]=fmat[mode].*repmat(lambda',size(X,mode),1) #fmat[mode]*diagm(lambda)
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

"""
    arrange!(X[,n::Integer])
    arrange!(X[,P::Vector])

Arrange the rank-1 components of a ktensor: normalize the columns of the factor matrices and then sort the ktensor components by magnitude, greatest to least.
Rewrite ktensor.

## Arguments:
- `n`: Absorb the weights into the nth factor matrix instead of lambda.
- `P`: Rearrange the components of X according to the permutation P. P should be a permutation of 1 to ncomponents(X).
"""
function arrange!{T<:Number}(X::ktensor{T},mode=-1)
    N=ndims(X)
    normalize!(X) #Ensure that matrices are normalized
    p=sortperm(X.lambda,rev=true) #Sort
    X.lambda=X.lambda[p]
    [X.fmat[n]=X.fmat[n][:,p] for n=1:N]
    #Absorb the weight into one factor, if requested
    if mode>0
        X.fmat[mode]=X.fmat[mode].*repmat(X.lambda',size(X,mode),1) #X.fmat[mode]*diagm(X.lambda)
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

#Compute a CP decomposition with R components of a tensor X. **Documentation in tensor.jl.
function cp_als{T<:Number}(X::ktensor{T},R::Integer;init="rand",tol=1e-4,maxit=1000,dimorder=[])
    N=ndims(X)
    nr=vecnorm(X)
    K=ktensor
    if length(dimorder) == 0
        dimorder=collect(1:N)
    end
    fmat=MatrixCell(N)
    if isa(init,Vector) || isa(init,MatrixCell)
        @assert(length(init)==N,"Wrong number of initial matrices.")
        for n in dimorder[2:end]
            @assert(size(init[n])==(size(X,n),R),"$(n)-th initial matrix is of wrong size.")
            fmat[n]=init[n]
        end
    elseif init=="rand"
        [fmat[n]=rand(size(X,n),R) for n in dimorder[2:end]]
    elseif init=="eigs" || init=="nvecs"
        [fmat[n]=nvecs(X,n,R) for n in dimorder[2:end]]
    else
        error("Initialization method wrong.")
    end
    G = zeros(R,R,N); #initalize gramians
    [G[:,:,n]=fmat[n]'*fmat[n] for n in dimorder[2:end]]
    fit=0
    for k=1:maxit
        fitold=fit
        lambda=[]
        for n in dimorder
            fmat[n]=mttkrp(X,fmat,n)
            W=reshape(prod(G[:,:,setdiff(collect(1:N),n)],3),Val{2})
            fmat[n]=fmat[n]/W
            if k == 1
                lambda = sqrt.(sum(fmat[n].^2,1))' #2-norm
            else
                lambda = maximum(maximum(abs.(fmat[n]),1),1)' #max-norm
            end
            fmat[n] = fmat[n]./lambda'
            G[:,:,n] = fmat[n]'*fmat[n]
        end
        K=ktensor(vec(lambda),fmat)
        if nr==0
            fit=vecnorm(K)^2-2*innerprod(X,K)
        else
            nr_res=sqrt.(nr^2+vecnorm(K)^2-2*innerprod(X,K))
            fir=1-nr_res/nr
        end
        fitchange=abs.(fitold-fit)
        if k>1 && fitchange<tol
            break
        end
    end
    arrange!(K)
    fixsigns!(K)
    K
end

#Display a ktensor. **Documentation in ttensor.jl.
function display{T<:Number}(X::ktensor{T},name="ktensor")
    println("Kruskal tensor of size ",size(X),":\n")
    println("$name.lambda: ")
    show(STDOUT, "text/plain", X.lambda)
    for n=1:ndims(X)
        println("\n\n$name.fmat[$n]:")
        show(STDOUT, "text/plain", X.fmat[n])
    end
end

"""
    extract(X,factors)

Create a new ktensor with only the specified factors.
"""
function extract{T<:Number,D<:Integer}(X::ktensor{T},factors::Vector{D})
    N=ndims(X)
    lambda = X.lambda[factors]
    fmat = MatrixCell(N)
    [fmat[n]=X.fmat[n][:,factors] for n=1:N]
    ktensor(lambda, fmat)
end
extract{T<:Number}(X::ktensor{T},factors::Integer)=extract(X,[factors])

"""
    fixsign(X)

Fix sign ambiguity of a ktensor.
"""
function fixsigns{T<:Number}(X::ktensor{T})
    N=ndims(X)
    sgn=zeros(Int,N)
    fmat=deepcopy(X.fmat)
     for r=1:ncomponents(X)
        for n=1:N
            maxind=indmax(abs.(X.fmat[n][:,r]))
            sgn[n]=sign(X.fmat[n][maxind,r])
        end
        negind=find(sgn.==-1)
        nflip=2*floor(Int,length(negind)/2)
        for i=1:nflip
            fmat[negind[i]][:,r]=-X.fmat[negind[i]][:,r]
        end
    end
    ktensor(X.lambda,fmat)
end
"""
    fixsign(X)

Fix sign ambiguity of a ktensor. Rewrite ktensor.
"""
function fixsigns!{T<:Number}(X::ktensor{T})
    N=ndims(X)
    sgn=zeros(Int,N)
     for r=1:ncomponents(X)
        for n=1:N
            maxind=indmax(abs.(X.fmat[n][:,r]))
            sgn[n]=sign(X.fmat[n][maxind,r])
        end
        negind=find(sgn.==-1)
        nflip=2*floor(Int,length(negind)/2)
        for i=1:nflip
            X.fmat[negind[i]][:,r]=-X.fmat[negind[i]][:,r]
        end
    end
end

#Makes full tensor out of a ktensor. **Documentation in ttensor.jl.
function full{T<:Number}(X::ktensor{T})
  reshape(Array(X.lambda'*khatrirao(X.fmat[end:-1:1])'),size(X)...)
end

#Inner product of two ktensors. **Documentation in tensor.jl.
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

#Two ktensors are equal if they have equal components. **Documentation in ttensor.jl.
function isequal{T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})
  if (X1.lambda == X2.lambda) && (X1.fmat == X2.fmat)
    true
  else
    false
  end
end
=={T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})=isequal(X1,X2)

#Subtraction of two ktensors. **Documentation in ttensor.jl.
function minus{T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})
  X1+(-1)*X2
end
-{T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})= minus(X1,X2)

#Scalar times tensor. **Documentation in ttensor.jl.
function mtimes{T<:Number}(α::Number,X::ktensor{T})
	ktensor(α*X.lambda,X.fmat);
end
*{T1<:Number,T2<:Number}(α::T1,X::ktensor{T2})=mtimes(α,X)
*{T1<:Number,T2<:Number}(X::ktensor{T1},α::T2)=*(α,X)

#Matricized ktensor times Khatri-Rao product. **Documentation in tensor.jl.
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

"""
    ncomponents(X)

Number of components of a ktensor.
"""
function ncomponents{T<:Number}(X::ktensor{T})
  length(X.lambda)
end

#Number of modes of a ttensor. **Documentation in Base.
function ndims{T<:Number}(X::ktensor{T})
  length(X.fmat)
end

"""
    normalize(X[,n;normtype,factor])

Normalize columns of factor matrices of a ktensor. Also ensures that lambda is positive.

## Arguments:
- `n`: Absorbe the excess weight into nth factor matrix.
   If n=0, equally divide the weights across the factor matrices and set lambda to vector of ones.
   If n="sort", sort the components by lambda value, from greatest to least.
- `normtype`: Default: 2.
- `factor`: Just normalize specified factor.
"""
function normalize{T<:Number}(X::ktensor{T},mode=-1;normtype=2,factor=-1)
  if mode=="sort"
    mode=-2
  end
  N=ndims(X)
  lambda=deepcopy(float(X.lambda))
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
    d=(lambda').^(1/N)
   [fmat[n]=fmat[n].*repmat(d,size(X,n),1) for n=1:N] #fmat[n]*diagm(d)
    lambda = ones(lambda)
  elseif mode > 0
    fmat[mode] = fmat[mode].*repmat(lambda',size(X,mode),1) #fmat[mode]*diagm(lambda)
    lambda = ones(lambda)
  elseif mode==-2
    p = sortperm(lambda,rev=true)
    return arrange(ktensor(lambda,fmat),p)
  end
  ktensor(lambda,fmat)
end

"""
    normalize!(X[,n,normtype,factor])

Normalize columns of factor matrices of a ktensor. Rewrite ktensor. Also ensures that lambda is positive.

## Arguments:
- `n`: Absorbe the excess weight into nth factor matrix.
   If n=0, equally divide the weights across the factor matrices and set lambda to vector of ones.
   If n="sort", sort the components by lambda value, from greatest to least.
- `normtype`: Default: 2.
- `factor`: Just normalize specified factor.
"""
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
     [X.fmat[n]=X.fmat[n].*repmat(d,size(X,n),1) for n=1:N] #X.fmat[n]*diagm(d)
     X.lambda = ones(X.lambda)
  elseif mode > 0
    X.fmat[mode] = X.fmat[mode].*repmat(X.lambda',size(X,mode),1) #X.fmat[n]*diagm(X.lambda)
    X.lambda = ones(X.lambda)
  elseif mode==-2
    p = sortperm(X.lambda,rev=true)
    X = arrange(X,p)
  end
   X
end

#Computes the r leading singular vectors of mode-n matricization of a tensor X.
# **Documentation in tensor.jl.
function nvecs{T<:Number}(X::ktensor{T},n::Integer,r=0;flipsign=false)
    M = X.lambda * X.lambda'
    for m in setdiff(1:ndims(X),n)
        M = M .* (X.fmat[m]' * X.fmat[m])
    end
    V=eigs(Symmetric(X.fmat[n] * M * X.fmat[n]'),nev=r,which=:LM)[2]
    if flipsign
        maxind = findmax(abs.(V),1)[2]
        for i = 1:r
            ind=ind2sub(size(V),maxind[i])
            if V[ind...] < 0
                V[:,ind[2]] = V[:,ind[2]] * -1
            end
        end
    end
    V
end

#Permute dimensions of a ktensor. **Documentation in Base.
function permutedims{T<:Number,D<:Integer}(X::ktensor{T},perm::Vector{D})
    N=ndims(X)
    @assert(collect(1:N)==sort(perm),"Invalid permutation.")
    ktensor(X.lambda,X.fmat[perm])
end

#Addition of two ktensors. **Documentation in ttensor.jl.
function plus{T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})
	@assert(size(X1) == size(X2),"Dimension mismatch.")
  lambda=[X1.lambda; X2.lambda]
	fmat=Matrix[[X1.fmat[n] X2.fmat[n]] for n=1:ndims(X1)] #concatenate factor matrices
	ktensor(lambda,fmat)
end
+{T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})=plus(X1,X2)

"""
    redistribute(X,n)

Distribute lambda values to a mode n.
"""
function redistribute{T<:Number}(X::ktensor{T},mode::Integer)
    lambda=deepcopy(float(X.lambda))
    fmat=deepcopy(X.fmat)
    for r = 1:ncomponents(X)
        fmat[mode][:,r] = fmat[mode][:,r] * lambda[r];
        lambda[r] = 1;
    end
    ktensor(lambda,fmat)
end

"""
    redistribute!(X,n)

Distribute lambda values to a mode n. Rewrite ktensor X.
"""
function redistribute!{T<:Number}(X::ktensor{T},mode::Integer)
    for r = 1:ncomponents(X)
        X.fmat[mode][:,r] = X.fmat[mode][:,r] * X.lambda[r];
        X.lambda[r] = 1;
    end
    X
end

#Size of a ktensor. **Documentation in Base.
function size{T<:Number}(X::ktensor{T})
  tuple([size(X.fmat[n],1) for n=1:length(X.fmat)]...)
end
#Size of n-th mode of a ktensor.
function size{T<:Number}(X::ktensor{T},n::Integer)
  size(X.fmat[n],1)
end

function Base.show{T<:Number}(io::IO,X::ktensor{T})
    display(X)
end

#Mode-n matricization of a ktensor. **Documentation in tensor.jl.
tenmat{T<:Number}(X::ktensor{T},n::Integer)=tenmat(full(X),n)

"""
tocell(X[,n])

Converts ktensor X into MatrixCell. If n specified, absorb the weights into the nth factor matrix.
"""
function tocell{T<:Number}(X::ktensor{T},mode::Integer)
    Y = normalize(X,mode)
    Y.fmat
end
function tocell{T<:Number}(X::ktensor{T})
    if isequal(X.lambda,ones(size(X.lambda)))
        return X.fmat;
    end
    N=ndims(X)
    d = (X.lambda').^(1/N)
    fmat=MatrixCell(N)
    for n = 1:N
        fmat[n] = X.fmat[n].*repmat(d,size(X.fmat[n],1),1)
    end
    fmat
end

"""
---
    ttensor(X::ktensor)

Create ttensor out of a ktensor.
"""
function ttensor{T<:Number}(X::ktensor{T})
    Xred=redistribute(X,1)
    ttensor(neye(ncomponents(X),order=ndims(X)),Xred.fmat)
end

#Mode-n multiplication of ktensor and matrix. **Documentation in tensor.jl.
#t='t' transposes matrices
function ttm{T<:Number,D<:Integer}(X::ktensor{T},M::MatrixCell,modes::Vector{D},t='n')
  if t=='t'
	 M=vec(M')
	end
	@assert(length(modes)<=length(M),"Too few matrices")
	@assert(length(M)<=ndims(X),"Too many matrices")
  if length(modes)<length(M)
    M=M[modes]
  end
  fmat=deepcopy(X.fmat)
  for n=1:length(modes)
    fmat[modes[n]]=M[n]*fmat[modes[n]]
  end
    ktensor(X.lambda,fmat)
end
ttm{T<:Number,D<:Integer}(X::ktensor{T},M::MatrixCell,modes::Range{D},t='n')=ttm(X,M,collect(modes),t)
ttm{T1<:Number,T2<:Number}(X::ktensor{T1},M::Matrix{T2},n::Integer,t='n')=ttm(X,[M],[n],t)
ttm{T<:Number}(X::ktensor{T},M::MatrixCell,t::Char)=ttm(X,M,1:length(M),t)
ttm{T<:Number}(X::ktensor{T},M::MatrixCell)=ttm(X,M,1:length(M))
function ttm{T<:Number}(X::ktensor{T},M::MatrixCell,n::Integer,t='n')
	if n>0
		ttm(X,M[n],n,t)
	else
		modes=setdiff(1:ndims(X),-n)
		ttm(X,M,modes,t)
	end
end
#If array of matrices isn't defined as MatrixCell, but as M=[M1,M2,...,Mn]:
ttm{T1<:Number,T2<:Number,D<:Integer}(X::ktensor{T1},M::Array{Matrix{T2}},modes::Vector{D},t='n')=ttm(X,MatrixCell(M),modes,t)
ttm{T1<:Number,T2<:Number,D<:Integer}(X::ktensor{T1},M::Array{Matrix{T2}},modes::Range{D},t='n')=ttm(X,MatrixCell(M),modes,t)
ttm{T1<:Number,T2<:Number}(X::ktensor{T1},M::Array{Matrix{T2}},t::Char)=ttm(X,MatrixCell(M),t)
ttm{T1<:Number,T2<:Number}(X::ktensor{T1},M::Array{Matrix{T2}})=ttm(X,MatrixCell(M))
ttm{T1<:Number,T2<:Number}(X::ktensor{T1},M::Array{Matrix{T2}},n::Integer,t='n')=ttm(X,MatrixCell(M),n,t)

#Mode-n multiplication ktensor and vector. **Documentation in tensor.jl.
#t='t' transposes matrices
function ttv{T<:Number,D<:Integer}(X::ktensor{T},V::VectorCell,modes::Vector{D})
  N=ndims(X)
  remmodes=setdiff(1:N,modes)
  lambda=deepcopy(float(X.lambda))
  if length(modes) < length(V)
    V=V[modes]
  end
  for n=1:length(modes)
      lambda=lambda.*(X.fmat[modes[n]]'*V[n])
  end
  if isempty(remmodes)
    sum(lambda)
  else
    ktensor(lambda,X.fmat[remmodes])
  end
end
ttv{T1<:Number,T2<:Number}(X::ktensor{T1},v::Vector{T2},n::Integer)=ttv(X,Vector[v],[n])
ttv{T<:Number,D<:Integer}(X::ktensor{T},V::VectorCell,modes::Range{D})=ttv(X,V,collect(modes))
ttv{T<:Number}(X::ktensor{T},V::VectorCell)=ttv(X,V,1:length(V))
function ttv{T<:Number}(X::ktensor{T},V::VectorCell,n::Integer)
	if n>0
		ttm(X,V[n],n)
	else
		modes=setdiff(1:ndims(X),-n)
		ttm(X,A,modes)
	end
end
#If array of vectors isn't defined as VectorCell, but as V=[v1,v2,...,vn]:
ttv{T1<:Number,T2<:Number,D<:Integer}(X::ktensor{T1},V::Array{Vector{T2}},modes::Vector{D})=ttv(X,VectorCell(V),modes)
ttv{T1<:Number,T2<:Number,D<:Integer}(X::ktensor{T1},V::Array{Vector{T2}},modes::Range{D})=ttv(X,VectorCell(V),modes)
ttv{T1<:Number,T2<:Number}(X::ktensor{T1},V::Array{Vector{T2}})=ttv(X,VectorCell(V))
ttv{T1<:Number,T2<:Number}(X::ktensor{T1},V::Array{Vector{T2}},n::Integer)=ttv(X,VectorCell(V),n)

#**Documentation in ttensor.jl.
uminus{T<:Number}(X::ktensor{T})=mtimes(-1,X)
-{T<:Number}(X::ktensor{T})=uminus(X)

#Frobenius norm of a ktensor. **Documentation in Base.
function vecnorm{T<:Number}(X::ktensor{T})
  nrm=X.lambda*X.lambda'
  for n=1:ndims(X)
    nrm=nrm.*(X.fmat[n]'*X.fmat[n])
  end
  sqrt.(abs.(sum(nrm[:])))
end
