#Tensors in Kruskal (CP) format + functions

export ktensor, randktensor, arrange, arrange!, cp_als, display, extract, fixsigns, fixsigns!, full, innerprod, isequal, krank, kten2TT, minus, mtimes, mttkrp, ncomponents, ndims
export normalize, normalize!, nvecs, permutedims, plus, redistribute, redistribute!, size, tenmat, tocell, ttensor, ttm, ttv, uminus, norm

"""
    ktensor(fmat)
    ktensor(lambda,fmat)

Tensor in Kruskal format defined by its vector of weights lambda and factor matrices. Default lambda: vector of ones.
For ktensor X, X.isorth=true if factor matrices are othonormal.
"""
mutable struct ktensor
	lambda::AbstractVector{<:Number}
	fmat::MatrixCell
	isorth::Bool
	function ktensor(lambda::AbstractVector{<:Number},fmat::MatrixCell,isorth::Bool)
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
#ktensor(lambda::AbstractArray{T1,1},fmat::MatrixCell,isorth::Bool) where {T1<:Number,T2<:Number}=ktensor{T1,T2}(lambda,fmat,isorth)
ktensor(lambda::AbstractVector{<:Number},fmat::MatrixCell)=ktensor(lambda,fmat,true)
ktensor(lambda::AbstractVector{<:Number},mat::AbstractMatrix{<:Number})=ktensor(lambda,collect(mat),true)
ktensor(fmat::MatrixCell)=ktensor(ones(size(fmat[1],2)),fmat)
# function ktensor(lambda::AbstractVector{<:Number},mat...)
#   fmat=MatrixCell(undef,0)
#   for M in mat
# 			push!(fmat,M)
# 	end
# 	ktensor(lambda,fmat,true)
# end
#If array of matrices isn't defined as MatrixCell, but as M=[M1,M2,...,Mn]:
ktensor(lambda::AbstractVector{<:Number},fmat::Array{Matrix{T},1},isorth::Bool) where {T<:Number}=ktensor(lambda,MatrixCell(fmat),isorth)
ktensor(lambda::AbstractVector{<:Number},fmat::Array{Matrix{T},1}) where {T<:Number}=ktensor(lambda,MatrixCell(fmat),true)
ktensor(fmat::Array{Matrix{T},1}) where {T<:Number}=ktensor(MatrixCell(fmat))
"""
    randktensor(I::Vector,R::Integer)
    randktensor(I::Integer,R::Integer,N::Integer)

Create random Kruskal tensor of size I with R components, or of order N and size I × ⋯ × I.
"""
function randktensor(sz::AbstractVector{<:Integer},R::Integer)
  fmat=[randn(sz[n],R) for n=1:length(sz)] #create random factor matrices
  ktensor(fmat)
end
randktensor(sz::Number,R::Number,N::Integer)=randktensor(repeat([sz],N),R)
#For input defined as tuples or nx1 matrices - ranttensor(([I,I,I],[R,R,R]))

"""
    arrange(X[,n::Integer])
    arrange(X[,P::Vector])

Arrange the rank-1 components of a ktensor: normalize the columns of the factor matrices and then sort the ktensor components by magnitude, greatest to least.

## Arguments:
- `n`: Absorb the weights into the nth factor matrix instead of lambda.
- `P`: Rearrange the components of X according to the permutation P. P should be a permutation of 1 to ncomponents(X).
"""
function arrange(X::ktensor,mode=-1)
    N=ndims(X)
    Xnorm=normalize(X) #Ensure that matrices are normalized
    p=sortperm(Xnorm.lambda,rev=true) #Sort
    lambda=Xnorm.lambda[p]
    fmat=MatrixCell(undef,N)
    [fmat[n]=Xnorm.fmat[n][:,p] for n=1:N]
    #Absorb the weight into one factor, if requested
    if mode>0
        fmat[mode]=fmat[mode].*repeat(lambda',size(X,mode),1) #fmat[mode]*diagm(lambda)
        lambda = fill(1,size(lambda))#ones(lambda)
    end
    ktensor(lambda,fmat)
end
function arrange(X::ktensor,perm::AbstractVector{<:Integer})
    N=ndims(X)
    @assert(collect(1:ncomponents(X))==sort(perm),"Invalid permutation.")
    lambda = X.lambda[perm]
    fmat=MatrixCell(undef,N)
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
function arrange!(X::ktensor,mode=-1)
    N=ndims(X)
    normalize!(X) #Ensure that matrices are normalized
    p=sortperm(X.lambda,rev=true) #Sort
    X.lambda=X.lambda[p]
    [X.fmat[n]=X.fmat[n][:,p] for n=1:N]
    #Absorb the weight into one factor, if requested
    if mode>0
        X.fmat[mode]=X.fmat[mode].*repeat(X.lambda',size(X,mode),1) #X.fmat[mode]*diagm(X.lambda)
        X.lambda = fill(1,size(X.lambda))#ones(X.lambda)
    end
    X
end
function arrange!(X::ktensor,perm::AbstractVector{<:Integer})
    N=ndims(X)
    @assert(collect(1:ncomponents(X))==sort(perm),"Invalid permutation.")
    X.lambda = X.lambda[perm]
    [X.fmat[n]=X.fmat[n][:,perm] for n=1:N]
    X
end

#Compute a CP decomposition with R components of a tensor X. **Documentation in tensor.jl.
function cp_als(X::ktensor,R::Integer;init="rand",tol=1e-4,maxit=1000,dimorder=[])
    N=ndims(X)
    nr=norm(X)
    K=ktensor
    if length(dimorder) == 0
        dimorder=collect(1:N)
    end
    fmat=MatrixCell(undef,N)
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
            W=reshape(prod(G[:,:,setdiff(collect(1:N),n)],dims=3),Val(2))
            fmat[n]=fmat[n]/W
            if k == 1
                lambda = sqrt.(sum(fmat[n].^2,dims=1))[:] #2-norm
            else
                lambda = maximum(maximum(abs.(fmat[n]),dims=1),dims=1)[:] #max-norm
            end
            fmat[n] = fmat[n]./lambda'
            G[:,:,n] = fmat[n]'*fmat[n]
        end
        K=ktensor(lambda,fmat)
        if nr==0
            fit=norm(K)^2-2*innerprod(X,K)
        else
            nr_res=sqrt.(nr^2+norm(K)^2-2*innerprod(X,K))
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
function display(X::ktensor,name="ktensor")
    println("Kruskal tensor of size ",size(X),":\n")
    println("$name.lambda: ")
    show(stdout, "text/plain", X.lambda)
    for n=1:ndims(X)
        println("\n\n$name.fmat[$n]:")
        show(stdout, "text/plain", X.fmat[n])
    end
end

"""
    extract(X,factors)

Create a new ktensor with only the specified factors.
"""
function extract(X::ktensor,factors::AbstractVector{<:Integer})
    N=ndims(X)
    lambda = X.lambda[factors]
    fmat = MatrixCell(undef,N)
    [fmat[n]=X.fmat[n][:,factors] for n=1:N]
    ktensor(lambda, fmat)
end
extract(X::ktensor,factors::Integer)=extract(X,[factors])

"""
    fixsign(X)

Fix sign ambiguity of a ktensor.
"""
function fixsigns(X::ktensor)
    N=ndims(X)
    sgn=zeros(Int,N)
    fmat=deepcopy(X.fmat)
     for r=1:ncomponents(X)
        for n=1:N
            maxind=argmax(abs.(X.fmat[n][:,r]))
            sgn[n]=sign(X.fmat[n][maxind,r])
        end
        negind=findall(sgn.==-1)
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
function fixsigns!(X::ktensor)
    N=ndims(X)
    sgn=zeros(Int,N)
     for r=1:ncomponents(X)
        for n=1:N
            maxind=argmax(abs.(X.fmat[n][:,r]))
            sgn[n]=sign(X.fmat[n][maxind,r])
        end
        negind=findall(sgn.==-1)
        nflip=2*floor(Int,length(negind)/2)
        for i=1:nflip
            X.fmat[negind[i]][:,r]=-X.fmat[negind[i]][:,r]
        end
    end
end

#Makes full tensor out of a ktensor. **Documentation in ttensor.jl.
function full(X::ktensor)
  reshape(Array(X.lambda'*khatrirao(X.fmat[end:-1:1])'),size(X)...)
end

#Inner product of two ktensors. **Documentation in tensor.jl.
function innerprod(X1::ktensor,X2::ktensor)
  @assert(size(X1)==size(X2),"Dimension mismatch.")
  inpr=X1.lambda*X2.lambda'
  for n=1:ndims(X1)
    inpr=inpr.*(X1.fmat[n]'*X2.fmat[n])
  end
  sum(inpr[:])
end
function innerprod(X1::ktensor,X2)
  @assert(isa(X2,ttensor) || isa(X2,Array), "Inner product not available for type $(typeof(X2)).")
  @assert(size(X1)==size(X2),"Dimension mismatch.")
  inpr=0
  for r=1:ncomponents(X1)
    inpr+=X1.lambda[r]*ttv(X2,[X1.fmat[n][:,r] for n=1:ndims(X1)])[1]
  end
  inpr
end
innerprod(X1,X2::ktensor)=innerprod(X2,X1)

#Two ktensors are equal if they have equal components. **Documentation in ttensor.jl.
function isequal(X1::ktensor,X2::ktensor)
  if (X1.lambda == X2.lambda) && (X1.fmat == X2.fmat)
    true
  else
    false
  end
end
==(X1::ktensor,X2::ktensor)=isequal(X1,X2)

"""
    krank(X::ktensor)

Represenation k-rank. Equal to number of columns of factor matrices if ktensor X.
"""
function krank(X::ktensor)
    size(X.fmat[1],2)
end

"""
    kten2TT(X::ktensor)

Transform ktensor to TTtensor.
"""
function kten2TT(X::ktensor)
    N=ndims(X)
    Isz=size(X)
    r=krank(X)
    G=CoreCell(undef,N)
    G[1]=reshape(X.fmat[1],(1,Isz[1],r))
    G[N]=reshape(X.fmat[N]',(r,Isz[N],1))
    for n=2:N-1
        G[n]=zeros(r,Isz[n],r)
        for i=1:Isz[n]
            G[n][:,i,:]=Diagonal(X.fmat[n][i,:])
        end
    end
    TTtensor(G)
end


#Subtraction of two ktensors. **Documentation in ttensor.jl.
function minus(X1::ktensor,X2::ktensor)
  1*X1+(-1)*X2
end
-(X1::ktensor,X2::ktensor)= minus(X1,X2)

#Scalar times tensor. **Documentation in ttensor.jl.
function mtimes(α::Number,X::ktensor)
	ktensor(α*X.lambda,X.fmat);
end
*(α::Number,X::ktensor)=mtimes(α,X)
*(X::ktensor,α::Number)=*(α,X)

#Matricized ktensor times Khatri-Rao product. **Documentation in tensor.jl.
function mttkrp(X::ktensor,M::MatrixCell,n::Integer)
  N=ndims(X)
  @assert(length(M) == N,"Wrong number of matrices")
  n==1 ? R=size(M[2],2) : R=size(M[1],2)
  modes=setdiff(1:N,n)
  W=repeat(X.lambda,1,R) #matrix of weights
  for m in modes
    W=W.*(X.fmat[m]'*M[m])
  end
  X.fmat[n]*W
end
mttkrp(X::ktensor,M::Array{Matrix{T},1},n::Integer) where {T<:Number}=mttkrp(X,MatrixCell(M),n)

"""
    ncomponents(X)

Number of components of a ktensor.
"""
function ncomponents(X::ktensor)
  length(X.lambda)
end

#Number of modes of a ttensor. **Documentation in Base.
function ndims(X::ktensor)
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
function normalize(X::ktensor,mode=-1;normtype=2,factor=-1)
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
  negind = findall(lambda .< 0)
  for i in negind
    lambda[i] = -1 * lambda[i]
    fmat[1][:,i] = -1 * fmat[1][:,i]
  end
  if mode == 0
    d=(lambda').^(1/N)
   [fmat[n]=fmat[n].*repeat(d,size(X,n),1) for n=1:N] #fmat[n]*diagm(d)
    lambda = fill(1,size(lambda))#ones(lambda)
  elseif mode > 0
    fmat[mode] = fmat[mode].*repeat(lambda',size(X,mode),1) #fmat[mode]*diagm(lambda)
    lambda = fill(1,size(lambda))#ones(lambda)
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
function normalize!(X::ktensor,mode=-1;normtype=2,factor=-1)
  if mode=="sort"
    mode=-2
  end
  N=ndims(X)
  X.lambda=float(X.lambda)
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
  negind = findall(X.lambda .< 0)
  for i in negind
    X.fmat[1][:,i] = -1 * X.fmat[1][:,i]
    X.lambda[i] = -1 * X.lambda[i]
  end
  if mode == 0
     d=(X.lambda').^(1/N)
     [X.fmat[n]=X.fmat[n].*repeat(d,size(X,n),1) for n=1:N] #X.fmat[n]*diagm(d)
     X.lambda = fill(1,size(X.lambda))#ones(X.lambda)
  elseif mode > 0
    X.fmat[mode] = X.fmat[mode].*repeat(X.lambda',size(X,mode),1) #X.fmat[n]*diagm(X.lambda)
    X.lambda = fill(1,size(X.lambda))#ones(X.lambda)
  elseif mode==-2
    p = sortperm(X.lambda,rev=true)
    X = arrange(X,p)
  end
   X
end

#Computes the r leading singular vectors of mode-n matricization of a tensor X.
# **Documentation in tensor.jl.
function nvecs(X::ktensor,n::Integer,r=0;flipsign=false)
    M = X.lambda * X.lambda'
    for m in setdiff(1:ndims(X),n)
        M = M .* (X.fmat[m]' * X.fmat[m])
    end
    #V=eigs(Symmetric(X.fmat[n] * M * X.fmat[n]'),nev=r,which=:LM)[2] #has bugs
    V=eigen(Symmetric(X.fmat[n] * M * X.fmat[n]')).vectors[:,end:-1:end-r+1]
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
function permutedims(X::ktensor,perm::AbstractVector{<:Integer})
    N=ndims(X)
    @assert(collect(1:N)==sort(perm),"Invalid permutation.")
    ktensor(X.lambda,X.fmat[perm])
end

#Addition of two ktensors. **Documentation in ttensor.jl.
function plus(X1::ktensor,X2::ktensor)
	@assert(size(X1) == size(X2),"Dimension mismatch.")
  	lambda=[X1.lambda; X2.lambda]
	fmat=[[X1.fmat[n] X2.fmat[n]] for n=1:ndims(X1)] #concatenate factor matrices
	ktensor(lambda,fmat)
end
+(X1::ktensor,X2::ktensor)=plus(X1,X2)

"""
    redistribute(X,n)

Distribute lambda values to a mode n.
"""
function redistribute(X::ktensor,mode::Integer)
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
function redistribute!(X::ktensor,mode::Integer)
    for r = 1:ncomponents(X)
        X.fmat[mode][:,r] = X.fmat[mode][:,r] * X.lambda[r];
        X.lambda[r] = 1;
    end
    X
end

#Size of a ktensor. **Documentation in Base.
function size(X::ktensor)
  tuple([size(X.fmat[n],1) for n=1:length(X.fmat)]...)
end
#Size of n-th mode of a ktensor.
function size(X::ktensor,n::Integer)
  size(X.fmat[n],1)
end

function Base.show(io::IO,X::ktensor)
    display(X)
end

#Mode-n matricization of a ktensor. **Documentation in tensor.jl.
tenmat(X::ktensor,n::Integer)=tenmat(full(X),n)

"""
tocell(X[,n])

Converts ktensor X into MatrixCell. If n specified, absorb the weights into the nth factor matrix.
"""
function tocell(X::ktensor,mode::Integer)
    Y = normalize(X,mode)
    Y.fmat
end
function tocell(X::ktensor)
    if isequal(X.lambda,ones(size(X.lambda)))
        return X.fmat;
    end
    N=ndims(X)
    d = (X.lambda').^(1/N)
    fmat=MatrixCell(undef,N)
    for n = 1:N
        fmat[n] = X.fmat[n].*repeat(d,size(X.fmat[n],1),1)
    end
    fmat
end

"""
---
    ttensor(X::ktensor)

Create ttensor out of a ktensor.
"""
function ttensor(X::ktensor)
    Xred=redistribute(X,1)
    ttensor(neye(ncomponents(X),order=ndims(X)),Xred.fmat)
end

#Mode-n multiplication of ktensor and matrix. **Documentation in tensor.jl.
#t='t' transposes matrices
function ttm(X::ktensor,M::MatrixCell,modes::AbstractVector{<:Integer},t='n')
  if t=='t'
	 [M[n]=M[n]' for n=1:length(M)]
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
ttm(X::ktensor,M::MatrixCell,modes::AbstractRange{<:Integer},t='n')=ttm(X,M,collect(modes),t)
ttm(X::ktensor,M::Matrix{<:Number},n::Integer,t='n')=ttm(X,MatrixCell([M]),[n],t)
ttm(X::ktensor,M::MatrixCell,t::Char)=ttm(X,M,1:length(M),t)
ttm(X::ktensor,M::MatrixCell)=ttm(X,M,1:length(M))
function ttm(X::ktensor,M::MatrixCell,n::Integer,t='n')
	if n>0
		ttm(X,M[n],n,t)
	else
		modes=setdiff(1:ndims(X),-n)
		ttm(X,M,modes,t)
	end
end
#If array of matrices isn't defined as MatrixCell, but as M=[M1,M2,...,Mn]:
ttm(X::ktensor,M::Array{Matrix{T},1},modes::AbstractVector{<:Integer},t='n') where {T<:Number}=ttm(X,MatrixCell(M),modes,t)
ttm(X::ktensor,M::Array{Matrix{T},1},modes::AbstractRange{<:Integer},t='n') where {T<:Number}=ttm(X,MatrixCell(M),modes,t)
ttm(X::ktensor,M::Array{Matrix{T},1},t::Char) where {T<:Number}=ttm(X,MatrixCell(M),t)
ttm(X::ktensor,M::Array{Matrix{T},1}) where {T<:Number}=ttm(X,MatrixCell(M))
ttm(X::ktensor,M::Array{Matrix{T},1},n::Integer,t='n') where {T<:Number}=ttm(X,MatrixCell(M),n,t)

#Mode-n multiplication ktensor and vector. **Documentation in tensor.jl.
#t='t' transposes matrices
function ttv(X::ktensor,V::VectorCell,modes::AbstractVector{<:Integer})
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
ttv(X::ktensor,v::AbstractVector{<:Number},n::Integer)=ttv(X,VectorCell([v]),[n])
ttv(X::ktensor,V::VectorCell,modes::AbstractRange{<:Integer})=ttv(X,V,collect(modes))
ttv(X::ktensor,V::VectorCell)=ttv(X,V,1:length(V))
function ttv(X::ktensor,V::VectorCell,n::Integer)
	if n>0
		ttm(X,V[n],n)
	else
		modes=setdiff(1:ndims(X),-n)
		ttm(X,A,modes)
	end
end
#If array of vectors isn't defined as VectorCell, but as V=[v1,v2,...,vn]:
ttv(X::ktensor,V::Array{Vector{T},1},modes::AbstractVector{<:Integer}) where {T<:Number}=ttv(X,VectorCell(V),modes)
ttv(X::ktensor,V::Array{Vector{T},1},modes::AbstractRange{<:Integer}) where {T<:Number}=ttv(X,VectorCell(V),modes)
ttv(X::ktensor,V::Array{Vector{T},1}) where {T<:Number}=ttv(X,VectorCell(V))
ttv(X::ktensor,V::Array{Vector{T},1},n::Integer) where {T<:Number}=ttv(X,VectorCell(V),n)

#**Documentation in ttensor.jl.
uminus(X::ktensor)=mtimes(-1,X)
-(X::ktensor)=uminus(X)

#Frobenius norm of a ktensor. **Documentation in Base.
function norm(X::ktensor)
  nrm=X.lambda*X.lambda'
  for n=1:ndims(X)
    nrm=nrm.*(X.fmat[n]'*X.fmat[n])
  end
  sqrt.(abs.(sum(nrm[:])))
end
