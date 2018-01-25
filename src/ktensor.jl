#Tensors in Kruskal (CP) format + functions

export ktensor, display, full, minus, mtimes, mttkrp, plus, ndims, normalize, normalize!, rank, size, tenmat, vecnorm, uminus

import Base: display, ndims, normalize, normalize!, rank, vecnorm

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

#@doc """Displays a ktensor.""" ->
function display{T<:Number}(X::ktensor{T},name="ktensor")
    println("$name.lambda: ")
    show(STDOUT, "text/plain", X.lambda)
    for n=1:ndims(X)
        println("\n\n$name.fmat[$n]:")
        show(STDOUT, "text/plain", X.fmat[n])
    end
end

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
  R=rank(X1)
  inpr=0
  for r=1:R
    inpr+=X1.lambda[r]*ttv(X2,[X1.fmat[n][:,r] for n=1:ndims(X1)])
  end
  inpr
end
innerprod{T<:Number}(X1,X2::ktensor{T})=innerprod(X2,X1)

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

#@doc """ Number of modes of a ktensor. """ ->
function ndims{T<:Number}(X::ktensor{T})
  length(X.fmat)
end

@doc """Normalize columns of factor matrices of ktensor, absorbing the excess weight into lambda."""->
function normalize{T<:Number}(X::ktensor{T})
    R=rank(X);N=ndims(X);
    lambda=zeros(X.lambda)
    fmat=MatrixCell(N);
    [fmat[n]=zeros(X.fmat[n]) for n=1:N]
    for r=1:R
        for n=1:N
            nr=norm(X.fmat[n][:,r])
            lambda[r]=X.lambda[r]*nr;
            if nr>0
                fmat[n][:,r]=X.fmat[n][:,r]/nr;
            end
        end
    end
    ktensor(lambda,fmat);
end
function normalize!{T<:Number}(X::ktensor{T})
    R=rank(X);N=ndims(X);
    for r=1:R
        for n=1:N
            nr=norm(X.fmat[n][:,r])
            X.lambda[r]=X.lambda[r]*nr;
            if nr>0
                X.fmat[n][:,r]=X.fmat[n][:,r]/nr;
            end
        end
    end
    X
end

function plus{T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})
	@assert(size(X1) == size(X2),"Dimension mismatch.")
  lambda=[X1.lambda; X2.lambda]
	fmat=Matrix[[X1.fmat[n] X2.fmat[n]] for n=1:ndims(X1)] #concatenate factor matrices
	ktensor(lambda,fmat)
end
+{T1<:Number,T2<:Number}(X1::ktensor{T1},X2::ktensor{T2})=plus(X1,X2)

function rank{T<:Number}(X::ktensor{T})
  length(X.lambda)
end


function size{T<:Number}(X::ktensor{T})
  tuple([size(X.fmat[n],1) for n=1:length(X.fmat)]...)
end
#n-th dimension of X
function size{T<:Number}(X::ktensor{T},n::Integer)
  size(X.fmat[n],1)
end

#@doc """ n-mode matricization of tensor. """ ->
tenmat{T<:Number}(X::ktensor{T},n::Integer)=tenmat(full(X),n)

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
