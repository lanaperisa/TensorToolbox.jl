using TensorToolbox

export tttensor, full, ndims, size, ttrank, ttsvd

import Base:ndims,size

type tttensor
  cten::TensorCell
  function tttensor(cten)
      N=length(cten)
      #@assert(1<ndims(cten[1])<=3 && 1<ndims(cten[N])<=3,"Core tensors must be tensors of order 3!")
      [@assert(ndims(cten[n])==3,"Core tensors must be tensors of order 3!") for n=1:N]
      @assert(size(cten[1],1)==size(cten[N],3)==1,"Core tensors are of incorrect sizes.")
      [@assert(size(cten[n],3)==size(cten[n+1],1),"Dimension mismatch.") for n=1:N-1]
      new(cten)
  end
end

function full(X::tttensor)
    squeeze(conprod(X.cten))
end

function ndims(X::tttensor)
	length(X.cten)
end

function size(X::tttensor)
    tuple([size(X.cten[n],2) for n=1:ndims(X)]...)
end

function ttrank(X::tttensor)
    tuple([size(X.cten[n],3) for n=1:ndims(X)]...)
end

function ttsvd{T<:Number,N}(X::Array{T,N},tol=1e-5)
    I=size(X)
    R=[I...]
    G=TensorCell(N)
    ep=tol/sqrt(N-1)
    δ=ep*vecnorm(X)
    C=X
    for n=1:N-1
        n==1 ?  r=1 : r=R[n-1]
        m=r*I[n]
        C=reshape(C,(m,Int(length(C)/m)))
        U,S,V=svd(C)
        R[n]=length(S)
        G[n]=reshape(U,(r,I[n],R[n]))
        C=diagm(S)*V'
    end
    G[N]=reshape(C,(size(C,1),size(C,2),1))
    tttensor(G)
end
