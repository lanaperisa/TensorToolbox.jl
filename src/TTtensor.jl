#using TensorToolbox

export TTtensor, randTTtensor, innerprod, ewprod, full, minus, mtimes, mtimes!, ndims, plus
export reorth, reorth!, size, TTrank, TTsvd, TTtv, norm

mutable struct TTtensor
  cores::CoreCell
  lorth::Bool
  rorth::Bool
  function TTtensor(cores::CoreCell,lorth::Bool,rorth::Bool)
      N=length(cores)
      #[@assert(ndims(cores[n])==3,"Core tensors must be tensors of order 3!") for n=1:N]
      @assert(size(cores[1],1)==size(cores[N],3)==1,"Core tensors are of incorrect sizes.")
      [@assert(size(cores[n],3)==size(cores[n+1],1),"Dimension mismatch.") for n=1:N-1]
      for G in cores[1:end-1]
          Gn=tenmat(G,row=1:2)
          if norm(eye(size(Gn,2))- transpose(Gn)*Gn)>(size(Gn,1)^2)*eps()
              lorth=false
              break
          end
      end
      for G in cores[2:end]
          Gn=tenmat(G,1)
          if norm(eye(size(Gn,1))- Gn*transpose(Gn))>(size(Gn,2)^2)*eps()
              rorth=false
              break
          end
      end
      new(cores,lorth,rorth)
  end
end
TTtensor(cores::CoreCell)=TTtensor(cores,true,true)
TTtensor(cores::Array{Array{T,3},1}) where {T<:Number}=TTtensor(CoreCell(cores),true,true)
TTtensor(cores::Array{AbstractArray{T,N},1}) where {T<:Number,N}=TTtensor(CoreCell(cores),true,true)

"""
    randTTtensor(I::Vector,R::Vector)
    randTTtensor(I::Integer,R::Integer,N::Integer)

Create random TTtensor of size I and TT-rank rank R, or of order N and size I × ⋯ × I and TT-rank (R,...,R).
"""
function randTTtensor(I::Vector{<:Integer},R::Vector{<:Integer})
  N=length(I)
  @assert(length(R)==N-1,"Dimension mismatch.")
  cores=CoreCell(undef,N) #create radnom cores
  Rfull=[1;R;1]
  [cores[n]=randn(Rfull[n],I[n],Rfull[n+1]) for n=1:N]
  TTtensor(cores)
end
randTTtensor(I::Integer,R::Integer,N::Integer)=randTTtensor(repmat([I],N),repmat([R],N-1));
#For input defined as tuples or nx1 matrices - ranttensor(([I,I,I],[R,R,R]))
function randTTtensor(arg...)
  randTTtensor([arg[1]...],[arg[2]...])
end

"""
---------------------------------------------------
    contract(T::TTtensor,X::Array,start_mode,modes)

Contract modes 1:ndims(X-1) of full tensor X to TTtensor T cores[start_mode:start_mode+modes-1].
Output: TTtensor.
"""
function contract(T::TTtensor,X::Array,start_mode=1,modes=0)
    if modes==0
        modes=ndims(X)-1
    end
    @assert(ndims(X)==modes+1,"Dimension mismatch.")
    for n=start_mode:start_mode+modes-1
        if n==start_mode
            X=contract(T.cores[n],2,X,1)
            X=permutedims(X,[collect(2:ndims(X))...;1])
        else
            X=contract(T.cores[n],collect(1:2),X,collect(1:2))
        end
    end
    X=permutedims(X,[3,2,1])
    N=ndims(T)-modes+1
    G=CoreCell(undef,N)
    for n=1:start_mode-1
        G[n]=T.cores[n]
    end
    G[start_mode]=X
    for n=start_mode+1:N
         G[n]=T.cores[modes+n-1]
    end
    TTtensor(G)
end
contract(X::Array,T::TTtensor,start_mode=1,modes=0)=contract(T,X,start_mode,modes)
function contract(X::CoreCell,squeeze=true)
    N=length(X)
    if N==1
        return X[1]
    end
    @assert(any([size(X[n])[end]==size(X[n+1])[1] for n=1:N-1]),"Dimensions mismatch.")
    Xcontr=contract(X[1],X[2])
    for n=3:N
        Xcontr=contract(Xcontr,X[n])
    end
    squeeze == true ? dropdims(Xcontr) : Xcontr
end
#
function contract(X1::CoreCell,X2::CoreCell)
    N=length(X2)
    @assert(length(X1)==N,"Dimensions mismatch.")
    @assert(size(X1)[1:end-1]==size(X2)[1:end-1],"Dimensions mismatch.")
    Xcontr=transpose(dropdims(contract(X1[1],2,X2[1],2)))
    for n=2:N
        Xcontr=contract(X1[n],1,Xcontr,2,[3,1,2])
        Xcontr=contract(X2[n],[1,2],Xcontr,[1,2])
    end
    Xcontr
end

function innerprod(X1::TTtensor,X2::TTtensor)
    Isz=size(X1)
    @assert(Isz==size(X2),"Dimension mismatch.")
    v=kron(X1.cores[1][:,1,:],X2.cores[1][:,1,:])
    for i=2:Isz[1]
        v+=kron(X1.cores[1][:,i,:],X2.cores[1][:,i,:])
    end
    for n=2:ndims(X1)
        w=VectorCell(undef,Isz[n])
        for i=1:Isz[n]
            w[i]=krontv(transpose(X1.cores[n][:,i,:]),transpose(X2.cores[n][:,i,:]),vec(v))
        end
        v=vec(sum(w))
    end
    v[1]
end

function innerprod(X1::CoreCell,X2::CoreCell)
    Isz=[size(X,2) for X in X1]
    @assert(Isz==[size(X,2) for X in X2],"Dimension mismatch.")
    v=kron(X1[1][:,1,:],X2[1][:,1,:])
    for i=2:Isz[1]
        v+=kron(X1[1][:,i,:],X2[1][:,i,:])
    end
    for n=2:ndims(X1)
        w=VectorCell(undef,Isz[n])
        for i=1:Isz[n]
            w[i]=krontv(transpose(X1[n][:,i,:]),transpose(X2[n][:,i,:]),vec(v))
        end
        v=vec(sum(w))
    end
    v[1]
end

function ewprod(X1::TTtensor,X2::TTtensor)
	Isz=size(X1)
	@assert(Isz == size(X2),"Dimension mismatch.")
	N=ndims(X1)
  	cores=CoreCell(undef,N) #initilize cores
	R=[1,TTrank(X1).*TTrank(X2)...,1]
  	for n=1:N
    	cores[n]=zeros(R[n],Isz[n],R[n+1])
    	for i=1:Isz[n]
        	cores[n][:,i,:]=kron(X1.cores[n][:,i,:],X2.cores[n][:,i,:]);
    	end
	end
  	TTtensor(cores)
end

function full(X::TTtensor)
    contract(X.cores,true)
end

"""
    minus(X::TTtensor,Y::TTtensor)

Subtraction of two TTtensors. Same as: X-Y.
"""
function minus(X1::TTtensor,X2::TTtensor)
	1*X1+(-1)*X2
end
-(X1::TTtensor,X2::TTtensor)=minus(X1,X2)

"""
    mtimes(a,X)

Scalar times TTtensor. Same as: a*X.
"""
function mtimes(α::Number,X::TTtensor)
    # G=deepcopy(X.cores)
    # G[1]=α*G[1]
	# TTtensor(G)
    TTtensor(mtimes(α,X.cores))
end
*(α::Number,X::TTtensor)=mtimes(α,X)
*(X::TTtensor,α::Number)=*(α,X)
function mtimes!(α::Number,X::TTtensor)
    X.cores[1]=α*X.cores[1]
end
function mtimes(α::Number,G::CoreCell)
    Gcopy=deepcopy(G)
    Gcopy[1]=α*G[1]
	Gcopy
end
*(α::T,G::CoreCell) where {T<:Number} = mtimes(α,G)
*(G::CoreCell,α::T) where {T<:Number} = *(α,G)
function mtimes!(α::Number,G::CoreCell)
    G[1]=α*G[1]
end

function ndims(X::TTtensor)
	length(X.cores)
end


#Frobenius norm of a TTtensor. **Documentation in Base.
function norm(X::TTtensor,orth="")
    if X.lorth==true
        sqrt(contract(X.cores[end],[1,2,3],X.cores[end],[1,2,3])[1])
    elseif X.rorth==true
        sqrt(contract(X.cores[1],[1,2,3],X.cores[1],[1,2,3])[1])
    else
        if orth=="reorth"
            Xorth=reorth(X)
            sqrt(abs.(innerprod(Xorth,Xorth)))
        else
            sqrt(abs.(innerprod(X,X)))
        end
    end
end

"""
    plus(X::TTtensor,Y::TTtensor)

Addition of two TTtensors. Same as: X+Y.
"""
function plus(X1::TTtensor,X2::TTtensor)
	@assert(size(X1) == size(X2),"Dimension mismatch.")
    TTtensor(plus(X1.cores,X2.cores))
end
+(X1::TTtensor,X2::TTtensor)=plus(X1,X2)

function plus(X1::CoreCell,X2::CoreCell)
    N=length(X1)
    @assert(N==length(X2),"Dimension mismatch.")
    Xres=CoreCell(undef,N)
    sz=size(X1[1]).+(0,0,size(X2[1])[3])
    Xres[1]=reshape([tenmat(X1[1],1) tenmat(X2[1],1)],sz)
    sz=size(X1[N]).+(size(X2[N])[1],0,0)
    Xres[N]=reshape([X1[N][:,:,1]; X2[N][:,:,1]],sz)
    for n=2:N-1
        sz1=size(X1[n])
        sz2=size(X2[n])
        sz=sz1.+(sz2[1],0,sz2[3])
        C=zeros(sz[1],sz[2]*sz[3])
        C[1:sz1[1],1:sz1[2]*sz1[3]]=tenmat(X1[n],1)
        C[sz1[1]+1:end,sz1[2]*sz1[3]+1:end]=tenmat(X2[n],1);
        Xres[n]=reshape(C,sz)
    end
	Xres
end
+(X1::CoreCell,X2::CoreCell)=plus(X1,X2)

function reorth(X::TTtensor,direction="left",full="no")
    TTtensor(reorth(X.cores,direction,full))
end
function reorth(X::CoreCell,direction="left",full="no")
    G=deepcopy(X)
    N=length(X)
    if direction=="left"
        for n=1:N-1
            Q,Rt=qr(tenmat(G[n],row=1:2))
            Q=Matrix(Q)
            sz=(size(G[n])[1:2]...,size(Q)[2])
            G[n]=reshape(Q,sz)
            G[n+1]=ttm(G[n+1],Rt,1)
        end
        if full=="full"
            # Q,Rt=qr(tenmat(G[N],row=1:2))
            # Q=Matrix(Q)
            # sz=(size(G[N])[1:2]...,size(Q)[2])
            # G[N]=reshape(Q,sz)
            G[N]=G[N]/norm(G[N])
        end
    elseif direction == "right"
        for n=N:-1:2
            Q,Rt=qr(transpose(tenmat(G[n],1)))
            Q=Matrix(Q)
			Rnorm=norm(Rt)
            sz=(size(Q)[2],size(G[n])[2:3]...)
            G[n]=reshape(Matrix(transpose(Q)),sz)
            G[n-1]=ttm(G[n-1],Rt,3)
        end
        if full=="full"
            # Q,Rt=qr(transpose(tenmat(G[1],1)))
            # Q=Matrix(Q)
            # sz=(size(Q)[2],size(G[1])[2:3]...)
            # G[1]=reshape(Matrix(transpose(Q)),sz)
            G[1]=G[1]/norm(G[1])
        end
    else
        error("Invalid direction. Should be either \"left\" or \"right\".")
    end
    G
end

function reorth!(X::TTtensor,direction="left",full="no")
    N=ndims(X)
    if direction=="left"
        for n=1:N-1
            Q,Rt=qr(tenmat(X.cores[n],row=1:2))
            Q=Matrix(Q)
            sz=(size(X.cores[n])[1:2]...,size(Q)[2])
            X.cores[n]=reshape(Q,sz)
            X.cores[n+1]=ttm(X.cores[n+1],Rt,1)
        end
        if full=="full"
            # Q,Rt=Matrix(qr(tenmat(X.cores[N],row=1:2)).Q)
            # sz=(size(X.cores[N])[1:2]...,size(Q)[2])
            # X.cores[N]=reshape(Q,sz)
            X.cores[N]=X.cores[N]/norm(X.cores[N])
        end
        X.lorth=true
    elseif direction == "right"
        for n=N:-1:2
            Q,Rt=qr(transpose(tenmat(X.cores[n],1)))
            Q=Matrix(Q)
            sz=(size(Q)[2],size(X.cores[n])[2:3]...)
            X.cores[n]=reshape(Matrix(transpose(Q)),sz)
            X.cores[n-1]=ttm(X.cores[n-1],Rt,3)
        end
        if full=="full"
            # Q=Matrix(qr(transpose(tenmat(X.cores[1],1))).Q)
            # sz=(size(Q)[2],size(X.cores[1])[2:3]...)
            # X.cores[1]=reshape(Matrix(transpose(Q)),sz)
            X.cores[1]=X.cores[1]/norm(X.cores[1])
        end
        X.rorth=true
    else
        error("Invalid direction. Should be either \"left\" or \"right\".")
    end
end
function reorth!(X::CoreCell,direction="left",full="no")
    N=length(X)
    if direction=="left"
        for n=1:N-1
            Q,Rt=qr(tenmat(X[n],row=1:2))
            Q=Matrix(Q)
            sz=(size(X[n])[1:2]...,size(Q)[2])
            X[n]=reshape(Q,sz)
            X[n+1]=ttm(X[n+1],Rt,1)
        end
        if full=="full"
            # Q=Matrix(qr(tenmat(X[N],row=1:2)).Q)
            # sz=(size(X[N])[1:2]...,size(Q)[2])
            # X[N]=reshape(Q,sz)
            X[N]=X[N]/norm(X[N])
        end
    elseif direction == "right"
        for n=N:-1:2
            Q,Rt=qr(tenmat(X[n],1)')
            Q=Matrix(Q)
            sz=(size(Q)[2],size(X[n])[2:3]...)
            X[n]=reshape(Q',sz)
            X[n-1]=ttm(X[n-1],Rt,3)
        end
        if full=="full"
            # Q=Matrix(qr(tenmat(X[1],1)').Q)
            # sz=(size(Q)[2],size(X[1])[2:3]...)
            # X[1]=reshape(Q',sz)
            X[1]=X[1]/norm(X[1])
        end
    else
        error("Invalid direction. Should be either \"left\" or \"right\".")
    end
end

#Size of a TTtensor. **Documentation in Base.
function size(X::TTtensor)
    tuple([size(X.cores[n],2) for n=1:ndims(X)]...)
end
function size(X::TTtensor,mode::Int)
    size(X.cores[mode],2)
end

"""
    TTrank(X::TTtensor,full=false)
    TTrank(X::CoreCell,full=false)

Representational TT-rank of X. If cores of X are of size R{n-1} x In x Rn, returns (R1,...,R{n-1}) for full=false or (1,R1,....,Rn) for full=true."""
function TTrank(X::TTtensor,full=false)
    TTrank(X.cores,full)
end
function TTrank(X::CoreCell,full=false)
    rnk=[size(X[n])[3] for n=1:length(X)-1]
    if full==true
        rnk=[1,rnk...,1]
    end
    tuple(rnk...)
end

"""
    TTsvd(X;tol=1e-8,reqrank=[])
    TTsvd(X::TTtensor;tol=1e-8,reqrank=[])

Decomposes full tensor X into its TT format. If input is already a TTtensor, recompresses it."""
function TTsvd(X::AbstractArray{<:Number,N};tol=1e-8,reqrank=[],tol_rel=0) where N
    Isz=size(X)
    if length(reqrank)>0
        @assert(length(reqrank)==N-1,"Dimension mismatch. For a N-dim tensor, reqrank has to be a vector of length N-1.")
        rnk=1
    else
        reqrank=[Isz...]
        rnk=0
		tol=tol/sqrt(N-1)
    end
    G=CoreCell(undef,N)
    Xtmp=copy(X)
    for n=1:N-1
        n==1 ?  R=1 : R=reqrank[n-1]
        m=R*Isz[n]
        Xtmp=reshape(Xtmp,m,:)
        U,S,V=svd(Xtmp)
        if rnk==1
            U=U[:,1:reqrank[n]]
            S=S[1:reqrank[n]]
            V=V[:,1:reqrank[n]]
        else
            if tol_rel != 0
                tol=tol_rel*S[1]
            end
            #K=findall(x-> x>tol ? true : false,S)
			Sum=0
			k=0
			while sqrt(Sum)<=tol && length(S)-k>0
				Sum=Sum+S[end-k]^2
				k+=1
			end
			K=length(S)-k+1
            U=U[:,1:K]
            S=S[1:K]
            V=V[:,1:K]
            reqrank[n]=length(S)
        end
        G[n]=reshape(U,(R,Isz[n],reqrank[n]))
        Xtmp=Diagonal(S)*transpose(V)
    end
    G[N]=reshape(Xtmp,(size(Xtmp,1),size(Xtmp,2),1))
    TTtensor(G)
end

function TTsvd(X::TTtensor;reqrank=[],tol=1e-8,tol_rel=0)
    TTtensor(TTsvd(X.cores,reqrank=reqrank,tol=tol,tol_rel=tol_rel))
end

function TTsvd(X::CoreCell;reqrank=[],tol=1e-8,tol_rel=0)
    N=length(X)
	tol=tol/sqrt(N-1)
	Isz=[size(X[n],2) for n=1:N]
 	G=reorth(X,"right")
    if length(reqrank)>0
        @assert(length(reqrank)==N-1,"Requested rank of inapropriate size.")
        if any(reqrank.<TTrank(G))==false
            @warn "Requested rank larger than the current rank."
            return X
        end
        for n=1:N-1
            U,S,V=svd(tenmat(G[n],row=1:2))
            if reqrank[n]<=size(U,2)
                U=U[:,1:reqrank[n]]
                V=V[:,1:reqrank[n]]*Diagonal(S[1:reqrank[n]])
			else
				@warn "Requested rank larger than the actual rank."
				reqrank[n]=size(U,2)
            end
            n==1 ? sz=(1,Isz[1],reqrank[1]) : sz=(reqrank[n-1],Isz[n],reqrank[n])
            G[n]=reshape(U,sz)
            G[n+1]=ttm(G[n+1],transpose(V),1)
        end
    else
        for n=1:N-1
            U,S,V=svd(tenmat(G[n],row=1:2))
            if tol_rel != 0
                tol=tol_rel*S[1]/sqrt(N-1)
            end
            Sum=0
			k=0
			while sqrt(Sum)<tol && length(S)-k>0
				Sum=Sum+S[end-k]^2
				k+=1
			end
			#K=findall(x-> x>δ ? true : false,S)
			if k>0
				K=length(S)-k+1        		
        		U=U[:,1:K]
        		V=V[:,1:K]
        		S=S[1:K]
			end
			#@show tol, norm(tenmat(G[n],row=1:2)-U*Diagonal(S)*transpose(V))
            G[n]=reshape(U,(size(G[n],1),Isz[n],:))
            G[n+1]=ttm(G[n+1],V*Diagonal(S),1,'t')
        end
    end
    G
end

"""
    TTtv(X::TTtensor,u::VectorCell,mode::Int)
    TTtv(X::TTtensor,U::AbstractMatrix{<:Number},mode::Int)
    TTtv(X::CoreCell,u::VectorCell)
    TTtv(X::CoreCell,U::AbstractMatrix{<:Number})

TT-tensor times vector. For N=ndims(X) and n=[mode,mode+1,...,N] contracts nth core
of the TT-tensor with vector u[N-n+1]. If input is a matrix, then vectors u are its columns.
Called on a subset of cores, it contracts all given cores to vectors.
"""
function TTtv(X::TTtensor,u::VectorCell,mode::Int)
    if mode==1
        TTtv(X.cores,u)
    else
        G=CoreCell(undef,mode-1)
        G[1:mode-1]=deepcopy(X.cores[1:mode-1])
        v=TTtv(X.cores[mode:end],u)
        G[mode-1]=contract(G[mode-1],v)
        mode==2 ? dropdims(G[1]) : TTtensor(G)
    end
end
function TTtv(X::TTtensor,U::AbstractMatrix{<:Number},mode::Int)
    L=size(U,2)
    u=VectorCell(undef,L)
    for l=1:L
        u[l]=U[:,l]
    end
    TTtrv(X,U,mode)
end
function TTtv(X::CoreCell,u::VectorCell)
    N=length(X)
    @assert(length(u)==N,"Dimension mismatch.")
    G=copy(X[N])
    for n=N:-1:1
        G=contract(G,2,u[n],1)
        if n!=1
            G=contract(X[n-1],G)
        end
    end
    G
end
function TTtv(X::CoreCell,U::AbstractMatrix{<:Number})
    Ucell=MatrixCell(undef,1)
    Ucell[1]=U
    TTtrv(X,Ucell)
end
