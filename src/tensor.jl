export blockdiag, cp_als, diagt, hosvd, innerprod, krontm, matten, mkrontm, mkrontv, mrank, mttkrp, neye, nrank, nvecs
export squeeze, sthosvd, tenmat, tkron, ttm, ttt, ttv

"""
   blockdiag(X,Y)

Create block diagonal tensor where tensors X and Y are block elements. If X and Y are matrices, equal to blkdiag for sparse matrices.
"""
function blockdiag{T1<:Number,T2<:Number}(X1::Array{T1},X2::Array{T2})
  sz=tuple([size(X1)...]+[size(X2)...]...)
  Xd=zeros(sz) #initialize core tensor
  I1=indicesmat(X1,zeros([size(X1)...]))
  I2=indicesmat(X2,[size(X1)...])
  idx1=indicesmat2vec(I1,size(Xd))
  idx2=indicesmat2vec(I2,size(Xd))
  Xd[idx1]=vec(X1) #first diagonal block
  Xd[idx2]=vec(X2) #second diagonal block
  Xd
end

"""
    cp_als(X,R;init,tol,maxit,dimorder)

Compute a CP decomposition with R components of a tensor X .

## Arguments:
- `init` ∈ {MatrixCell,"rand","nvecs","eigs"}. Initial guess for factor matrices. If init="nvecs" (same as "eigs") initialize matrices with function nvecs.
- `tol`: Tolerance. Defualt: 1e-4.
- `maxit`: Maximal number of iterations. Default: 1000.
- `dimorder': Order of dimensions. Default: 1:ndims(A).
"""
function cp_als{T<:Number}(X::Array{T},R::Integer;init="rand",tol=1e-4,maxit=1000,dimorder=[])
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
    for n in dimorder[2:end]
      if !isempty(fmat[n])
        G[:,:,n]=fmat[n]'*fmat[n]
      end
    end
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
            nr_res=sqrt.(abs.(nr^2+vecnorm(K)^2-2*innerprod(X,K)))
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

"""
    diagt(v[,dims])

Create a diagonal tensor for a given vector of diagonal elements. Generalization of diagm.
"""
function diagt{T<:Number}(v::Vector{T})
    N=length(v)
    dims=repmat([N],N)
    I=zeros(tuple(dims...))
    diagt(I,v,dims)
end
function diagt{T<:Number,D<:Integer}(v::Vector{T},dims::Vector{D})
    I=zeros(tuple(dims...))
    diagt(I,v,dims)
end
@generated function diagt{T1<:Number,T2<:Number,D<:Integer,N}(I::Array{T1,N},v::Vector{T2},dims::Vector{D})
  quote
      @assert(length(v)==minimum(dims),"Dimension mismatch.")
      n=1
      @nloops $N i I begin
	 	    ind = [(@ntuple $N i)...]
        if length(unique(ind))==1
            I[ind...]=v[n]
            n+=1;
        end
	  end
	  I
  end
end

"""
    hosvd(X; <keyword arguments>)

Higher-order singular value decomposition.

## Arguments:
- `X`: Tensor (multidimensional array) or ttensor.
- `method` ∈ {"lapack","lanczos","randsvd"} Method for SVD. Default: "lapack".
- `reqrank::Vector`: Requested mutlilinear rank. Optional.
- `eps_abs::Number/Vector`: Drop singular values (of mode-n matricization) below eps_abs. Default: 1e-8.
- `eps_rel::Number/Vector`: Drop singular values (of mode-n matricization) below eps_rel*sigma_1. Optional.
- `p::Integer`: Oversampling parameter. Defaul p=10.
"""
function hosvd{T<:Number,N}(X::Array{T,N};method="lapack",reqrank=[],eps_abs=[],eps_rel=[],p=10)
	fmat=MatrixCell(N)

  reqrank=check_vector_input(reqrank,N,0)
  eps_abs=check_vector_input(eps_abs,N,1e-8)
  eps_rel=check_vector_input(eps_rel,N,0)

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
      fmat[n]=fmat[n][:,1:reqrank[n]]
    else
      eps_rel[n] != 0 ? tol=eps_rel[n]*S[1] : tol=eps_abs[n]
      I=find(x-> x>tol ? true : false,S)
      fmat[n]=fmat[n][:,I]
    end
	end
  ttensor(ttm(X,fmat,'t'),fmat)
end

"""
   innerprod(X,Y)
   innerprod(X::ttensor,Y::ttensor)
   innerprod(X::ktensor,Y::ktensor)

Inner product of two tensors.
"""
function innerprod{T1<:Number,T2<:Number}(X1::Array{T1},X2::Array{T2})
	@assert(size(X1) == size(X2),"Dimension mismatch")
	sum(X1.*conj(X2))
end

"""
   krontm(X,Y,M[,modes,t='n'])

Kronecker product of two tensors times matrix (n-mode product): (X ⊗ Y) x₁ M₁ x₂ M₂ x₃ ⋯ xₙ Mₙ.

## Arguments:
- `X::Array`
- `Y::Array`
- `M::Matrix/MatrixCell`
- `modes::Integer/Vector` : Modes for multiplication. Default: 1:length(M).
- `t='t'`: Transpose matrices from M.
"""
function krontm{T1<:Number,T2<:Number,D<:Integer,N}(X1::Array{T1,N},X2::Array{T2,N},M::MatrixCell,modes::Vector{D},t='n')
  if t=='t'
    M=vec(M')
	end
	@assert(length(modes)<=length(M)<=N,"Dimension mismatch.")
  I=[size(X1)...].*[size(X2)...]
  R=copy(I)
  if length(modes) != length(M)
    M=M[modes] #discard matrices not needed for multiplication
  end
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
#  end
  X
end
krontm{T1<:Number,T2<:Number,T3<:Number}(X1::Array{T1},X2::Array{T2},M::Matrix{T3},n::Integer,t='n')=krontm(X1,X2,[M],[n],t)
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


"""
    matten(A,n,dims)
    matten(A,R,C,dims)

Fold matrix A into a tensor of dimension dims by mode n or by row and column vectors R and C.
"""
function matten{T<:Number,D<:Integer}(A::Matrix{T},n::Integer,dims::Vector{D})
	@assert(dims[n]==size(A,1),"Dimensions mismatch")
	m = setdiff(1:length(dims), n)
	@assert prod(dims[m])==size(A,2)
	X = reshape(A,[dims[n];dims[m]]...)
	permutedims(X,invperm([n;m]))
end

function matten{T<:Number,D<:Integer}(A::Matrix{T},R::Vector{D},C::Vector{D},dims::Vector{D})
	@assert(prod(dims[R])==size(A,1) && prod(dims[C])==size(A,2),"Dimensions mismatch")
	X = reshape(A,[dims[R];dims[C]]...)
	permutedims(X,invperm([R;C]))
end

"""
    mkrontv(X,Y,v,n,t='n')

Matricized Kronecker product of tensors X and Y times vector v (n-mode multiplication): (X ⊗ Y)ₙv.
If t='t', transpose matricized Kronecker product.
If v is a matrix, multiply column by column.
"""
#for t='n' calculates tenmat(tkron(X1,X2),n)*v
#for t='t' calculates tenmat(tkron(X1,X2),n)'*v
function mkrontv{T1<:Number,T2<:Number,T3<:Number,N}(X1::Array{T1,N},X2::Array{T2,N},v::Vector{T3},n::Integer,t='n')
  I1=size(X1)
  I2=size(X2)
  kronsize=tuple(([I1...].*[I2...])...);
  ind=setdiff(1:N,n) #all indices but n
  X1n=tenmat(X1,n);
  X2n=tenmat(X2,n);
  perfect_shuffle=Int[ [2*k-1 for k=1:N-1]; [2*k for k=1:N-1] ]
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

"""
    mrank(X[,tol])
    mrank(X::ttensor[,tol])

Multilinear rank of a tensor with optionally given tolerance.
"""
function mrank{T<:Number,N}(X::Array{T,N})
   ntuple(n->nrank(X,n),N)
end
function mrank{T<:Number,N}(X::Array{T,N},tol::Number)
   ntuple(n->nrank(X,n,tol),N)
end

"""
    mttkrp(X,M,n)
    mttkrp(X::ttensor,M,n)
    mttkrp(X::ktensor,M,n)

Mode-n matricized tensor X times Khatri-Rao product of matrices from M (except nth) in reverse order.
"""
function mttkrp{T<:Number,N}(X::Array{T,N},M::MatrixCell,n::Integer)
  @assert(N-1<=length(M)<=N,"Wrong number of matrices")
  if length(M)==N-1  #if nth matrix not defined
    push!(M,M[end])
    [M[m]=M[m-1] for m=N-1:-1:n+1]
  end
  modes=setdiff(1:N,n)
  I=[size(X)...]
  K=size(M[modes[1]],2)
  @assert(!any(map(Bool,[size(M[m],2)-K for m in modes])),"Matrices must have the same number of columns.")
  @assert(!any(map(Bool,[size(M[m],1)-I[m] for m in modes])),"Matrices are of wrong size.")
  Xn=tenmat(X,n)
  Xn*khatrirao(reverse(M[modes]))
end
mttkrp{T1<:Number,T2<:Number,N}(X::Array{T1,N},M::Array{Matrix{T2}},n::Integer)=mttkrp(X,MatrixCell(M),n)

"""
    neye(dims)

Identity tensor of a given dimension. Generalization of eye.
"""
function neye{D<:Integer}(dims::Vector{D})
    I=zeros(tuple(dims...))
    neye(I,dims)
end
function neye(dims::Integer;order=0)
  @assert(order>0,"Wrong input.")
  neye(repmat([dims],order,1)[:])
end
function neye(d1,d2...)
  dims=[d1]
  for d in d2
		push!(dims,d)
	end
	neye(dims)
end
@generated function neye{T<:Number,D<:Integer,N}(I::Array{T,N},dims::Vector{D})
  quote
  	@nloops $N i I begin
	 	ind = [(@ntuple $N i)...]
    if length(unique(ind))==1
         I[ind...]=1
    end
	end
	I
  end
end

"""
    nrank(X,n[,tol])
    nrank(X::ttensor,n[,tol])

Rank of the n-mode matricization of a tensor X (n-rank).
"""
function nrank{T<:Number}(X::Array{T},n::Integer)
  rank(tenmat(X,n))
end
function nrank{T<:Number}(X::Array{T},n::Integer,tol::Number)
  rank(tenmat(X,n),tol)
end

"""
    nvecs(X,n,r=0;flipsign=false,svds=false)
    nvecs(X::ttensor,n,r=0;flipsign=false)
    nvecs(X::ktensor,n,r=0;flipsign=false)

Computes the r leading singular vectors of mode-n matricization of a tensor X.
Works with XₙXₙᵀ.

## Arguments:
- `flipsign=true`: Make the largest magnitude element be positive.
- `svds=true`: Use svds on Xₙ rather than eigs on XₙXₙᵀ.
"""
function nvecs{T<:Number}(X::Array{T},n::Integer,r=0;flipsign=false,svds=false)
  if r==0
    r=size(X,n)
  end
  Xn=tenmat(X,n)
  if svds
    #U=svds(Xn,nsv=r)[1][:U]
    #if size(U,2)<r
      U=svdfact(Xn)[:U][1:r]
    #end
  else
    G=Symmetric(Xn*Xn') #Gramian
    #U=eigs(G,nev=r,which=:LM)[2] #has bugs!
    #if size(U,2)<r
       U=eigfact(G)[:vectors][:,end:-1:end-r+1]
    #end
  end
  if flipsign
      maxind = findmax(abs.(U),1)[2]
      for i = 1:r
          ind=ind2sub(size(U),maxind[i])
          if U[ind...] < 0
             U[:,ind[2]] = U[:,ind[2]] * -1
          end
      end
  end
  U
end

#Squeeze all singleton dimensions. **Documentation in Base.jl.
function squeeze{T<:Number}(A::Array{T})
  sz=size(A)
  sdims=find(sz.==1) #singleton dimensions
  squeeze(A,tuple(sdims...))
end

"""
    sthosvd(X,reqrank,p)

Sequentially truncated HOSVD of a tensor X of predifined rank and processing order p.
"""
function sthosvd{T<:Number,D<:Integer,N}(X::Array{T,N},reqrank::Vector{D},p::Vector{D})
	@assert(N==length(reqrank)==length(p),"Dimensions mismatch")
	I=[size(X)...]
	fmat=MatrixCell(N)
	for n=1:N
		fmat[n]=zeros(I[n],reqrank[n])
	end
	for n in p
		Xn=tenmat(X,n)
		U,S,V=svd(Xn)
		fmat[n]=U[:,1:reqrank[n]]
		Xn=diagm(S[1:reqrank[n]])*V'[1:reqrank[n],:]
		I[n]=reqrank[n]
		X=matten(Xn,n,I)
	end
	ttensor(X,fmat)
end

"""
    tenmat(X,n)
    tenmat(X,R=[],C=[])
    tenmat(X::ttensor,n)
    tenmat(X::ktensor,n)

Mode-n matricization of a tensor or matricization by row and column vectors R and C.
"""
function tenmat{T<:Number,N}(X::Array{T,N},n::Integer)
	@assert(n<=ndims(X),"Mode exceedes number of dimensions")
	I=size(X)
	m=setdiff(1:N,n)
	reshape(permutedims(X,[n;m]),I[n],prod(I[m]))
end

function tenmat{T<:Number,N}(X::Array{T,N};R=[],C=[])
    @assert(R!=[] || C!=[],"Al least one of R and C needs to be specified.")
    if R!=[] && C!=[]
        @assert(sort([R;C])==collect(1:N),"Incorrect mode partitioning.")
    elseif R==[]
        @assert(!(false in [c in collect(1:N) for c in C]),"Incorrect modes.")
        if isa(C,Integer)
            C=[C]
        end
        R=collect(1:N);deleteat!(R,sort(C));
    else
        @assert(!(false in [r in collect(1:N) for r in R]),"Incorrect modes.")
        if isa(R,Integer)
            R=[R]
        end
        C=collect(1:N);deleteat!(C,sort(R));
    end
	  I=size(X);
    J=prod(I[R]);K=prod(I[C]);
	reshape(permutedims(X,[R;C]),J,K)
end

"""
    tkron(X,Y)

Kronecker product of two tensors X and Y. Direct generalization of Kronecker product of matrices.
"""
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

"""
    ttm(X,M[,modes,t='n'])
    ttm(X::ttensor,M[,modes,t='n'])

Tensor times matrix (n-mode product):  X x₁ M₁ x₂ M₂ x₃ ⋯ xₙ Mₙ
Default modes: 1:length(M).
If t='t', transpose matrices from M.
"""
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

"""
    ttv(X,Y)

Outer product of two tensors.
"""
function ttt{T1<:Number,T2<:Number}(X1::Array{T1},X2::Array{T2})
  sz=tuple([[size(X1)...];[size(X2)...]]...);
  Xprod=vec(X1)*vec(X2)';
  reshape(Xprod,sz)
end

"""
    ttv(X,V[,modes])

Tensor times vectors (n-mode product):  X x₁ V₁ x₂ V₂ x₃ ⋯ xₙ Vₙ.
Default modes: 1:length(M).
"""
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
