#Tensors in Hierarchical Tucker format + functions

export htensor, randhtensor, display, full, hrank, htdecomp, innerprod, isequal, minus, mtimes, ndims, plus, reorth, reorth!
export size, squeeze, trten2mat, trten2ten, trtensor, ttm, ttv, vecnorm

"""
    htensor(tree,trten,fmat)
    htensor(tree,trten,U1,U2,...)

Hierarchical Tucker tensor.

## Arguments:
- `tree::dimtree`: Dimension tree.
- `trten::TensorCell`: Transfer tensors. One for each internal node of the tree.
- `fmat::MatrixCell`: Frame matrices. One for each leaf of the tree.
For htensor X, X.isorth=true if frame matrices are othonormal.
"""
type htensor#{T<:Number}
	tree::dimtree
	trten::TensorCell
  fmat::MatrixCell
  isorth::Bool
	function htensor(tree::dimtree,trten::TensorCell,fmat::MatrixCell,isorth::Bool)# where T<:Number
    @assert(length(fmat)==length(tree.leaves),"Dimension mismatch.") #order of a tensor
    @assert(length(trten)==length(tree.internal_nodes),"Dimension mismatch.")
		for U in fmat
			if norm(U'*U-eye(size(U,2)))>(size(U,1)^2)*eps()
				isorth=false
			end
		end
		new(tree,trten,fmat,isorth)
	end
end
#htensor(tree::dimtree,trten::TensorCell,fmat::MatrixCell,isorth::Bool)=htensor(tree,trten,fmat,isorth)
htensor(tree::dimtree,trten::TensorCell,fmat::MatrixCell)=htensor(tree,trten,fmat,true)
htensor(tree::dimtree,trten::TensorCell,mat::Matrix)=htensor(tree,trten,collect(mat),true)
function htensor(tree::dimtree,trten::TensorCell,mat...)
  fmat=MatrixCell(0)
  for M in mat
			push!(fmat,M)
	end
	htensor(tree,trten,fmat,true)
end
#If array of matrices isn't defined as MatrixCell, but as M=[M1,M2,...,Mn]:
htensor{T<:Number}(tree::dimtree,trten::TensorCell,fmat::Array{Matrix{T}},isorth::Bool)=htensor(tree,trten,MatrixCell(fmat),isorth)
htensor{T<:Number}(tree::dimtree,trten::TensorCell,fmat::Array{Matrix{T}})=htensor(tree,trten,MatrixCell(fmat),true)

"""
    randhtensor(I::Vector,R::Vector)
    randhtensor(I::Vector,R::Vector,T::dimtree)

Create random htensor of size I, hierarchical rank R and balanced tree T. If T not specified use balanced tree.
"""
function randhtensor{D<:Integer}(sz::Vector{D},rnk::Vector{D},T::dimtree)
  N=length(sz)
  @assert(length(rnk)==non(T),"Size and rank dimension mismatch.")
  I=length(T.internal_nodes) #number of internal nodes
  cmps=TensorCell(N+I)
  [cmps[T.leaves[n]]=rand(sz[n],rnk[I+n]) for n=1:N];
  for n in T.internal_nodes[end:-1:1]
    c=children(T,n)
    lsz=size(cmps[c[1]],2)
    rsz=size(cmps[c[2]],2)
    cmps[n]=rand(lsz,rsz,rnk[n])
  end
  htensor(T,cmps[T.internal_nodes],MatrixCell(cmps[T.leaves]))
end
function randhtensor{D<:Integer}(sz::Vector{D},rnk::Vector{D})
    T=create_dimtree(length(sz))
    randhtensor(sz,rnk,T)
end

#Display a htensor. **Documentation in ttensor.jl
function display(X::htensor,name="htensor")
  println("Hierarchical Tucker tensor of size ",size(X),":\n")
  println("$name.tree: ")
  display(X.tree)
  #show(STDOUT, "text/plain", display(X.tree))
  for n=1:length(X.trten)
    println("\n\n$name.trten[$n]:")
    show(STDOUT, "text/plain", X.trten[n])
  end
  for n=1:length(X.fmat)
    println("\n\n$name.fmat[$n]:")
    show(STDOUT, "text/plain", X.fmat[n])
  end
end

#Makes full tensor out of a ktensor. **Documentation in ttensor.jl.
function full(X::htensor)
  t,tl,tr=structure(X.tree)
  I=length(X.tree.internal_nodes)
  V=MatrixCell(I)
  j=I
  for i=I:-1:1
    if length(tr[i])==1
      Ur=X.fmat[tr[i]...]
    else
      Ur=V[j]
      j-=1;
    end
    if length(tl[i])==1
      Ul=X.fmat[tl[i]...]
    else
      Ul=V[j];
      j-=1;
    end
     V[i]=kron(Ur,Ul)*trten2mat(X.trten[i]);
     #V[i]=krontv(Ur,Ul,B[i]);
  end
  reshape(V[1],size(X))
end

"""
    fmat(X)

Hierarchical ranks of a htensor.
"""
function hrank(X::htensor)
  order=[X.tree.internal_nodes;X.tree.leaves]
  r1=[size(B,3) for B in X.trten]
  r2=[size(U,2) for U in X.fmat]
  [r1;r2][invperm(order)]
end

"""
    htdecomp(X[,tree])

Decompose full tensor X into a htensor for a given tree. If tree not specified use balanced tree.
"""
function htdecomp{T<:Number,N}(X::Array{T,N},tree::dimtree;method="lapack",maxrank=10,atol=1e-8,rtol=0)
  @assert(N==length(tree.leaves),"Dimension mismatch.")
  I=length(tree.internal_nodes)
  U=MatrixCell(N)
  #[U[n]=svdfact(tenmat(X,n))[:U] for n=1:N]
	for n=1:N
    Xn=float(tenmat(X,n))
    U[n]=colspace(Xn,method=method,reqrank=maxrank,atol=atol,rtol=rtol)
    #println("size U[$n] = ",size(U[n]))
	end
  B=TensorCell(I)
  t,tl,tr=structure(tree);
  for i=1:I
     B[i]=trtensor(X,t=t[i],tl=tl[i],tr=tr[i],method=method,maxrank=maxrank,atol=atol,rtol=rtol);
  end
  htensor(tree,B,U)
end
htdecomp{T<:Number,N}(X::Array{T,N})=htdecomp(X,create_dimtree(X))

#Inner product of two htensors. **Documentation in tensor.jl.
function innerprod(X1::htensor,X2::htensor)
	@assert(size(X1) == size(X2),"Dimension mismatch.")
  @assert(X1.tree == X2.tree, "Dimension trees must be equal.")
  N=ndims(X1)
  I=length(X1.tree.internal_nodes) #number of internal nodes
  M=MatrixCell(N+I)
  [M[X1.tree.leaves[n]]=X1.fmat[n]'*X2.fmat[n] for n=1:N]
  for n in X1.tree.internal_nodes[end:-1:1]
    c=children(X1.tree,n)
    M[n]=trten2mat(X1.trten[node2ind(X1.tree,n)])'*kron(M[c[2]],M[c[1]])*trten2mat(X2.trten[node2ind(X2.tree,n)])
  end
  M[1][:]
end

#True if htensors have equal components. **Documentation in ttensor.jl.
function isequal(X1::htensor,X2::htensor)
  if (X1.trten == X2.trten) && (X1.fmat == X2.fmat)
    true
  else
    false
  end
end
==(X1::htensor,X2::htensor)=isequal(X1,X2)

#Subtraction of two htensors. **Documentation in ttensor.jl
function minus(X1::htensor,X2::htensor)
  X1+(-1)*X2
end
-(X1::htensor,X2::htensor)=minus(X1,X2)

#Scalar times htensor. **Documentation in ttensor.jl
function mtimes(α::Number,X::htensor)
  B=deepcopy(X.trten)
  B[1]=α*B[1]
	htensor(X.tree,B,X.fmat)
end
*{T<:Number}(α::T,X::htensor)=mtimes(α,X)
*{T<:Number}(X::htensor,α::T)=*(α,X)

#Number of modes of a htensor. **Documentation in Base.
function ndims(X::htensor)
	length(X.fmat)
end

#Addition of two htensors. **Documentation in ttensor.jl
function plus(X1::htensor,X2::htensor)
	@assert(X1.tree == X2.tree,"Input mismatch.")
	fmat=Matrix[[X1.fmat[n] X2.fmat[n]] for n=1:ndims(X1)] #concatenate fmat
  trten=TensorCell(length(X1.tree.internal_nodes))
  tmp=[squeeze(X1.trten[1],3) zeros(size(X1.trten[1],1),size(X2.trten[1],2)); zeros(size(X1.trten[1],2),size(X2.trten[1],1)) squeeze(X2.trten[1],3)]
  trten[1]=reshape(tmp,size(tmp,1),size(tmp,2),1)
  for i=2:length(X1.tree.internal_nodes)
    sz=tuple([size(X1.trten[i])...]+[size(X2.trten[i])...]...)
    trten[i]=zeros(sz) #initialize core tensor
	  I1=indicesmat(X1.trten[i],zeros([size(X1.trten[i])...]))
	  I2=indicesmat(X2.trten[i],[size(X1.trten[i])...])
    idx1=indicesmat2vec(I1,size(trten[i]))
	  idx2=indicesmat2vec(I2,size(trten[i]))
	  trten[i][idx1]=vec(X1.trten[i]) #first diagonal block
	  trten[i][idx2]=vec(X2.trten[i]) #second diagonal block
  end
  htensor(X1.tree,trten,fmat)
end
+(X1::htensor,X2::htensor)=plus(X1,X2)

#Orthogonalize factor matrices of a tensor. **Documentation in ttensor.jl.
function reorth(X::htensor)
	if X.isorth
		return X
  end
  N=ndims(X)
  I=length(X.tree.internal_nodes)
  trten=TensorCell(I)
  fmat=MatrixCell(N)
	R=MatrixCell(N+I)
  n=1;
	for U in X.fmat
		fmat[n],R[X.tree.leaves[n]]=qr(U)
    n+=1
	end
  for n in X.tree.internal_nodes[end:-1:1]
    c=children(X.tree,n)
    Rl=R[c[1]]
    Rr=R[c[2]]
    ind=node2ind(X.tree,n)
    trten[ind]=kron(Rr,Rl)*trten2mat(X.trten[ind])
    if n!=1
      trten[ind],R[n]=qr(trten[ind])
    end
    trten[ind]=trten2ten(trten[ind],size(Rl,2),size(Rr,2))
  end
	htensor(X.tree,trten,fmat)
end

#Orthogonalize factor matrices of a tensor. Rewrite tensor. **Documentation in ttensor.jl.
function reorth!(X::htensor)
  if X.isorth==true
    return X
  end
  N=ndims(X)
  I=length(X.tree.internal_nodes)
  R=MatrixCell(N+I)
  n=1;
  for U in X.fmat
    X.fmat[n],R[X.tree.leaves[n]]=qr(U)
    n+=1
  end
  for n in X.tree.internal_nodes[end:-1:1]
    c=children(X.tree,n)
    Rl=R[c[1]]
    Rr=R[c[2]]
    ind=node2ind(X.tree,n)
    X.trten[ind]=kron(Rr,Rl)*trten2mat(X.trten[ind])
    if n!=1
      X.trten[ind],R[n]=qr(X.trten[ind])
    end
    X.trten[ind]=trten2ten(X.trten[ind],size(Rl,2),size(Rr,2))
  end
  X.isorth=true
  X
end

function Base.show(io::IO,X::htensor)
    display(X)
end

#Size of a htensor. **Documentation in Base.
function size(X::htensor)
	tuple([size(X.fmat[n],1) for n=1:ndims(X)]...)
end
#Size of n-th mode of a htensor.
function size(X::htensor,n::Integer)
  size(X)[n]
end

"""
    squeeze(X)

Remove singleton dimensions from htensor.
"""
function squeeze(X::htensor)
  B=deepcopy(X.trten)
  U=deepcopy(X.fmat)
  I=copy(X.tree.internal_nodes)
  sz=[size(X)...]
  sdims=find(sz.==1) #singleton dimensions
  for d in sdims
    node=X.tree.leaves[d]
    sibling_node=sibling(X.tree,node)
    parent_node=parent(X.tree,node)
    prnt=I[parent_node]
    sblng=node2ind(X.tree,sibling_node)
    if is_left(X.tree,node)
      tmp=ttm(B[prnt],U[d],1)
      tmp=reshape(tmp,size(tmp,2),size(tmp,3))
      lft=1
    else
      tmp=ttm(B[prnt],U[d],2)
      tmp=reshape(tmp,size(tmp,1),size(tmp,3))
      lft=0
    end
    if is_leaf(X.tree,sibling_node)
      if lft==1
        U[d]=U[d+1]*tmp
        deleteat!(U,d+1)
      else
        U[d-1]=U[d-1]*tmp
        deleteat!(U,d)
      end
      deleteat!(B,prnt)
      deleteat!(I,prnt)
    else
      B[prnt]=ttm(B[sblng],tmp',3)
      deleteat!(B,sblng)
      deleteat!(I,sblng)
      deleteat!(U,d)
    end
  end
  T=dimtree(ndims(X)-length(sdims),internal_nodes=I)
  htensor(T,B,U)
end

"""
    trtensor

Create transfer tensor for a given tensor, node representation t and its left and right children representations tl and tr.
"""
function trtensor{T<:Number,N}(X::Array{T,N};t=collect(1:N),tl=collect(1:ceil(Int,N/2)),tr=[],method="lapack",maxrank=10,atol=1e-8,rtol=0)
  if isa(tl,Number)
    tl=[tl]
  end
  if isa(tr,Number)
    tr=[tr]
  end
  @assert(!(length(tl)==0 && length(tr)==0),"Left or right child needs to be specified.")
  if length(tl)==0
    tl=t[1:length(t)-length(tr)]
  elseif length(tr)==0
    tr=t[length(tl)+1:end]
  end
  if t==collect(1:N)
    Ut=vec(X)
  else
    Xt=tenmat(X,R=t)
    #Ut=svdfact(Xt)[:U]
    Ut=colspace(Xt,method=method,reqrank=maxrank,atol=atol,rtol=rtol)
  end
  Xl=tenmat(X,R=tl)
  Xr=tenmat(X,R=tr)
  #Ul=svdfact(Xl)[:U]
  Ul=colspace(Xl,method=method,reqrank=maxrank,atol=atol,rtol=rtol)
  #Ur=svdfact(Xr)[:U]
  Ur=colspace(Xr,method=method,reqrank=maxrank,atol=atol,rtol=rtol)
  B=kron(Ur',Ul')*Ut
  #B=krontv(Ur',Ul',Ut)
  #reshape(B,(size(Ur,2),size(Ul,2),size(Ut,2)))
  trten2ten(B,size(Ul,2),size(Ur,2))


end

"""
    trten2mat(B::Array)
    trten2mat(B::TensorCell)

Transfer tensor to matrix. If transfer tensor is given as a tensor of order 3 and size `(r1,r2,r3)`, reshape it into a matrix of size `(r1r2,r3)`.
"""
function trten2mat{T<:Number}(B::Array{T,3})
  (r1,r2,r3)=size(B)
  reshape(B,r1*r2,r3)
end
function trten2mat(B::TensorCell)
  I=length(B)
  @assert([ndims(B[i])==3 for i=1:I],"Transfer tensors should be tensors of order 3.")
  Bmat=MatrixCell(I)
  for i=1:I
    (r1,r2,r3)=size(B[i])
    Bmat[i]=reshape(B[i],r1*r2,r3)
  end
  Bmat
end

"""
    trten2ten(B::Matrix,r1::Integer,r2::Integer)
    trten2ten(B::MatrixCell,r1::Vector,r2::Vector)

Transfer tensor to tensor. If transfer tensor is given as a matrix, reshape it into a tensor of order 3 and size r1×r2×r3, where `r3=size(B,2)`.
"""
function trten2ten{T<:Number}(B::Matrix{T},r1::Integer,r2::Integer)
  @assert(r1*r2==size(B,1),"Dimension mismatch.")
  r3=size(B,2)
  reshape(B,r1,r2,r3)
end
trten2ten{T<:Number}(B::Vector{T},r1::Integer,r2::Integer)=trten2ten(B[:,:],r1,r2)
function trten2ten(B::MatrixCell,r1::Vector{Int},r2::Vector{Int})
  I=length(B)
  @assert([r1[i]*r2[i]==size(B[i],1) for i=1:I],"Dimension mismatch.")
  Bten=TensorCell(I)
  for i=1:I
    r3=size(B[i],2)
    Bten[i]=reshape(B[i],r1[i],r2[i],r3)
  end
  Bten
end

#htensor times matrix (n-mode product). **Documentation in tensor.jl.
#t='t' transposes matrices
function ttm{D<:Integer}(X::htensor,M::MatrixCell,modes::Vector{D},t='n')
  if t=='t'
	 M=vec(M')
	end
	@assert(length(modes)<=length(M),"Too few matrices")
	@assert(length(M)<=ndims(X),"Too many matrices")
  if length(modes)<length(M)
    M=M[modes]
  end
  U=copy(X.fmat)
  for n=1:length(modes)
    U[modes[n]]=M[n]*X.fmat[modes[n]]
  end
    htensor(X.tree,X.trten,U)
end
ttm{D<:Integer}(X::htensor,M::MatrixCell,modes::Range{D},t='n')=ttm(X,M,collect(modes),t)
ttm{T<:Number}(X::htensor,M::Matrix{T},n::Integer,t='n')=ttm(X,[M],[n],t)
ttm(X::htensor,M::MatrixCell,t::Char)=ttm(X,M,1:length(M),t)
ttm(X::htensor,M::MatrixCell)=ttm(X,M,1:length(M))
function ttm(X::htensor,M::MatrixCell,n::Integer,t='n')
	if n>0
		ttm(X,M[n],n,t)
	else
		modes=setdiff(1:ndims(X),-n)
		ttm(X,M,modes,t)
	end
end
#If array of matrices isn't defined as MatrixCell, but as M=[M1,M2,...,Mn]:
ttm{T<:Number,D<:Integer}(X::htensor,M::Array{Matrix{T}},modes::Vector{D},t='n')=ttm(X,MatrixCell(M),modes,t)
ttm{T<:Number,D<:Integer}(X::htensor,M::Array{Matrix{T}},modes::Range{D},t='n')=ttm(X,MatrixCell(M),modes,t)
ttm{T<:Number}(X::htensor,M::Array{Matrix{T}},t::Char)=ttm(X,MatrixCell(M),t)
ttm{T<:Number}(X::htensor,M::Array{Matrix{T}})=ttm(X,MatrixCell(M))
ttm{T<:Number}(X::htensor,M::Array{Matrix{T}},n::Integer,t='n')=ttm(X,MatrixCell(M),n,t)

#htensor times vector (n-mode product). **Documentation in tensor.jl.
#t='t' transposes matrices
function ttv{D<:Integer}(X::htensor,V::VectorCell,modes::Vector{D})
  N=ndims(X)
  sz=[size(X)...]
  remmodes=setdiff(1:N,modes)
  U=deepcopy(X.fmat)
  if length(modes) < length(V)
    V=V[modes]
  end
  for n=1:length(modes)
    U[modes[n]]=V[n]'*X.fmat[modes[n]]
    deleteat!(sz,modes[n])
  end
  #reshape(htensor(X.tree,X.trten,U),tuple(sz))
  htensor(X.tree,X.trten,U)
end
ttv{T<:Number}(X::htensor,v::Vector{T},n::Integer)=ttv(X,Vector[v],[n])
ttv{D<:Integer}(X::htensor,V::VectorCell,modes::Range{D})=ttv(X,V,collect(modes))
ttv(X::htensor,V::VectorCell)=ttv(X,V,1:length(V))
function ttv(X::htensor,V::VectorCell,n::Integer)
	if n>0
		ttm(X,V[n],n)
	else
		modes=setdiff(1:ndims(X),-n)
		ttm(X,A,modes)
	end
end
#If array of vectors isn't defined as VectorCell, but as V=[v1,v2,...,vn]:
ttv{T<:Number,D<:Integer}(X::htensor,V::Array{Vector{T}},modes::Vector{D})=ttv(X,VectorCell(V),modes)
ttv{T<:Number,D<:Integer}(X::htensor,V::Array{Vector{T}},modes::Range{D})=ttv(X,VectorCell(V),modes)
ttv{T<:Number}(X::htensor,V::Array{Vector{T}})=ttv(X,VectorCell(V))
ttv{T<:Number}(X::htensor,V::Array{Vector{T}},n::Integer)=ttv(X,VectorCell(V),n)

#Frobenius norm of a htensor. **Documentation in Base.
function vecnorm(X::htensor;orth=true)
  if orth
    reorth!(X)
    vecnorm(X.trten[1])
  else
    sqrt(innerprod(X,X))
  end
end
