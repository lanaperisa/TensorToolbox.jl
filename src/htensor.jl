#Tensors in Hierarchical Tucker format + functions
export htensor, randhtensor, cat, display, dropdims, full, hrank, htrunc, innerprod, isequal, minus, mtimes, ndims, plus, reorth, reorth!
export size, trten2mat, trten2ten, ttm, ttv, uminus, norm

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
mutable struct htensor
	tree::dimtree
	trten::TensorCell
  	fmat::MatrixCell
  	isorth::Bool
	function htensor(tree::dimtree,trten::TensorCell,fmat::MatrixCell,isorth::Bool)
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
#htensor(tree::dimtree{D1,D2},trten::TensorCell{T1},fmat::MatrixCell{T2},isorth::Bool) where {D1<:Integer,D2<:Integer,T1<:Number,T2<:Number}=htensor{D1,D2,T1,T2}(tree,trten,fmat,isorth)
htensor(tree::dimtree,trten::TensorCell,fmat::MatrixCell)=htensor(tree,trten,fmat,true)
htensor(tree::dimtree,trten::TensorCell,mat::AbstractMatrix{<:Number})=htensor(tree,trten,collect(mat),true)
# function htensor(tree::dimtree,trten::TensorCell,mat...)
#   fmat=MatrixCell(undef,0)
#   for M in mat
# 			push!(fmat,M)
# 	end
# 	htensor(tree,trten,fmat,true)
# end
#If array of matrices and tensors aren't defined as TensorCell and MatrixCell, but as M=[M1,M2,...,Mn]:
htensor(tree::dimtree,trten::Array{Array{T1,3}},fmat::Array{Matrix{T2}},isorth::Bool) where {T1<:Number,T2<:Number}=htensor(tree,TensorCell(trten),MatrixCell(fmat),isorth)
htensor(tree::dimtree,trten::Array{Array{T1,3}},fmat::Array{Matrix{T2}}) where {T1<:Number,T2<:Number}=htensor(tree,TensorCell(trten),MatrixCell(fmat),true)

"""
    randhtensor(I::Vector,R::Vector)
    randhtensor(I::Vector,R::Vector,T::dimtree)

Create random htensor of size I, hierarchical rank R and balanced tree T. If T not specified use balanced tree.
"""
function randhtensor(sz::AbstractVector{<:Integer},rnk::AbstractVector{<:Integer},T::dimtree)
  N=length(sz)
  @assert(length(rnk)==non(T),"Size and rank dimension mismatch.")
  In=length(T.internal_nodes) #number of internal nodes
  cmps=TensorCell(undef,N+In)
  [cmps[T.leaves[n]]=rand(sz[n],rnk[T.leaves[n]]) for n=1:N];
  for n in reverse(T.internal_nodes)
    c=children(T,n)
    is_leaf(T,c[1]) ? lsz=size(cmps[c[1]],2) : lsz=size(cmps[c[1]],3)
    is_leaf(T,c[2]) ? rsz=size(cmps[c[2]],2) : rsz=size(cmps[c[2]],3)
    cmps[n]=rand(lsz,rsz,rnk[n])
  end
  htensor(T,cmps[T.internal_nodes],MatrixCell(cmps[T.leaves]))
end
function randhtensor(sz::AbstractVector{<:Integer},rnk::AbstractVector{<:Integer})
    T=dimtree(length(sz))
    randhtensor(sz,rnk,T)
end
function randhtensor(sz::AbstractVector{<:Integer},T::dimtree)
  N=length(sz)
  rnk=ones(Int,non(T))
  for l in T.leaves
    rnk[l]=sz[node2ind(T,l)]
  end
  for n in reverse(T.internal_nodes)[1:end-1]
    c=children(T,n)
    ind=setdiff(1:N,node2ind(T,c))
    rnk[n]=prod(sz[ind])
  end
  randhtensor(sz,rnk)
end
function randhtensor(sz::AbstractVector{<:Integer})
  T=dimtree(length(sz))
  randhtensor(sz,T)
end

"""
    cat(X,Y,n)

Concatente htensors X and Y by mode n.
"""
function cat(X1::htensor,X2::htensor,n::Integer)
    @assert(X1.tree == X2.tree, "htensors must have equal dimension trees.")
    @assert(0<n<=ndims(X1),"Invalid concatenation mode.")
    eqm=setdiff(1:ndims(X1),n) #equal modes
    @assert(size(X1)[eqm] == size(X2)[eqm],"Dimensions mismatch.")
    Xc=deepcopy(X1)
    rnk1=hrank(X1)
    rnk2=hrank(X2)
    rnk=rnk1+rnk2
    rnk[1]=1
    for l=1:ndims(X1)
        if l!=n
            Xc.fmat[l]=[X1.fmat[l] X2.fmat[l]]
        else
            Xc.fmat[l]=blockdiag(X1.fmat[l],X2.fmat[l])
        end
    end
    for i in X1.tree.internal_nodes
        ind=node2ind(X1.tree,i)
        c=children(X1.tree,i)
        Xc.trten[ind]=zeros(rnk[c[1]],rnk[c[2]],rnk[i])
        if i!=1
            Xc.trten[ind][1:rnk1[c[1]],1:rnk1[c[2]],1:rnk1[i]]=X1.trten[ind]
            Xc.trten[ind][rnk1[c[1]]+1:end,rnk1[c[2]]+1:end,rnk1[i]+1:end]=X2.trten[ind]
        else
            Xc.trten[ind][1:rnk1[c[1]],1:rnk1[c[2]]]=X1.trten[ind]
            Xc.trten[ind][rnk1[c[1]]+1:end,rnk1[c[2]]+1:end]=X2.trten[ind]
        end
    end
    Xc
end

#Display a htensor. **Documentation in ttensor.jl
function display(X::htensor,name="htensor")
  println("Hierarchical Tucker tensor of size ",size(X),":\n")
  println("$name.tree: ")
  display(X.tree)
  #show(stdout, "text/plain", display(X.tree))
  for n=1:length(X.trten)
    println("\n\n$name.trten[$n]:")
    show(stdout, "text/plain", X.trten[n])
  end
  for n=1:length(X.fmat)
    println("\n\n$name.fmat[$n]:")
    show(stdout, "text/plain", X.fmat[n])
  end
end

#Makes full tensor out of a ktensor. **Documentation in ttensor.jl.
function full(X::htensor)
  t,tl,tr=structure(X.tree)
  In=length(X.tree.internal_nodes)
  V=MatrixCell(undef,In)
  j=In
  for i=In:-1:1
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
    htrunc(X[,tree])

Truncate full tensor X into a htensor for a given tree. If tree not specified use balanced tree.
"""
function htrunc(X::Array{<:Number,N},tree::dimtree;method="svd",maxrank=[],atol=1e-8,rtol=0) where N
  @assert(N==length(tree.leaves),"Dimension mismatch.")
  maxrank=check_vector_input(maxrank,2*N-1,10);
  In=length(tree.internal_nodes)
  U=MatrixCell(undef,N)
	for n=1:N
    Xn=float(tenmat(X,n))
    U[n]=colspace(Xn,method=method,maxrank=maxrank[tree.leaves[n]],atol=atol,rtol=rtol) #[U[n]=svdfact(tenmat(X,n))[:U][1:maxrank[tree.leaves[n]]] for n=1:N]
	end
  B=TensorCell(undef,In)
  t,tl,tr=structure(tree)
  for i in reverse(tree.internal_nodes)
    ind=node2ind(tree,i)
    if i!=1
      Xt=tenmat(X,row=t[ind])
      Ut=colspace(Xt,method=method,maxrank=maxrank[ind],atol=atol,rtol=rtol) #Ut=svdfact(Xt)[:U][1:maxrank[ind]]
    else
      Ut=vec(X)
    end
    Xl=tenmat(X,row=tl[ind])
    Xr=tenmat(X,row=tr[ind])
    Ul=colspace(Xl,method=method,maxrank=maxrank[ind],atol=atol,rtol=rtol) #Ul=svdfact(Xl)[:U][1:maxrank[ind]]
    Ur=colspace(Xr,method=method,maxrank=maxrank[ind],atol=atol,rtol=rtol) #Ur=svdfact(Xr)[:U][1:maxrank[ind]]
    #B=krontv(Ur',Ul',Ut)
    #B[ind]=trten2ten(kron(Ur',Ul')*Ut,size(Ul,2),size(Ur,2))

    B[ind]=trten2ten(krontv(copy(Ur'),copy(Ul'),Ut),size(Ul,2),size(Ur,2))
  end
  htensor(tree,B,U)
end
htrunc(X::Array{<:Number,N};method="lapack",maxrank=[],atol=1e-8,rtol=0) where N=htrunc(X,dimtree(ndims(X)),method=method,maxrank=maxrank,atol=atol,rtol=rtol)

#Inner product of two htensors. **Documentation in tensor.jl.
function innerprod(X1::htensor,X2::htensor)
	@assert(size(X1) == size(X2),"Dimension mismatch.")
  @assert(X1.tree == X2.tree, "Dimension trees must be equal.")
  N=ndims(X1)
  In=length(X1.tree.internal_nodes) #number of internal nodes
  M=MatrixCell(undef,N+In)
  [M[X1.tree.leaves[n]]=X1.fmat[n]'*X2.fmat[n] for n=1:N]
  for n in reverse(X1.tree.internal_nodes)
    c=children(X1.tree,n)
    M[n]=trten2mat(X1.trten[node2ind(X1.tree,n)])'*kron(M[c[2]],M[c[1]])*trten2mat(X2.trten[node2ind(X2.tree,n)])
  end
  M[1][1]
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
*(α::Number,X::htensor)=mtimes(α,X)
*(X::htensor,α::Number)=*(α,X)

#Number of modes of a htensor. **Documentation in Base.
function ndims(X::htensor)
	length(X.fmat)
end

#Used by dropdims
function next_single_node(T::dimtree,dims::AbstractVector{<:Integer})
    #println("dims = $dims")
    nodes=T.leaves[dims]
    #println("nodes = $nodes")
    if isempty(nodes)
        return -1
    end
    parents=parent(T,nodes)
    #println("parents = $parents")
    if allunique(parents)
       return nodes[1]
    end
    for p in parents
        inds=findall(parents.==p)
        if length(inds)>1
            return nodes[inds[1]]
        end
    end
end
next_single_node(T::dimtree,dims::Integer)=next_single_node(X,[dims])

#Addition of two htensors. **Documentation in ttensor.jl
function plus(X1::htensor,X2::htensor)
  @assert(X1.tree == X2.tree,"Input mismatch.")
  fmat=[[X1.fmat[n] X2.fmat[n]] for n=1:ndims(X1)] #concatenate fmat
  trten=TensorCell(undef,length(X1.tree.internal_nodes))
  tmp=[dropdims(X1.trten[1],dims=3) zeros(size(X1.trten[1],1),size(X2.trten[1],2)); zeros(size(X1.trten[1],2),size(X2.trten[1],1)) dropdims(X2.trten[1],dims=3)]
  trten[1]=reshape(tmp,size(tmp,1),size(tmp,2),1)
  for i=2:length(X1.tree.internal_nodes)
    trten[i]=blockdiag(X1.trten[i],X2.trten[i])
  end
  htensor(X1.tree,trten,MatrixCell(fmat))
end
+(X1::htensor,X2::htensor)=plus(X1,X2)

#Orthogonalize factor matrices of a tensor. **Documentation in ttensor.jl.
function reorth(X::htensor)
	if X.isorth
		return X
  end
  N=ndims(X)
  In=length(X.tree.internal_nodes)
  trten=TensorCell(undef,In)
  fmat=MatrixCell(undef,N)
	R=MatrixCell(undef,N+In)
  n=1;
	for U in X.fmat
		Q,R[X.tree.leaves[n]]=qr(U)
		fmat[n]=Matrix(Q)
    n+=1
	end
  for n in reverse(X.tree.internal_nodes)
    c=children(X.tree,n)
    Rl=R[c[1]]
    Rr=R[c[2]]
    ind=node2ind(X.tree,n)
    trten[ind]=kron(Rr,Rl)*trten2mat(X.trten[ind])
    if n!=1
      Q,R[n]=qr(trten[ind])
	  trten[ind]=Matrix(Q)
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
  In=length(X.tree.internal_nodes)
  R=MatrixCell(undef,N+In)
  n=1;
  for U in X.fmat
    Q,R[X.tree.leaves[n]]=qr(U)
	X.fmat[n]=Matrix(Q)
    n+=1
  end
  for n in reverse(X.tree.internal_nodes)
    c=children(X.tree,n)
    Rl=R[c[1]]
    Rr=R[c[2]]
    ind=node2ind(X.tree,n)
    X.trten[ind]=kron(Rr,Rl)*trten2mat(X.trten[ind])
    if n!=1
      Q,R[n]=qr(X.trten[ind])
	  X.trten[ind]=Matrix(Q)
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
TensorToolbox:

    dropdims(X)

Remove singleton dimensions from htensor.
"""
function dropdims(X::htensor)
  B=deepcopy(X.trten)
  U=deepcopy(X.fmat)
  T=deepcopy(X.tree)
  In=copy(X.tree.internal_nodes)
  L=copy(X.tree.leaves)
  sz=[size(X)...]
  sdims=findall(sz.==1) #singleton dimensions
  while length(findall(sz.!=-1)) >2
    node=next_single_node(T,sdims)
    if node == -1
        break
    end
    sibling_node=sibling(T,node)
    parent_node=parent(T,node)
    ind=node2ind(T,node)
    inds=node2ind(T,sibling_node)
    indp=node2ind(T,parent_node)
    if is_left(T,node)
      tmp=ttm(B[indp],U[ind],1)
      tmp=reshape(tmp,size(tmp,2),size(tmp,3))
      lft=1
    else
      tmp=ttm(B[indp],U[ind],2)
      tmp=reshape(tmp,size(tmp,1),size(tmp,3))
      lft=0
    end
    if is_leaf(T,sibling_node)
      v=setdiff(findall(L.>L[inds]),ind)
      L[v].-=2
      L[inds]=parent_node
      #deleteat!(sdims,findall(sdims.==inds))
      sz[ind]=-1
      if lft==1
        U[ind]=U[ind+1]*tmp
        deleteat!(U,ind+1)
      else
        U[ind-1]=U[ind-1]*tmp
        deleteat!(U,ind)
      end
      deleteat!(B,indp)
      deleteat!(In,indp)
    else
      sibling_subnodes=subnodes(T,sibling_node)
      len=length(nodes_on_lvl(T,lvl(T,node)))
      h=height(subtree(T,parent_node))
      hsib=height(subtree(T,sibling(T,parent_node)))
      for n in sibling_subnodes[2:end]
        if lvl(T,n)>hsib+1 && is_left(T,node)
           break
        end
        if is_leaf(T,n)
           L[node2ind(T,n)]-=len;
        else
           In[node2ind(T,n)]-=len;
        end
      end
      if h<=hsib && is_left(T,node)
        len=length(nodes_on_lvl(subtree(T,parent_node),h-1))
        for level=lvl(T,parent_node)+h-1:height(T)-1
          lvl_nodes=nodes_on_lvl(T,level)
           for n in lvl_nodes[findall((in)(subnodes(T,sibling(T,parent_node))),lvl_nodes)]
             if is_leaf(T,n)
                L[node2ind(T,n)]-=len
              else
                In[node2ind(T,n)]-=len
              end
           end
        end
      end
      B[indp]=ttm(B[inds],copy(tmp'),3)
      deleteat!(B,inds)
      deleteat!(In,inds)
      deleteat!(U,ind)
    end
    deleteat!(L,ind)
    deleteat!(sdims,findall(sdims.==ind))
    sdims[findall(sdims.>ind)].-=1
    T=dimtree(copy(L))
    #display(T)
  end
  H=htensor(T,B,U)
end

"""
    trten2mat(B::Array)
    trten2mat(B::TensorCell)

Transfer tensor to matrix. If transfer tensor is given as a tensor of order 3 and size `(r1,r2,r3)`, reshape it into a matrix of size `(r1r2,r3)`.
"""
function trten2mat(B::Array{<:Number,3})
  (r1,r2,r3)=size(B)
  reshape(B,r1*r2,r3)
end
function trten2mat(B::TensorCell)
  In=length(B)
  @assert(any([ndims(B[i])==3 for i=1:In]),"Transfer tensors should be tensors of order 3.")
  Bmat=MatrixCell(undef,In)
  for i=1:In
    (r1,r2,r3)=size(B[i])
    Bmat[i]=reshape(B[i],r1*r2,r3)
  end
  Bmat
end
trten2mat(B::Array{Vector{T}}) where T<:Number= trten2mat(TensorCell(B))

"""
    trten2ten(B::Matrix,r1::Integer,r2::Integer)
    trten2ten(B::MatrixCell,r1::Vector,r2::Vector)

Transfer tensor to tensor. If transfer tensor is given as a matrix, reshape it into a tensor of order 3 and size r1×r2×r3, where `r3=size(B,2)`.
"""
function trten2ten(B::AbstractMatrix{<:Number},r1::Integer,r2::Integer)
  @assert(r1*r2==size(B,1),"Dimension mismatch.")
  r3=size(B,2)
  reshape(B,r1,r2,r3)
end
trten2ten(B::AbstractVector{<:Number},r1::Integer,r2::Integer)=trten2ten(B[:,:],r1,r2)
function trten2ten(B::MatrixCell,r1::AbstractVector{<:Integer},r2::AbstractVector{<:Integer})
  L=length(B)
  @assert([r1[i]*r2[i]==size(B[i],1) for i=1:L],"Dimension mismatch.")
  Bten=TensorCell(undef,L)
  for i=1:L
    r3=size(B[i],2)
    Bten[i]=reshape(B[i],r1[i],r2[i],r3)
  end
  Bten
end
function trten2ten(B::AbstractMatrix{<:Number})
  reshape(B,size(B,1),size(B,2),1)
end

#htensor times matrix (n-mode product). **Documentation in tensor.jl.
#t='t' transposes matrices
function ttm(X::htensor,M::MatrixCell,modes::AbstractVector{<:Integer},t='n')
  if t=='t'
	 [M[n]=M[n]' for n=1:length(M)]
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
ttm(X::htensor,M::MatrixCell,modes::AbstractRange{<:Integer},t='n')=ttm(X,M,collect(modes),t)
ttm(X::htensor,M::AbstractMatrix{<:Number},n::Integer,t='n')=ttm(X,MatrixCell([M]),[n],t)
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
ttm(X::htensor,M::Array{Matrix{T},1},modes::AbstractVector{<:Integer},t='n') where {T<:Number}=ttm(X,MatrixCell(M),modes,t)
ttm(X::htensor,M::Array{Matrix{T},1},modes::AbstractRange{<:Integer},t='n') where {T<:Number}=ttm(X,MatrixCell(M),modes,t)
ttm(X::htensor,M::Array{Matrix{T},1},t::Char) where {T<:Number}=ttm(X,MatrixCell(M),t)
ttm(X::htensor,M::Array{Matrix{T},1}) where {T<:Number}=ttm(X,MatrixCell(M))
ttm(X::htensor,M::Array{Matrix{T},1},n::Integer,t='n') where {T<:Number}=ttm(X,MatrixCell(M),n,t)

#htensor times vector (n-mode product). **Documentation in tensor.jl.
#t='t' transposes matrices
function ttv(X::htensor,V::VectorCell,modes::AbstractVector{<:Integer})
  N=ndims(X)
  remmodes=setdiff(1:N,modes)
  U=deepcopy(X.fmat)
  if length(modes) < length(V)
    V=V[modes]
  end
  for n=1:length(modes)
    U[modes[n]]=V[n]'*X.fmat[modes[n]]
  end
  #reshape(htensor(X.tree,X.trten,U),tuple(sz))
  htensor(X.tree,X.trten,U)
end
ttv(X::htensor,v::AbstractVector{<:Number},n::Integer)=ttv(X,VectorCell([v]),[n])
ttv(X::htensor,V::VectorCell,modes::AbstractRange{<:Integer})=ttv(X,V,collect(modes))
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
ttv(X::htensor,V::Array{Vector{T}},modes::AbstractVector{<:Integer}) where {T<:Number}=ttv(X,VectorCell(V),modes)
ttv(X::htensor,V::Array{Vector{T}},modes::AbstractRange{<:Integer}) where {T<:Number}=ttv(X,VectorCell(V),modes)
ttv(X::htensor,V::Array{Vector{T}}) where {T<:Number}=ttv(X,VectorCell(V),1:length(V))
ttv(X::htensor,V::Array{Vector{T}},n::Integer) where {T<:Number}=ttv(X,VectorCell(V),n)

#**Documentation in ttensor.jl.
uminus(X::htensor)=mtimes(-1,X)
-(X::htensor)=uminus(X)

#Frobenius norm of a htensor. **Documentation in Base.
function norm(X::htensor;orth=true)
  if orth
    Y=reorth(X)
    norm(Y.trten[1])
  else
    sqrt(innerprod(X,X))
  end
end
