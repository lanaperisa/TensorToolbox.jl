#Tensors in Hierarchical Tucker format + functions

export htensor, display, full, htdecomp, ndims, size, trtensor
export trten2mat, trten2ten

"""
    htensor(tree,trten,frames)
    htensor(tree,trten,U1,U2,...)

Hierarchical Tucker tensor.

## Arguments:
- `tree::dimtree`: Dimension tree.
- `trten::TensorCell`: Transfer tensors. One for each internal node of the tree.
- `frames::MatrixCell`: Frame matrices. One for each leaf of the tree.
For htensor X, X.isorth=true if frame matrices are othonormal.
"""
type htensor#{T<:Number}
	tree::dimtree
	trten::TensorCell
  frames::MatrixCell
  isorth::Bool
	function htensor(tree::dimtree,trten::TensorCell,frames::MatrixCell,isorth::Bool)# where T<:Number
    N=length(tree.leaves) #order of tensor = number of leaves in a tree
    I=length(tree.internal_nodes) #number of internal nodes;
    @assert(length(frames)==N,"Dimension mismatch.")
    @assert(length(trten)==I,"Dimension mismatch.")
		for U in frames
			if norm(U'*U-eye(size(U,2)))>(size(U,1)^2)*eps()
				isorth=false
			end
		end
		new(tree,trten,frames,isorth)
	end
end
#htensor(tree::dimtree,trten::TensorCell,frames::MatrixCell,isorth::Bool)=htensor(tree,trten,frames,isorth)
htensor(tree::dimtree,trten::TensorCell,frames::MatrixCell)=htensor(tree,trten,frames,true)
htensor(tree::dimtree,trten::TensorCell,mat::Matrix)=htensor(tree,trten,collect(mat),true)
function htensor(tree::dimtree,trten::TensorCell,mat...)
  frames=MatrixCell(0)
  for M in mat
			push!(frames,M)
	end
	htensor(tree,trten,frames,true)
end
#If array of matrices isn't defined as MatrixCell, but as M=[M1,M2,...,Mn]:
htensor{T<:Number}(tree::dimtree,trten::TensorCell,frames::Array{Matrix{T}},isorth::Bool)=htensor(tree,trten,MatrixCell(frames),isorth)
htensor{T<:Number}(tree::dimtree,trten::TensorCell,frames::Array{Matrix{T}})=htensor(tree,trten,MatrixCell(frames),true)

#Display a htensor. **Documentation in ttensor.jl
function display(X::htensor,name="htensor")
    println("Hierarchical Tucker tensor of size ",size(X),":\n")
    println("$name.tree: ")
    show(STDOUT, "text/plain", X.tree)
    for n=1:length(X.trten)
        println("\n\n$name.trten[$n]:")
        show(STDOUT, "text/plain", X.trten[n])
    end
    for n=1:length(X.frames)
        println("\n\n$name.frames[$n]:")
        show(STDOUT, "text/plain", X.frames[n])
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
      Ur=X.frames[tr[i]...]
    else
      Ur=V[j]
      j-=1;
    end
    if length(tl[i])==1
      Ul=X.frames[tl[i]...]
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
    htdecomp(X[,tree])

Decompose full tensor X into a htensor for a given tree. If tree not specified use balanced tree.
"""
function htdecomp{T<:Number,N}(X::Array{T,N},tree::dimtree)
    @assert(N==length(tree.leaves),"Dimension mismatch.");
    I=length(tree.internal_nodes);
    U=MatrixCell(N);
    [U[n]=svdfact(tenmat(X,n))[:U] for n=1:N];
    B=TensorCell(I);
    t,tl,tr=structure(tree);
    for i=1:I
       B[i]=trtensor(X,t=t[i],tl=tl[i],tr=tr[i]);
    end
    htensor(tree,B,U)
end
htdecomp{T<:Number,N}(X::Array{T,N})=htdecomp(X,create_dimtree(X))

#Number of modes of a htensor. **Documentation in Base.
function ndims(X::htensor)
	length(X.frames)
end

function Base.show(io::IO,X::htensor)
    display(X)
end

#Size of a htensor. **Documentation in Base.
function size(X::htensor)
	tuple([size(X.frames[n],1) for n=1:ndims(X)]...)
end
#Size of n-th mode of a htensor.
function size(X::htensor,n::Integer)
  size(X)[n]
end

"""
    trtensor

Create transfer tensor for a given tensor, node representation t and its left and right children representations tl and tr.
"""
function trtensor{T<:Number,N}(X::Array{T,N};t=collect(1:N),tl=collect(1:ceil(Int,N/2)),tr=[])
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
    Ut=svdfact(Xt)[:U]
  end
  Xl=tenmat(X,R=tl)
  Xr=tenmat(X,R=tr)
  Ul=svdfact(Xl)[:U]
  Ur=svdfact(Xr)[:U]
  B=kron(Ur',Ul')*Ut
  #B=krontv(Ur',Ul',Ut)
  #reshape(B,(size(Ur,2),size(Ul,2),size(Ut,2)))
  trten2ten(B,size(Ur,2),size(Ul,2))
end

"""
    trten2mat(B::Array)
    trten2mat(B::TensorCell)

Transfer tensor to matrix.
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






