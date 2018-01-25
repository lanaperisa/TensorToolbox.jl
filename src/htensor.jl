#Tensors in Hierarchical Tucker format + functions

export htensor, trtensor, htdecomp
export display, full

import Base: display, full

@doc """Hierarchical Tucker tensor."""->
type htensor#{T<:Number}
	tree::dimtree
	trten::Array{Array,1}
  frames::MatrixCell
  isorth::Bool
	function htensor(tree::dimtree,trten::Array{Array,1},frames::MatrixCell,isorth::Bool)# where T<:Number
    N=length(tree.leaves); #order of tensor = number of leaves in a tree
    i=size(tree.mat,1)-N; #number of internal nodes;
    @assert(length(frames)==N,"Dimension mismatch.")
    @assert(length(trten)==i,"Dimension mismatch.")
		for U in frames
			if norm(U'*U-eye(size(U,2)))>(size(U,1)^2)*eps()
				isorth=false
			end
		end
		new(tree,trten,frames,isorth)
	end
end
#htensor(tree::dimtree,trten::Array{Array,1},frames::MatrixCell,isorth::Bool)=htensor(tree,trten,frames,isorth)
htensor(tree::dimtree,trten::Array{Array,1},frames::MatrixCell)=htensor(tree,trten,frames,true)
htensor(tree::dimtree,trten::Array{Array,1},mat::Matrix)=htensor(tree,trten,collect(mat),true)
@doc """ Hierarchical Tucker tensor is defined by a dimensional tree, its transfer tensor and frames (matrices). """ ->
function htensor(tree::dimtree,trten::Array{Array,1},mat...)
  frames=MatrixCell(0)
  for M in mat
			push!(frames,M)
	end
	htensor(tree,trten,frames,true)
end
#If array of matrices isn't defined as MatrixCell, but as M=[M1,M2,...,Mn]:
htensor{T<:Number}(tree::dimtree,trten::Array{Array,1},frames::Array{Matrix{T}},isorth::Bool)=htensor(tree,trten,MatrixCell(frames),isorth)
htensor{T<:Number}(tree::dimtree,trten::Array{Array,1},frames::Array{Matrix{T}})=htensor(tree,trten,MatrixCell(frames),true)

@doc """Creates transfer tensor for a given tensor, node t and its left and right children tl and tr."""->
function trtensor{T<:Number,N}(X::Array{T,N};t=collect(1:N),tl=collect(1:ceil(Int,N/2)),tr=[])
    if isa(tl,Number)
        tl=[tl];
    end
    if isa(tr,Number)
        tr=[tr];
    end
    @assert(!(length(tl)==0 && length(tr)==0),"Left or right child needs to be specified.")
    if length(tl)==0
        tl=t[1:length(t)-length(tr)]
    elseif length(tr)==0
        tr=t[length(tl)+1:end]
    end
    if t==collect(1:N)
        Ut=vec(X);
    else
        Xt=tenmat(X,R=t);
        Ut=svdfact(Xt)[:U];
    end
    Xl=tenmat(X,R=tl);
    Xr=tenmat(X,R=tr);
    Ul=svdfact(Xl)[:U];
    Ur=svdfact(Xr)[:U];
    #kron(Ur',Ul')*Ut
    krontv(Ur',Ul',Ut)
end

@doc """Decomposes tensor into a htensor for a given tree. If tree not specified it uses balanced tree."""->
function htdecomp{T<:Number,N}(X::Array{T,N},tree::dimtree)
    @assert(N==length(tree.leaves),"Dimension mismatch.");
    M=length(tree.internal_nodes);
    U=MatrixCell(N);
    [U[n]=svdfact(tenmat(X,n))[:U] for n=1:N];
    B=TensorCell(M);
    t,tl,tr=structure(tree);
    for m=1:M
        B[m]=trtensor(X,t=t[m],tl=tl[m],tr=tr[m]);
    end
    htensor(tree,B,U)
end
htdecomp{T<:Number,N}(X::Array{T,N})=htdecomp(X,balanced_dimtree(X))



#@doc """Displays a htensor.""" ->
function display(X::htensor,name="htensor")
    println("$name.tree: ")
    show(STDOUT, "text/plain", X.tree.mat)
    for n=1:length(X.trten)
        println("\n\n$name.trten[$n]:")
        show(STDOUT, "text/plain", X.trten[n])
    end
    for n=1:length(X.frames)
        println("\n\n$name.frames[$n]:")
        show(STDOUT, "text/plain", X.frames[n])
    end
end

#@doc """Creates full tensor out of htensor."""->
function full(H::htensor)
    B=H.trten;
    U=H.frames;
    T=H.tree;
    dims=[size(H.frames[n],1) for n=1:length(H.frames)]
    t,tl,tr=structure(T);
    M=length(T.internal_nodes);
    V=MatrixCell(M);
    n=M;
    for m=M:-1:1
        if length(tr[m])==1
            Ur=U[tr[m]...]
        else
            Ur=V[n]
            n-=1;
        end
        if length(tl[m])==1
            Ul=U[tl[m]...]
        else
            Ul=V[n];
            n-=1;
        end
        if m==1
             #V[m]=(kron(Ur,Ul)*B[m])[:,:];
            V[m]=(krontv(Ur,Ul,B[m]))[:,:];
        else
             #V[m]=kron(Ur,Ul)*B[m];
            V[m]=krontv(Ur,Ul,B[m]);
        end
    end
    reshape(V[1],dims...)
end
