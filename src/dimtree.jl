export dimtree, number_of_nodes, parent, children, structure, left_child_length, count_leaves, balanced_dimtree

import Base.parent

@doc """Dimension tree."""->
type dimtree
    mat::Matrix{Int}
    leaves::Vector{Int}
    internal_nodes::Vector{Int}
    function dimtree(mat::Matrix{Int},leaves::Vector{Int},internal_nodes::Vector{Int})
        @assert(size(mat,2)==2,"Incorrecty defined tree.")
        @assert(!any([(mat[i,1]==0 && mat[i,2]!=0) || (mat[i,1]!=0 && mat[i,2]==0) for i=1:size(mat,1)]),"Incorrecty defined tree.")
        new(mat,leaves,internal_nodes)
    end
end
function dimtree(mat::Matrix{Int})
    leaves=findn(mat[:,1].==0)
    internal_nodes=findn(mat[:,1].!=0)
    dimtree(mat,leaves,internal_nodes)
end

function number_of_nodes(tree::dimtree)
    size(tree.mat,1)
end

@doc """Vectors of parents for each node of a given dimtree."""->
function parent(tree::dimtree)
    nmbr_nodes=number_of_nodes(tree)
    parent=zeros(Int,nmbr_nodes) #parent[node]
    parent[1]=0
    for node=2:nmbr_nodes
        parent[node]=ind2sub(tree.mat,find(x->x==node,tree.mat))[1][1];
    end
    parent
end
function parent(tree::dimtree,node::Integer)
    parent(tree)[node]
end

@doc """Vector of children for each node of a given dimtree."""->
function children(tree::dimtree)
    nmbr_nodes=number_of_nodes(tree)
    children=VectorCell(nmbr_nodes)
    for node=1:nmbr_nodes
        children[node]=tree.mat[node,:]
    end
    children
end
function children(tree::dimtree,node::Integer)
    children(tree)[node]
end

@doc """For each internal node of a given dimtree, returns its representation t, its left child and right child representations tl and tr."""->
function structure(tree::dimtree)
  N=length(tree.leaves)
  M=length(tree.internal_nodes)
  t=VectorCell(M)
  tl=VectorCell(M)
  tr=VectorCell(M)
  t[1]=collect(1:N)
  i=2;
  for m=1:M
    l=left_child_length(tree,tree.internal_nodes[m])
    tl[m]=collect(t[m][1]:t[m][1]+l-1)
    tr[m]=collect(t[m][1]+l:t[m][1]+length(t[m])-1)
    if length(tl[m])>1
        t[i]=tl[m]
        i+=1
    end
     if length(tr[m])>1
        t[i]=tr[m]
        i+=1
    end
  end
  t,tl,tr
end

@doc """For a given dimtree and node, returns the length of the left child of the node."""->
function left_child_length(tree::dimtree,node::Int)
    if node in tree.leaves
        return 0
    else
        return count_leaves(tree,tree.mat[node,1])
    end
end
left_child_length(tree::dimtree)=left_child_length(tree,1)

@doc """For a given dimtree and node, returns the number of leaves under the node."""->
function count_leaves(tree::dimtree,node::Int,l::Int)
    @assert(0<node<=number_of_nodes(tree),"Input exceeds number of nodes.")
    if node in tree.leaves
        l+=1
        return l
    end
    for c in children(tree,node)
        l=count_leaves(tree,c,l)
    end
    l
end
count_leaves(tree::dimtree,node::Int)=count_leaves(tree,node,0)

@doc """Creates balanced dimtree for a given tensor."""->
function balanced_dimtree{T<:Number,N}(X::Array{T,N})
    nmbr_nodes=2*N-1;
    Tmat=zeros(Int,nmbr_nodes,2);
    Tmat[1,:]=[2,3]
    node_count=4;
    parent=zeros(Int,nmbr_nodes);
    parent[1]=0;parent[2]=1;parent[3]=1;
    node_length=zeros(Int,nmbr_nodes);
    node_length[1]=N;
    for node=2:nmbr_nodes
        if iseven(node)
            node_length[node]=ceil(Int,node_length[parent[node]]/2);
            if node_length[node] > 1
                Tmat[node,:]=[node_count;node_count+1];
                parent[node_count]=node;
                parent[node_count+1]=node;
                node_count+=2;
            end
        else
            node_length[node]=floor(Int,node_length[parent[node]]/2);
            if node_length[node] > 1
                Tmat[node,:]=[node_count;node_count+1];
                parent[node_count]=node;
                parent[node_count+1]=node;
                node_count+=2;
            end
        end
    end
    dimtree(Tmat)
end
