export dimtree, children, non, parent,  structure, left_child_length, count_leaves, balanced_dimtree
export level, height, is_leaf, subtree, is_left, is_right, sibling, nodes_on_level,set_positions, dims, show, display

import Base: parent, show

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



@doc """Vector of children for each node of a given dimtree."""->
function children(tree::dimtree)
    nmbr_nodes=non(tree)
    children=VectorCell(nmbr_nodes)
    for node=1:nmbr_nodes
        children[node]=tree.mat[node,:]
    end
    children
end
function children(tree::dimtree,node::Integer)
    children(tree)[node]
end

function is_left(tree::dimtree,node::Integer)
    ind=findn(tree.mat[:,1].==node)
    if length(ind)>0
        return true
    else
        return false
    end
end

function is_right(tree::dimtree,node::Integer)
    ind=findn(tree.mat[:,2].==node)
    if length(ind)>0
        return true
    else
        return false
    end
end

function sibling(tree::dimtree,node::Integer)
    i,j=findn(tree.mat.==node)
    if j==[1]
        tree.mat[i,2]
    else
        tree.mat[i,1]
    end
end

#Number of nodes
function non(tree::dimtree)
    size(tree.mat,1)
end


"""
    parent(tree[,node])

Vectors of parents for each node of a given dimtree.
"""
function parent(tree::dimtree)
    nmbr_nodes=non(tree)
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
    @assert(0<node<=non(tree),"Input exceeds number of nodes.")
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
function balanced_dimtree(N::Integer)
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
balanced_dimtree{T<:Number,N}(X::Array{T,N})=balanced_dimtree(N)


function level(tree::dimtree)
    nn=non(tree)
    L=zeros(Int,nn)
    for n in collect(2:nn)
        L[n]=L[parent(tree,n)]+1
    end
    L
end
function level(tree::dimtree,node::Integer)
    level(tree)[node]
end

function height(tree::dimtree)
    maximum(level(tree))+1
end

function is_leaf(tree::dimtree,node::Integer)
   node in tree.leaves ? true : false
end

#Return all nodes in the subtree of a node.
function subtree(tree::dimtree,node::Integer)
    sub_ind=[node]
    n=1
    for i=node:non(tree)
        if !is_leaf(tree,sub_ind[n])
            sub_ind=[sub_ind; children(tree,sub_ind[n])]
        end
        if n!=length(sub_ind)
            n+=1
        end
    end
    sub_ind
end


#nodes on level
function nodes_on_level(tree::dimtree,l::Integer)
    L=level(tree)
    findn(L.==l)
end


function set_positions(tree::dimtree)
    h=height(tree)
    pos_nmbr=2^h-1
    position=zeros(Int,non(tree))
    position[1]=1;
    i=2;
    maxit=100;
    for l=1:h
        nds=nodes_on_level(tree,l)
        #println("level = $l, nodes = $nds")
        if length(nds)==2^l
            for n in nds
                position[n]=i
                #println("correct length - position[$n] = $i")
                i+=1
            end
        else
            for n in nds
                if position[n]!=0
                    continue
                end
                p=parent(tree,n)
                #println("i = $i, n = $n, p = $p")
                while i!=position[p]*2 && i<maxit
                    i+=2
                    #println("i = $i, n = $n, p = $p")
                end
                #println("i = $i, n = $n, p = $p")
                position[n]=i
                #println("incorrect length - position[$n] = $i")
                i+=1
                position[n+1]=i
                #println("incorrect length - position[$(n+1)] = $i")
                i+=1
                n+=1
            end
        end
    end
    position
end

function dims(tree::dimtree,node::Integer)
    t,tl,tr=structure(tree)
    internal_ind=findin(tree.internal_nodes,node)
    if length(internal_ind)>0
        t[internal_ind]
    else
        p=parent(tree,node)
        internal_ind=findin(tree.internal_nodes,p)
        #println("p = $p")
        if is_left(tree,node)
            tl[internal_ind]
        else
            tr[internal_ind]
        end
    end
end

function Base.show(T::dimtree)
    display(T)
end

function display(T::dimtree)
    t,tl,tr=structure(T)
    h=height(T)
    blank_len=h
    initial_blank_len=4*blank_len
    [print(" ") for i=1:initial_blank_len]; initial_blank_len-=2;
    print(t[1]) #internal node 1
    println()
    [print(" ") for i=1:initial_blank_len]; initial_blank_len-=2;
    print(tl[1])
    [print(" ") for i=1:blank_len]
    print(tr[1]) #children
    println()
    [print(" ") for i=1:initial_blank_len]; initial_blank_len-=2;
    blank_len-=1
    m=2
    nodes=collect(1:non(T)) #nodes
    nodepos=set_positions(T) #node position
    nnmbr=4 #node ncounter
    lvl_old=2
    bingo=0
    for p=4:2^h-1  #loop over all positions in a binary tree
        if bingo == 1 || nnmbr>non(T)
            bingo=0
            continue
        end
        lvl=level(T,nodes[nnmbr])
        if lvl!=lvl_old
            println()
            [print(" ") for i=1:initial_blank_len];  initial_blank_len-=2;
            lvl_old=lvl;
            lvl+=1
            blank_len-=1
        end
        if nodepos[nnmbr]==p
            print(tl[m])
            [print(" ") for i=1:blank_len]
            print(tr[m])
            [print(" ") for i=1:blank_len+1]
            m+=1
            nnmbr+=2
            bingo=1
        else
            #print(" ")
            [print(" ") for i=1:blank_len]
            print("")
        end
    end
end
