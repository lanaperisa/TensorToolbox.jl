export dimtree, children, count_leaves, create_dimtree, dims, display, height, isequal, ==, is_leaf, is_left, is_right
export left_child_length, lvl, nodes_on_lvl, node2ind, non, parent, positions, show, sibling, sort_leaves, structure, subnodes, subtree

"""
    dimtree(leaves::Vector{Int},internal_nodes::Vector{Int})
    dimtree(children::Matrix{Int})
    create_tree(N;leaves=[],internal_nodes=[])

Dimension tree. Create by:
- vectors of leaves and internal nodes,
- matrix of children,
- order of a tensor and one of leaves or internal_nodes.
"""
type dimtree
  leaves::Vector{Int}
  internal_nodes::Vector{Int}
  function dimtree(leaves::Vector{Int},internal_nodes::Vector{Int})
    @assert(length(leaves)==length(internal_nodes)+1,"Incorrect nodes.")
    nodes=sort([leaves;internal_nodes])
    @assert(nodes==collect(1:maximum(nodes)),"Missing nodes.")
    new(leaves,internal_nodes)
  end
end
function dimtree(children::Matrix{Int})
  @assert(size(children,2)==2,"Incorrect children matrix.")
  @assert(!any([(children[i,1]==0 && children[i,2]!=0) || (children[i,1]!=0 && children[i,2]==0) for i=1:size(children,1)]),"Incorrect children matrix.")
  leaves=findn(children[:,1].==0)
  internal_nodes=findn(children[:,1].!=0)
  sort_leaves(dimtree(leaves,internal_nodes))
end


function dimtree(N::Integer;leaves=[],internal_nodes=[])
    @assert(length(leaves)>0 || length(internal_nodes)>0,"Too few input arguments.")
    nn=2*N-1
    if length(leaves)>0
        internal_nodes=setdiff(collect(1:nn),leaves)
    else
        leaves=setdiff(collect(1:nn),internal_nodes)
    end
    @assert(sort([leaves;internal_nodes])==collect(1:nn),"Wrong input.")
    sort_leaves(dimtree(leaves,internal_nodes))
end

"""
    children(T[,node])

Matrix of children for each node of a dimtree T.
"""
function children(T::dimtree)
    l=length(T.leaves)+length(T.internal_nodes)
    C=zeros(Int,l,2)
    C[1,:]=[2 3]
    next_node=4
    for n=2:l
        if n in T.internal_nodes
            C[n,:]=[next_node next_node+1]
            next_node+=2
        end
    end
    C
end
function children(T::dimtree,node::Integer)
    children(T)[node,:]
end

"""
    count_leaves(T,node)

Number of leaves under a node of a dimtree T.
"""
function count_leaves(T::dimtree,node::Int)
  @assert(0<node<=non(T),"Input exceeds number of nodes.")
  nodes=subnodes(T,node)
  l=0
  for n in nodes
    if is_leaf(T,n)
      l+=1
    end
  end
  l
end

"""
    create_dimtree(N[,treetype])

Create a dimtree of type treetype for a tensor of order N. Default: treetype="balanced".
"""
function create_dimtree(N::Integer,treetype="balanced")
  @assert(treetype=="balanced","Only balanced trees are supported.")
  nmbr_nodes=2*N-1
  Tmat=zeros(Int,nmbr_nodes,2)
  Tmat[1,:]=[2,3]
  node_count=4
  parent=zeros(Int,nmbr_nodes)
  parent[1]=0;parent[2]=1;parent[3]=1
  node_length=zeros(Int,nmbr_nodes)
  node_length[1]=N
  for node=2:nmbr_nodes
    if iseven(node)
      node_length[node]=ceil(Int,node_length[parent[node]]/2)
      if node_length[node] > 1
        Tmat[node,:]=[node_count;node_count+1]
        parent[node_count]=node
        parent[node_count+1]=node
        node_count+=2
      end
    else
      node_length[node]=floor(Int,node_length[parent[node]]/2)
      if node_length[node] > 1
        Tmat[node,:]=[node_count;node_count+1]
        parent[node_count]=node
        parent[node_count+1]=node
        node_count+=2
      end
    end
  end
  dimtree(Tmat)
end
create_dimtree{T<:Number,N}(X::Array{T,N},treetype="balanced")=create_dimtree(N,treetype)

"""
    dims(T,node)

Content of a node of a dimtree T.
"""
function dims(T::dimtree,node::Integer)
    t,tl,tr=structure(T)
    internal_ind=findin(T.internal_nodes,node)
    if length(internal_ind)>0
        t[internal_ind[1]]
    else
        p=parent(T,node)
        internal_ind=findin(T.internal_nodes,p)
        if is_left(T,node)
            tl[internal_ind[1]]
        else
            tr[internal_ind[1]]
        end
    end
end

"""
    display(T)

Display a dimtree T.
"""
function display(T::dimtree)
  t,tl,tr=structure(T)
  h=height(T)
  initial_blank_len=zeros(Int,h)
  a=2
  for l=2:h
    initial_blank_len[l]=initial_blank_len[l-1]+a
    a=a*2
  end
  blank_len=2*initial_blank_len[1:end-1]+1
  l=h;k=h-1;
  [print(" ") for i=1:initial_blank_len[l]]; l-=1
  print(t[1][1],"-",t[1][end])
  println()
  [print(" ") for i=1:initial_blank_len[l]]; l-=1
  length(tl[1]) == 1 ? print(" ",tl[1][1]," ") : print(tl[1][1],"-",tl[1][end])
  [print(" ") for i=1:blank_len[k]]; k-=1
  length(tr[1]) == 1 ? print(" ",tr[1][1]," ") : print(tr[1][1],"-",tr[1][end])
  println()
  [print(" ") for i=1:initial_blank_len[l]]; l-=1
  m=2
  nodes=collect(1:non(T)) #nodes
  nodepos=positions(T) #node position
  nnmbr=4 #node counter
  level_old=2
  bingo=0
  for p=4:2^h-1  #loop over all positions in a binary tree
    if bingo == 1 || nnmbr>non(T)
      bingo=0
      continue
    end
    level=lvl(T,nodes[nnmbr])
     if level!=level_old && p==2^level
       println()
       [print(" ") for i=1:initial_blank_len[l]];  l-=1
       level_old=level;
       level+=1
       k-=1
     end
    #println("p = $p, nodepos[$nnmbr] = $(nodepos[nnmbr])")
    if nodepos[nnmbr]==p
      length(tl[m]) == 1 ? print(" ",tl[m][1]," ") : print(tl[m][1],"-",tl[m][end])
      [print(" ") for i=1:blank_len[k]]
      length(tr[m]) == 1 ? print(" ",tr[m][1]," ") : print(tr[m][1],"-",tr[m][end])
      [print(" ") for i=1:blank_len[k]]
      m+=1
      nnmbr+=2
      bingo=1
    else
      #print(" ")
      [print(" ") for i=1:3+blank_len[k]]
      #print("")
    end
  end
  println()
end

"""
    height(T)

Height of a dimtree T.
"""
function height(T::dimtree)
    maximum(lvl(T))+1
end

"""
    isequal(T1,T2)

True if two dimensional trees are equal, false otherwise.
"""
function isequal(T1::dimtree,T2::dimtree)
  if children(T1)==children(T2)
    return true
  else
    return false
  end
end
==(T1::dimtree,T2::dimtree)=isequal(T1,T2)

"""
    is_leaf(T,node)

True if a node is a leaf in a dimtree T. False otherwise.
"""
function is_leaf(T::dimtree,node::Integer)
   node in T.leaves ? true : false
end

"""
    is_left(T,node)

True if a node is a left node of dimtree T. False otherwise.
"""
function is_left(T::dimtree,node::Integer)
    ind=findn(children(T)[:,1].==node)
    if length(ind)>0
        return true
    else
        return false
    end
end

"""
    is_right(T,node)

True if a node is a right node of dimtree T. False otherwise.
"""
function is_right(T::dimtree,node::Integer)
    ind=findn(children(T)[:,2].==node)
    if length(ind)>0
        return true
    else
        return false
    end
end

"""
    left_child_length(T[,node])

Length of the left child of a node of a dimtree T. Default: node=1.
"""
function left_child_length(T::dimtree,node::Int)
    if node in T.leaves
        return 0
    else
        return count_leaves(T,children(T)[node,1])
    end
end
left_child_length(T::dimtree)=left_child_length(T,1)

"""
    lvl(T[,node])

Vector of levels for each node of a dimtree T.
"""
function lvl(T::dimtree)
    nn=non(T)
    L=zeros(Int,nn)
    for n in collect(2:nn)
        L[n]=L[parent(T,n)]+1
    end
    L
end
function lvl(T::dimtree,node::Integer)
    lvl(T)[node]
end

"""
    nodes_on_lvl(T,l)

Nodes on a level l in a dimtree T.
"""
function nodes_on_lvl(T::dimtree,l::Integer)
    findn(lvl(T).==l)
end

"""
    node2ind(T,nodes)

Convert node numbers to transfer tensor or frames indices ina dimtree T.
"""
function node2ind(T::dimtree,nodes::Vector{Int})
    ind=zeros(Int,length(nodes))
    k=1;
    for n in nodes
        if is_leaf(T,n)
            ind[k]=findin(T.leaves,n)[1]
        else
            ind[k]=findin(T.internal_nodes,n)[1]
        end
        k+=1
    end
    ind
end
node2ind(T::dimtree,nodes::Integer)=node2ind(T,[nodes])[1]
node2ind(T::dimtree)=node2ind(T,collect(1:non(T)))

"""
    non(T)

Number of nodes of a dimtree T.
"""
function non(T::dimtree)
  maximum([T.leaves;T.internal_nodes])
end

"""
    parent(T[,node])

Vectors of parents for each node of a dimtree T.
"""
function parent(T::dimtree)
  nn=non(T)
  C=children(T)
  P=zeros(Int,nn) #parent[node]
  P[1]=0
  for n=2:nn
    P[n]=ind2sub(C,find(x->x==n,C))[1][1];
  end
  P
end
function parent(T::dimtree,node::Integer)
    parent(T)[node]
end

"""
    positions(T)

Positions of nodes of a dimtree T in a full binary tree.
"""
function positions(T::dimtree)
    h=height(T)
    pos_nmbr=2^h-1
    position=zeros(Int,non(T))
    position[1]=1;
    i=2;
    maxit=100;
    for l=1:h
        nds=nodes_on_lvl(T,l)
        if length(nds)==2^l
            for n in nds
                position[n]=i
                i+=1
            end
        else
            for n in nds
                if position[n]!=0
                    continue
                end
                p=parent(T,n)
                while i!=position[p]*2 && i<maxit
                    i+=2
                end
                position[n]=i
                i+=1
                position[n+1]=i
                i+=1
                n+=1
            end
        end
    end
    position
end

function Base.show(T::dimtree)
    display(T)
end

"""
    sibling(T,node)

Sibling of a node in a dimtree T.
"""
function sibling(T::dimtree,node::Integer)
  C=children(T)
  i,j=findn(C.==node)
  if j==[1]
      C[i...,2]
  else
      C[i...,1]
  end
end

"""
    sort_leaves(T)

Sort leaves such that leaves[m] represents mode m.
"""
function sort_leaves(T::dimtree)
  L=[]
  for l in T.leaves
    L=[L;dims(T,l)[1]]
  end
  leaves=T.leaves[sortperm(L)]
  dimtree(leaves,T.internal_nodes)
end

"""
    structure(T)

For each internal node of a given dimtree, returns its representation t, its left child and right child representations tl and tr.
"""
function structure(T::dimtree)
  llen=length(T.leaves)
  inlen=length(T.internal_nodes)
  t=VectorCell(inlen)
  tl=VectorCell(inlen)
  tr=VectorCell(inlen)
  t[1]=collect(1:llen)
  i=2;
  for n=1:inlen
    l=left_child_length(T,T.internal_nodes[n])
    tl[n]=collect(t[n][1]:t[n][1]+l-1)
    tr[n]=collect(t[n][1]+l:t[n][1]+length(t[n])-1)
    if length(tl[n])>1
        t[i]=tl[n]
        i+=1
    end
     if length(tr[n])>1
        t[i]=tr[n]
        i+=1
    end
  end
  t,tl,tr
end

"""
    subnodes(T,node)

Nodes in the subtree of a node in a dimtree T.
"""
function subnodes(T::dimtree,node::Integer)
    sub_ind=[node]
    n=1
    for i=node:non(T)
        if !is_leaf(T,sub_ind[n])
            sub_ind=[sub_ind; children(T,sub_ind[n])]
        end
        if n!=length(sub_ind)
            n+=1
        end
    end
    sub_ind
end

"""
    subtree(T,node)

Dimensional tree with the same structure as a subtree of a node in a dimtree T.
"""
function subtree(T::dimtree,node::Integer)
    if is_leaf(T,node)
        return [0 0]
    end
    nodes=subnodes(T,node)
    nodes_nmbr=length(nodes)
    C=zeros(Int,nodes_nmbr,2)
    C[1,:]=[2 3]
    i=4
    for n=2:nodes_nmbr
        if is_leaf(T,nodes[n])
            C[n,:]=[0 0]
        else
            C[n,:]=[i i+1]
            i+=2
        end
    end
  dimtree(C)
end
