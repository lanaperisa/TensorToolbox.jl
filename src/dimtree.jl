export dimtree, children, count_leaves, dims, display, height, isequal, ==, is_leaf, is_left, is_right
export left_child_length, lvl, nodes_on_lvl, node2ind, non, parent, positions, show, sibling, structure, subnodes, subtree

"""
    dimtree(leaves::AbstractVector{<:Integer}[,internal_nodes::AbstractVector{<:Integer}])
    dimtree(N[,treetype])

Dimension tree. Create from:
- a vector of leaves (and a vector of internal nodes),
- an order of a tensor N and a type treetype of a dimtree. Default: treetype="balanced".
"""
mutable struct dimtree
  leaves::AbstractVector{<:Integer}
  internal_nodes::AbstractVector{<:Integer}
  function dimtree(leaves::AbstractVector{<:Integer},internal_nodes::AbstractVector{<:Integer})
    @assert(length(leaves)==length(internal_nodes)+1,"Incorrect nodes.")
    nodes=sort([leaves;internal_nodes])
    @assert(nodes==collect(1:maximum(nodes)),"Missing nodes.")
    new(leaves,internal_nodes)
  end
end
#dimtree(leaves::AbstractVector{<:Integer},internal_nodes::AbstractVector{<:Integer})=dimtree{D1,D2}(leaves,internal_nodes)
function dimtree(leaves::AbstractVector{<:Integer})
  N=length(leaves)
  nn=2*N-1
  internal_nodes=setdiff(collect(1:nn),leaves)
  @assert(sort([leaves;internal_nodes])==collect(1:nn),"Wrong input.")
  dimtree(leaves,internal_nodes)
end

function dimtree(N::Integer,treetype="balanced")
  @assert(treetype=="balanced","Only balanced trees are supported.")
  C=zeros(Int,2*N-1,2)
  dims=Array{Any}(undef,2*N-1)
  dims[1]=collect(1:N)
  nn=1
  i=1
  while i<=nn
    if length(dims[i]) == 1
      C[i,:]=[0 0]
    else
      ind_left=nn+1
      ind_right=nn+2
      nn+=2
      C[i,:]=[ind_left,ind_right]
      dims[ind_left]=dims[i][1:ceil(Int,end/2)]
      dims[ind_right]=dims[i][ceil(Int,end/2)+1:end]
    end
    i+=1
  end
  L=findall((in)([0,0]),C[:,1])
  leaves=zeros(Int,N)
  [leaves[dims[l]].=l for l in L]
  dimtree(leaves)
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
    dims(T,node)

Content of a node of a dimtree T.
"""
function dims(T::dimtree)
    dims_mat=Array{Vector}(undef,non(T))
    [dims_mat[T.leaves[n]]=[n] for n=1:length(T.leaves)]
    for n in reverse(T.internal_nodes)
        c=children(T,n)
        dims_mat[n]=[dims_mat[c[1]]...,dims_mat[c[2]]...]
    end
    dims_mat
end
dims(T::dimtree,node::Integer)=dims(T)[node]

"""
    display(T)

Display a dimtree T.
"""
function display(T::dimtree)
  println("Dimensional tree:" )
  if T.leaves==[1]
    println("   1   ")
    return
  end
  t,tl,tr=structure(T)
  h=height(T)
  initial_blank_len=zeros(Int,h)
  a=2
  for l=2:h
    initial_blank_len[l]=initial_blank_len[l-1]+a
    a=a*2
  end
  blank_len=2*initial_blank_len[1:end-1].+1
  l=h;k=h-1;
  [print(" ") for i=1:initial_blank_len[l]]; l-=1
  print(minimum(t[1]),"-",maximum(t[1]))
  println()
  [print(" ") for i=1:initial_blank_len[l]]; l-=1
  length(tl[1]) == 1 ? print(" ",minimum(tl[1])," ") : print(minimum(tl[1]),"-",maximum(tl[1]))
  [print(" ") for i=1:blank_len[k]]; k-=1
  length(tr[1]) == 1 ? print(" ",minimum(tr[1])," ") : print(minimum(tr[1]),"-",maximum(tr[1]))
  println()
  if l>0
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
      length(tl[m]) == 1 ? print(" ",tl[m][1]," ") : print(tl[m][1],"-",maximum(tl[m]))
      [print(" ") for i=1:blank_len[k]]
      length(tr[m]) == 1 ? print(" ",tr[m][1]," ") : print(tr[m][1],"-",maximum(tr[m]))
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
  if T1.leaves==T2.leaves
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
    ind=findall(children(T)[:,1].==node)
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
    ind=findall(children(T)[:,2].==node)
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
lvl(T::dimtree,node::Integer)=lvl(T)[node]
function lvl(T::dimtree,nodes::AbstractVector{<:Integer})
    [lvl(T)[n] for n in nodes]
end

"""
    nodes_on_lvl(T,l)

Nodes on a level l in a dimtree T.
"""
function nodes_on_lvl(T::dimtree,l::Integer)
    findall(lvl(T).==l)
end

"""
    node2ind(T,nodes)

Convert node numbers to transfer tensor or frames indices in a dimtree T.
"""
function node2ind(T::dimtree,nodes::AbstractVector{<:Integer})
    ind=zeros(Int,length(nodes))
    k=1
    for n in nodes
        if is_leaf(T,n)
            ind[k]=findall((in)(n),T.leaves)[1]
        else
            ind[k]=findall((in)(n),T.internal_nodes)[1]
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
    P[n]=CartesianIndices(C)[findall(C.==n)][1][1]
  end
  P
end
parent(T::dimtree,node::Integer)=parent(T)[node]
parent(T::dimtree,nodes::AbstractVector{<:Integer})=[parent(T)[n] for n in nodes]

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
  ind=findall(C.==node)[1]
  if ind[2]==1
      C[ind[1]...,2]
  else
      C[ind[1]...,1]
  end
end

"""
    structure(T)

For each internal node of a given dimtree, returns its representation t, its left child and right child representations tl and tr.
"""
function structure(T::dimtree)
  t=dims(T)[T.internal_nodes]
  tl=dims(T)[children(T)[T.internal_nodes,1]]
  tr=dims(T)[children(T)[T.internal_nodes,2]]
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

Dimensional tree with the same structure as a subtree of a node in a dimtree T. See also: subnodes.
"""
function subtree(T::dimtree,node::Integer)
  if is_leaf(T,node)
    return dimtree([1],Int[])
  end
  nodes=subnodes(T,node)
  #L=T.leaves[findin(T.leaves,nodes)]
  L=T.leaves[findall((in)(nodes),T.leaves)]
  #I=T.internal_nodes[findin(T.internal_nodes,nodes)]
  I=T.internal_nodes[findall((in)(nodes),T.internal_nodes)]
  dimtree(sortperm(L).+length(I))
end
